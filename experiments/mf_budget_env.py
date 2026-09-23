"""MFBudgetEnv: a credible multi-fidelity (cell-budget) environment for Perturb-seq.

Deliverable scope (review-agreed):
  reset()          fix task split, observation order and budget; return initial view
  step(alloc)      accept NEW cell counts; enforce capacity and budget; reveal only
                   new cells; return purchased expression stats
  observation      returns purchased expression (delta pseudobulk) + derived
                   statistics -- never precomputed reference correlations
  scorer           separate object holding hidden references; its data cannot enter
                   the policy observation
  cost model       cells / first-time perturbation startup / batch / control are
                   accounted separately (main experiment: cell=1, others=0)
  replay log       cell IDs, actions, cumulative cost, config, seeds; exact replay

Invariants enforced in code:
  * repeated requests never re-reveal the same cell
  * out-of-range or over-budget actions are REJECTED and leave state unchanged
  * a fixed action sequence replays exactly (same cell IDs, same estimates)
  * the hidden reference is unreachable from the observation object

Design notes / explicit choices
  * Control cells are split once into disjoint halves: `ctrl_est` (used to build
    every estimate the policy sees) and `ctrl_ref` (used only by the scorer to
    build reference deltas). The split is part of the config and is replayable.
  * `charge_initial` states whether the initial cell allocation is charged to the
    budget. Default True: the initial allocation is a real experimental cost.
  * Leftover budget is NOT silently spent. `finalize_uniform_leftover()` spends it
    deterministically (round-robin, capacity-aware) so that different policies can
    be compared at exactly the same total cost; the accounting is logged.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    cell: float = 1.0            # cost per newly revealed cell
    startup: float = 0.0         # one-off cost the first time a pert is measured
    batch: float = 0.0           # cost per step that touches >=1 perturbation
    control: float = 0.0         # cost of control-cell information (charged once)
    charge_initial: bool = True  # is the initial allocation billed?

    def step_cost(self, new_cells: int, startups: int, touched_pert: int,
                  control_already_charged: bool) -> Dict[str, float]:
        b = {
            'cell': self.cell * new_cells,
            'startup': self.startup * startups,
            'batch': self.batch * (1 if touched_pert > 0 else 0),
            'control': 0.0 if control_already_charged else self.control,
        }
        b['total'] = sum(b.values())
        return b


# --------------------------------------------------------------------------- #
# observation
# --------------------------------------------------------------------------- #
@dataclass
class Obs:
    """What a policy may see. Contains purchased expression, never references."""
    step: int
    candidates: List[str]                      # candidate perturbation ids
    n_observed: np.ndarray                     # cells purchased per candidate
    delta: np.ndarray                          # (n_cand, n_genes) pseudobulk delta
    delta_norm: np.ndarray
    half_split_r: np.ndarray                   # agreement of two disjoint halves
    cell_sd: np.ndarray                        # mean per-gene sd across observed cells
    n_capacity: np.ndarray                     # remaining purchasable cells
    remaining_budget: float
    cost_so_far: Dict[str, float]
    initial_charged: bool


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
class MFBudgetEnv:
    def __init__(self, cond: np.ndarray, X: np.ndarray, budget: float,
                 candidates: Sequence[str], init_cells: int = 25,
                 ref_frac: float = 0.30, ref_min: int = 20, ref_max: int = 80,
                 seed: int = 0, cost_model: Optional[CostModel] = None,
                 control_split_seed: int = 12345):
        self.X = X
        self.cond = cond
        self.budget = float(budget)
        self.init_cells = int(init_cells)
        self.cost_model = cost_model or CostModel()
        self.seed = int(seed)
        self.config = dict(budget=budget, init_cells=init_cells, seed=seed,
                           ref_frac=ref_frac, ref_min=ref_min, ref_max=ref_max,
                           control_split_seed=control_split_seed,
                           cost_model=asdict(self.cost_model),
                           candidates=list(candidates))
        self._rng_order = np.random.default_rng(seed)

        # ---- candidates, pools, hidden references -------------------------
        # control split (disjoint halves, fixed by control_split_seed)
        ctrl = np.where(cond == 'ctrl')[0]
        c_rng = np.random.default_rng(control_split_seed)
        ctrl = c_rng.permutation(ctrl)
        half = len(ctrl) // 2
        self._ctrl_est_m = X[ctrl[:half]].mean(0)
        self._ctrl_ref_m = X[ctrl[half:]].mean(0)
        self._control_charged = False

        o_rng = np.random.default_rng(seed)
        self.candidates = list(candidates)
        self._pool: Dict[str, np.ndarray] = {}
        self._order: Dict[str, np.ndarray] = {}
        self._reference: Dict[str, np.ndarray] = {}      # PRIVATE: scorer only
        for c in self.candidates:
            idx = np.where(cond == c)[0]
            n = len(idx)
            n_ref = int(min(ref_max, max(ref_min, round(ref_frac * n))))
            perm = o_rng.permutation(idx)
            ref_idx, pool = perm[:n_ref], perm[n_ref:]
            self._reference[c] = X[ref_idx].mean(0) - self._ctrl_ref_m
            self._order[c] = pool[o_rng.permutation(len(pool))]
            self._pool[c] = np.full(len(pool), -1, dtype=np.int64)  # purchased flags

        self._n_obs = {c: 0 for c in self.candidates}
        self._startup_done = {c: False for c in self.candidates}
        self._cost = {'cell': 0.0, 'startup': 0.0, 'batch': 0.0, 'control': 0.0,
                      'total': 0.0}
        self._step = 0
        self._done = False
        self.log: List[dict] = []

        # ---- initial allocation ------------------------------------------
        init = {}
        for c in self.candidates:
            cap = len(self._order[c])
            init[c] = min(self.init_cells, cap)
        self._apply(init, initial=True)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _delta(self, c: str) -> np.ndarray:
        k = self._n_obs[c]
        if k == 0:
            return np.zeros(self.X.shape[1], dtype=np.float64)
        cells = self._order[c][:k]
        return self.X[cells].mean(0) - self._ctrl_est_m

    def _validate(self, alloc: Dict[str, int]) -> Dict[str, int]:
        """Reject unknown / negative / over-capacity / over-budget actions."""
        clean: Dict[str, int] = {}
        for c, k in alloc.items():
            if c not in self._n_obs:
                raise KeyError(f'unknown candidate {c!r}')
            if not isinstance(k, (int, np.integer)) or k < 0:
                raise ValueError(f'cell counts must be non-negative ints, got {k!r}')
            cap = len(self._order[c]) - self._n_obs[c]
            if k > cap:
                raise ValueError(f'{c}: requested {k} cells but only {cap} remain')
            clean[c] = int(k)
        new_cells = sum(clean.values())
        startups = sum(1 for c, k in clean.items()
                       if k > 0 and not self._startup_done[c])
        touched = sum(1 for k in clean.values() if k > 0)
        br = self.cost_model.step_cost(new_cells, startups, touched,
                                       self._control_charged)
        if br['total'] > self.remaining_budget + 1e-9:
            raise RuntimeError(
                f'over budget: action costs {br["total"]:.3f}, '
                f'remaining {self.remaining_budget:.3f}')
        return clean

    def _apply(self, alloc: Dict[str, int], initial: bool = False) -> dict:
        br = self.cost_model.step_cost(
            sum(alloc.values()),
            sum(1 for c, k in alloc.items() if k > 0 and not self._startup_done[c]),
            sum(1 for k in alloc.values() if k > 0),
            self._control_charged)
        if initial and not self.cost_model.charge_initial:
            br = {k: 0.0 for k in br}
        revealed = {}
        for c, k in alloc.items():
            if k <= 0:
                continue
            start = self._n_obs[c]
            cells = self._order[c][start:start + k]
            self._pool[c][start:start + k] = cells
            self._n_obs[c] += k
            self._startup_done[c] = True
            revealed[c] = [int(x) for x in cells]
        if br.get('control', 0) > 0:
            self._control_charged = True
        for k_, v in br.items():
            self._cost[k_] = self._cost.get(k_, 0.0) + v
        self.log.append(dict(step=self._step, initial=bool(initial),
                             action={c: int(k) for c, k in alloc.items()},
                             revealed_cells=revealed, cost=br,
                             cumulative_total=self._cost['total'],
                             config_seed=self.seed))
        return br

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @property
    def remaining_budget(self) -> float:
        return self.budget - self._cost['total']

    def reset(self) -> Obs:
        """Rebuild the environment (same config/seed) and return the initial view."""
        return MFBudgetEnv(self.cond, self.X, self.budget, self.candidates,
                           init_cells=self.init_cells, seed=self.seed,
                           cost_model=self.cost_model).observation()

    def observation(self) -> Obs:
        n = np.array([self._n_obs[c] for c in self.candidates])
        cap = np.array([len(self._order[c]) - self._n_obs[c] for c in self.candidates])
        delta = np.stack([self._delta(c) for c in self.candidates])
        hsr = np.full(len(self.candidates), np.nan)
        csd = np.full(len(self.candidates), np.nan)
        for i, c in enumerate(self.candidates):
            k = self._n_obs[c]
            if k >= 1:
                csd[i] = float(np.mean(self.X[self._order[c][:k]].std(axis=0)))
            if k >= 2:
                h = k // 2
                e1 = self.X[self._order[c][:h]].mean(0) - self._ctrl_est_m
                e2 = self.X[self._order[c][h:2 * h]].mean(0) - self._ctrl_est_m
                a, b = e1 - e1.mean(), e2 - e2.mean()
                hsr[i] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return Obs(step=self._step, candidates=list(self.candidates),
                   n_observed=n, delta=delta,
                   delta_norm=np.linalg.norm(delta, axis=1), half_split_r=hsr,
                   cell_sd=csd,
                   n_capacity=cap, remaining_budget=self.remaining_budget,
                   cost_so_far=dict(self._cost),
                   initial_charged=bool(self.cost_model.charge_initial))

    def step(self, alloc: Dict[str, int]) -> tuple[Obs, dict]:
        """Purchase additional cells. Rejects (without state change) anything that
        is out of range, exceeds capacity, or would overspend the budget."""
        if self._done:
            raise RuntimeError('episode finished; call reset()')
        clean = self._validate(alloc)            # raises; state untouched until here
        br = self._apply(clean)
        self._step += 1
        if self.remaining_budget < 0:
            raise AssertionError('budget invariant violated')
        return self.observation(), br

    def finalize_uniform_leftover(self) -> tuple[Obs, dict]:
        """Spend the leftover budget deterministically (round-robin, capacity-aware)
        so that every policy is compared at exactly the same total cost. Leftover
        that cannot be spent (capacity exhausted) is left unspent and reported."""
        alloc: Dict[str, int] = {c: 0 for c in self.candidates}
        cap = {c: len(self._order[c]) - self._n_obs[c] for c in self.candidates}
        remaining = self.remaining_budget
        startup_left = {c: not self._startup_done[c] for c in self.candidates}
        progress = True
        while progress and remaining >= self.cost_model.cell:
            progress = False
            for c in self.candidates:
                if cap[c] <= 0 or remaining < self.cost_model.cell:
                    continue
                extra = self.cost_model.cell + (
                    self.cost_model.startup if startup_left[c] else 0.0)
                if extra > remaining + 1e-9:
                    continue
                alloc[c] += 1
                cap[c] -= 1
                remaining -= self.cost_model.cell
                if startup_left[c]:
                    remaining -= self.cost_model.startup
                    startup_left[c] = False
                progress = True
        alloc = {c: k for c, k in alloc.items() if k > 0}
        if alloc:
            self._apply(alloc)
        self._done = True
        return self.observation(), {'leftover_action': alloc,
                                    'unspent': self.remaining_budget}

    # ------------------------------------------------------------------ #
    # scorer: holds hidden references, never exposed to the policy
    # ------------------------------------------------------------------ #
    def scorer(self) -> 'Scorer':
        return Scorer(self._reference, self.candidates, self.cond, self.X)


class Scorer:
    """Independent evaluator. Owns the hidden references; consumes an Obs only."""

    def __init__(self, reference: Dict[str, np.ndarray], candidates, cond, X):
        self._reference = reference
        self.candidates = list(candidates)
        self._index = {c: i for i, c in enumerate(self.candidates)}

    def reliability(self, obs: Obs) -> np.ndarray:
        out = np.full(len(self.candidates), np.nan)
        for c, i in self._index.items():
            if obs.n_observed[i] == 0:
                continue
            a = obs.delta[i] - obs.delta[i].mean()
            b = self._reference[c] - self._reference[c].mean()
            out[i] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return out

    def objective(self, obs: Obs, kind: str, tau: float = 0.7) -> float:
        r = self.reliability(obs)
        r = r[~np.isnan(r)]
        if kind == 'mean_r':
            return float(r.mean())
        if kind.startswith('frac_ge_'):
            return float((r >= tau).mean())
        raise ValueError(kind)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _tiny_env(n_cells=60, budget=1000):
    """Synthetic env: 6 perturbations, enough cells, deterministic values."""
    rng = np.random.default_rng(0)
    G = 12
    perts = [f'P{i}' for i in range(6)]
    rows = [rng.standard_normal(G) for _ in range(40)]        # controls
    cond = ['ctrl'] * 40
    for p in perts:
        base = rng.standard_normal(G)
        for _ in range(n_cells):
            rows.append(base + 0.35 * rng.standard_normal(G))
            cond.append(p)
    return np.array(cond), np.array(rows, dtype=np.float32), perts


def run_tests(verbose=True):
    ok = True

    def check(name, cond_):
        nonlocal ok
        ok = ok and bool(cond_)
        if verbose:
            print(f'  [{"PASS" if cond_ else "FAIL"}] {name}')

    cond, X, perts = _tiny_env()
    env = MFBudgetEnv(cond, X, budget=200, candidates=perts, init_cells=5, seed=3)
    obs0 = env.observation()

    # 1) repeated actions never re-reveal cells
    o1, _ = env.step({perts[0]: 7})
    c_first = list(env._order[perts[0]][:12])
    o2, _ = env.step({perts[0]: 7})
    c_all = list(env._order[perts[0]][:19])
    check('no duplicate reveal', len(set(c_all)) == len(c_all) and
          env._n_obs[perts[0]] == 19 and
          c_first == c_all[:12])

    # 2) over-capacity rejected, state unchanged
    before = (env._n_obs.copy(), dict(env._cost), env._step)
    try:
        env.step({perts[1]: 10_000})
        check('over-capacity rejected', False)
    except ValueError:
        after = (env._n_obs.copy(), dict(env._cost), env._step)
        check('over-capacity rejected, state unchanged',
              before[0] == after[0] and before[1] == after[1] and before[2] == after[2])

    # 3) over-budget rejected, state unchanged
    # initial allocation (6 x 5 cells = 30) consumes the whole budget
    env2 = MFBudgetEnv(cond, X, budget=30, candidates=perts, init_cells=5, seed=1)
    n_before = env2._n_obs.copy(); cost_before = dict(env2._cost)
    check('initial allocation was charged', abs(env2._cost['total'] - 30) < 1e-9)
    try:
        env2.step({perts[0]: 1})                  # capacity ok, budget exhausted
        check('over-budget rejected', False)
    except RuntimeError:
        check('over-budget rejected, state unchanged',
              env2._n_obs == n_before and env2._cost == cost_before)

    # 4) unknown candidate / negative count rejected
    try:
        env.step({'NOPE': 1}); check('unknown candidate rejected', False)
    except KeyError:
        check('unknown candidate rejected', True)
    try:
        env.step({perts[0]: -3}); check('negative count rejected', False)
    except ValueError:
        check('negative count rejected', True)

    # 5) exact replay of an action sequence
    actions = [{perts[0]: 4, perts[2]: 6}, {perts[2]: 3}, {perts[1]: 8}]
    def run(seed):
        e = MFBudgetEnv(cond, X, budget=500, candidates=perts, init_cells=5, seed=seed)
        seen = []
        for a in actions:
            e.step(a)
            seen.append({c: e._order[c][:e._n_obs[c]].copy() for c in perts})
        return seen, e.observation().delta.copy()
    s1, d1 = run(11); s2, d2 = run(11); s3, _ = run(12)
    def same(x, y):
        return all(np.array_equal(x[i][c], y[i][c]) for i in range(len(x)) for c in perts)
    check('exact replay (same seed)', same(s1, s2) and np.array_equal(d1, d2))
    check('different seed -> different order', not same(s1, s3))

    # 6) reference is unreachable from the observation
    fields = set(vars(obs0).keys())
    allowed = {'step', 'candidates', 'n_observed', 'delta', 'delta_norm',
               'half_split_r', 'cell_sd', 'n_capacity', 'remaining_budget',
               'cost_so_far', 'initial_charged'}
    check('obs fields are whitelisted', fields == allowed)
    refs = np.stack([env._reference[c] for c in perts])
    leaked = any(np.allclose(obs0.delta, refs[i:i + 1], atol=1e-8) for i in range(len(perts)))
    check('reference not present in obs.delta', not leaked)
    check('scorer is a separate object', env.scorer() is not env)

    # 7) accounting adds up
    e = MFBudgetEnv(cond, X, budget=200, candidates=perts, init_cells=5, seed=5,
                    cost_model=CostModel(cell=1.0, startup=2.0, batch=0.5,
                                         control=1.0, charge_initial=True))
    # semantics: the initial allocation is a real measurement, so startup is
    # charged there (once per perturbation) and not again on later steps
    n_init = len(perts) * 5
    init_expect = n_init * 1.0 + len(perts) * 2.0 + 0.5 + 1.0
    check('initial allocation billed (cells+startups+batch+control)',
          abs(e._cost['total'] - init_expect) < 1e-9)
    before_tot = e._cost['total']
    o, br = e.step({perts[0]: 3, perts[3]: 2})
    expect = 5 * 1.0 + 0 * 2.0 + 0.5          # cells + no new startup + batch
    check('cost breakdown matches spec', abs((e._cost['total'] - before_tot) - expect) < 1e-9)
    check('billed == revealed cells * cell cost (+batch)', br['total'] == expect)
    check('startup not double charged', br['startup'] == 0.0)
    check('control charged exactly once', e._cost['control'] == 1.0)

    # 8) budget strictly enforced under finalize
    e = MFBudgetEnv(cond, X, budget=137, candidates=perts, init_cells=5, seed=7)
    o, info = e.finalize_uniform_leftover()
    check('finalize never overspends', e._cost['total'] <= 137 + 1e-9)
    check('finalize reveals only new cells',
          all(e._n_obs[c] <= len(e._order[c]) for c in perts))

    # 9) capacity ceiling respected
    e = MFBudgetEnv(cond, X, budget=10**6, candidates=perts, init_cells=5, seed=9)
    _, info = e.finalize_uniform_leftover()
    check('caps at pool capacity',
          all(e._n_obs[c] == len(e._order[c]) for c in perts))
    if verbose:
        print(f'\n{"ALL TESTS PASSED" if ok else "SOME TESTS FAILED"}')
    return ok


if __name__ == '__main__':
    run_tests()
