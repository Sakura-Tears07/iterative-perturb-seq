"""Stage 8: acceptance experiment ON THE ENVIRONMENT (real K562 data).

Runs the review-agreed acceptance test inside MFBudgetEnv, so the same run both
integrates the environment and screens the science question:

    "does a feedback-driven allocation beat the best fixed allocation chosen on
     a development split, at equal total cell budget, on a held-out test split?"

Protocol
    * perturbations split once into dev / test (probe fitted on dev only)
    * test candidates are allocated, billed and scored only through the env
    * exact integer cells via env.step(); leftover budget is spent by the env's
      fixed round-robin rule, so every policy ends at the SAME total cost
    * per-objective oracle for the two-level (25 -> D) family: ranks by the
      marginal value of that objective, NOT by gain. It is the best allocation
      WITHIN the two-level family, not a global upper bound.
    * results reported per split (paired differences) with mean and sample sd;
      this is resampling stability on one dataset, not statistical significance.

Usage: iterpert_env/bin/python stage8_env_acceptance.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/lihaoran/ai4s/analysis')
from mf_budget_env import MFBudgetEnv, CostModel  # noqa: E402

H5AD = ('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
        'replogle_k562_essential_1000hvg/perturb_processed.h5ad')
OUT = '/home/lihaoran/ai4s/analysis/results'
D_SHALLOW, D_DEEP = 25, 100
SPLITS = [0, 1, 2]
BUDGETS = [40, 60, 90]          # nominal cells per perturbation


def main():
    import anndata as ad
    adata = ad.read_h5ad(H5AD, backed='r')
    cond = adata.obs['condition'].astype(str).values
    X = np.asarray(adata.to_memory().X, dtype=np.float32)
    perts = np.unique(cond)
    # keep perturbations whose pool (70% of cells) can reach D_DEEP
    info = {}
    for c in perts:
        if c == 'ctrl':
            continue
        n = int((cond == c).sum())
        info[c] = n - int(min(80, max(20, round(0.30 * n))))
    cand = [c for c, pool in info.items() if pool >= D_DEEP]
    print(f'candidates with pool>={D_DEEP}: {len(cand)} of {len(info)} perturbations')

    rows = []
    for sp in SPLITS:
        rng = np.random.default_rng(100 + sp)
        perm = rng.permutation(len(cand))
        dev = [cand[i] for i in perm[:int(0.7 * len(cand))]]
        test = [cand[i] for i in perm[int(0.7 * len(cand)):]]
        print(f'\n=== split {sp}: dev={len(dev)} test={len(test)} ===')

        env0 = MFBudgetEnv(cond, X, budget=10 ** 9, candidates=cand, init_cells=0,
                           seed=sp, cost_model=CostModel(cell=1.0, charge_initial=True))
        # --- dev observations used ONLY to fit the probe -------------------
        d_obs = env0.observation()
        idx = {c: i for i, c in enumerate(cand)}
        d_shallow = d_obs.n_observed.copy()
        # measure dev at 25 and 100 cells for probe fitting
        env_dev = MFBudgetEnv(cond, X, budget=10 ** 9, candidates=dev, init_cells=D_SHALLOW,
                              seed=sp)
        o25 = env_dev.observation()
        env_dev.step({c: D_DEEP - D_SHALLOW for c in dev})
        o100 = env_dev.observation()
        sc_dev = env_dev.scorer()
        r25, r100 = sc_dev.reliability(o25), sc_dev.reliability(o100)
        gain = r100 - r25
        Fd = np.c_[o25.half_split_r, o25.delta_norm, o25.cell_sd]
        mu, sd = np.nanmean(Fd, 0), np.nanstd(Fd, 0) + 1e-12
        w = np.linalg.lstsq(np.c_[(Fd - mu) / sd, np.ones(len(dev))], gain,
                            rcond=None)[0]
        from scipy.stats import spearmanr
        print(f'  probe (dev-fitted) spearman on dev gain: '
              f'{spearmanr((Fd-mu)/sd @ w[:-1], gain).statistic:+.3f}')

        for B in BUDGETS:
            # ---------------- build each policy's action sequence ----------
            def run_policy(score=None, rule='uniform'):
                env = MFBudgetEnv(cond, X, budget=float(B * len(test)),
                                  candidates=test, init_cells=D_SHALLOW, seed=sp)
                if rule == 'uniform':
                    env.finalize_uniform_leftover()
                    return env
                k_deep = int((B - D_SHALLOW) * len(test) / (D_DEEP - D_SHALLOW))
                order = np.argsort(-score)[:k_deep]
                alloc = {test[i]: D_DEEP - D_SHALLOW for i in order}
                env.step(alloc)
                env.finalize_uniform_leftover()
                return env

            o_s = env_dev.observation()   # reuse dev measurements as stand-in? no
            # test-side observable features come from a fresh shallow pass
            env_t = MFBudgetEnv(cond, X, budget=float(B * len(test)),
                                candidates=test, init_cells=D_SHALLOW, seed=sp)
            ot = env_t.observation()
            Ft = np.c_[ot.half_split_r, ot.delta_norm, ot.cell_sd]
            score_probe = ((Ft - mu) / sd) @ w[:-1] + w[-1]
            sc_t = env_t.scorer()
            r_s, r_d = sc_t.reliability(ot), None
            # true gain for the diagnostic oracle (needs the hidden reference)
            env_full = MFBudgetEnv(cond, X, budget=10 ** 9, candidates=test,
                                   init_cells=D_DEEP, seed=sp)
            r_deep_true = env_full.scorer().reliability(env_full.observation())
            env_sh = MFBudgetEnv(cond, X, budget=10 ** 9, candidates=test,
                                 init_cells=D_SHALLOW, seed=sp)
            r_sh_true = env_sh.scorer().reliability(env_sh.observation())

            policies = {
                'uniform': None,
                'probe(dev-fitted)': score_probe,
                'est_norm': ot.delta_norm,
                'random': np.random.default_rng(sp).standard_normal(len(test)),
            }
            for obj, tau in [('mean_r', None), ('frac_ge_0.7', 0.7)]:
                if tau is None:
                    marg = r_deep_true - r_sh_true
                else:
                    marg = ((r_deep_true >= tau).astype(float)
                            - (r_sh_true >= tau).astype(float))
                policies[f'oracle_2lvl({obj})'] = marg
            for name, sc in policies.items():
                env = run_policy(score=sc, rule='uniform' if sc is None else 'two_stage')
                o = env.observation()
                scorer = env.scorer()
                for obj, tau in [('mean_r', None), ('frac_ge_0.7', 0.7)]:
                    val = scorer.objective(o, obj, tau=tau if tau else 0.7)
                    rows.append(dict(split=sp, budget=B, policy=name,
                                     objective=obj, value=val,
                                     cost=env._cost['total'],
                                     billed=float(B * len(test))))
        print(f'  split {sp} done')

    import pandas as pd
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'stage8_env_acceptance.csv'), index=False)

    print('\n=== acceptance results (test pool, env-billed, equal total cost) ===')
    for obj in ['mean_r', 'frac_ge_0.7']:
        sub = res[res.objective == obj]
        piv = sub.pivot_table(index='policy', columns=['budget', 'split'], values='value')
        print(f'\n--- objective {obj} ---')
        base = sub[sub.policy == 'uniform'].set_index(['budget', 'split']).value
        diffs = {}
        for name in piv.index:
            d = (sub[sub.policy == name].set_index(['budget', 'split']).value - base).dropna()
            diffs[name] = d
        print(f'{"policy":24s} ' + ' '.join(f'{b:>10d}' for b in BUDGETS) +
              f'{"mean diff":>11s}{"sd":>8s}')
        for name in diffs:
            per_b = [diffs[name].loc[b].mean() for b in BUDGETS if b in diffs[name].index.get_level_values(0)]
            print(f'{name:24s} ' + ' '.join(f'{v:+10.4f}' for v in per_b) +
                  f'{diffs[name].mean():+11.4f}{diffs[name].std():8.4f}')
    print('\ncost check (billed == actual for every row): '
          f'{bool((res.cost.round(6) == res.billed.round(6)).all())}')
    print('wrote', os.path.join(OUT, 'stage8_env_acceptance.csv'))


if __name__ == '__main__':
    main()
