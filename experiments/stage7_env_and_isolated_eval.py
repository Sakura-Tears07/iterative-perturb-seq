"""Stage 7: credible cell-allocation environment + strictly isolated evaluation.

This replaces stage6/6b/6c for decision-making. It fixes every issue raised in
review:

  * NO interpolation, NO cumulative-max smoothing. For every perturbation (and
    every reference split) the reliability curve r(d) is measured EXACTLY at each
    integer depth d = 1..D_max by taking the first d cells of a fixed, shared
    cell order. Strategies can therefore request real integer cell counts.
  * HIDDEN REFERENCE. The reference pseudobulk is built from cells that never
    enter any measurement, and from a disjoint half of the control cells. Only
    the scorer sees it.
  * SHARED ORDER. All strategies observe prefixes of the same permutation, so
    comparisons differ only in the allocation decision, not in sampling noise.
  * STRICT DEV/TEST ISOLATION. Perturbations are split once into dev and test.
    Any fitted rule (the probe) is fitted on dev only; all strategies are
    allocated, billed and scored on the test pool only.
  * STRICT BILLING. Each strategy's allocation is an integer vector; its actual
    average cost is reported next to the target budget.
  * REFERENCE STABILITY. Every number is computed over several independent
    reference splits and reported with its across-split spread.
  * BUDGET CURVES. Per-objective trapezoidal AUC over the budget axis, not an
    average of a few budget points.

Objectives: mean reliability across perturbations, fraction with r >= 0.5,
fraction with r >= 0.7.

Usage:
    iterpert_env/bin/python stage7_env_and_isolated_eval.py --measure   # cache
    iterpert_env/bin/python stage7_env_and_isolated_eval.py             # evaluate
"""
import argparse
import json
import os
import sys

import numpy as np

H5AD = ('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
        'replogle_k562_essential_1000hvg/perturb_processed.h5ad')
OUT = '/home/lihaoran/ai4s/analysis/results'
CACHE_R = os.path.join(OUT, 'stage7_r_curves.npy')
CACHE_F = os.path.join(OUT, 'stage7_features.npy')
CACHE_M = os.path.join(OUT, 'stage7_meta.json')

K_SPLITS = 3
D_MAX = 200
REF_FRAC, MIN_REF, MAX_REF = 0.30, 20, 80
D_SHALLOW, D_DEEP = 25, 100
FEATS = ['half_split_r', 'est_norm', 'cell_sd']


def measure():
    import anndata as ad
    adata = ad.read_h5ad(H5AD, backed='r')
    cond = adata.obs['condition'].astype(str).values
    X = np.asarray(adata.to_memory().X, dtype=np.float32)
    ctrl = np.where(cond == 'ctrl')[0]
    conds = [c for c in np.unique(cond) if c != 'ctrl']
    per = {c: np.where(cond == c)[0] for c in conds}
    print(f'cells={len(cond)} perts={len(conds)}', flush=True)

    meta = []
    curves, feats = [], []
    for k in range(K_SPLITS):
        rng = np.random.default_rng(1000 + k)
        ctrl_p = rng.permutation(ctrl)
        ctrl_est_m = X[ctrl_p[:len(ctrl_p) // 2]].mean(0)
        ctrl_ref_m = X[ctrl_p[len(ctrl_p) // 2:]].mean(0)
        for c in conds:
            idx = per[c]
            n = len(idx)
            n_ref = int(min(MAX_REF, max(MIN_REF, round(REF_FRAC * n))))
            pool_n = n - n_ref
            if k == 0:
                meta.append(dict(condition=str(c), n_cells=int(n), n_pool=int(pool_n)))
            if pool_n < 1:
                # keep row alignment across splits: pad with NaN
                curves.append(np.full(D_MAX, np.nan, np.float32))
                feats.append([np.nan] * len(FEATS))
                continue
            perm = rng.permutation(idx)
            ref = X[perm[:n_ref]].mean(0) - ctrl_ref_m
            pool = perm[n_ref:][rng.permutation(pool_n)]
            dm = min(D_MAX, pool_n)
            cs = np.cumsum(X[pool[:dm]], axis=0, dtype=np.float64)
            refc = ref - ref.mean()
            refn = np.linalg.norm(refc) + 1e-12
            rr = np.full(D_MAX, np.nan, np.float32)
            for d in range(1, dm + 1):
                est = cs[d - 1] / d - ctrl_est_m
                ec = est - est.mean()
                rr[d - 1] = float(ec @ refc / (np.linalg.norm(ec) * refn + 1e-12))
            row = [rr]
            # online features from the SAME first 25 cells every strategy sees
            if pool_n >= 50:
                obs = pool[:D_SHALLOW]
                e1 = X[obs[:D_SHALLOW // 2]].mean(0) - ctrl_est_m
                e2 = X[obs[D_SHALLOW // 2:]].mean(0) - ctrl_est_m
                a, b = e1 - e1.mean(), e2 - e2.mean()
                hsr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                est25 = cs[D_SHALLOW - 1] / D_SHALLOW - ctrl_est_m
                row += [hsr, float(np.linalg.norm(est25)),
                        float(np.mean(X[obs].std(axis=0)))]
            else:
                row += [np.nan, np.nan, np.nan]
            curves.append(row[0])
            feats.append(row[1:])
        print(f'  split {k} done', flush=True)

    n_used = len(meta)                      # one row per perturbation, per split
    assert len(curves) == K_SPLITS * n_used, (len(curves), K_SPLITS, n_used)
    R = np.stack(curves).reshape(K_SPLITS, n_used, D_MAX)
    F = np.stack(feats).reshape(K_SPLITS, n_used, len(FEATS))
    os.makedirs(OUT, exist_ok=True)
    np.save(CACHE_R, R)
    np.save(CACHE_F, F)
    json.dump({'meta': meta, 'K': K_SPLITS, 'd_max': D_MAX,
               'feats': FEATS, 'd_shallow': D_SHALLOW, 'd_deep': D_DEEP},
              open(CACHE_M, 'w'), indent=1)
    print('cached', CACHE_R, R.shape, CACHE_F, F.shape)


def r_at(R_k, depth, rows=None):
    """Exact measured reliability at integer depth (no interpolation)."""
    d = np.clip(np.asarray(depth).astype(int), 1, D_MAX)
    r = np.arange(R_k.shape[0]) if rows is None else np.asarray(rows)
    return R_k[r, d - 1]


def eval_rules():
    R = np.load(CACHE_R)            # (K, N_used, D_MAX)
    F = np.load(CACHE_F)            # (K, N_used, n_feats)
    meta = json.load(open(CACHE_M))
    K, N, _ = R.shape
    pool = np.array([m['n_pool'] for m in meta['meta']])
    ok = pool >= D_DEEP                       # deep allocation feasible
    print(f'perturbations measured={N}; with pool>={D_DEEP}: {ok.sum()} '
          f'({ok.mean()*100:.1f}%) -- restriction is reported, not hidden')

    idx_all = np.where(ok)[0]
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(idx_all))
    dev, test = idx_all[perm[:int(0.7 * len(idx_all))]], idx_all[perm[int(0.7 * len(idx_all)):]]
    print(f'dev={len(dev)} test={len(test)} (probe fitted on dev only)')

    OBJ = {
        'mean_r': lambda r: float(np.mean(r)),
        'frac_r_ge_0.5': lambda r: float(np.mean(r >= 0.5)),
        'frac_r_ge_0.7': lambda r: float(np.mean(r >= 0.7)),
    }
    budgets = [25, 30, 40, 50, 60, 75, 90]
    rows = []
    for k in range(K):
        Rk, Fk = R[k], F[k]
        # ---- probe fitted on DEV only (target: true gain 25->100 on dev) ----
        ydev = r_at(Rk, np.full(len(dev), D_DEEP), dev) - r_at(Rk, np.full(len(dev), D_SHALLOW), dev)
        Fd = Fk[dev]
        mu, sd = Fd.mean(0), Fd.std(0) + 1e-12
        w = np.linalg.lstsq(np.c_[ (Fd - mu) / sd, np.ones(len(dev))], ydev,
                            rcond=None)[0]
        score_probe = ((Fk[test] - mu) / sd) @ w[:-1] + w[-1]
        ytest_true = r_at(Rk, np.full(len(test), D_DEEP), test) - r_at(Rk, np.full(len(test), D_SHALLOW), test)
        from scipy.stats import spearmanr
        rho = spearmanr(score_probe, ytest_true).statistic
        Nt = len(test)
        for B in budgets:
            d_u = int(min(B, D_MAX))
            k_deep = int(round((B - D_SHALLOW) / (D_DEEP - D_SHALLOW) * Nt))
            k_deep = max(0, min(Nt, k_deep))
            rules = {'uniform': np.full(Nt, d_u), 'oracle': None, 'probe': None,
                     'est_norm': None, 'half_split_r': None, 'random': None}
            for rname, sc in [('oracle', ytest_true), ('probe', score_probe),
                              ('est_norm', Fk[test][:, 1]),
                              ('half_split_r', Fk[test][:, 0]),
                              ('random', rng.standard_normal(Nt))]:
                depth = np.full(Nt, D_SHALLOW)
                if k_deep > 0:
                    depth[np.argsort(-sc)[:k_deep]] = D_DEEP
                rules[rname] = depth
            for rname, depth in rules.items():
                r = r_at(Rk, depth, test)
                for oname, fn in OBJ.items():
                    rows.append(dict(split=k, budget=B, rule=rname, objective=oname,
                                     value=fn(r), actual_cost=float(depth.mean()),
                                     k_deep=int((depth == D_DEEP).sum())))
        print(f'  split {k}: probe test spearman={rho:+.3f}', flush=True)

    import pandas as pd
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'stage7_isolated_eval.csv'), index=False)

    def budget_auc(g):
        x = g.budget.values.astype(float)
        y = g.value.values
        o = np.argsort(x)
        return float(np.trapezoid(y[o], x[o]) / (x.max() - x.min()))

    print('\n=== strictly isolated evaluation (test pool only, exact integer depths) ===')
    for oname in OBJ:
        print(f'\n--- objective: {oname} ---')
        sub = res[res.objective == oname]
        tab = sub.groupby(['rule', 'budget']).value.mean().unstack()
        aucs = sub.groupby(['rule', 'split']).apply(
            lambda g: budget_auc(g), include_groups=False).groupby('rule')
        base = aucs.mean()['uniform']
        print(f'{"rule":14s} ' + ' '.join(f'{b:>8d}' for b in budgets) + f'{"budgetAUC":>11s}{"vs unif":>9s}')
        for rname in ['uniform', 'oracle', 'probe', 'est_norm', 'half_split_r', 'random']:
            if rname not in tab.index:
                continue
            line = f'{rname:14s} ' + ' '.join(f'{tab.loc[rname, b]:8.4f}' for b in budgets)
            line += f'{aucs.mean()[rname]:11.4f}{aucs.mean()[rname]-base:+9.4f}'
            print(line)
        print(f'   across-reference-split std of the budgetAUC: ' +
              ', '.join(f'{r}={aucs.std()[r]:.4f}' for r in aucs.mean().index))
    json.dump({'probe_spearman': None}, open(os.path.join(OUT, 'stage7_summary.json'), 'w'), indent=1)
    print('\nwrote', os.path.join(OUT, 'stage7_isolated_eval.csv'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--measure', action='store_true')
    a = ap.parse_args()
    measure() if a.measure else eval_rules()
