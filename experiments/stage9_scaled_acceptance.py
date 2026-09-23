"""Stage 9: scaled acceptance on MFBudgetEnv.

Three additions over stage8, per review:
  (1) SPLITS: >=10 independent (reference-split, order, dev/test) realizations, so the
      probe's effect size comes with a paired confidence interval.
  (2) RICHER ONLINE FEATURES: stage8's three (half_split_r, delta_norm, cell_sd) plus
      noise_ratio (cell dispersion relative to effect size), n_pool (capacity), and two
      prior-informed, label-free features computed from the 8 prior kernels:
        prior_typiclust : mean similarity of the perturbation to its k nearest
                          candidate neighbours under the fused prior kernel
        prior_nbr_halfsplit : mean half_split_r of those neighbours (observable only)
  (3) DOWNSTREAM ENDPOINT: at the SAME total cost, how well do the measured candidates
      predict the responses of unseen evaluation perturbations? Predictor = kernel-
      weighted kNN in the fused prior kernel (deployable; uses no truth of eval perts).
      Metric = mean Pearson(predicted delta, true delta) over eval perturbations.

Objectives: mean reliability, fraction r >= 0.5, fraction r >= 0.7. Budget curves are
integrated (trapezoidal AUC over the budget axis). Oracles are objective-matched and
are labelled as the best allocation WITHIN the 25->100 two-level family.

Usage: iterpert_env/bin/python stage9_scaled_acceptance.py --splits 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/lihaoran/ai4s/analysis')
from mf_budget_env import MFBudgetEnv, CostModel  # noqa: E402
import kernels_common as kc                       # noqa: E402

H5AD = ('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
        'replogle_k562_essential_1000hvg/perturb_processed.h5ad')
OUT = '/home/lihaoran/ai4s/analysis/results'
D_SHALLOW, D_DEEP = 25, 100
BUDGETS = [40, 55, 70, 90]
K_NN = 10
N_EVAL = 200
FEATS = ['half_split_r', 'delta_norm', 'cell_sd', 'noise_ratio', 'n_pool',
         'prior_typiclust', 'prior_nbr_halfsplit']


def load_all_data():
    import anndata as ad
    adata = ad.read_h5ad(H5AD, backed='r')
    cond = adata.obs['condition'].astype(str).values
    X = np.asarray(adata.to_memory().X, dtype=np.float32)
    kd = kc.load_all()                       # priors on the 2042-pert kernel axis
    fused = kc.normalize_kernel(np.mean([kc.normalize_kernel(kd['priors'][n])
                                         for n in kc.PRIOR_NAMES], axis=0))
    return cond, X, kd, fused


def build_pools(cond, kd, n_cells_min_pool=100):
    """candidates = perts whose 70% pool can reach D_DEEP; eval = disjoint perts."""
    counts = {c: int((cond == c).sum()) for c in np.unique(cond) if c != 'ctrl'}
    pool = {}
    for c, n in counts.items():
        n_ref = int(min(80, max(20, round(0.30 * n))))
        pool[c] = n - n_ref
    cand = sorted([c for c, v in pool.items() if v >= n_cells_min_pool])
    base_of = lambda s: s.split('+')[0]
    kern_axis = {p: i for i, p in enumerate(kd['perts'])}
    cand = [c for c in cand if base_of(c) in kern_axis]
    rest = [c for c in pool if c not in set(cand) and counts[c] >= 40
            and base_of(c) in kern_axis]
    rng = np.random.default_rng(0)
    ev = [rest[i] for i in rng.permutation(len(rest))[:N_EVAL]]
    return cand, ev, counts


def prior_features(fused, cand, kern_axis, base_of):
    """Label-free prior-based features for each candidate."""
    idx = np.array([kern_axis[base_of(c)] for c in cand])
    S = fused[np.ix_(idx, idx)].copy()
    np.fill_diagonal(S, -np.inf)
    nb = np.argsort(-S, axis=1)[:, :K_NN]
    typ = S[np.arange(len(idx))[:, None], nb].mean(axis=1)
    return idx, nb, typ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=int, default=10)
    ap.add_argument('--tag', default='s10')
    args = ap.parse_args()

    t0 = time.time()
    cond, X, kd, fused = load_all_data()
    cand, ev, counts = build_pools(cond, kd)
    base_of = lambda s: s.split('+')[0]
    kern_axis = {p: i for i, p in enumerate(kd['perts'])}
    idx_c, nb_c, typ_c = prior_features(fused, cand, kern_axis, base_of)
    print(f'candidates={len(cand)} eval={len(ev)} loaded in {time.time()-t0:.0f}s', flush=True)

    # true deltas for eval perturbations (independent cells; control_est frame)
    ctrl = np.where(cond == 'ctrl')[0]
    crng = np.random.default_rng(999)
    ctrl = crng.permutation(ctrl)
    ctrl_est_m = X[ctrl[:len(ctrl) // 2]].mean(0)
    ev_true = {}
    for c in ev:
        ids = np.where(cond == c)[0]
        ids = ids[crng.permutation(len(ids))[:max(20, len(ids) // 2)]]
        ev_true[c] = X[ids].mean(0) - ctrl_est_m

    G = X.shape[1]
    rows = []
    ds_rows = []
    for sp in range(args.splits):
        rng = np.random.default_rng(500 + sp)
        perm = rng.permutation(len(cand))
        dev = [cand[i] for i in perm[:int(0.7 * len(cand))]]
        test = [cand[i] for i in perm[int(0.7 * len(cand)):]]
        it = {c: i for i, c in enumerate(cand)}

        # ---- observations at 25 and 100 cells (dev: fit; test: decide) ----
        env_dev = MFBudgetEnv(cond, X, budget=1e9, candidates=dev,
                              init_cells=D_SHALLOW, seed=sp)
        o25d = env_dev.observation()
        env_dev.step({c: D_DEEP - D_SHALLOW for c in dev})
        r100d = env_dev.scorer().reliability(env_dev.observation())
        gain = r100d - env_dev.scorer().reliability(o25d)
        Fd = np.c_[o25d.half_split_r, o25d.delta_norm, o25d.cell_sd,
                   o25d.delta_norm / (np.sqrt(G) * (o25d.cell_sd + 1e-9)),
                   np.array([len(env_dev._order[c]) for c in dev]),
                   typ_c[[it[c] for c in dev]],
                   np.full(len(dev), np.nan)]
        # prior neighbour half_split: mean of dev-neighbour half_split_r
        pos_dev = {c: i for i, c in enumerate(dev)}
        for i, c in enumerate(dev):
            vals = [o25d.half_split_r[pos_dev[cand[g]]]
                    for g in nb_c[it[c]] if cand[g] in pos_dev]
            Fd[i, -1] = np.nanmean(vals) if vals else np.nan
        col_mean = np.nanmean(Fd, axis=0)
        Fd = np.where(np.isnan(Fd), col_mean, Fd)
        mu, sd = Fd.mean(0), Fd.std(0) + 1e-12
        w = np.linalg.lstsq(np.c_[(Fd - mu) / sd, np.ones(len(dev))], gain,
                            rcond=None)[0]
        from scipy.stats import spearmanr
        rho_dev = spearmanr((Fd - mu) / sd @ w[:-1], gain).statistic

        env_t = MFBudgetEnv(cond, X, budget=1e9, candidates=test,
                            init_cells=D_SHALLOW, seed=sp)
        o25t = env_t.observation()
        env_t.step({c: D_DEEP - D_SHALLOW for c in test})
        sc_t = env_t.scorer()
        r100t = sc_t.reliability(env_t.observation())
        r25t = sc_t.reliability(o25t)
        it_t = {c: i for i, c in enumerate(test)}
        Ft = np.c_[o25t.half_split_r, o25t.delta_norm, o25t.cell_sd,
                   o25t.delta_norm / (np.sqrt(G) * (o25t.cell_sd + 1e-9)),
                   np.array([len(env_t._order[c]) for c in test]),
                   typ_c[[it[c] for c in test]],
                   np.full(len(test), np.nan)]
        for i, c in enumerate(test):
            vals = [o25t.half_split_r[it_t[cand[g]]]
                    for g in nb_c[it[c]] if cand[g] in it_t]
            Ft[i, -1] = np.nanmean(vals) if vals else np.nan
        Ft = np.where(np.isnan(Ft), col_mean, Ft)
        score_probe = ((Ft - mu) / sd) @ w[:-1] + w[-1]
        rho_test = spearmanr(score_probe, r100t - r25t).statistic

        # ---- policies at equal total cost, objective-matched oracles ------
        for B in BUDGETS:
            env = MFBudgetEnv(cond, X, budget=float(B * len(test)),
                              candidates=test, init_cells=D_SHALLOW, seed=sp)
            obs0 = env.observation()
            ot = obs0
            policies = {
                'uniform': None,
                'probe(dev-fitted)': score_probe,
                'est_norm': ot.delta_norm,
                'half_split_r': ot.half_split_r,
                'noise_ratio': -ot.delta_norm / (np.sqrt(G) * (ot.cell_sd + 1e-9)),
                'prior_typiclust': typ_c[[it[c] for c in test]],
                'random': np.random.default_rng(sp).standard_normal(len(test)),
            }
            for obj, tau in [('mean_r', None), ('frac_ge_0.5', 0.5), ('frac_ge_0.7', 0.7)]:
                if tau is None:
                    marg = r100t - r25t
                else:
                    marg = ((r100t >= tau).astype(float) - (r25t >= tau).astype(float))
                policies[f'oracle_{obj}'] = marg
            for name, sc in policies.items():
                e = MFBudgetEnv(cond, X, budget=float(B * len(test)),
                                candidates=test, init_cells=D_SHALLOW, seed=sp)
                if sc is not None:
                    k = int((B - D_SHALLOW) * len(test) / (D_DEEP - D_SHALLOW))
                    k = max(0, min(len(test), k))
                    e.step({test[i]: D_DEEP - D_SHALLOW
                            for i in np.argsort(-sc)[:k]})
                e.finalize_uniform_leftover()
                o = e.observation()
                s = e.scorer()
                for obj, tau in [('mean_r', 0.7), ('frac_ge_0.5', 0.5),
                                 ('frac_ge_0.7', 0.7)]:
                    rows.append(dict(split=sp, budget=B, policy=name, objective=obj,
                                     value=s.objective(o, obj, tau=tau),
                                     cost=e._cost['total']))
                # ---- downstream endpoint: predict unseen eval perturbations --
                Dtr = o.delta
                S = fused[np.ix_(idx_c[[it[c] for c in test]],
                                 [kern_axis[base_of(c)] for c in ev])]
                kk = min(K_NN, Dtr.shape[0])
                pred = np.zeros((len(ev), G))
                for j in range(len(ev)):
                    col = S[:, j]
                    top = np.argpartition(-col, kk - 1)[:kk]
                    ww = np.clip(col[top], 0, None)
                    ww = ww / ww.sum() if ww.sum() > 0 else np.ones(kk) / kk
                    pred[j] = ww @ Dtr[top]
                rr = []
                for j, c in enumerate(ev):
                    a = pred[j] - pred[j].mean()
                    b = ev_true[c] - ev_true[c].mean()
                    rr.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
                ds_rows.append(dict(split=sp, budget=B, policy=name,
                                    downstream_pearson=float(np.mean(rr)),
                                    cost=e._cost['total']))
        print(f'  split {sp}: rho_dev={rho_dev:+.3f} rho_test={rho_test:+.3f} '
              f'({time.time()-t0:.0f}s)', flush=True)

    import pandas as pd
    res = pd.DataFrame(rows)
    ds = pd.DataFrame(ds_rows)
    res.to_csv(os.path.join(OUT, f'stage9_acceptance_{args.tag}.csv'), index=False)
    ds.to_csv(os.path.join(OUT, f'stage9_downstream_{args.tag}.csv'), index=False)

    def report(df, value, label):
        print(f'\n=== {label} ===')
        for obj in sorted(df.objective.unique()):
            sub = df[df.objective == obj]
            base = sub[sub.policy == 'uniform'].set_index(['budget', 'split'])[value]
            print(f'\n--- {obj} ---')
            print(f'{"policy":22s}{"mean d":>9s}{"sd":>8s}{"95% CI":>18s}{"n":>5s}')
            for name in sorted(sub.policy.unique()):
                d = (sub[sub.policy == name].set_index(['budget', 'split'])[value]
                     - base).dropna()
                if name == 'uniform':
                    print(f'{name:22s}{0.0:9.4f}{0.0:8.4f}{"—":>18s}{len(d):5d}')
                    continue
                se = d.std(ddof=1) / np.sqrt(len(d))
                ci = 1.96 * se
                print(f'{name:22s}{d.mean():9.4f}{d.std(ddof=1):8.4f}'
                      f'{"["+f"{d.mean()-ci:+.4f},{d.mean()+ci:+.4f}"+"]":>18s}{len(d):5d}')
        return

    report(res, 'value', 'acceptance: reliability objectives (paired vs uniform)')
    print('\n=== downstream endpoint: mean Pearson on unseen perturbations ===')
    base = ds[ds.policy == 'uniform'].set_index(['budget', 'split']).downstream_pearson
    print(f'{"policy":22s}{"mean d":>9s}{"sd":>8s}{"95% CI":>18s}')
    for name in sorted(ds.policy.unique()):
        d = (ds[ds.policy == name].set_index(['budget', 'split']).downstream_pearson
             - base).dropna()
        if name == 'uniform':
            continue
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'{name:22s}{d.mean():9.4f}{d.std(ddof=1):8.4f}'
              f'{"["+f"{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}"+"]":>18s}')
    print(f'\nuniform absolute downstream Pearson: {base.mean():.4f} '
          f'(sd across {base.groupby(level="split").mean().shape[0]} splits '
          f'{base.groupby(level="split").mean().std(ddof=1):.4f})')
    print('wrote', os.path.join(OUT, f'stage9_acceptance_{args.tag}.csv'))


if __name__ == '__main__':
    main()
