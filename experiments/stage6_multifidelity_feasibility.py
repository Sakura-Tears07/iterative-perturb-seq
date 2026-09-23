"""Stage 6: corrected multi-fidelity (cell-budget) feasibility diagnostic.

Fixes the three flaws of stage3_cell_budget.py:

  (1) INDEPENDENCE: the reference is built from cells that are NEVER used in the
      estimate, and from a disjoint half of the control cells. stage3 correlated
      a subsample against the mean that contained it, which inflates r as the
      subsample approaches the full data.
  (2) POPULATION: reports the ALL-perturbation population (censored curves, with
      the censored fraction shown) separately from a high-cell subset, instead of
      generalising from perturbations with >=250 cells.
  (3) DECISION VALUE: does not stop at "more cells is more precise". It asks
      whether the marginal benefit of extra cells (a) differs across
      perturbations and (b) is predictable from online-observable statistics,
      and then simulates uniform vs feedback-driven vs oracle allocation at a
      fixed total cell budget.

Measurement protocol
--------------------
Each perturbation's cells are split once (seeded) into a hidden REFERENCE block
and a measurement POOL. Measurements are nested prefixes of a random permutation
of the pool, so "going deeper" means adding new cells. Reliability r(d) is the
Pearson correlation between the pool-prefix pseudobulk delta and the hidden
reference delta. Control cells are split into disjoint halves for estimates and
references.

Online-observable features (no reference, no test labels):
    half_split_r   agreement between two disjoint halves of the observed cells
    est_norm       magnitude of the observed delta
    cells_avail    pool cells available for this perturbation
    cell_sd        mean per-gene sd across the observed cells

Run: iterpert_env/bin/python stage6_multifidelity_feasibility.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/lihaoran/ai4s/analysis')

H5AD = ('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
        'replogle_k562_essential_1000hvg/perturb_processed.h5ad')
OUT = '/home/lihaoran/ai4s/analysis/results'
DEPTHS = [10, 25, 50, 100, 150, 200]
REF_FRAC = 0.30
MIN_REF = 20
MAX_REF = 80
MIN_POOL_FOR = {d: d for d in DEPTHS}          # need >= d pool cells for depth d


def load_data():
    import anndata as ad
    adata = ad.read_h5ad(H5AD, backed='r')
    cond = adata.obs['condition'].astype(str).values
    X = np.asarray(adata.to_memory().X, dtype=np.float32)
    return cond, X


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    rng = np.random.default_rng(0)
    cond, X = load_data()
    ctrl = np.where(cond == 'ctrl')[0]
    rng.shuffle(ctrl)
    half = len(ctrl) // 2
    ctrl_est, ctrl_ref = ctrl[:half], ctrl[half:]
    ctrl_est_m = X[ctrl_est].mean(axis=0)
    ctrl_ref_m = X[ctrl_ref].mean(axis=0)
    print(f'cells={len(cond)} genes={X.shape[1]} ctrl_est={len(ctrl_est)} ctrl_ref={len(ctrl_ref)}')

    conds = [c for c in np.unique(cond) if c != 'ctrl']
    per = {c: np.where(cond == c)[0] for c in conds}

    rows = []
    for c in conds:
        idx = per[c]
        n = len(idx)
        n_ref = int(min(MAX_REF, max(MIN_REF, round(REF_FRAC * n))))
        if n - n_ref < 10:                    # cannot even measure 10 cells
            continue
        perm = rng.permutation(idx)
        ref_idx, pool = perm[:n_ref], perm[n_ref:]
        # hidden reference (independent control half)
        ref = X[ref_idx].mean(axis=0) - ctrl_ref_m
        # nested pool prefixes
        order = rng.permutation(len(pool))
        pool_sorted = pool[order]
        cs = np.cumsum(X[pool_sorted], axis=0, dtype=np.float64)
        rec = {'condition': str(c), 'n_cells': int(n), 'n_pool': int(len(pool))}
        for d in DEPTHS:
            if len(pool) < MIN_POOL_FOR[d]:
                rec[f'r{d}'] = None
                continue
            est = cs[d - 1] / d - ctrl_est_m
            rec[f'r{d}'] = pearson(est, ref)
        # online-observable features at depth 25 (no reference involved)
        d0 = 25
        if len(pool) >= 2 * d0:
            obs = pool_sorted[:d0]
            h1, h2 = obs[:d0 // 2], obs[d0 // 2:]
            rec['half_split_r'] = pearson(X[h1].mean(0) - ctrl_est_m,
                                          X[h2].mean(0) - ctrl_est_m)
            est25 = cs[d0 - 1] / d0 - ctrl_est_m
            rec['est_norm'] = float(np.linalg.norm(est25))
            rec['cell_sd'] = float(np.mean(X[obs].std(axis=0)))
            # magnitude of cell-level dispersion relative to effect size
            rec['noise_ratio'] = float(np.mean(X[obs].std(axis=0)) /
                                       (np.linalg.norm(est25) / np.sqrt(X.shape[1]) + 1e-9))
        else:
            for k in ('half_split_r', 'est_norm', 'cell_sd', 'noise_ratio'):
                rec[k] = None
        rows.append(rec)

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, 'stage6_multifidelity_per_pert.csv'), index=False)
    print(f'perturbations measured: {len(df)}')

    # ---------------- (2) population: ALL vs HIGH, censoring shown -----------
    print('\n=== (1) reliability vs depth, with censoring (disjoint reference) ===')
    print(f'{"depth":>6s} {"all: n":>7s} {"all: mean r":>11s} {"all: median":>11s} '
          f'{"high: n":>7s} {"high: mean r":>12s} {"high: median":>12s}')
    high = df[df.n_pool >= 200]
    for d in DEPTHS:
        col = f'r{d}'
        a = df[col].dropna()
        h = high[col].dropna()
        print(f'{d:6d} {len(a):7d} {a.mean():11.3f} {a.median():11.3f} '
              f'{len(h):7d} {h.mean():12.3f} {h.median():12.3f}')
    frac_lt = float((df.n_cells < 100).mean())
    print(f'\n  perturbations with <100 cells overall: {frac_lt*100:.1f}% '
          f'(they can never reach depth 100)')

    # ---------------- (3a) heterogeneity of the marginal benefit -------------
    print('\n=== (2) is the marginal benefit of extra cells heterogeneous? ===')
    sub = df.dropna(subset=['r25', 'r100'])
    gain = sub.r100 - sub.r25
    print(f'  n={len(sub)}  gain r(25)->r(100): mean={gain.mean():.3f} '
          f'sd={gain.std():.3f}  p10={np.percentile(gain,10):.3f} '
          f'p50={np.percentile(gain,50):.3f} p90={np.percentile(gain,90):.3f}')
    print(f'  fraction of perturbations with gain<=0.02: {(gain<=0.02).mean()*100:.1f}%')
    print(f'  fraction with gain>=0.15                : {(gain>=0.15).mean()*100:.1f}%')
    # how much of the variance is predictable at all (oracle feature)?
    r2_true = float(np.corrcoef(sub.r25, gain)[0, 1] ** 2)
    print(f'  R^2 of gain predicted by the (unobservable) r25 itself: {r2_true:.3f}')

    # ---------------- (3b) predictability from online features ---------------
    print('\n=== (3) can the gain be predicted from ONLINE-observable features? ===')
    feats = ['half_split_r', 'est_norm', 'cell_sd', 'noise_ratio']
    sub2 = sub.dropna(subset=feats)
    y = (sub2.r100 - sub2.r25).values
    from scipy.stats import spearmanr
    for f in feats:
        rho = spearmanr(sub2[f].values, y).statistic
        print(f'  spearman(gain, {f:14s}) = {rho:+.3f}')
    # leave-one-perturbation-out linear probe (holds out the whole perturbation)
    Fm = sub2[feats].values
    mu, sd = Fm.mean(0), Fm.std(0) + 1e-12
    Fs = (Fm - mu) / sd
    pred = np.zeros(len(Fs))
    for i in range(len(Fs)):
        keep = np.arange(len(Fs)) != i
        w = np.linalg.lstsq(np.c_[Fs[keep], np.ones(keep.sum())], y[keep],
                            rcond=None)[0]
        pred[i] = Fs[i] @ w[:-1] + w[-1]
    r2 = 1 - ((pred - y) ** 2).mean() / ((y - y.mean()) ** 2).mean()
    print(f'  leave-one-perturbation-out R^2 (linear, {len(feats)} features) = {r2:.3f}')

    # ---------------- (4) allocation simulation at a fixed cell budget -------
    print('\n=== (4) allocation at a FIXED total cell budget (high-cell subset) ===')
    sim_pop = df.dropna(subset=['r25', 'r100', 'half_split_r']).copy()
    sim_pop = sim_pop[sim_pop.n_pool >= 100]
    N = len(sim_pop)
    d_shallow, d_deep = 25, 100
    C = N * 50                                   # 50 cells/perturbation on average
    leftover = C - d_shallow * N
    k = max(0, int(leftover // (d_deep - d_shallow)))
    # predicted gain from observable features (fit on 70%, apply to the rest)
    tr = rng.permutation(N)
    fit_idx, app_idx = tr[:int(0.7 * N)], tr[int(0.7 * N):]
    Fm = sim_pop[feats].values
    mu, sd = Fm.mean(0), Fm.std(0) + 1e-12
    Fs = (Fm - mu) / sd
    yv = (sim_pop.r100 - sim_pop.r25).values
    w = np.linalg.lstsq(np.c_[Fs[fit_idx], np.ones(len(fit_idx))], yv[fit_idx],
                        rcond=None)[0]
    pred_gain = Fs @ w[:-1] + w[-1]
    true_gain = yv
    results = {}
    for name, chosen in [
        ('uniform (everyone d=50)', np.zeros(N, bool)),
        ('uniform (everyone d=25, keep budget)', np.zeros(N, bool)),
        ('two-stage, OBSERVABLE features', np.zeros(N, bool)),
        ('two-stage, ORACLE (true gain)', np.zeros(N, bool)),
    ]:
        if name.startswith('uniform (everyone d=50'):
            depth = np.full(N, 50.0)
        elif name.startswith('uniform (everyone d=25'):
            depth = np.full(N, d_shallow, float)
        else:
            sel = np.zeros(N, bool)
            score = pred_gain if 'OBSERVABLE' in name else true_gain
            sel[np.argsort(-score)[:k]] = True
            if name.startswith('two-stage, ORACLE'):
                chosen = sel
            depth = np.where(sel, d_deep, d_shallow).astype(float)
        # realised reliability at the allocated depth (from the measured curves)
        def rel_at(row, d):
            for cand in DEPTHS:
                if cand >= d:
                    v = row[f'r{cand}']
                    return float(v) if v is not None and not pd.isna(v) else np.nan
            return float(row['r200']) if not pd.isna(row['r200']) else np.nan
        rr = np.array([rel_at(sim_pop.iloc[i], depth[i]) for i in range(N)])
        results[name] = float(np.nanmean(rr))
        print(f'  {name:38s} budget={int(depth.sum()):7d}  mean reliability={results[name]:.4f}'
              + (f'  (deep-assigned {int((depth==d_deep).sum())})'
                 if 'two-stage' in name else ''))
    print(f'\n  budget C = {C} cells for N = {N} perturbations '
          f'(= {C/N:.1f} cells/perturbation on average)')
    head = results['two-stage, ORACLE (true gain)'] - results['uniform (everyone d=50)']
    ach = results['two-stage, OBSERVABLE features'] - results['uniform (everyone d=50)']
    print(f'  headroom  (oracle  - uniform d=50) = {head:+.4f}')
    print(f'  achievable(observ. - uniform d=50) = {ach:+.4f}')

    json.dump({'n_pert': int(len(df)), 'frac_lt100': frac_lt,
               'reliability': {str(d): dict(
                   all_n=int(df[f'r{d}'].notna().sum()),
                   all_mean=float(df[f'r{d}'].mean()),
                   high_n=int(high[f'r{d}'].notna().sum()),
                   high_mean=float(high[f'r{d}'].mean())) for d in DEPTHS},
               'gain_25_100': dict(mean=float(gain.mean()), sd=float(gain.std()),
                                   p10=float(np.percentile(gain, 10)),
                                   p90=float(np.percentile(gain, 90)),
                                   frac_le002=float((gain <= 0.02).mean()),
                                   frac_ge015=float((gain >= 0.15).mean())),
               'probe_r2_loo': float(r2), 'allocation': results,
               'allocation_headroom': float(head),
               'allocation_achievable': float(ach)},
              open(os.path.join(OUT, 'stage6_multifidelity.json'), 'w'), indent=2)
    print('\nwrote', os.path.join(OUT, 'stage6_multifidelity.json'))


if __name__ == '__main__':
    main()
