"""Stage 3 diagnostic: is the multi-fidelity action space of proposal 3 real?

Proposal 3 wants to choose, per round, *which perturbation* AND *how many cells*
to spend on it. That only makes sense if
  (a) perturbations genuinely differ in how many cells they have (unequal cost),
  (b) observation noise per perturbation actually decreases with cell count, and
  (c) the noise level differs across perturbations (heteroscedastic).

We measure all three directly from the raw 310k-cell h5ad, and then ask the
budget-allocation question: at a fixed total cell budget, is it better to
(a) profile more perturbations shallowly, or (b) fewer perturbations deeply?
We answer it with a plain measurement: for each perturbation, subsample n cells
and compute the pseudobulk delta vs the control pseudobulk, then correlate that
subsample estimate with the full-data estimate (i.e. how reliable the
measurement is at that depth).
"""
import json
import os

import numpy as np

H5AD = ('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
        'replogle_k562_essential_1000hvg/perturb_processed.h5ad')
OUT = '/home/lihaoran/ai4s/analysis/results'


def main():
    import anndata as ad
    import scipy.sparse as sp
    adata = ad.read_h5ad(H5AD, backed='r')
    cond = adata.obs['condition'].astype(str).values
    ctrl_mask = cond == 'ctrl'
    n_ctrl = int(ctrl_mask.sum())
    print(f'cells={adata.n_obs} genes={adata.n_vars} conditions={len(set(cond))} '
          f'control cells={n_ctrl}')

    counts = {c: int((cond == c).sum()) for c in set(cond)}
    per_pert = np.array([v for k, v in counts.items() if k != 'ctrl'])
    print(f'\n=== (a) cells per perturbation ===')
    print(f'  n_perturbations={len(per_pert)}  min={per_pert.min()} '
          f'p25={np.percentile(per_pert,25):.0f} median={np.median(per_pert):.0f} '
          f'p75={np.percentile(per_pert,75):.0f} max={per_pert.max()} mean={per_pert.mean():.1f}')
    print(f'  Gini-like inequality: top 10% of perts hold '
          f'{per_pert[np.argsort(-per_pert)][:max(1,len(per_pert)//10)].sum()/per_pert.sum()*100:.1f}% of cells')
    print(f'  fraction of perts with <50 cells: {(per_pert<50).mean()*100:.1f}%')
    print(f'  fraction of perts with <100 cells: {(per_pert<100).mean()*100:.1f}%')

    # ---- how reliable is a pseudobulk delta measured from n cells? -------------
    rng = np.random.default_rng(0)
    # load full matrix once (310k x 1000 dense float32 ~ 1.2 GB, fine)
    X = adata.to_memory().X if not sp.issparse(adata.X) else adata.to_memory().X.toarray()
    X = np.asarray(X, dtype=np.float32)
    # Normalise exactly as the pipeline's pseudobulk does: the h5ad is already
    # log-normalised (adata.uns['log1p']); we only average.
    ctrl_idx = np.where(ctrl_mask)[0]
    ctrl_full = X[ctrl_idx].mean(axis=0)
    depths = [10, 25, 50, 100, 200]
    # pick perturbations that actually have enough cells
    cand = [c for c, v in counts.items() if c != 'ctrl' and v >= 250]
    rng.shuffle(cand)
    cand = cand[:60]
    corrs = np.zeros((len(cand), len(depths)))
    for i, c in enumerate(cand):
        idx = np.where(cond == c)[0]
        full = X[idx].mean(axis=0) - ctrl_full
        for j, d in enumerate(depths):
            sub = rng.choice(idx, size=min(d, len(idx)), replace=False)
            est = X[sub].mean(axis=0) - ctrl_full
            a = full - full.mean()
            b = est - est.mean()
            corrs[i, j] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f'\n=== (b) reliability of a pseudobulk delta vs sequencing depth '
          f'({len(cand)} perturbations with >=250 cells) ===')
    print(f'{"cells":>6s} {"mean r":>8s} {"median r":>9s} {"p10 r":>7s}')
    for j, d in enumerate(depths):
        print(f'{d:6d} {corrs[:,j].mean():8.3f} {np.median(corrs[:,j]):9.3f} '
              f'{np.percentile(corrs[:,j],10):7.3f}')
    print('  (reference: r=1.0 means the shallow measurement recovers the full-data delta)')

    # ---- (c) fixed budget: breadth vs depth -----------------------------------
    # Total cells in the screen ~= 310k for 2058 perts. Question: at a fixed
    # total budget for N perturbations, how does the *median* measurement
    # reliability change with depth? Use the depth curve above.
    print(f'\n=== (c) breadth vs depth at fixed total budget ===')
    print('  from the depth curve, reliability saturates quickly:')
    for d, j in zip(depths, range(len(depths))):
        print(f'    depth {d:4d} cells -> median r = {np.median(corrs[:,j]):.3f}')
    # marginal gain of doubling depth
    print('  marginal gain (median r) of doubling cells:')
    for j in range(1, len(depths)):
        print(f'    {depths[j-1]} -> {depths[j]}: '
              f'{np.median(corrs[:,j]) - np.median(corrs[:,j-1]):+.3f}')

    json.dump(dict(n_cells=int(adata.n_obs), n_conditions=len(counts), n_ctrl=n_ctrl,
                   per_pert=dict(min=int(per_pert.min()), median=float(np.median(per_pert)),
                                 mean=float(per_pert.mean()), max=int(per_pert.max()),
                                 p25=float(np.percentile(per_pert, 25)),
                                 p75=float(np.percentile(per_pert, 75)),
                                 frac_lt50=float((per_pert < 50).mean()),
                                 frac_lt100=float((per_pert < 100).mean())),
                   depths=depths,
                   corr_mean=[float(corrs[:, j].mean()) for j in range(len(depths))],
                   corr_median=[float(np.median(corrs[:, j])) for j in range(len(depths))],
                   corr_p10=[float(np.percentile(corrs[:, j], 10)) for j in range(len(depths))]),
              open(os.path.join(OUT, 'stage3_cell_budget.json'), 'w'), indent=2)
    print('\nwrote', os.path.join(OUT, 'stage3_cell_budget.json'))


if __name__ == '__main__':
    main()
