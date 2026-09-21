"""Stage 4: can any planner (LLM or otherwise) learn to beat uniform fusion?

A planner only ever sees observable statistics: how well each prior kernel
explains the responses already measured. Its maximum possible value is bounded
by how well those observable statistics predict the quantity it actually cares
about (quality on unseen perturbations).

This script measures that upper bound with a feature matrix that contains
everything a text-based planner could reasonably be given as numbers:

  per round and per prior kernel:
    - alignment on the labelled (observed) subset      <- observable
    - mean / sd of the kernel similarity with truth    <- observable
  outcome to predict:
    - the aligned quality on held-out perturbations    <- NOT observable

If a strong regressor cannot predict held-out quality from the observable
features (leave-one-round-out CV), then no planner reading those statistics can
choose better than the uniform default, and the ceiling for "learned prior
selection" is ~0. If it *can* predict them, the interesting question becomes why
uniform still wins -- which is a matter of decision value, not of perception.
"""
import json
import os

import numpy as np
import kernels_common as kc

OUT = '/home/lihaoran/ai4s/analysis/results'


def main():
    d = kc.load_all()
    truth, tr, te = d['truth'], d['train_idx'], d['test_idx']
    priors = {n: kc.normalize_kernel(d['priors'][n]) for n in kc.PRIOR_NAMES}
    rng = np.random.default_rng(0)

    rows = []
    for seed in range(40):
        lab = rng.choice(tr, size=120, replace=False)          # ~round-1 budget
        T_lab = kc.submatrix(truth, lab)
        T_te = kc.submatrix(truth, te)
        feats = {}
        outs = {}
        for n in kc.PRIOR_NAMES:
            K_lab = kc.submatrix(priors[n], lab)
            K_te = kc.submatrix(priors[n], te)
            feats[n] = dict(
                align_lab=kc.kernel_alignment(K_lab, T_lab),
                sim_mean=float(K_lab.mean()),
                sim_sd=float(K_lab.std()),
                diag_mean=float(np.diag(K_lab).mean()),
                # observable proxy: how well the prior orders perturbations that
                # the model has already seen
                rank_corr_lab=kendall_like(K_lab, T_lab),
            )
            outs[n] = dict(align_te=kc.kernel_alignment(K_te, T_te),
                           rank_corr_te=kendall_like(K_te, T_te))
        rows.append((seed, feats, outs))

    names = kc.PRIOR_NAMES
    # ---- how well does observable alignment predict held-out alignment? ------
    x = np.array([[r[1][n]['align_lab'] for n in names] for r in rows]).ravel()
    y = np.array([[r[2][n]['align_te'] for n in names] for r in rows]).ravel()
    x2 = np.array([[r[1][n]['rank_corr_lab'] for n in names] for r in rows]).ravel()
    y2 = np.array([[r[2][n]['rank_corr_te'] for n in names] for r in rows]).ravel()
    print('=== per-prior: observable vs held-out quality '
          f'({len(rows)} labelled subsets x {len(names)} priors) ===')
    print(f'  corr(align_lab, align_te)      = {np.corrcoef(x,y)[0,1]:.3f}')
    print(f'  corr(rank_corr_lab, rank_corr_te) = {np.corrcoef(x2,y2)[0,1]:.3f}')
    # how often is the observable-best prior also the held-out-best prior?
    A = np.array([[r[1][n]['align_lab'] for n in names] for r in rows])
    B = np.array([[r[2][n]['align_te'] for n in names] for r in rows])
    pick_obs = A.argmax(axis=1)
    pick_true = B.argmax(axis=1)
    print(f'  argmax agreement (observable-best == held-out-best): '
          f'{(pick_obs==pick_true).mean()*100:.1f}%')
    print(f'  held-out alignment of observable-best prior: {B[np.arange(len(B)),pick_obs].mean():.4f}')
    print(f'  held-out alignment of best-fixed prior     : {B.mean(axis=0).max():.4f}')
    print(f'  held-out alignment of UNIFORM fusion       : '
          f'{float(np.mean([kc.kernel_alignment(kc.submatrix(np.mean([priors[n] for n in names],axis=0), te), kc.submatrix(truth, te))])):.4f}')
    print(f'  held-out alignment of per-subset oracle    : {B.max(axis=1).mean():.4f}')

    # ---- can a regressor turn observable features into the right choice? -----
    # leave-one-subset-out CV over a simple linear model on standardised features
    F = np.stack([[r[1][n]['align_lab'], r[1][n]['rank_corr_lab'],
                   r[1][n]['sim_mean'], r[1][n]['sim_sd'], r[1][n]['diag_mean']]
                  for r in rows for n in names])
    Y = B.ravel()
    mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-12
    Fs = (F - mu) / sd
    errs = []
    for i in range(len(Fs)):
        keep = np.arange(len(Fs)) != i
        w = np.linalg.lstsq(np.c_[Fs[keep], np.ones(keep.sum())], Y[keep], rcond=None)[0]
        pred = Fs[i] @ w[:-1] + w[-1]
        errs.append((pred - Y[i]) ** 2)
    base_err = ((Y - Y.mean()) ** 2).mean()
    print('\n=== can observable features rank priors for held-out quality? ===')
    print(f'  leave-one-out MSE of linear probe: {np.mean(errs):.6f}')
    print(f'  variance of the target           : {base_err:.6f}')
    print(f'  R^2 = {1 - np.mean(errs)/base_err:.3f}')

    json.dump(dict(n_subsets=len(rows), n_priors=len(names),
                   corr_align=float(np.corrcoef(x, y)[0, 1]),
                   corr_rank=float(np.corrcoef(x2, y2)[0, 1]),
                   argmax_agreement=float((pick_obs == pick_true).mean()),
                   heldout_align_obs_best=float(B[np.arange(len(B)), pick_obs].mean()),
                   heldout_align_best_fixed=float(B.mean(axis=0).max()),
                   heldout_align_oracle=float(B.max(axis=1).mean()),
                   probe_r2=float(1 - np.mean(errs) / base_err),
                   prior_names=names),
              open(os.path.join(OUT, 'stage4_planner_learnability.json'), 'w'), indent=2)
    print('\nwrote', os.path.join(OUT, 'stage4_planner_learnability.json'))


def kendall_like(K, T):
    """Rank agreement between two sub-kernel matrices, via a random pair sample.

    Computes the fraction of sampled point pairs whose similarity ordering agrees
    between the prior kernel and the truth kernel. Cheap and scale-free.
    """
    n = K.shape[0]
    rng = np.random.default_rng(12345)
    i = rng.integers(0, n, 20000)
    j = rng.integers(0, n, 20000)
    m = i != j
    a, b = K[i[m], j[m]], T[i[m], j[m]]
    # compare each sampled pair against a reference partner pair
    i2 = rng.integers(0, n, len(a))
    m2 = i2 != i[m]
    a2, b2 = K[i[m][m2], i2[m2]], T[i[m][m2], i2[m2]]
    da, db = a[m2] - a2, b[m2] - b2
    keep = (da != 0) & (db != 0)
    if keep.sum() == 0:
        return 0.0
    return float(np.mean(np.sign(da[keep]) == np.sign(db[keep])))


if __name__ == '__main__':
    main()
