"""Stage 1 diagnostic: is there any room for *adaptive* prior weighting?

(a) How different are the 8 priors in quality, and how good is the model kernel?
(b) Is prior quality *context specific* (varies across perturbations) or global?
    Global ordering  -> fixed weights suffice, adaptivity buys little.
    Context specific -> adaptive weighting has headroom.
(c) Under a realistic labelled budget (100..500 perts), do data-driven weights
    generalise to held-out perturbations better than uniform mean fusion?
(d) Does the oracle-best weighting beat uniform at all? (upper bound on headroom)
"""
import os
import json
import sys
import numpy as np
import kernels_common as kc

OUT = '/home/lihaoran/ai4s/analysis/results'
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)


def ka(A, B):
    """Kernel alignment = Frobenius inner product of flattened matrices."""
    return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-300))


def ka_all_vec(ks, T):
    """Alignment of every kernel in ks against T."""
    t = np.linalg.norm(T)
    return np.array([(K * T).sum() / (np.linalg.norm(K) * t + 1e-300) for K in ks])


def main():
    d = kc.load_all()
    perts, truth = d['perts'], d['truth']
    tr, te = d['train_idx'], d['test_idx']
    n_prior = len(kc.PRIOR_NAMES)

    # cache: per index subset, the individual kernel blocks
    block_cache = {}

    def blocks(idx):
        key = (len(idx), idx[0], idx[-1], int(idx.sum()))
        if key not in block_cache:
            block_cache[key] = [kc.submatrix(kc.normalize_kernel(d['priors'][n]), idx)
                                for n in kc.PRIOR_NAMES]
        return block_cache[key]

    # ---------------------------------------------------------------- (a) global
    prior_norm = {n: kc.normalize_kernel(K) for n, K in d['priors'].items()}
    rows = []
    for n in kc.PRIOR_NAMES:
        K = prior_norm[n]
        rows.append(dict(name=n,
                         align_full=ka(K, truth),
                         align_test=ka(kc.submatrix(K, te), kc.submatrix(truth, te)),
                         align_train=ka(kc.submatrix(K, tr), kc.submatrix(truth, tr))))
    rows.sort(key=lambda r: -r['align_test'])
    print('\n=== (a) per-prior alignment with ground-truth delta kernel ===', flush=True)
    print(f'{"prior":22s} {"full(2042)":>11s} {"train(1850)":>12s} {"test(192)":>11s}')
    for r in rows:
        print(f'{r["name"]:22s} {r["align_full"]:11.4f} {r["align_train"]:12.4f} {r["align_test"]:11.4f}')

    T_te, T_tr = kc.submatrix(truth, te), kc.submatrix(truth, tr)
    K_te = [kc.submatrix(prior_norm[n], te) for n in kc.PRIOR_NAMES]
    unif_te = np.mean(K_te, axis=0)
    print(f'\nuniform-8 mean: align(full)={ka(np.mean([prior_norm[n] for n in kc.PRIOR_NAMES],axis=0), truth):.4f} '
          f'align(test)={ka(unif_te, T_te):.4f}', flush=True)

    # ------------------------------------------------------------- (b) context?
    print('\n=== (b) is prior quality context specific? ===', flush=True)
    n_sub, size = 200, 200
    rank_mat = np.zeros((n_sub, n_prior))
    al = np.zeros((n_sub, n_prior))
    for s in range(n_sub):
        idx = RNG.choice(len(perts), size=size, replace=False)
        T = kc.submatrix(truth, idx)
        ks = [kc.submatrix(prior_norm[n], idx) for n in kc.PRIOR_NAMES]
        al[s] = ka_all_vec(ks, T)
        rank_mat[s] = (-al[s]).argsort().argsort()
    glob_best = kc.PRIOR_NAMES.index(rows[0]['name'])
    print(f'across {n_sub} random subsets of {size}: best prior == global best in '
          f'{(rank_mat[:,0]==glob_best).mean()*100:.1f}% of subsets')
    kendall = np.mean([np.corrcoef(rank_mat[s], rank_mat[0])[0, 1] for s in range(1, n_sub)])
    print(f'mean rank-correlation between subsets and subset #0: {kendall:.3f}')
    for j, n in enumerate(kc.PRIOR_NAMES):
        print(f'  {n:22s} align mean={al[:,j].mean():.4f} sd={al[:,j].std():.4f} '
              f'mean_rank={rank_mat[:,j].mean():.2f}')
    print(f'  best fixed prior  : {al.mean(axis=0).max():.4f}')
    print(f'  per-subset oracle : {al.max(axis=1).mean():.4f}  '
          f'(headroom over best-fixed = {al.max(axis=1).mean()-al.mean(axis=0).max():+.4f})', flush=True)

    # ------------------------------------------- (c) weighting at budget + (d)
    print('\n=== (c)/(d) held-out alignment of adaptively fitted weights ===', flush=True)
    schemes = ['uniform', 'align_lab', 'ridge_lab', 'shrink_align', 'oracle_test', 'single_best']
    budgets = [20, 50, 100, 200, 300, 500]
    res = {s: {b: [] for b in budgets} for s in schemes}
    lam_grid = [0.0, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    n_rep = 30
    for b in budgets:
        for rep in range(n_rep):
            lab = RNG.choice(tr, size=b, replace=False)
            L = blocks(lab)
            T_lab = kc.submatrix(truth, lab)
            align = ka_all_vec(L, T_lab)
            A = np.stack([K.ravel() for K in L], axis=1)
            y = T_lab.ravel()
            lam = 1e-3 * np.trace(A.T @ A) / A.shape[1]
            w_ridge = np.clip(np.linalg.solve(A.T @ A + lam * np.eye(n_prior), A.T @ y), 0, None)
            w_unif = np.ones(n_prior) / n_prior
            w_align = np.clip(align, 0, None)
            w_align = w_align / w_align.sum() if w_align.sum() > 0 else w_unif
            w_ridge = w_ridge / w_ridge.sum() if w_ridge.sum() > 0 else w_unif
            # leave-one-out CV, subsampled for speed
            sub = RNG.choice(b, size=min(b, 60), replace=False) if b > 60 else np.arange(b)
            best_lam, best_cv = 1.0, -np.inf
            for ls in lam_grid:
                cvs = []
                for i in sub:
                    keep = np.setdiff1d(np.arange(b), [i], assume_unique=True)
                    kk = [K[np.ix_(keep, keep)] for K in L]
                    tt = T_lab[np.ix_(keep, keep)]
                    aa = np.clip(ka_all_vec(kk, tt), 0, None)
                    aa = aa / aa.sum() if aa.sum() > 0 else w_unif
                    wi = ls * w_unif + (1 - ls) * aa
                    fused = np.tensordot(wi, np.stack(kk), axes=1)
                    cvs.append(ka(fused, tt))
                m = float(np.mean(cvs))
                if m > best_cv:
                    best_cv, best_lam = m, ls
            w = {
                'uniform': w_unif,
                'align_lab': w_align,
                'ridge_lab': w_ridge,
                'shrink_align': best_lam * w_unif + (1 - best_lam) * w_align,
            }
            for s in schemes:
                if s == 'oracle_test':
                    ww = np.clip(ka_all_vec(K_te, T_te), 0, None)
                    ww = ww / ww.sum()
                elif s == 'single_best':
                    ww = np.zeros(n_prior)
                    ww[int(np.argmax(align))] = 1.0
                else:
                    ww = w[s]
                res[s][b].append(ka(np.tensordot(ww, np.stack(K_te), axes=1), T_te))
        print(f'  budget {b:4d} done (lam chosen most often example: {best_lam})', flush=True)

    print(f'\n{"scheme":16s}' + ''.join(f'{"n="+str(b):>11s}' for b in budgets))
    for s in schemes:
        print(f'{s:16s}' + ''.join(f'{np.mean(res[s][b]):11.4f}' for b in budgets))
    print(f'\n{"std":16s}' + ''.join(f'{"":>11s}' for _ in budgets))
    for s in ['uniform', 'align_lab', 'shrink_align', 'ridge_lab']:
        print(f'{s:16s}' + ''.join(f'{np.std(res[s][b]):11.4f}' for b in budgets))

    # paired win-rate of adaptive schemes vs uniform, same labelled subset
    print('\npaired comparison vs uniform (held-out alignment, same budget):')
    for s in ['align_lab', 'shrink_align', 'ridge_lab', 'oracle_test']:
        line = f'  {s:14s}'
        for b in budgets:
            a = np.array(res[s][b]) - np.array(res['uniform'][b])
            line += f'  n={b}: {np.mean(a):+.4f} (win {np.mean(a>0)*100:.0f}%)'
        print(line)

    json.dump({'per_prior': rows,
               'uniform_full': ka(np.mean([prior_norm[n] for n in kc.PRIOR_NAMES], axis=0), truth),
               'uniform_test': ka(unif_te, T_te),
               'ranking_stability_best_frac': float((rank_mat[:, 0] == glob_best).mean()),
               'subset_rank_corr': float(kendall),
               'subset_align_mean': al.mean(axis=0).tolist(),
               'subset_align_sd': al.std(axis=0).tolist(),
               'per_subset_oracle': float(al.max(axis=1).mean()),
               'weighting': {s: {str(b): [float(x) for x in res[s][b]] for b in budgets} for s in schemes},
               'prior_names': kc.PRIOR_NAMES},
              open(os.path.join(OUT, 'stage1_prior_diagnostics.json'), 'w'), indent=2)
    print('\nwrote', os.path.join(OUT, 'stage1_prior_diagnostics.json'))


if __name__ == '__main__':
    main()
