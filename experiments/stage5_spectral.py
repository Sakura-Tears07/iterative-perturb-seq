"""Stage 5: why does the uniform fusion survive priors that are obviously broken?

The proxy experiments show that averaging 8 prior kernels keeps working even when
4-8 of them are replaced by random similarity matrices. Two mechanism questions:

 (1) Does a prior kernel carry its information in its leading directions, i.e.
     does "this prior is good" mean "its top eigendirection matches the truth's"?
     If so, a mean over kernels should be destroyed by adding random kernels.
 (2) Under random corruption, is the corrupt kernel's leading direction
     *orthogonal* to the signal (harmless dilution) or does it actively point
     away from it (harmful)?

We measure: for each prior (clean and corrupted) the eigenvalue spectrum, the
alignment of its top eigenvector with the truth's top eigenvectors, and how much
of the kernel alignment comes from the top-k signal directions.
"""
import json
import os

import numpy as np
import kernels_common as kc

OUT = '/home/lihaoran/ai4s/analysis/results'


def top_eig(K, k=5):
    """Top-k eigenpairs via randomised SVD (full eigh on 2042x2042 is ~35s)."""
    K = (K + K.T) / 2
    n = K.shape[0]
    rng = np.random.default_rng(0)
    p = min(n, k + 10)
    Omega = rng.standard_normal((n, p))
    Y = K @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ K @ Q
    B = (B + B.T) / 2
    w, v = np.linalg.eigh(B)
    order = np.argsort(-w)[:k]
    return w[order], Q @ v[:, order]


def main():
    d = kc.load_all()
    truth = d['truth']
    T = truth - truth.mean()          # centre so "direction" means signal direction

    wT, vT = top_eig(T, 5)
    print('=== truth kernel spectrum (centred) ===')
    print('  top-5 eigenvalues:', np.round(wT, 1))
    print('  share of top-1 / top-5 of total positive mass: '
          f'{wT[0]/wT[wT>0].sum():.3f} / {wT[:5].sum()/wT[wT>0].sum():.3f}')

    rng = np.random.default_rng(11)
    N = truth.shape[0]
    R = rng.standard_normal((N, N))
    R = (R + R.T) / 2

    rows = []
    for n in kc.PRIOR_NAMES + ['__random__']:
        K = R if n == '__random__' else d['priors'][n]
        K = kc.normalize_kernel(K, 'max')
        Kc = K - K.mean()
        w, v = top_eig(Kc, 5)
        # alignment of this kernel's top eigenvector with truth's top-5 subspace
        proj = v[:, 0] @ vT
        rows.append(dict(
            name=n,
            top1_share=float(w[0] / w[w > 0].sum()),
            align_truth=kc.kernel_alignment(K, truth),
            top1_align_toptop=float(abs(v[:, 0] @ vT[:, 0])),
            top1_in_truth_subspace=float(np.linalg.norm(proj)),
            eig=[float(x) for x in w],
        ))
    print('\n=== per-kernel leading direction vs truth ===')
    print(f'{"kernel":24s} {"top1 share":>11s} {"align(truth)":>13s} '
          f'{"|cos(v1,vT1)|":>13s} {"||proj on vT1..5||":>18s}')
    for r in rows:
        print(f'{r["name"]:24s} {r["top1_share"]:11.3f} {r["align_truth"]:13.4f} '
              f'{r["top1_align_toptop"]:13.4f} {r["top1_in_truth_subspace"]:18.4f}')

    # how much of the alignment of a prior comes from the signal subspace?
    print('\n=== alignment decomposition ===')
    print('  For each prior, alignment computed on (a) full kernel and (b) the')
    print('  kernel with its top-1 eigendirection removed (residual information).')
    for n in kc.PRIOR_NAMES:
        K = kc.normalize_kernel(d['priors'][n], 'max')
        Kc = K - K.mean()
        w, v = top_eig(Kc, 1)
        K_res = Kc - w[0] * np.outer(v[:, 0], v[:, 0])
        print(f'  {n:24s} align(full)={kc.kernel_alignment(Kc, T):.4f}  '
              f'align(residual)={kc.kernel_alignment(K_res, T):.4f}')

    # does a random kernel hurt a mean fusion, and in what direction?
    print('\n=== effect of adding random kernels to the mean fusion ===')
    clean_mean = np.mean([kc.normalize_kernel(d['priors'][n], 'max') for n in kc.PRIOR_NAMES], axis=0)
    print(f'  mean of 8 clean priors           : align={kc.kernel_alignment(clean_mean, truth):.4f}')
    for n_rand in [1, 2, 4, 8]:
        accs = []
        for rep in range(10):
            rr = np.random.default_rng(100 + rep)
            ks = [kc.normalize_kernel(d['priors'][n], 'max') for n in kc.PRIOR_NAMES]
            for _ in range(n_rand):
                M = rr.standard_normal((N, N))
                M = (M + M.T) / 2
                ks.append(kc.normalize_kernel(M, 'max'))
            accs.append(kc.kernel_alignment(np.mean(ks, axis=0), truth))
        print(f'  + {n_rand} random kernel(s) (total {8+n_rand:2d}) : '
              f'align={np.mean(accs):.4f} (sd {np.std(accs):.4f})')

    json.dump(rows, open(os.path.join(OUT, 'stage5_spectral.json'), 'w'), indent=2)
    print('\nwrote', os.path.join(OUT, 'stage5_spectral.json'))


if __name__ == '__main__':
    main()
