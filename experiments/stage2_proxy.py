"""Stage 2: GEARS-free active-learning proxy for comparing selection strategies.

Why this exists
---------------
The full IterPert loop retrains GEARS 6x per method (tens of GPU-minutes each),
so it cannot be used to screen many strategies. But the *only* place strategies
differ is the fused kernel used by maxdist (farthest-first) selection. We
therefore evaluate selection strategies directly, with ground-truth deltas
standing in for the wet-lab readout:

  round r:  agent has labelled set L (ground-truth deltas known)
            -> build fused kernel K_hat from priors (+ a simulated model kernel)
            -> maxdist select n_query perturbations
            -> look up their ground-truth deltas (this is the "experiment")
  metric:   how well the resulting labelled set covers / predicts the
            held-out 192 perturbations (same split as the paper, seed 1)

Each method runs its own independent closed loop, and sees exactly the same
initial labelled set, the same split and the same priors. Every weighting scheme
is fitted ONLY on the labelled set (no test leakage); 'oracle_truth' is an
explicitly-cheating upper bound.

Usage: python stage2_proxy.py --scenario clean|corrupt|corrupt_all --seeds 0,1,2
"""
import argparse
import json
import os
import numpy as np
import kernels_common as kc

OUT = '/home/lihaoran/ai4s/analysis/results'
PRIOR_NAMES = kc.PRIOR_NAMES

METHODS = ['random', 'priors_only_uniform', 'base_only', 'uniform',
           'align', 'shrink_align', 'ridge', 'topk_align', 'oracle_truth']
LABEL = {'random': 'Random', 'priors_only_uniform': 'Priors only (unif)',
         'base_only': 'Model kernel only', 'uniform': 'IterPert uniform fusion',
         'align': 'Alignment weights', 'shrink_align': 'Shrink-to-uniform align',
         'ridge': 'Ridge kernel regression', 'topk_align': 'Top-1 prior by align',
         'oracle_truth': 'ORACLE (sees truth)'}


def normalize(K):
    return kc.normalize_kernel(K, 'max')


def ka(A, B):
    return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-300))


def align_vec(blocks, T_lab):
    t = np.linalg.norm(T_lab)
    return np.array([(K * T_lab).sum() / (np.linalg.norm(K) * t + 1e-300) for K in blocks])


def maxdist_select(K_full, labelled, pool, n_select):
    """Replicates bmdal MaxDistSelectionMethodwithPrior (farthest-first)."""
    diag = np.diag(K_full)
    min_sq = np.full(len(pool), np.inf)
    for t in labelled:
        np.minimum(min_sq, diag[t] + diag[pool] - 2.0 * K_full[t, pool], out=min_sq)
    selected = []
    for _ in range(min(n_select, len(pool))):
        j = int(np.argmax(min_sq))
        t = pool[j]
        np.minimum(min_sq, diag[t] + diag[pool] - 2.0 * K_full[t, pool], out=min_sq)
        min_sq[j] = -np.inf
        selected.append(t)
    return np.array(selected)


def fit_weights(method, blocks, T_lab, unif, rng):
    nk = len(blocks)
    a = align_vec(blocks, T_lab)
    if method == 'uniform':
        return unif
    if method == 'align':
        w = np.clip(a, 0, None)
        return w / w.sum() if w.sum() > 0 else unif
    if method == 'topk_align':
        w = np.zeros(nk)
        w[int(np.argmax(a))] = 1.0
        return w
    if method == 'ridge':
        A = np.stack([K.ravel() for K in blocks], axis=1)
        y = T_lab.ravel()
        lam = 1e-3 * np.trace(A.T @ A) / A.shape[1]
        w = np.clip(np.linalg.solve(A.T @ A + lam * np.eye(nk), A.T @ y), 0, None)
        return w / w.sum() if w.sum() > 0 else unif
    if method == 'shrink_align':
        n = T_lab.shape[0]
        sub = rng.choice(n, size=min(n, 60), replace=False) if n > 60 else np.arange(n)
        w_a = np.clip(a, 0, None)
        w_a = w_a / w_a.sum() if w_a.sum() > 0 else unif
        stack = np.stack(blocks)                       # (nk, n, n)
        best_lam, best_cv = 1.0, -np.inf
        for ls in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
            cvs = []
            for i in sub:
                keep = np.setdiff1d(np.arange(n), [i], assume_unique=True)
                kk = stack[:, keep][:, :, keep]
                tt = T_lab[np.ix_(keep, keep)]
                aa = np.clip(align_vec(list(kk), tt), 0, None)
                aa = aa / aa.sum() if aa.sum() > 0 else unif
                wj = ls * unif + (1 - ls) * aa
                cvs.append(ka(np.einsum('k,kij->ij', wj, kk), tt))
            m = float(np.mean(cvs))
            if m > best_cv:
                best_cv, best_lam = m, ls
        return best_lam * unif + (1 - best_lam) * w_a
    raise ValueError(method)


def simulated_model_kernel(base_kernel, labelled, Y, k_build=15):
    """Stand-in for the GEARS model kernel K = f(X) f(X)^T.

    The real pipeline trains GEARS on the labelled perturbations and forms a
    linear kernel over its predictions. We approximate that with a
    kernel-weighted kNN prediction of the observed deltas over the most
    variable genes. Depends on what has been observed, so it is the "free"
    feedback channel every method gets (as GEARS is retrained for all of them).
    """
    S = base_kernel[:, labelled]
    k = min(k_build, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    W = np.clip(np.take_along_axis(S, idx, axis=1), 0, None)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    W = W / rs
    P = np.einsum('nk,nkg->ng', W, Y[labelled][idx])
    top = np.argsort(-P.var(axis=0))[:64]
    P = P[:, top]
    P = P - P.mean(axis=0, keepdims=True)
    return P @ P.T


def evaluate_selection(sel_idx, Y, te, genes_top, ref_kernel=None, alpha=1e-2,
                       ref_truth=None):
    """Metrics. Main one: KRR on a fixed RBF kernel built from true deltas ->
    Pearson r on held-out perturbations. Scale-free and well-conditioned."""
    Ysel = Y[sel_idx]
    Yn = Y[te] / (np.linalg.norm(Y[te], axis=1, keepdims=True) + 1e-12)
    Sn = Ysel / (np.linalg.norm(Ysel, axis=1, keepdims=True) + 1e-12)
    sim = Yn @ Sn.T
    best = sim.max(axis=1)
    nn = Sn[np.argmax(sim, axis=1)]
    de = (np.sign(Yn[:, genes_top]) == np.sign(nn[:, genes_top])).mean(axis=1)
    out = dict(coverage_mean=float(best.mean()),
               coverage_p10=float(np.percentile(best, 10)),
               de_sign_agree=float(de.mean()))
    if ref_kernel is not None:
        # The metric that actually matches the paper: fit a predictor on the
        # perturbations that were "profiled" and ask how well it predicts the
        # held-out ones. We use kernel ridge regression on the observed deltas,
        # with a FIXED reference kernel (uniform prior fusion) so that the
        # comparison isolates *which perturbations were selected* rather than
        # which kernel the strategy prefers for prediction.
        Yl = Y[sel_idx]
        Yt = Y[te]
        Kll = ref_kernel[np.ix_(sel_idx, sel_idx)]
        Ktl = ref_kernel[np.ix_(te, sel_idx)]
        A = Kll + alpha * np.eye(len(sel_idx))
        try:
            coef = np.linalg.solve(A, Yl)
            pred = Ktl @ coef
        except np.linalg.LinAlgError:
            pred = Ktl @ (np.linalg.pinv(A) @ Yl)
        yc = Yt - Yt.mean(axis=1, keepdims=True)
        pc = pred - pred.mean(axis=1, keepdims=True)
        num = (yc * pc).sum(axis=1)
        den = np.linalg.norm(yc, axis=1) * np.linalg.norm(pc, axis=1) + 1e-12
        r = num / den
        top = np.argsort(-Yt.var(axis=0))[:50]
        yt = Yt[:, top]
        yc2 = yt - yt.mean(axis=1, keepdims=True)
        pc2 = pred[:, top] - pred[:, top].mean(axis=1, keepdims=True)
        r2 = (yc2 * pc2).sum(axis=1) / (np.linalg.norm(yc2, axis=1) * np.linalg.norm(pc2, axis=1) + 1e-12)
        out['krr_pearson_all'] = float(np.mean(r))
        out['krr_pearson_top50var'] = float(np.mean(r2))
        out['krr_mse'] = float(np.mean((pred - Yt) ** 2))
    return out


def _init_like_real(n_init, n_pool):
    """Reproduce Data.initialize_labels exactly (it hardcodes np.random.seed(42))."""
    tmp = np.arange(n_pool)
    st = np.random.get_state()
    np.random.seed(42)
    np.random.shuffle(tmp)
    np.random.set_state(st)
    return np.sort(tmp[:n_init])


def run_method(method, seed, sc, n_init, n_round, n_query, k_model=15,
               real_init=False):
    rng = np.random.default_rng(1000 + seed)
    d = sc['d']
    Y = d['truth_feat']
    tr, te = d['train_idx'], d['test_idx']
    priors, base_kernel = sc['priors'], sc['base_kernel']
    genes_top = np.argsort(-Y[tr].var(axis=0))[:50]
    prior_full = np.stack([priors[n] for n in PRIOR_NAMES])
    # fixed reference kernel for the prediction metric: RBF on cosine similarity
    # between ground-truth deltas. Positive definite, fixed bandwidth, so the
    # metric does not inherit the scale of whatever prior bank the scenario uses
    # (a max-normalised fused kernel made the old metric swing 10x between
    # scenarios with 0.95-aligned kernels -- it was measuring scale, not skill).
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    ref_kernel = np.exp(-(1.0 - np.clip(Yn @ Yn.T, -1, 1)) / 0.2)

    if real_init:
        # match the real closed loop: Data.initialize_labels(n_init) over the pool
        init = tr[_init_like_real(n_init, len(tr))]
    else:
        init = rng.choice(tr, size=n_init, replace=False)  # identical across methods
    labelled = np.array(sorted(init))
    pool = np.setdiff1d(tr, labelled, assume_unique=True)
    curve, model_cache = [], {}
    for r in range(n_round + 1):
        curve.append(evaluate_selection(labelled, Y, te, genes_top,
                                         ref_kernel=ref_kernel))
        if r == n_round:
            break
        if method == 'random':
            sel = rng.choice(pool, size=min(n_query, len(pool)), replace=False)
        else:
            T_lab = kc.submatrix(Y @ Y.T, labelled)
            stack_lab = np.stack([kc.submatrix(priors[n], labelled) for n in PRIOR_NAMES]
                                 + [kc.submatrix(base_kernel, labelled)])
            unif = np.ones(len(stack_lab)) / len(stack_lab)
            if method == 'base_only':
                K_full = base_kernel
            elif method == 'priors_only_uniform':
                K_full = prior_full.mean(axis=0)
            elif method == 'oracle_truth':
                # upper bound: prior weights fitted on the LABELLED truth kernel,
                # i.e. an oracle that knows every observed perturbation's real
                # response (the agent does not). Weights are then applied to the
                # full prior kernels for selection.
                a = np.clip(align_vec([kc.submatrix(priors[n], labelled) for n in PRIOR_NAMES],
                                      T_lab), 0, None)
                K_full = np.tensordot(a / a.sum(), prior_full, axes=1)
            else:
                key = (len(labelled), int(labelled[0]), int(labelled[-1]), int(labelled.sum()))
                if key not in model_cache:
                    model_cache[key] = normalize(simulated_model_kernel(base_kernel, labelled, Y, k_model))
                stack_full = np.concatenate([prior_full, model_cache[key][None]], axis=0)
                if method == 'uniform':
                    K_full = stack_full.mean(axis=0)
                else:
                    w = fit_weights(method, list(stack_lab), T_lab, unif, rng)
                    K_full = np.tensordot(w, stack_full, axes=1)
            sel = maxdist_select(K_full, labelled, pool, n_query)
        labelled = np.sort(np.concatenate([labelled, sel]))
        pool = np.setdiff1d(pool, sel, assume_unique=True)
    return curve


def build_scenario(scenario, d):
    priors = {n: normalize(d['priors'][n]) for n in PRIOR_NAMES}
    if scenario == 'noise2':
        # Targeted failure mode: the two STRONGEST priors are replaced by random
        # similarity, the mediocre ones stay. A weight-learning method should be
        # able to detect and down-weight the garbage. This is the most favourable
        # realistic setting for adaptive prior weighting.
        rng = np.random.default_rng(11)
        N = d['truth'].shape[0]
        for n in ['rpe1_kernel', 'node2vec_kernel']:
            R = rng.standard_normal((N, N))
            R = (R + R.T) / 2
            priors[n] = normalize(R)
    if scenario == 'noise4':
        # Halve the prior bank: 4 of 8 replaced by random matrices.
        rng = np.random.default_rng(13)
        N = d['truth'].shape[0]
        for n in ['rpe1_kernel', 'node2vec_kernel', 'pops_kernel', 'esm_kernel']:
            R = rng.standard_normal((N, N))
            R = (R + R.T) / 2
            priors[n] = normalize(R)
    # 'gears_kernel' = the precomputed GEARS model-kernel artifact shipped with
    # IterPert; used here only as the embedding space for the simulated model
    # kernel. It is NOT one of the eight prior kernels.
    base_kernel = normalize(d['gears']) if 'gears' in d else None
    if base_kernel is None:
        raise KeyError('gears kernel missing from load_all()')
    if scenario in ('corrupt', 'corrupt_all'):
        rng = np.random.default_rng(7)
        N = d['truth'].shape[0]
        targets = ['ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel', 'biogpt_kernel', 'esm_kernel']
        if scenario == 'corrupt_all':
            targets += ['rpe1_kernel', 'node2vec_kernel', 'pops_kernel', 'ops_A549_kernel']
        for n in targets:
            perm = rng.permutation(N)
            priors[n] = normalize(priors[n][np.ix_(perm, perm)])
    return dict(d=d, priors=priors, base_kernel=base_kernel,
                corrupt=targets if scenario in ('corrupt', 'corrupt_all') else scenario)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='clean',
                    choices=['clean', 'corrupt', 'corrupt_all', 'noise2', 'noise4'])
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--n_init', type=int, default=100)
    ap.add_argument('--n_query', type=int, default=100)
    ap.add_argument('--n_round', type=int, default=5)
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--real_init', type=int, default=0,
                    help='1 = reproduce the exact initial set of the real loop (seed 42)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    d = kc.load_all()
    sc = build_scenario(args.scenario, d)
    print(f'scenario={args.scenario} corrupted priors: {sc["corrupt"]}', flush=True)
    methods = args.methods.split(',')
    results = {}
    for m in methods:
        curves = []
        for s in seeds:
            curves.append(run_method(m, s, sc, args.n_init, args.n_round, args.n_query,
                                     real_init=bool(args.real_init)))
        results[m] = curves
        print(f'{m:22s} ' + ' | '.join(
            ' '.join(f'{x["coverage_mean"]:.4f}' for x in c) for c in curves), flush=True)

    summary = {}
    for m in methods:
        aucs = [float(np.mean([x['coverage_mean'] for x in c])) for c in results[m]]
        kauc = [float(np.mean([x.get('krr_pearson_all', np.nan) for x in c])) for c in results[m]]
        kfin = [c[-1].get('krr_pearson_all', float('nan')) for c in results[m]]
        finals = [c[-1]['coverage_mean'] for c in results[m]]
        de = [c[-1]['de_sign_agree'] for c in results[m]]
        summary[m] = dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                          final=float(np.mean(finals)), de_final=float(np.mean(de)),
                          krr_auc=float(np.nanmean(kauc)), krr_final=float(np.nanmean(kfin)),
                          curve=[float(np.mean([c[r]['coverage_mean'] for c in results[m]]))
                                 for r in range(args.n_round + 1)],
                          krr_curve=[float(np.nanmean([c[r].get('krr_pearson_all', np.nan)
                                                       for c in results[m]]))
                                     for r in range(args.n_round + 1)])
    base = summary['uniform']['auc']
    print(f'\n=== scenario={args.scenario} | mean over seeds={seeds} | metric=held-out coverage (higher better) ===')
    kb = summary['uniform']['krr_auc']
    print(f'{"method":24s} {"covAUC":>8s} {"sd":>7s} {"krrAUC":>8s} {"krrFinal":>9s} '
          f'{"DEagree":>8s} {"d(cov)":>9s} {"d(krr)":>9s}')
    for m in sorted(summary, key=lambda x: -summary[x]['auc']):
        print(f'{m:24s} {summary[m]["auc"]:8.4f} {summary[m]["auc_sd"]:7.4f} '
              f'{summary[m]["krr_auc"]:8.4f} {summary[m]["krr_final"]:9.4f} '
              f'{summary[m]["de_final"]:8.4f} {summary[m]["auc"]-base:+9.4f} '
              f'{summary[m]["krr_auc"]-kb:+9.4f}')
    out = args.out or os.path.join(OUT, f'stage2_proxy_{args.scenario}.json')
    json.dump(dict(scenario=args.scenario, seeds=seeds, n_init=args.n_init,
                   real_init=bool(args.real_init),
                   n_query=args.n_query, n_round=args.n_round,
                   corrupted=sc['corrupt'], summary=summary,
                   curves={m: {str(s): results[m][i] for i, s in enumerate(seeds)}
                           for m in methods}),
              open(out, 'w'), indent=2)
    print('wrote', out)


if __name__ == '__main__':
    main()
