"""Stage 11 (plan A): downstream-value-driven allocation, with isolated dev/test TASKS.

Question: can we predict, from only the feedback and priors a policy is allowed to
see, how much *deepening* a candidate will help the DOWNSTREAM task -- and does
using that prediction improve the whole-batch allocation on an independent test
task?

Design (per review spec)
  Task split: perturbations are partitioned into four DISJOINT blocks
      C_dev, C_test   (candidate pools we may allocate cells to)
      T_dev, T_test   (target perturbations we try to predict, never measured)
    The dev task uses (C_dev, T_dev) to generate labels and to choose the fixed
    strategy; the test task uses (C_test, T_test) for the final acceptance. Target
    responses are used only for scoring, never for fitting on the test task.
  Labels (dev only): marg_i = downstream(T_dev | deepen i) - downstream(T_dev | all shallow)
    measured with the kNN-in-fused-prior predictor; this is the register that the
    policy is trying to predict.
  Inputs (observable): purchased-response stats (half_split_r, delta_norm, cell_sd,
    noise_ratio, n_pool), a prior-quality feature (prior_typiclust), and STRUCTURAL
    INFLUENCE on the target set (max / top-k mean fused-kernel similarity to targets,
    and how often the candidate is among a target's k nearest candidates).
  Policies compared on the test task at equal total cost (env-billed):
      uniform, random, dev-chosen fixed strategy, reliability-driven probe,
      downstream-value-driven probe, structural-only rule (cheap control).
  Primary metric: per split, the budget curve of downstream Pearson is integrated
    (normalised trapezoidal AUC); we then take the PAIRED AUC difference vs uniform
    over splits and report its distribution and resampling interval. Per-budget
    means are reported as diagnostics only.

Usage: iterpert_env/bin/python stage11_downstream_value.py --splits 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/lihaoran/ai4s/analysis')
from mf_budget_env import MFBudgetEnv  # noqa: E402
import kernels_common as kc            # noqa: E402

H5AD = ('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
        'replogle_k562_essential_1000hvg/perturb_processed.h5ad')
OUT = '/home/lihaoran/ai4s/analysis/results'
D_SHALLOW, D_DEEP = 25, 100
BUDGETS = [40, 55, 70, 90]
K_NN = 10
FRAC_C_DEV, FRAC_C_TEST = 0.40, 0.20      # remaining 0.40 -> targets


def auc_budget(budgets, values):
    x = np.asarray(budgets, float)
    y = np.asarray(values, float)
    o = np.argsort(x)
    if x[o].max() == x[o].min():
        return float(y.mean())
    return float(np.trapezoid(y[o], x[o]) / (x[o].max() - x[o].min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=int, default=10)
    ap.add_argument('--tag', default='s10')
    args = ap.parse_args()
    t0 = time.time()

    import anndata as ad
    adata = ad.read_h5ad(H5AD, backed='r')
    cond = adata.obs['condition'].astype(str).values
    X = np.asarray(adata.to_memory().X, dtype=np.float32)
    G = X.shape[1]
    kd = kc.load_all()
    fused = kc.normalize_kernel(np.mean([kc.normalize_kernel(kd['priors'][n])
                                         for n in kc.PRIOR_NAMES], axis=0))
    base_of = lambda s: s.split('+')[0]
    kern_axis = {p: i for i, p in enumerate(kd['perts'])}
    counts = {c: int((cond == c).sum()) for c in np.unique(cond) if c != 'ctrl'}
    pool = {c: n - int(min(80, max(20, round(0.30 * n)))) for c, n in counts.items()}
    usable = sorted([c for c, v in pool.items()
                     if v >= D_DEEP and base_of(c) in kern_axis and counts[c] >= 60])
    ctrl = np.where(cond == 'ctrl')[0]
    crng = np.random.default_rng(999)
    ctrl = crng.permutation(ctrl)
    ctrl_est_m = X[ctrl[:len(ctrl) // 2]].mean(0)
    # true deltas for every usable perturbation (scoring only; independent cells)
    true_delta = {}
    for c in usable:
        ids = np.where(cond == c)[0]
        ids = ids[crng.permutation(len(ids))[:max(20, len(ids) // 2)]]
        true_delta[c] = X[ids].mean(0) - ctrl_est_m
    print(f'usable perturbations={len(usable)}  prep={time.time()-t0:.0f}s', flush=True)

    rel_rows, ds_rows, diag_rows = [], [], []
    for sp in range(args.splits):
        rng = np.random.default_rng(9000 + sp)
        perm = rng.permutation(len(usable))
        n = len(usable)
        n_cd = int(FRAC_C_DEV * n)
        n_ct = int(FRAC_C_TEST * n)
        C_dev = [usable[i] for i in perm[:n_cd]]
        C_test = [usable[i] for i in perm[n_cd:n_cd + n_ct]]
        rest = [usable[i] for i in perm[n_cd + n_ct:]]
        T_dev = rest[:len(rest) // 2]
        T_test = rest[len(rest) // 2:]

        def kern_idx(lst):
            return np.array([kern_axis[base_of(c)] for c in lst])

        def knn_downstream(D_cand, cands, targets):
            """Predict targets from candidate deltas with fused-kernel kNN."""
            S = fused[np.ix_(kern_idx(cands), kern_idx(targets))]
            kk = min(K_NN, D_cand.shape[0])
            rr = []
            for j, tc in enumerate(targets):
                col = S[:, j]
                top = np.argpartition(-col, kk - 1)[:kk]
                w = np.clip(col[top], 0, None)
                w = w / w.sum() if w.sum() > 0 else np.ones(kk) / kk
                pred = w @ D_cand[top]
                a = pred - pred.mean()
                b = true_delta[tc] - true_delta[tc].mean()
                rr.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            return float(np.mean(rr))

        def structural_features(cands, targets):
            S = fused[np.ix_(kern_idx(cands), kern_idx(targets))]
            cov_max = S.max(axis=1)
            kk = min(K_NN, S.shape[1])
            part = np.partition(-S, kk - 1, axis=1)[:, :kk]
            cov_topk = -part.mean(axis=1)
            # rank[i, j] = rank of candidate i among candidates for target j
            rank = np.argsort(np.argsort(-S, axis=0), axis=0)
            in_nn = (rank < K_NN).sum(axis=1).astype(float)   # per candidate
            return np.c_[cov_max, cov_topk, in_nn]

        # ---------------- DEV task: labels + fitted probes -----------------
        env_cd = MFBudgetEnv(cond, X, budget=1e9, candidates=C_dev,
                             init_cells=D_SHALLOW, seed=sp)
        o25 = env_cd.observation()
        D_shallow = o25.delta.copy()
        r25 = env_cd.scorer().reliability(o25)
        env_cd.step({c: D_DEEP - D_SHALLOW for c in C_dev})
        o100 = env_cd.observation()
        D_deep = o100.delta.copy()
        r100 = env_cd.scorer().reliability(o100)
        base_ds = knn_downstream(D_shallow, C_dev, T_dev)
        marg = np.zeros(len(C_dev))
        for i in range(len(C_dev)):
            D = D_shallow.copy()
            D[i] = D_deep[i]
            marg[i] = knn_downstream(D, C_dev, T_dev) - base_ds
        stf = structural_features(C_dev, T_dev)

        def fit(target):
            F = np.c_[o25.half_split_r, o25.delta_norm, o25.cell_sd,
                      o25.delta_norm / (np.sqrt(G) * (o25.cell_sd + 1e-9)),
                      np.array([len(env_cd._order[c]) for c in C_dev]),
                      stf]
            mu, sd = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-12
            Fz = np.where(np.isnan(F), mu, F)
            Fz = (Fz - mu) / sd
            w = np.linalg.lstsq(np.c_[Fz, np.ones(len(Fz))], target, rcond=None)[0]
            return mu, sd, w

        from scipy.stats import spearmanr
        mu_ds, sd_ds, w_ds = fit(marg)
        mu_rl, sd_rl, w_rl = fit(r100 - r25)
        Fd = np.c_[o25.half_split_r, o25.delta_norm, o25.cell_sd,
                   o25.delta_norm / (np.sqrt(G) * (o25.cell_sd + 1e-9)),
                   np.array([len(env_cd._order[c]) for c in C_dev]), stf]
        Fdz = (np.where(np.isnan(Fd), mu_ds, Fd) - mu_ds) / sd_ds
        nz = np.abs(marg) > 1e-12
        diag_rows.append(dict(split=sp,
                              label_spearman_dev=float(spearmanr(Fdz @ w_ds[:-1], marg).statistic),
                              marg_mean=float(marg.mean()), marg_sd=float(marg.std()),
                              marg_frac_zero=float(1 - nz.mean()),
                              marg_p90=float(np.percentile(np.abs(marg), 90)),
                              frac_structural_influential=float((stf[:, 2] > 0).mean()),
                              base_downstream_dev=base_ds,
                              n_cand_dev=len(C_dev), n_cand_test=len(C_test),
                              n_target_dev=len(T_dev), n_target_test=len(T_test)))
        # dev-chosen fixed strategy: evaluate a small set of FIXED rules on dev
        fixed_rules = {'est_norm': o25.delta_norm, 'half_split_r': o25.half_split_r,
                       'structural_in_nn': stf[:, 2],
                       'random': rng.standard_normal(len(C_dev))}
        dev_scores = {}
        for nm, sc_ in fixed_rules.items():
            D = D_shallow.copy()
            kk = max(1, int(0.25 * len(C_dev)))
            D[np.argsort(-sc_)[:kk]] = D_deep[np.argsort(-sc_)[:kk]]
            dev_scores[nm] = knn_downstream(D, C_dev, T_dev)
        best_fixed = max(dev_scores, key=dev_scores.get)
        print(f'  split {sp}: dev-chosen fixed={best_fixed} '
              f'({dev_scores[best_fixed]:.4f} vs base {base_ds:.4f}), '
              f'label spearman={diag_rows[-1]["label_spearman_dev"]:+.3f}', flush=True)

        # ---------------- TEST task: acceptance ----------------------------
        env_ct = MFBudgetEnv(cond, X, budget=1e9, candidates=C_test,
                             init_cells=D_SHALLOW, seed=sp)
        ot = env_ct.observation()
        stf_t = structural_features(C_test, T_test)
        F_t = np.c_[ot.half_split_r, ot.delta_norm, ot.cell_sd,
                    ot.delta_norm / (np.sqrt(G) * (ot.cell_sd + 1e-9)),
                    np.array([len(env_ct._order[c]) for c in C_test]), stf_t]
        Ft_ds = (np.where(np.isnan(F_t), mu_ds, F_t) - mu_ds) / sd_ds
        Ft_rl = (np.where(np.isnan(F_t), mu_rl, F_t) - mu_rl) / sd_rl
        policies = {
            'uniform': None, 'random': np.random.default_rng(sp).standard_normal(len(C_test)),
            'reliability_driven': Ft_rl @ w_rl[:-1],
            'downstream_driven': Ft_ds @ w_ds[:-1],
            'structural_only': stf_t[:, 2],
            f'fixed_dev_chosen({best_fixed})': fixed_rules_t if False else None,
        }
        fixed_test = {'est_norm': ot.delta_norm, 'half_split_r': ot.half_split_r,
                      'structural_in_nn': stf_t[:, 2],
                      'random': np.random.default_rng(sp + 1).standard_normal(len(C_test))}
        policies[f'fixed_dev_chosen({best_fixed})'] = fixed_test[best_fixed]
        ds_curve, rel_curve = {}, {}
        for B in BUDGETS:
            k = int((B - D_SHALLOW) * len(C_test) / (D_DEEP - D_SHALLOW))
            k = max(0, min(len(C_test), k))
            for name, score in policies.items():
                env = MFBudgetEnv(cond, X, budget=float(B * len(C_test)),
                                  candidates=C_test, init_cells=D_SHALLOW, seed=sp)
                if score is not None and k > 0:
                    env.step({C_test[i]: D_DEEP - D_SHALLOW
                              for i in np.argsort(-score)[:k]})
                env.finalize_uniform_leftover()
                o = env.observation()
                r_alloc = env.scorer().reliability(o)
                ds_curve[(name, B)] = knn_downstream(o.delta, C_test, T_test)
                rel_curve[(name, B)] = float(np.mean(r_alloc >= 0.7))
        for name in policies:
            ds_rows.append(dict(split=sp, policy=name,
                                auc_downstream=auc_budget(BUDGETS, [ds_curve[(name, B)] for B in BUDGETS]),
                                **{f'ds_b{B}': ds_curve[(name, B)] for B in BUDGETS}))
            rel_rows.append(dict(split=sp, policy=name,
                                 auc_frac70=auc_budget(BUDGETS, [rel_curve[(name, B)] for B in BUDGETS])))

    import pandas as pd
    pd.DataFrame(ds_rows).to_csv(os.path.join(OUT, f'stage11_downstream_auc_{args.tag}.csv'), index=False)
    pd.DataFrame(rel_rows).to_csv(os.path.join(OUT, f'stage11_reliability_auc_{args.tag}.csv'), index=False)
    pd.DataFrame(diag_rows).to_csv(os.path.join(OUT, f'stage11_diagnostics_{args.tag}.csv'), index=False)
    ds = pd.DataFrame(ds_rows)
    rl = pd.DataFrame(rel_rows)

    def report(df, col, label):
        print(f'\n=== {label}: paired AUC difference vs uniform over {args.splits} splits ===')
        base = df[df.policy == 'uniform'].set_index('split')[col]
        print(f'{"policy":30s}{"mean dAUC":>11s}{"sd":>8s}{"95% CI":>20s}'
              f'{"per-budget direction":>24s}')
        for name in sorted(df.policy.unique()):
            d = (df[df.policy == name].set_index('split')[col] - base).dropna()
            if name == 'uniform':
                print(f'{name:30s}{0.0:11.4f}{0.0:8.4f}{"—":>20s}')
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            ci = 1.96 * se
            print(f'{name:30s}{d.mean():11.4f}{d.std(ddof=1):8.4f}'
                  f'{"["+f"{d.mean()-ci:+.4f},{d.mean()+ci:+.4f}"+"]":>20s}')
        print(f'uniform absolute AUC: {base.mean():.4f} '
              f'(sd over splits {base.std(ddof=1):.4f})')

    report(ds, 'auc_downstream', 'PRIMARY: downstream Pearson')
    report(rl, 'auc_frac70', 'secondary: reliability frac>=0.7')
    dg = pd.DataFrame(diag_rows)
    print(f'\nlabel diagnostics: dev marg mean={dg.marg_mean.mean():+.4f} '
          f'({dg.marg_mean.min():+.4f}..{dg.marg_mean.max():+.4f}), '
          f'label spearman dev={dg.label_spearman_dev.mean():+.3f}')
    print('wrote', os.path.join(OUT, f'stage11_*_{args.tag}.csv'))


if __name__ == '__main__':
    main()
