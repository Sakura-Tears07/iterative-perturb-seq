"""Stage 10: does ANY allocation improve the DOWNSTREAM endpoint?

stage9 showed that (a) measurement-reliability headroom is real on tail objectives
(oracle +0.040 on frac r>=0.7), (b) a dev-fitted probe captures ~31% of it
(+0.0125 [+0.0092,+0.0158]), but (c) on the downstream endpoint -- predicting the
responses of unseen perturbations -- every policy, including the objective-matched
reliability oracle, is slightly NEGATIVE.

That raises the decisive question: is there an allocation that improves the
downstream endpoint at all, or is breadth simply optimal for prediction at these
budgets? We answer it with a downstream-matched oracle:

    oracle_downstream : within the same 25->100 two-level family, rank candidates by
    their TRUE marginal effect on downstream Pearson (computed by singleton deepening),
    then deepen the top-k. Diagnostic only (uses eval truth).

We also quantify the tension between the two objectives by correlating the
per-(split,budget) deltas of the two metrics.

Usage: iterpert_env/bin/python stage10_downstream_oracle.py --splits 10
"""
import argparse
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
N_EVAL = 200


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
    kd = kc.load_all()
    fused = kc.normalize_kernel(np.mean([kc.normalize_kernel(kd['priors'][n])
                                         for n in kc.PRIOR_NAMES], axis=0))
    base_of = lambda s: s.split('+')[0]
    kern_axis = {p: i for i, p in enumerate(kd['perts'])}
    counts = {c: int((cond == c).sum()) for c in np.unique(cond) if c != 'ctrl'}
    pool = {c: n - int(min(80, max(20, round(0.30 * n)))) for c, n in counts.items()}
    cand = sorted([c for c, v in pool.items() if v >= D_DEEP and base_of(c) in kern_axis])
    rest = [c for c in pool if c not in set(cand) and counts[c] >= 40
            and base_of(c) in kern_axis]
    rng0 = np.random.default_rng(0)
    ev = [rest[i] for i in rng0.permutation(len(rest))[:N_EVAL]]
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
    print(f'candidates={len(cand)} eval={len(ev)} prep {time.time()-t0:.0f}s', flush=True)

    rows = []
    for sp in range(args.splits):
        rng = np.random.default_rng(500 + sp)
        perm = rng.permutation(len(cand))
        dev = [cand[i] for i in perm[:int(0.7 * len(cand))]]
        test = [cand[i] for i in perm[int(0.7 * len(cand)):]]
        it_t = {c: i for i, c in enumerate(test)}
        idx_test = np.array([kern_axis[base_of(c)] for c in test])
        idx_ev = np.array([kern_axis[base_of(c)] for c in ev])
        S = fused[np.ix_(idx_test, idx_ev)]           # candidate x eval similarity

        env_t = MFBudgetEnv(cond, X, budget=1e9, candidates=test,
                            init_cells=D_SHALLOW, seed=sp)
        o25 = env_t.observation()
        D_shallow = o25.delta.copy()
        env_t.step({c: D_DEEP - D_SHALLOW for c in test})
        o100 = env_t.observation()
        D_deep = o100.delta.copy()
        sc = env_t.scorer()
        r25 = sc.reliability(o25)
        r100 = sc.reliability(o100)
        # probe fitted on dev (3 features + prior neighbourhood)
        env_d = MFBudgetEnv(cond, X, budget=1e9, candidates=dev, init_cells=D_SHALLOW, seed=sp)
        od25 = env_d.observation()
        r25_d = env_d.scorer().reliability(od25)
        env_d.step({c: D_DEEP - D_SHALLOW for c in dev})
        gd = env_d.scorer().reliability(env_d.observation()) - r25_d
        Fd = np.c_[od25.half_split_r, od25.delta_norm, od25.cell_sd]
        mu, sd = Fd.mean(0), Fd.std(0) + 1e-12
        w = np.linalg.lstsq(np.c_[(Fd - mu) / sd, np.ones(len(dev))], gd, rcond=None)[0]
        Ft = np.c_[o25.half_split_r, o25.delta_norm, o25.cell_sd]
        score_probe = ((Ft - mu) / sd) @ w[:-1] + w[-1]

        def downstream(Dtr):
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
            return float(np.mean(rr))

        ds_uniform = downstream(D_shallow)
        # singleton marginal effect on downstream of deepening candidate i
        marg = np.zeros(len(test))
        for i in range(len(test)):
            D = D_shallow.copy()
            D[i] = D_deep[i]
            marg[i] = downstream(D) - ds_uniform
        # reliability marginal for the tail objective
        marg_rel = (r100 >= 0.7).astype(float) - (r25 >= 0.7).astype(float)

        for B in BUDGETS:
            k = int((B - D_SHALLOW) * len(test) / (D_DEEP - D_SHALLOW))
            k = max(0, min(len(test), k))
            for name, score in [('uniform', None),
                                ('probe(dev-fitted)', score_probe),
                                ('oracle_frac_ge_0.7', marg_rel),
                                ('oracle_downstream', marg),
                                ('random', np.random.default_rng(sp).standard_normal(len(test)))]:
                # every arm goes through the env: init 25, deepen k, then the env
                # spends the leftover uniformly -> all arms share the same total cost
                env = MFBudgetEnv(cond, X, budget=float(B * len(test)),
                                  candidates=test, init_cells=D_SHALLOW, seed=sp)
                if score is not None and k > 0:
                    env.step({test[i]: D_DEEP - D_SHALLOW
                              for i in np.argsort(-score)[:k]})
                env.finalize_uniform_leftover()
                obs = env.observation()
                r_alloc = env.scorer().reliability(obs)
                row = dict(split=sp, budget=B, policy=name,
                           mean_r=float(np.mean(r_alloc)),
                           downstream=downstream(obs.delta),
                           cost=env._cost['total'])
                row['frac_ge_0.7'] = float(np.mean(r_alloc >= 0.7))
                rows.append(row)
        print(f'  split {sp} ({time.time()-t0:.0f}s)', flush=True)

    import pandas as pd
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, f'stage10_downstream_oracle_{args.tag}.csv'), index=False)
    print('\n=== paired deltas vs uniform (10 splits x 4 budgets) ===')
    base_f = res[res.policy == 'uniform'].set_index(['budget', 'split'])['frac_ge_0.7']
    base_d = res[res.policy == 'uniform'].set_index(['budget', 'split'])['downstream']
    print(f'{"policy":22s}{"d(frac>=0.7)":>14s}{"95% CI":>20s}{"d(downstream)":>15s}{"95% CI":>20s}')
    for name in sorted(res.policy.unique()):
        if name == 'uniform':
            continue
        df_ = (res[res.policy == name].set_index(['budget', 'split'])['frac_ge_0.7'] - base_f).dropna()
        dd_ = (res[res.policy == name].set_index(['budget', 'split'])['downstream'] - base_d).dropna()
        cf = 1.96 * df_.std(ddof=1) / np.sqrt(len(df_))
        cd = 1.96 * dd_.std(ddof=1) / np.sqrt(len(dd_))
        print(f'{name:22s}{df_.mean():+14.4f}'
              f'{"["+f"{df_.mean()-cf:+.4f},{df_.mean()+cf:+.4f}"+"]":>20s}'
              f'{dd_.mean():+15.4f}'
              f'{"["+f"{dd_.mean()-cd:+.4f},{dd_.mean()+cd:+.4f}"+"]":>20s}')
    print(f'\nuniform absolute: frac>=0.7={base_f.mean():.4f}, '
          f'downstream={base_d.mean():.4f}')
    # tension between the two objectives across policies
    base = (res[res.policy == 'uniform']
            .set_index(['budget', 'split'])[['frac_ge_0.7', 'downstream']]
            .rename(columns={'frac_ge_0.7': 'b_rel', 'downstream': 'b_ds'}))
    m = res.join(base, on=['budget', 'split'])
    m = m[m.policy != 'uniform']
    m['d_rel'] = m['frac_ge_0.7'] - m['b_rel']
    m['d_ds'] = m['downstream'] - m['b_ds']
    cc = np.corrcoef(m.d_rel, m.d_ds)[0, 1]
    print(f'corr across policies of d(frac>=0.7) vs d(downstream) = {cc:+.3f}')
    print('cost equality across policies: '
          f'{bool(res.groupby(["budget","split"]).cost.nunique().eq(1).all())}')
    print('wrote', os.path.join(OUT, f'stage10_downstream_oracle_{args.tag}.csv'))


if __name__ == '__main__':
    main()
