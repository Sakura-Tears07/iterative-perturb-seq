"""Stage 6b: robustness sweep of the cell-allocation decision problem.

Reads the cached per-perturbation reliability curves from stage6 and asks, over a
range of average budgets and objectives:

    At a fixed total cell budget, can ANY allocation rule beat uniform
    allocation, and is the winner implementable from online-observable data?

Rules compared
    uniform        : same depth for every perturbation (largest measured depth <= B)
    oracle         : rank by TRUE gain r(d_deep)-r(d_shallow)
    probe          : rank by a linear probe on online features (fit on 70%, applied to 30%)
    est_norm       : rank by observed effect magnitude at depth 25 (simple heuristic)
    half_split_r   : rank by split-half agreement at depth 25
    random         : random subset gets the deep allocation (control)

Objectives
    mean_r         : mean reliability across perturbations
    frac_r_ge_0.5  : fraction of perturbations measured with r >= 0.5
    frac_r_ge_0.7  : fraction measured with r >= 0.7
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CSV = '/home/lihaoran/ai4s/analysis/results/stage6_multifidelity_per_pert.csv'
OUT = '/home/lihaoran/ai4s/analysis/results/stage6b_allocation_sweep.json'
DEPTHS = [10, 25, 50, 100, 150, 200]
FEATS = ['half_split_r', 'est_norm', 'cell_sd', 'noise_ratio']
rng = np.random.default_rng(0)


def rel_at(row, d):
    for cand in DEPTHS:
        if cand >= d:
            v = row.get(f'r{cand}')
            return float(v) if v is not None and not pd.isna(v) else np.nan
    return np.nan


def objective(vals, kind):
    v = vals[~np.isnan(vals)]
    if kind == 'mean_r':
        return float(v.mean())
    if kind == 'frac_r_ge_0.5':
        return float((v >= 0.5).mean())
    if kind == 'frac_r_ge_0.7':
        return float((v >= 0.7).mean())
    raise ValueError(kind)


def main():
    df = pd.read_csv(CSV)
    df = df.dropna(subset=['r25', 'r50', 'r100', 'half_split_r']).copy()
    df = df[df.n_pool >= 100]
    N = len(df)
    y_gain = (df.r100 - df.r25).values
    F = df[FEATS].values
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Fs = (F - mu) / sd
    tr = rng.permutation(N)
    fit, app = tr[:int(0.7 * N)], tr[int(0.7 * N):]
    w = np.linalg.lstsq(np.c_[Fs[fit], np.ones(len(fit))], y_gain[fit], rcond=None)[0]
    probe = Fs @ w[:-1] + w[-1]
    print(f'population: N={N} perturbations with pool>=100 (of {len(pd.read_csv(CSV))} measured)')
    print(f'probe test-set spearman(probe, true gain) = '
          f'{spearmanr(probe[app], y_gain[app]).statistic:+.3f}')

    rows = []
    for B in [25, 37.5, 50, 62.5, 75, 100]:
        d_shallow = max([d for d in DEPTHS if d <= B], default=25)
        d_deep = min([d for d in DEPTHS if d > d_shallow], default=d_shallow)
        if d_deep == d_shallow:
            continue
        k = int((B - d_shallow) * N / (d_deep - d_shallow))
        k = max(0, min(N, k))
        for rule, score in [
            ('uniform', None),
            ('oracle', y_gain),
            ('probe', probe),
            ('est_norm', df.est_norm.values),
            ('half_split_r', df.half_split_r.values),
            ('random', rng.standard_normal(N)),
        ]:
            if rule == 'uniform':
                depth = np.full(N, float(d_shallow))
            else:
                depth = np.full(N, float(d_shallow))
                depth[np.argsort(-score)[:k]] = d_deep
            rr = np.array([rel_at(df.iloc[i], depth[i]) for i in range(N)])
            for kind in ['mean_r', 'frac_r_ge_0.5', 'frac_r_ge_0.7']:
                rows.append(dict(avg_budget=B, d_shallow=d_shallow, d_deep=d_deep,
                                 k_deep=k, rule=rule, objective=kind,
                                 value=objective(rr, kind)))
    res = pd.DataFrame(rows)
    res.to_csv('/home/lihaoran/ai4s/analysis/results/stage6b_allocation_sweep.csv', index=False)

    for kind in ['mean_r', 'frac_r_ge_0.5', 'frac_r_ge_0.7']:
        print(f'\n=== objective: {kind} ===')
        piv = res[res.objective == kind].pivot(index='avg_budget', columns='rule', values='value')
        base = piv['uniform']
        order = ['uniform', 'oracle', 'probe', 'est_norm', 'half_split_r', 'random']
        print(f'{"avg budget":>10s} {"d_s->d_d":>10s} ' +
              ' '.join(f'{r:>13s}' for r in order) + '   (delta vs uniform)')
        for B in sorted(piv.index):
            delta = ' '.join(f'{piv.loc[B, r]-base[B]:+13.4f}' for r in order if r != 'uniform')
            print(f'{B:10.1f} {int(piv.loc[B].name if False else 0):10d} ' +
                  ' '.join(f'{piv.loc[B, r]:13.4f}' for r in order) + f'   {delta}')
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
