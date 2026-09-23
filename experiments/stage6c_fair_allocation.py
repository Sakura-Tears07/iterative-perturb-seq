"""Stage 6c: fair allocation comparison (equal cost, continuous depth).

stage6b compared two-stage mixtures against a uniform baseline that could only
use the *measured* depths, so at intermediate budgets (e.g. 37.5 cells/pert) the
uniform arm was forced to waste budget (all at 25) while the mixture spent it.
That inflates the apparent value of reallocation.

Here both arms are evaluated at exactly the same total cost using per-
perturbation linear interpolation of the measured reliability curve r(d)
(measured at 10/25/50/100/150/200). The oracle for each objective ranks by the
TRUE marginal objective value of the deep assignment, not by gain.

Objectives: mean_r, frac r>=0.5, frac r>=0.7.
Run: iterpert_env/bin/python stage6c_fair_allocation.py
"""
import json

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CSV = '/home/lihaoran/ai4s/analysis/results/stage6_multifidelity_per_pert.csv'
DEPTHS = np.array([0, 10, 25, 50, 100, 150, 200], float)
FEATS = ['half_split_r', 'est_norm', 'cell_sd', 'noise_ratio']
rng = np.random.default_rng(0)


def build_curves(df):
    """r(d) for each perturbation on DEPTHS, with r(0)=0, monotone-corrected."""
    M = np.zeros((len(df), len(DEPTHS)))
    for j, d in enumerate(DEPTHS[1:], start=1):
        col = f'r{int(d)}'
        v = df[col].values.astype(float)
        M[:, j] = np.where(np.isnan(v), np.nan, v)
    # fill censored (NaN) entries by carrying the last observed value forward
    for i in range(M.shape[0]):
        last = 0.0
        for j in range(1, M.shape[1]):
            if np.isnan(M[i, j]):
                M[i, j] = last
            else:
                last = M[i, j]
    # enforce monotone non-decreasing (measurement noise can break this)
    M = np.maximum.accumulate(M, axis=1)
    return M


def interp(M, d):
    """Vectorised linear interpolation of r(d); d may be fractional."""
    out = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        out[i] = np.interp(d[i], DEPTHS, M[i])
    return out


def main():
    df = pd.read_csv(CSV)
    min_pool = int(os.environ.get('S6C_MIN_POOL', '100'))
    df = df.dropna(subset=['r25', 'r50', 'r100', 'half_split_r']).copy()
    df = df[df.n_pool >= min_pool].reset_index(drop=True)
    N = len(df)
    M = build_curves(df)
    y_gain = M[:, DEPTHS.tolist().index(100.0)] - M[:, DEPTHS.tolist().index(25.0)]
    F = df[FEATS].values
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Fs = (F - mu) / sd
    tr = rng.permutation(N)
    fit, app = tr[:int(0.7 * N)], tr[int(0.7 * N):]
    w = np.linalg.lstsq(np.c_[Fs[fit], np.ones(len(fit))], y_gain[fit], rcond=None)[0]
    probe = Fs @ w[:-1] + w[-1]
    print(f'population N={N} (pool>=200; note: this is the high-cell subset, biases r up)')
    print(f'probe: test spearman vs true gain = '
          f'{spearmanr(probe[app], y_gain[app]).statistic:+.3f}')

    OBJ = {
        'mean_r': lambda r: float(np.mean(r)),
        'frac_r_ge_0.5': lambda r: float(np.mean(r >= 0.5)),
        'frac_r_ge_0.7': lambda r: float(np.mean(r >= 0.7)),
    }
    D_SHALLOW, D_DEEP = 25.0, 100.0
    rows = []
    for B in [30, 40, 50, 60, 75, 90]:
        frac_deep = (B - D_SHALLOW) / (D_DEEP - D_SHALLOW)
        if not (0 < frac_deep < 1):
            continue
        k = max(1, min(N - 1, int(round(frac_deep * N))))
        for name, kind in OBJ.items():
            # per-objective oracle: true marginal objective value of going deep
            r_shallow = M[:, DEPTHS.tolist().index(D_SHALLOW)]
            r_deep = M[:, DEPTHS.tolist().index(D_DEEP)]
            if kind == 'mean_r':
                marg = r_deep - r_shallow
            elif kind == 'frac_r_ge_0.5':
                marg = (r_deep >= 0.5).astype(float) - (r_shallow >= 0.5).astype(float)
            else:
                marg = (r_deep >= 0.7).astype(float) - (r_shallow >= 0.7).astype(float)
            rules = {
                'uniform': np.zeros(N, bool),
                'oracle': np.zeros(N, bool),
                'probe': np.zeros(N, bool),
                'est_norm': np.zeros(N, bool),
                'half_split_r': np.zeros(N, bool),
                'random': np.zeros(N, bool),
            }
            rules['oracle'][np.argsort(-marg)[:k]] = True
            rules['probe'][np.argsort(-probe)[:k]] = True
            rules['est_norm'][np.argsort(-df.est_norm.values)[:k]] = True
            rules['half_split_r'][np.argsort(-df.half_split_r.values)[:k]] = True
            rules['random'][rng.permutation(N)[:k]] = True
            for rule, sel in rules.items():
                if rule == 'uniform':
                    # the fixed-allocation arm spends exactly B cells on every
                    # perturbation; r(B) is read off the interpolated curve
                    depth = np.full(N, float(B))
                else:
                    depth = np.where(sel, D_DEEP, D_SHALLOW)
                actual = float(depth.mean())
                # the integer mixture can only approximate the target cost
                assert abs(actual - B) < (D_DEEP - D_SHALLOW) / N + 1e-9, (actual, B)
                r = interp(M, depth)
                rows.append(dict(avg_budget=B, objective=name, rule=rule,
                                 actual_budget=actual, value=OBJ[name](r),
                                 k_deep=int(sel.sum())))
    res = pd.DataFrame(rows)
    res.to_csv('/home/lihaoran/ai4s/analysis/results/stage6c_fair_allocation.csv', index=False)
    for name in OBJ:
        print(f'\n=== objective {name} (fair: equal cost, interpolated uniform) ===')
        piv = res[res.objective == name].pivot(index='avg_budget', columns='rule', values='value')
        order = ['uniform', 'oracle', 'probe', 'est_norm', 'half_split_r', 'random']
        print(f'{"cells/pert":>10s} ' + ' '.join(f'{r:>13s}' for r in order))
        for B in sorted(piv.index):
            print(f'{B:10.1f} ' + ' '.join(f'{piv.loc[B, r]:13.4f}' for r in order))
        print('   delta vs uniform: ' + ' '.join(
            f'{r}={piv[r].mean()-piv["uniform"].mean():+.4f}' for r in order[1:]))
    tag = f'pool{min_pool}'
    res.to_csv(f'/home/lihaoran/ai4s/analysis/results/stage6c_fair_allocation_{tag}.csv', index=False)
    json.dump(res.to_dict('records'),
              open(f'/home/lihaoran/ai4s/analysis/results/stage6c_fair_allocation_{tag}.json', 'w'),
              indent=2)
    print(f'\nwrote stage6c_fair_allocation_{tag}.csv/.json')


if __name__ == '__main__':
    main()
