#!/usr/bin/env python3
"""Analyse the cell-budget feedback ablation.

Primary endpoint (review-specified): prediction quality on UNSEEN perturbations at a
pre-specified final budget (round 3 = 100 initial + 3x100 = 400 perturbations, every
one measured with exactly 100 cells; 40,000 purchased cells + 10,691 control cells).

Acceptance question: does the UPDATED selection policy beat (a) the static-prior
Core-Set and (b) the frozen-model-kernel variant? Beating Random alone is not enough.
All comparisons are paired over runs (same data split seed, same initial set, same
training seeds).
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

RES = '/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/res'
ARMS = ['random', 'static_prior', 'iterpert_full', 'iterpert_frozen']
METRICS = ['pearson_delta', 'pearson_delta_top20_de_non_dropout',
           'mse_top20_de_non_dropout', 'frac_opposite_direction_top20_non_dropout']


def load():
    rows = []
    for arm in ARMS:
        for f in sorted(glob.glob(f'{RES}/cb_{arm}_r*_metrics.json')):
            d = json.load(open(f))
            run = d['run']
            budget = json.load(open(f.replace('_metrics.json', '_budget.json'))) \
                if os.path.exists(f.replace('_metrics.json', '_budget.json')) else {}
            for r in d['curve']:
                if 'pearson_delta' not in r:
                    continue
                rows.append(dict(arm=arm, run=run, round=r['round'],
                                 n_labeled=r.get('n_test_perts') and None or None,
                                 purchased_cells=budget.get('purchased_cells_per_campaign'),
                                 **{m: r.get(m) for m in METRICS}))
    df = pd.DataFrame(rows)
    if len(df):
        df['n_labeled'] = 100 + 100 * df['round']
        df['purchased_cells'] = df['n_labeled'] * 100
    return df


def paired(df, metric, ref='iterpert_full', target='static_prior', rounds=(3,)):
    a = df[(df.arm == ref) & (df['round'].isin(rounds))].set_index(['run', 'round'])[metric]
    b = df[(df.arm == target) & (df['round'].isin(rounds))].set_index(['run', 'round'])[metric]
    d = (a - b).dropna()
    return d


def main():
    df = load()
    if not len(df):
        print('no results yet'); return
    pd.set_option('display.width', 200)
    print('=== per-arm, per-round means (n runs) ===')
    for m in METRICS:
        piv = df.pivot_table(index='round', columns='arm', values=m)
        cnt = df.groupby('arm').run.nunique().to_dict()
        print(f'\n--- {m} (runs per arm: {cnt}) ---')
        print(piv.round(4).to_string())
    print('\n=== PRIMARY: final budget (round 3, 400 perts, 40k purchased cells) ===')
    for m in METRICS:
        fin = df[df['round'] == 3].pivot_table(index='run', columns='arm', values=m)
        print(f'\n--- {m} ---')
        print(fin.round(4).to_string())
        for tgt in ['static_prior', 'iterpert_frozen', 'random']:
            if 'iterpert_full' in fin and tgt in fin:
                d = (fin['iterpert_full'] - fin[tgt]).dropna()
                if len(d):
                    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                    print(f'   full - {tgt:16s} mean={d.mean():+.4f}  '
                          f'per-run={[round(x,4) for x in d.values]}'
                          + (f'  t-CI=[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}]' if len(d) > 1 else ''))
    print('\n=== secondary: budget-curve AUC over rounds 0..3 (paired) ===')
    for m in ['pearson_delta', 'mse_top20_de_non_dropout']:
        w = df.pivot_table(index=['run', 'round'], columns='arm', values=m)
        def auc(s):
            s = s.dropna()
            if s.empty: return np.nan
            x = np.array([100 + 100 * i for i in s.index])
            return float(np.trapezoid(s.values, x) / (x[-1] - x[0]))
        a = w.groupby(level='run').apply(lambda g: auc(g['iterpert_full']), include_groups=False)
        for tgt in ['static_prior', 'iterpert_frozen', 'random']:
            b = w.groupby(level='run').apply(lambda g: auc(g[tgt]), include_groups=False)
            d = (a - b).dropna()
            if len(d):
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                print(f'{m}: full - {tgt:16s} AACU={d.mean():+.5f} per-run={[round(x,5) for x in d.values]}'
                      + (f' t-CI=[{d.mean()-1.96*se:+.5f},{d.mean()+1.96*se:+.5f}]' if len(d) > 1 else ''))


if __name__ == '__main__':
    main()
