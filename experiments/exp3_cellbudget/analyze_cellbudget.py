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


def tcrit(n, level=0.95):
    from scipy.stats import t as _t
    return float(_t.ppf(1 - (1 - level) / 2, df=max(1, n - 1)))


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
    # ---- actual cell bill: min(available cells, cap 100) per selected perturbation
    import pickle
    import anndata as ad
    H5 = ('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
          'replogle_k562_essential_1000hvg/perturb_processed.h5ad')
    cond = ad.read_h5ad(H5, backed='r').obs['condition'].astype(str).values
    counts = {c: int((cond == c).sum()) for c in np.unique(cond)}
    bills = []
    for arm in ARMS:
        for f in sorted(glob.glob(f'{RES}/cb_{arm}_r*_metrics.json')):
            run = json.load(open(f))['run']
            pf = f.replace('_metrics.json', '.pkl')
            if not os.path.exists(pf):
                continue
            d = pickle.load(open(pf, 'rb'))
            sel = {str(x) for r in d for x in d[r]}
            sel = {x.split('+')[0] for x in sel}
            avail = np.array([counts.get(x + '+ctrl', counts.get(x, 0)) for x in sel])
            bills.append(dict(arm=arm, run=run, n_pert=len(sel),
                              n_short_of_cap=int((avail < 100).sum()),
                              bill=int(np.minimum(avail, 100).sum())))
    bl = pd.DataFrame(bills)
    print('\n=== actual cell bill (cap = at most 100 cells/perturbation) ===')
    print(bl.to_string(index=False))
    print('bill spread across arms: '
          f'{bl.bill.min()} - {bl.bill.max()} cells ({(bl.bill.max()/bl.bill.min()-1)*100:.1f}% apart); '
          f'control cells 10,691 charged separately, NOT in the perturbation bill')
    print('NOTE: the campaign therefore measures a FIXED NUMBER of perturbations with '
          'AT MOST 100 cells each -- not an exactly equal-cell budget.')

    print('\n=== PRIMARY: final budget (round 3, 400 perturbations, cap 100 cells each) ===')
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
                          + (f'  t-CI(df={len(d)-1},crit={tcrit(len(d)):.3f})='
                             f'[{d.mean()-tcrit(len(d))*se:+.4f},{d.mean()+tcrit(len(d))*se:+.4f}]'
                             if len(d) > 1 else ''))
    print('\n=== secondary: budget-curve AUC over rounds 0..3 (paired) ===')
    for m in ['pearson_delta', 'mse_top20_de_non_dropout']:
        w = df.pivot_table(index=['run', 'round'], columns='arm', values=m)
        def auc(s):
            s = s.dropna()
            if s.empty:
                return np.nan
            rounds = np.array([r for (_, r) in s.index], dtype=float)
            x = 100.0 + 100.0 * rounds
            return float(np.trapezoid(s.values, x) / (x[-1] - x[0]))
        a = w.groupby(level='run').apply(lambda g: auc(g['iterpert_full']), include_groups=False)
        for tgt in ['static_prior', 'iterpert_frozen', 'random']:
            b = w.groupby(level='run').apply(lambda g: auc(g[tgt]), include_groups=False)
            d = (a - b).dropna()
            if len(d):
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                print(f'{m}: full - {tgt:16s} AACU={d.mean():+.5f} per-run={[round(x,5) for x in d.values]}'
                      + (f' t-CI(df={len(d)-1},crit={tcrit(len(d)):.3f})='
                         f'[{d.mean()-tcrit(len(d))*se:+.5f},{d.mean()+tcrit(len(d))*se:+.5f}]'
                         if len(d) > 1 else ''))


if __name__ == '__main__':
    main()
