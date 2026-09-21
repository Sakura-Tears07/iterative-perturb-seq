"""Is the cheap proxy a valid screen for the real GEARS loop?

The proxy (stage2_proxy.py) replaces the 2-hour GEARS closed loop with a
3-minute selection-only simulation. That is only useful if its ranking agrees
with the real thing. This script does two checks with the real runs that are
already on disk:

  (1) Ranking agreement. For every pair of methods, does the proxy order match
      the real closed-loop order?
  (2) Selection-level transfer. Independently of any strategy, take the
      perturbation sets the real runs actually selected, score those exact sets
      with the proxy metric, and correlate that score with GEARS' observed
      pearson_delta. This isolates "does the proxy metric see what GEARS sees?"
      from "does the proxy's own selection loop match GEARS'?".

Both are limited by having few real runs, which is reported explicitly.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, '/home/lihaoran/ai4s/analysis')
import kernels_common as kc            # noqa: E402
import stage2_proxy as s2              # noqa: E402

RES = '/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/res'
OUT = '/home/lihaoran/ai4s/analysis/results'


def pearson_auc(d):
    vals = [r['pearson_delta'] for r in d['curve'] if 'pearson_delta' in r]
    return float(np.mean(vals)), vals


def main():
    d = kc.load_all()
    sc = s2.build_scenario('clean', d)
    Y, te = d['truth_feat'], d['test_idx']
    tr = d['train_idx']
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    ref_kernel = np.exp(-(1.0 - np.clip(Yn @ Yn.T, -1, 1)) / 0.2)
    genes_top = np.argsort(-Y[tr].var(axis=0))[:50]
    idx_of = {p: i for i, p in enumerate(d['perts'])}

    runs = []
    for mf in sorted(glob.glob(os.path.join(RES, 'rvs_*_metrics.json'))):
        tag = json.load(open(mf))['exp_name']
        pf = os.path.join(RES, tag + '.pkl')
        if not os.path.exists(pf):
            print(f'  (skip {tag}: no selection pickle yet)')
            continue
        rq = pickle.load(open(pf, 'rb'))
        d_metrics = json.load(open(mf))
        auc, vals = pearson_auc(d_metrics)
        # labelled set at the last round = round 0 set + every later query set
        perts = []
        for r in sorted(rq, key=lambda x: int(x)):
            perts.extend([str(p).split('+')[0] for p in rq[r]])
        sel = np.array(sorted({idx_of[p] for p in perts if p in idx_of}))
        n_expected = d_metrics['n_init_labeled'] + d_metrics['n_round'] * d_metrics['n_query']
        if abs(len(sel) - n_expected) > 2:
            print(f'  WARNING {tag}: {len(sel)} perts recovered, expected ~{n_expected}')
        met = s2.evaluate_selection(sel, Y, te, genes_top, ref_kernel=ref_kernel)
        runs.append(dict(tag=tag, n_perts=len(sel), real_pearson_auc=auc,
                         real_final=vals[-1], real_vals=vals,
                         proxy_krr_auc=met['krr_pearson_all'],
                         proxy_cov=met['coverage_mean'],
                         proxy_krr_final=met['krr_pearson_top50var']))

    print('\n=== selection-level transfer: proxy score of the REAL selected sets ===')
    print(f'{"run":26s} {"n":>4s} {"real pearson AUC":>16s} {"proxy KRR":>10s} {"proxy cov":>10s}')
    for r in sorted(runs, key=lambda x: -x['real_pearson_auc']):
        print(f'{r["tag"]:26s} {r["n_perts"]:4d} {r["real_pearson_auc"]:16.4f} '
              f'{r["proxy_krr_auc"]:10.4f} {r["proxy_cov"]:10.4f}')

    if len(runs) >= 3:
        a = np.array([r['real_pearson_auc'] for r in runs])
        b = np.array([r['proxy_krr_auc'] for r in runs])
        c = np.array([r['proxy_cov'] for r in runs])
        print(f'\n  corr(real, proxy KRR) = {np.corrcoef(a, b)[0,1]:+.3f}  (n={len(runs)})')
        print(f'  corr(real, proxy cov) = {np.corrcoef(a, c)[0,1]:+.3f}  (n={len(runs)})')
        spread_real = a.max() - a.min()
        spread_proxy = b.max() - b.min()
        print(f'  real spread across runs   = {spread_real:.4f}')
        print(f'  proxy spread across runs  = {spread_proxy:.4f}')
        print(f'  ratio proxy/real spread   = {spread_proxy/max(spread_real,1e-9):.3f}')
        print('  (a proxy whose spread is tiny relative to real spread cannot rank methods)')

    json.dump(runs, open(os.path.join(OUT, 'proxy_validation.json'), 'w'), indent=2)
    print('\nwrote', os.path.join(OUT, 'proxy_validation.json'))


if __name__ == '__main__':
    main()
