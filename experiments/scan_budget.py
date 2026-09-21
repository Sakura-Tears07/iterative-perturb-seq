"""Budget scan: where in (n_init, n_query) does the selection strategy matter?

Motivation
----------
At the paper's budget (100 init + 5x100) every selection strategy lands within
0.003 KRR-AUC of uniform fusion, but at a much smaller budget (20 init + 5x10)
the spread opens up to 0.02. Before proposing any new selection method we need to
know whether that is a real effect or noise, and where the useful regime is.

This driver sweeps (n_init, n_query) with the same closed-loop proxy used in
stage2_proxy, several seeds, and reports for each cell of the grid:

    spread   = best method - worst method      (does the choice matter at all?)
    gain     = best method - uniform fusion    (is there anything to win?)
    gain_ada = best ADAPTIVE method - uniform  (can learning beat a fixed rule?)

Run: iterpert_env/bin/python scan_budget.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/lihaoran/ai4s/analysis')
import kernels_common as kc            # noqa: E402
import stage2_proxy as s2              # noqa: E402

OUT = '/home/lihaoran/ai4s/analysis/results'
METHODS = ['random', 'uniform', 'priors_only_uniform', 'base_only',
           'align', 'shrink_align', 'ridge', 'topk_align', 'oracle_truth']
ADAPTIVE = ['align', 'shrink_align', 'ridge', 'oracle_truth']


def main():
    grid = [(20, 5), (20, 10), (20, 25), (20, 50),
            (40, 5), (40, 10), (40, 25),
            (60, 10), (60, 25),
            (100, 10), (100, 25), (100, 50), (100, 100)]
    seeds = [0, 1, 2]
    n_round = 5
    scenario = 'clean'
    d = kc.load_all()
    sc = s2.build_scenario(scenario, d)

    rows = []
    t_start = time.time()
    for (ni, nq) in grid:
        per_method = {}
        for m in METHODS:
            curves = [s2.run_method(m, s, sc, ni, n_round, nq, real_init=(s == 0))
                      for s in seeds]
            per_method[m] = [float(np.nanmean([c[r].get('krr_pearson_all', np.nan)
                                               for r in range(n_round + 1)]))
                             for c in curves]
        mean_auc = {m: float(np.mean(v)) for m, v in per_method.items()}
        order = sorted(mean_auc, key=lambda x: -mean_auc[x])
        best, worst = order[0], order[-1]
        ada = max(ADAPTIVE, key=lambda x: mean_auc[x])
        budget = ni + n_round * nq
        rows.append(dict(n_init=ni, n_query=nq, total_budget=budget,
                         mean_auc=mean_auc, best=best, worst=worst,
                         spread=mean_auc[best] - mean_auc[worst],
                         gain=mean_auc[best] - mean_auc['uniform'],
                         gain_ada=mean_auc[ada] - mean_auc['uniform'],
                         best_adaptive=ada))
        el = time.time() - t_start
        print(f'n_init={ni:4d} n_query={nq:4d} budget={budget:4d} | '
              f'best={best:20s} spread={rows[-1]["spread"]:+.4f} '
              f'gain={rows[-1]["gain"]:+.4f} gain_ada={rows[-1]["gain_ada"]:+.4f} '
              f'| {el:.0f}s', flush=True)
        json.dump(rows, open(os.path.join(OUT, 'budget_scan.json'), 'w'), indent=2)

    print('\n=== summary: strategy value vs total budget ===')
    print(f'{"budget":>7s} {"n_init":>7s} {"n_query":>8s} {"spread":>8s} '
          f'{"gain":>8s} {"gain_ada":>9s}  best method')
    for r in sorted(rows, key=lambda x: x['total_budget']):
        print(f'{r["total_budget"]:7d} {r["n_init"]:7d} {r["n_query"]:8d} '
              f'{r["spread"]:8.4f} {r["gain"]:8.4f} {r["gain_ada"]:9.4f}  {r["best"]}')
    print('\nwrote', os.path.join(OUT, 'budget_scan.json'))


if __name__ == '__main__':
    main()
