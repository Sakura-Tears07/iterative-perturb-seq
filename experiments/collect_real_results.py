"""Collect the real closed-loop runs and compare them round by round.

Reads every *_metrics.json written by analysis/real_loop.py (via the patched
IterPert.start) and prints the learning curves side by side, plus the paired
difference against the uniform-fusion baseline.
"""
import glob
import json
import os
import sys

RES = '/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/res'
KEYS = ['pearson_delta', 'mse_non_dropout', 'mse_top20_de_non_dropout',
        'pearson_delta_top20_de_non_dropout',
        'frac_opposite_direction_top20_non_dropout']


def main(tags=None):
    files = sorted(glob.glob(os.path.join(RES, '*_metrics.json')))
    runs = {}
    for f in files:
        d = json.load(open(f))
        name = d['exp_name']
        if name.startswith('smokecheck'):
            continue
        if tags and name not in tags:
            continue
        runs[name] = d
    if not runs:
        print('no metrics files yet under', RES)
        return
    print(f'found {len(runs)} runs: {sorted(runs)}\n')
    for key in ['pearson_delta', 'mse_top20_de_non_dropout',
                'pearson_delta_top20_de_non_dropout',
                'frac_opposite_direction_top20_non_dropout']:
        print(f'=== {key} (round 0..5) ===')
        print(f'{"tag":26s}' + ''.join(f'{"r"+str(r):>9s}' for r in range(6)) + f'{"AUC":>9s}')
        base = None
        for name in sorted(runs):
            vals = []
            for row in runs[name]['curve']:
                if row.get('round') == 0 and key not in row:
                    continue
                vals.append(row.get(key))
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            if name == 'rl_iterpert_mean':
                base = sum(vals) / len(vals)
        for name in sorted(runs):
            vals = []
            for row in runs[name]['curve']:
                if row.get('round') == 0 and key not in row:
                    continue
                vals.append(row.get(key))
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            auc = sum(vals) / len(vals)
            line = f'{name:26s}' + ''.join(f'{v:9.4f}' for v in vals)
            line += f'{auc:9.4f}'
            if base is not None and name != 'rl_iterpert_mean':
                line += f'  ({auc-base:+.4f} vs mean-fusion)'
            print(line)
        print()


if __name__ == '__main__':
    main(sys.argv[1:] or None)
