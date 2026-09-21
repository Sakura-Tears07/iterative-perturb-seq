"""Time one real GEARS training cycle + evaluation on the IterPert pipeline.

Purpose: decide whether the *real* closed loop (6 GEARS retrains per method) is
affordable for the small experiments we want to run, and give the user a hard
number instead of a guess.

Run:  iterpert_env/bin/python bench_gears_epoch.py --n_labeled 100 --epochs 2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/lihaoran/ai4s/iterative-perturb-seq')
from iterpert.iterpert import IterPert  # noqa: E402

PATH = '/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
DATASET = 'replogle_k562_essential_1000hvg'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_labeled', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--strategy', default='IterPert')
    ap.add_argument('--out', default='/home/lihaoran/ai4s/analysis/results/bench_gears.json')
    args = ap.parse_args()

    t0 = time.time()
    itf = IterPert(weight_bias_track=False, exp_name='bench', device='cuda', seed=1)
    itf.initialize_data(path=PATH, dataset_name=DATASET, batch_size=args.batch_size)
    t_data = time.time() - t0
    print(f'[bench] data ready in {t_data:.1f}s  '
          f'n_pool={itf.dataset.n_pool} n_test={itf.dataset.n_test}', flush=True)

    t0 = time.time()
    itf.initialize_model(epochs=args.epochs, hidden_size=args.hidden)
    print(f'[bench] model init in {time.time()-t0:.1f}s', flush=True)

    t0 = time.time()
    itf.initialize_active_learning_strategy(strategy=args.strategy)
    t_strat = time.time() - t0
    print(f'[bench] strategy init (kernels) in {t_strat:.1f}s', flush=True)

    init_idx = itf.dataset.initialize_labels(args.n_labeled)
    print(f'[bench] initial labels: {args.n_labeled}', flush=True)

    t0 = time.time()
    itf.strategy.train()
    t_train = time.time() - t0
    print(f'[bench] GEARS train ({args.epochs} epochs) = {t_train:.1f}s '
          f'({t_train/args.epochs:.1f}s/epoch)', flush=True)

    t0 = time.time()
    res, out = itf.strategy.eval(itf.test_data)
    t_eval = time.time() - t0
    pear = np.mean([j['pearson_delta'] for i, j in out.items() if 'pearson_delta' in j])
    mse = np.mean([j['mse_non_dropout'] for i, j in out.items() if 'mse_non_dropout' in j])
    print(f'[bench] eval = {t_eval:.1f}s  pearson_delta={pear:.4f} mse={mse:.4f}', flush=True)

    t0 = time.time()
    q = itf.strategy.query(20, round=2)
    t_query = time.time() - t0
    print(f'[bench] query(20) = {t_query:.1f}s -> {len(q)} idxs', flush=True)

    proj = dict(data_s=t_data, train_s=t_train, eval_s=t_eval, query20_s=t_query,
                epochs=args.epochs, s_per_epoch=t_train / args.epochs,
                n_labeled=args.n_labeled, hidden=args.hidden,
                pearson_delta=float(pear), mse_non_dropout=float(mse))
    # cost of a full 6-round loop, 20 epochs each, with one query per round
    per_round = 20 * proj['s_per_epoch'] + t_eval + t_query
    proj['est_full_loop_20ep_s'] = 6 * per_round
    proj['est_full_loop_20ep_min'] = 6 * per_round / 60
    print(f'[bench] ESTIMATE full 6-round loop @20 epochs: '
          f'{proj["est_full_loop_20ep_min"]:.1f} min per method', flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(proj, open(args.out, 'w'), indent=2)
    print('[bench] wrote', args.out)


if __name__ == '__main__':
    main()
