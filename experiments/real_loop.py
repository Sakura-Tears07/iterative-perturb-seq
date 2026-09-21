"""Run the real IterPert closed loop with a selectable prior-fusion mode.

This is the honest, full-fidelity experiment: GEARS is retrained every round and
the reported metric is the same hold-out evaluation the paper uses. It is
expensive (~77 min per method at 20 epochs), so we run one method per GPU.

Examples:
  CUDA_VISIBLE_DEVICES=0 python real_loop.py --strategy Random     --tag random
  CUDA_VISIBLE_DEVICES=1 python real_loop.py --strategy IterPert --mode mean_new  --tag iterpert_mean
  CUDA_VISIBLE_DEVICES=2 python real_loop.py --strategy IterPert --mode alignment --tag iterpert_align
  CUDA_VISIBLE_DEVICES=3 python real_loop.py --strategy IterPert --mode coeff     --tag iterpert_coeff
"""
import argparse
import os
import sys
import time

sys.path.insert(0, '/home/lihaoran/ai4s/iterative-perturb-seq')

PATH = '/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
DATASET = 'replogle_k562_essential_1000hvg'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='IterPert')
    ap.add_argument('--mode', default='mean_new',
                    choices=['mean_new', 'mean', 'alignment', 'coeff', 'learn',
                             'product', 'max', 'softmax_align', 'best_align',
                             'ridge_lab'])
    ap.add_argument('--tag', default=None)
    ap.add_argument('--seed', type=int, default=1, help='data split / initial set seed')
    ap.add_argument('--run', type=int, default=1, help='torch training seed')
    ap.add_argument('--n_init', type=int, default=100)
    ap.add_argument('--n_round', type=int, default=5)
    ap.add_argument('--n_query', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--fix_evaluation', type=int, default=1,
                    help='1 = evaluate at the last epoch (pipeline default)')
    ap.add_argument('--fix_typiclust', type=int, default=0,
                    help='1 = use the intended linear_fix_ctrl kernel for TypiClust')
    args = ap.parse_args()

    from iterpert.iterpert import IterPert
    tag = args.tag or (args.strategy + '_' + args.mode)
    t0 = time.time()
    print(f'[real_loop] tag={tag} strategy={args.strategy} mode={args.mode} '
          f'seed={args.seed} run={args.run} CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}',
          flush=True)

    itf = IterPert(weight_bias_track=False, exp_name=tag, device='cuda',
                   seed=args.seed, run=args.run)
    itf.initialize_data(path=PATH, dataset_name=DATASET, batch_size=args.batch_size)
    print(f'[real_loop] data ready {time.time()-t0:.0f}s  '
          f'pool={itf.dataset.n_pool} test={itf.dataset.n_test}', flush=True)

    itf.initialize_model(epochs=args.epochs, hidden_size=args.hidden,
                         fix_evaluation=bool(args.fix_evaluation))
    itf.initialize_active_learning_strategy(strategy=args.strategy,
                                            integrate_mode=args.mode,
                                            fix_typiclust_branch=bool(args.fix_typiclust))
    print(f'[real_loop] model+strategy ready {time.time()-t0:.0f}s', flush=True)

    itf.start(n_init_labeled=args.n_init, n_round=args.n_round,
              n_query=args.n_query)
    print(f'[real_loop] DONE tag={tag} total={time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
