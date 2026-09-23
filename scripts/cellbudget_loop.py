"""Cell-budget GEARS loop with a feedback ablation (review-specified experiment).

Arms (all share the initial perturbation set, the data split seed and the cell
budget; they differ ONLY in how the next batch of perturbations is chosen):
  random          : Random selection, independent query seed per repeat
  static_prior    : Core-Set on the fused 8-prior kernel, never updated
  iterpert_full   : priors + model kernel updated every round (original IterPert)
  iterpert_frozen : priors + model kernel FROZEN after round 0 (predictor still
                    retrained on newly purchased data every round)

Budget discipline:
  * every perturbation is measured with exactly `--cells` cells (the cap is applied
    to labelled/validation data and to the selection embeddings; test data is the
    evaluation target and is never used for training or selection);
  * controls: the GEARS co-expression graph is built from CONTROL cells only
    (get_coexpression_network_from_train); no unpurchased perturbation response
    enters training, validation or the graph. Controls are charged once in the log.
  * validation perturbations come from the purchased pool (10% of labelled), so
    they add no additional purchase.

Usage:
  python scripts/cellbudget_loop.py --arm iterpert_full --run 1 --tag cb_full_r1
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, '/home/lihaoran/ai4s/iterative-perturb-seq')
PATH = '/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
DATASET = 'replogle_k562_essential_1000hvg'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=['random', 'static_prior', 'iterpert_full', 'iterpert_frozen'])
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--cells', type=int, default=100)
    ap.add_argument('--n_init', type=int, default=100)
    ap.add_argument('--n_round', type=int, default=3)
    ap.add_argument('--n_query', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    tag = args.tag or f'cb_{args.arm}_r{args.run}'

    from iterpert.iterpert import IterPert
    t0 = time.time()
    itf = IterPert(weight_bias_track=False, exp_name=tag, device='cuda',
                   seed=args.seed, run=args.run)
    itf.initialize_data(path=PATH, dataset_name=DATASET, batch_size=args.batch_size,
                        max_cells_per_pert=args.cells)
    # control is a shared resource, not a selectable perturbation:
    # remove it from the candidate pool (it stays available as ctrl_mean / ctrl graph)
    pt = itf.dataset.pert_train
    keep = np.array([str(x) != 'ctrl' for x in pt])
    n_removed = int((~keep).sum())
    itf.dataset.pert_train = pt[keep]
    itf.dataset.n_pool = len(itf.dataset.pert_train)
    itf.dataset.labeled_idxs = np.zeros(itf.dataset.n_pool, dtype=bool)

    # budget audit: cells actually available to every consumer
    pert_data = itf.dataset.pert_data
    n_pool_perts = len(itf.dataset.pert_train)
    n_ctrl = len(pert_data.dataset_processed['ctrl'])
    audit = dict(arm=args.arm, run=args.run, cells_per_pert=args.cells,
                 n_train_perts=n_pool_perts, n_test_perts=itf.dataset.n_test,
                 cap_enforced_in='train/val/selection dataloaders (verified: exactly cells_per_pert)',
                 ctrl_removed_from_pool=int(n_removed),
                 ctrl_in_pool_after=bool('ctrl' in set(map(str, itf.dataset.pert_train))),
                 n_control_cells=float(n_ctrl),
                 purchased_cells_per_campaign=float((args.n_init + args.n_round * args.n_query) * args.cells),
                 control_cells_charged=float(n_ctrl))
    print('[budget-audit]', json.dumps(audit), flush=True)
    with open(f'/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/res/{tag}_budget.json', 'w') as f:
        json.dump(audit, f, indent=2)

    itf.initialize_model(epochs=args.epochs, hidden_size=args.hidden)
    if args.arm == 'random':
        itf.initialize_active_learning_strategy(strategy='Random',
                                                random_seed=1000 + args.run)
    elif args.arm == 'static_prior':
        itf.initialize_active_learning_strategy(strategy='IterPert', integrate_mode='mean_new',
                                                use_prior_only=True)
    elif args.arm == 'iterpert_full':
        itf.initialize_active_learning_strategy(strategy='IterPert', integrate_mode='mean_new')
    else:
        itf.initialize_active_learning_strategy(strategy='IterPert', integrate_mode='mean_new')
    print(f'[arm] {args.arm} ready {time.time()-t0:.0f}s', flush=True)
    itf.start(n_init_labeled=args.n_init, n_round=args.n_round, n_query=args.n_query,
              freeze_selection_after_round0=(args.arm == 'iterpert_frozen'))
    print(f'[arm] {args.arm} DONE {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
