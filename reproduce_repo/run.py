import argparse
import numpy as np
import torch
import os
from local_paths import (
    DATA_ROOT, GEARS_DATA_PATH, SAVE_DIR, RES_DIR, KERNEL_PATHS,
    CUSTOM_TEST_SPLIT, ESSENTIAL_GENE_PATHS, resolve_result_dir,
)
from utils import get_strategy
from data_pert import Data
from nets_pert import Net
from pprint import pprint
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--run', type=int, default=1, help="random run")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--test_fraction', type=float, default=0.1, help="test_fraction")
parser.add_argument('--hidden_size', type=int, default=64, help="hidden_size")
parser.add_argument('--uncertainty', default=True, action="store_false")
parser.add_argument('--uncertainty_reg', type=float, default=1, help="uncertainty_reg")
parser.add_argument('--direction_lambda', type=float, default=1, help="direction_lambda")
parser.add_argument('--epoch_per_cycle', type=int, default=5, help="epoch_per_cycle")
parser.add_argument('--simple_loss', default=False, action="store_true")

parser.add_argument('--retrain', default=False, action="store_true")
parser.add_argument('--wandb', default=False, action="store_true")
parser.add_argument('--wb_proj_name', type=str, default='active_gears', help="cuda device")
parser.add_argument('--wb_exp_name', type=str, default='gears', help="cuda device")
parser.add_argument('--model_name', type=str, default='GEARS', choices = ['GEARS', 'scGPT'])

parser.add_argument('--device', type=str, default='cuda:0', help="cuda device (use cuda:0 with CUDA_VISIBLE_DEVICES)")
parser.add_argument('--n_init_labeled', type=int, default=10, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=10, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="adamson", choices=["adamson", 
                                                                            "replogle_k562_gw_1000hvg",
                                                                            "replogle_k562_essential_1000hvg",
                                                                            "replogle_k562_essential_1000hvg+pert_in_gene"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="kernel_based_active_learning", 
                    choices=["RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KMeansUncertainty",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "AdversarialDeepFool", 
                             "kernel_based_active_learning", 
                             "MaxDist",
                             "TypiClust",
                             "EssentialSampling",
                             "EssentialWeighted"], help="query strategy")

#parser.add_argument('--selection_method', type=str, default="maxdet", choices=['random', 'maxdiag', 'maxdet', 'bait', 'fw', 'maxdist', 'kmeanspp', 'lcmd'])
parser.add_argument('--kernel_strategy', type=str, choices=['Random', 'BALD', 'BatchBALD', 
                                                            'BAIT', 'ACS-FW', 'Core-Set', 
                                                            'LCMD', 'BADGE', 'MAXDIST', 
                                                            'KMEANSPP', 'MAXDET', 'MAXDIAG', 
                                                            'BAIT', 'FW', 'D-OptimalDesign', 
                                                            'DIR'])
parser.add_argument('--base_kernel', type=str, default="linear", choices=['ll', 'grad', 'linear', 
                                                                          'nngp', 'ntk', 'laplace', 
                                                                          'before_gene_specific_layer1', 
                                                                          'before_cross_gene', 'cross_gene_embed',
                                                                          'cross_gene_out', 'diff_effect', 'linear_fix_ctrl'])
#parser.add_argument('--kernel_transforms', type=str, default="train", choices=["train", 'pool', 'scale', 'rp', 'acs-rf'])
#parser.add_argument('--not_sel_with_train', action = 'store_true', default=False)
#parser.add_argument('--sigma', type=int, default=0.1)
#parser.add_argument('--factor', type=int, default=None)
parser.add_argument('--reduce_latent_feat_dim_via_pca', action = 'store_true', default=False)
parser.add_argument('--custom_split', action = 'store_true', default=False)
parser.add_argument('--save_kernel', action = 'store_true', default=False)

parser.add_argument('--use_prior', action = 'store_true', default=False)
parser.add_argument('--use_prior_only', action = 'store_true', default=False)
parser.add_argument('--integrate_mode',  type=str, choices=['mean', 'coeff', 'learn', 'mean_new', 'product', 'max', 'alignment'], default='mean')
parser.add_argument('--normalize_mode',  type=str, choices=['diagonal', 'mean', 'max', 'trace', 'frobenius', 'row_sum', 'centering', 'ID', 'diag'], default='diag')
parser.add_argument('--use_single_prior', action = 'store_true', default=False)
parser.add_argument('--single_prior', type=str, choices=['kg_kernel', 'pops_kernel', 'rpe1_kernel', 
                                                         'esm_kernel', 'biogpt_kernel', 'node2vec_kernel', 
                                                         'ground_truth_delta', 'gears_kernel', 'ops_kernel', 
                                                         'ops_A549_kernel', 'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel'], default='ground_truth_delta')
parser.add_argument('--use_kernel_for_kmeans', action = 'store_true', default=False)
parser.add_argument('--ops_cell',  type=str, choices=['A549', 'HeLa'], default='A549')
parser.add_argument('--ops_plate',  type=str, default='A')

parser.add_argument('--normalize_kernel', action = 'store_true', default=False)
parser.add_argument('--normalize_method',  type=str, choices=['min-max', 'feature-scale', 
                                                              'unit-length', 'centering', 'NA'], default='NA')
parser.add_argument('--sample_cells_training', action = 'store_true', default=False)
parser.add_argument('--fix_evaluation', action = 'store_true', default=False)
parser.add_argument('--kernel_normalize_feat', action = 'store_true', default=False)

parser.add_argument('--cluster_mode',  type=str, choices=['kmeans', 'kmeans++', 
                                                          'gmm_max', 'gmm_min',
                                                          'typicust', 'Agglomerative', 'Spectral'], default='kmeans')

parser.add_argument('--num_dim', type=int, default=100, help="number of init labeled samples")
parser.add_argument('--dim_reduce', type=str, default='NA', help="number of init labeled samples")

parser.add_argument('--essential_assay',  type=str, choices=['hap1', 'kbm7', 'NA'], default='NA')
parser.add_argument('--hvg_kernel', action = 'store_true', default=False)
parser.add_argument('--hvg_num', type=int, default=100)
parser.add_argument('--valid_perts', action = 'store_true', default=False)

parser.add_argument('--lamb', type=float, default=2)
parser.add_argument('--batch_exp', action = 'store_true', default=False)
parser.add_argument('--selection_log', action='store_true', default=False)
parser.add_argument('--selection_log_dir', type=str, default=os.path.join(DATA_ROOT, 'round_states'))
parser.add_argument('--duplicate_prior', type=str, default='')
parser.add_argument('--duplicate_copies', type=int, default=1)
parser.add_argument('--corrupt_prior', type=str, default='')
parser.add_argument('--corrupt_mode', type=str, choices=['permute', 'none'], default='none')
parser.add_argument('--corrupt_lambda', type=float, default=1.0,
                    help="K_λ = (1-λ)K + λ P K P^T. 0=clean, 1=full permutation")
parser.add_argument('--corrupt_seed', type=int, default=20260826,
                    help="Fixed permutation seed (independent of campaign --seed/--run)")
parser.add_argument('--drop_prior', type=str, default='',
                    help="Oracle diagnostic: remove this prior from equal fusion")
parser.add_argument('--detect_drop', action='store_true', default=False,
                    help="Hard-reject a prior when Observed KA on S_t is below --detect_tau")
parser.add_argument('--detect_tau', type=float, default=-1.0,
                    help="Reject if z < tau. Frozen in results/analysis/idea3_intervention/frozen_tau.json")
parser.add_argument('--detect_target', type=str, default='rpe1',
                    help="Prior name in fusion_alignments (without _kernel). v1 = rpe1 only")
parser.add_argument('--detect_min_round', type=int, default=1,
                    help="First round id that may reject (1 = first acquisition)")
parser.add_argument('--result_tag', type=str, default='',
                    help="If set, write metrics under results/<tag>/runs")
parser.add_argument('--model_weight', type=float, default=-1.0,
                    help="Weight on the model kernel; remaining mass is split equally across priors. <0 = equal mean")
parser.add_argument('--weight_schedule', type=str, choices=['fixed', 'early_prior', 'early_model'], default='fixed',
                    help="fixed: use --model_weight every round; early_prior: small w_model then larger; early_model: opposite")
parser.add_argument('--labeled_genes_file', type=str, default='',
                    help="Start from this labeled gene list (common-state fork)")
parser.add_argument('--load_checkpoint', type=str, default='',
                    help="Load GEARS checkpoint (config.pkl + model.pt) and skip initial train")
parser.add_argument('--dump_state_dir', type=str, default='',
                    help="If set, after each train dump labeled/pool genes + GEARS checkpoint under this dir")
parser.add_argument('--dump_n_labeled', type=str, default='100,300,500',
                    help="Comma-separated n_labeled values to dump when --dump_state_dir is set")

args = parser.parse_args()

if args.batch_exp:
    args.dataset_name += '_batch_exp'

if args.single_prior == 'ops_kernel':
    args.single_prior = 'ops_' + args.ops_cell + '_' + args.ops_plate + '_kernel'
    print('Using ' + args.single_prior)

if args.base_kernel == 'linear_fix_ctrl':
    print('using linear_fix_ctrl...')
    args.base_kernel = 'diff_effect'
    print(args.base_kernel)
    add_ctrl = True
else:
    add_ctrl = False

args.wb_exp_name = '_'.join([args.model_name, 
                             str(args.n_query), str(args.n_round), 
                             str(args.n_init_labeled), str(args.batch_size), 
                             str(args.seed)])

if args.retrain:
    args.wb_exp_name += '_rt'
else:
    args.wb_exp_name += '_nrt'

if args.epoch_per_cycle != 20:
    args.wb_exp_name += '_epo' + str(args.epoch_per_cycle)

if args.reduce_latent_feat_dim_via_pca:
    args.wb_exp_name += '_reduce_dim'

if args.simple_loss:
    args.wb_exp_name += '_simple_loss_v2'
    args.uncertainty = False

if args.dataset_name == 'replogle_k562_gw_1000hvg':
    args.wb_exp_name += '_gw'
elif args.dataset_name == 'replogle_k562_essential_1000hvg':
    args.wb_exp_name += '_ess_1k'
elif args.dataset_name == 'replogle_rpe1_essential_1000hvg':
    args.wb_exp_name += '_rpe1'

if args.normalize_kernel:
    args.wb_exp_name += '_' + args.normalize_method

if args.fix_evaluation:
    args.wb_exp_name += '_fix_eval'

if args.cluster_mode != 'kmeans':
    args.wb_exp_name += '_' + args.cluster_mode

if args.strategy_name == 'EssentialWeighted':
    args.wb_exp_name += '_' + args.essential_assay

if args.use_prior:
    args.wb_exp_name += '_prior'
    if args.use_prior_only:
        print('Using prior, without the base kernel from the model!')
        args.wb_exp_name += '_only'
    args.wb_exp_name += args.integrate_mode + '_'+ args.normalize_mode
    if args.use_single_prior:
        print('Just using one prior: ' + args.single_prior)
        args.wb_exp_name += '_single_' + args.single_prior
    if args.strategy_name == 'KMeansSampling':
        if args.use_kernel_for_kmeans:
            print('use kernel integration...')
            args.wb_exp_name += '_kernel'
    if args.kernel_normalize_feat:
        args.wb_exp_name += '_feat_norm'

if add_ctrl:
    args.wb_exp_name += '_add_ctrl'

if args.duplicate_prior:
    args.wb_exp_name += f'_dup{args.duplicate_copies}_{args.duplicate_prior.replace("_kernel", "")}'

if args.drop_prior:
    args.wb_exp_name += f'_drop_{args.drop_prior.replace("_kernel", "")}'

if args.detect_drop:
    args.wb_exp_name += (
        f'_detect_{args.detect_target}_t{float(args.detect_tau):.4f}_r{int(args.detect_min_round)}'
    )

if args.corrupt_prior and args.corrupt_mode != 'none':
    args.wb_exp_name += (
        f'_corrupt_{args.corrupt_mode}_{args.corrupt_prior.replace("_kernel", "")}'
        f'_l{float(args.corrupt_lambda):g}_cs{int(args.corrupt_seed)}'
    )

if args.weight_schedule != 'fixed':
    args.wb_exp_name += f'_sched_{args.weight_schedule}'
elif args.model_weight >= 0:
    args.wb_exp_name += f'_mw{args.model_weight}'

if args.labeled_genes_file:
    args.wb_exp_name += '_fork'
    if args.load_checkpoint:
        # Encode source state n if present in path (.../n300/...)
        import re as _re
        m = _re.search(r'/n(\d+)/', args.load_checkpoint.replace('\\', '/') + '/')
        if m:
            args.wb_exp_name += f'_n{m.group(1)}'

if args.sample_cells_training:
    args.wb_exp_name += '_sct'

if args.custom_split:
    if args.dataset_name != 'replogle_k562_gw_1000hvg':
        raise ValueError
    args.wb_exp_name += '_test_ess'
    custom_test = CUSTOM_TEST_SPLIT
else:
    custom_test = None
    
if args.lamb != 2:
    args.wb_exp_name += '_lamb' + str(args.lamb)

if args.strategy_name == 'EssentialSampling':
    custom_test = CUSTOM_TEST_SPLIT
    essential_gene = pickle.load(open(custom_test, 'rb'))['test'] + pickle.load(open(custom_test, 'rb'))['train']

args.wb_exp_name += '_run' + str(args.run)


if args.batch_exp:
    args.wb_exp_name += '_batch_exp'

#args.wb_exp_name += '_test_shuffle'

print(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.run)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device(args.device if use_cuda else "cpu")
print(device)

path = GEARS_DATA_PATH
dataset = Data(path, args.dataset_name, args.batch_size, args.test_fraction, args.seed, custom_test)
n_features = dataset.pert_data.adata.X.shape[1]
test_data = dataset.get_test_data()
print('Using overall framework:' + args.strategy_name)
if args.strategy_name == 'kernel_based_active_learning':
    print('Using strategy:' + args.kernel_strategy)
    args.wb_exp_name += '_' + args.kernel_strategy
    if args.base_kernel != 'linear':
        args.wb_exp_name += '_' + args.base_kernel
    print(args.wb_exp_name)
    if args.kernel_strategy == 'Random':
        selection_method = 'random'
        kernel_transforms=[]
        sel_with_train = True
    elif args.kernel_strategy == 'BALD':
        selection_method='maxdiag'
        #kernel_transforms=[('rp', [512]), ('train', [0.1, None])]
        kernel_transforms=[('train', [0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'BatchBALD':
        selection_method='maxdet'
        #kernel_transforms=[('rp', [512]), ('train', [0.1, None])]
        kernel_transforms=[('train', [0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'BAIT':
        selection_method='bait'
        #kernel_transforms=[('rp', [512]), ('train', [0.1, None])]
        kernel_transforms=[('train', [0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'ACS-FW':
        selection_method = 'fw'
        #kernel_transforms=[('rp', [512]), ('acs-rf', [512, 0.1, None])]
        kernel_transforms=[('acs-rf', [512, 0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'Core-Set':
        selection_method = 'maxdist'
        #kernel_transforms=[('rp', [512]), ('train', [0.1, None])]
        kernel_transforms=[]
        sel_with_train = True
    elif args.kernel_strategy == 'BADGE':
        selection_method = 'kmeanspp'
        #kernel_transforms=[('rp', [512]), ('acs-rf', [512, 0.1, None])]
        kernel_transforms=[('train', [0.1, None])]
        sel_with_train = False
    elif args.kernel_strategy == 'MAXDIST':
        selection_method = 'maxdist'
        kernel_transforms=[]
        sel_with_train = False
    elif args.kernel_strategy == 'KMEANSPP':
        selection_method = 'kmeanspp'
        kernel_transforms=[]
        sel_with_train = False
    elif args.kernel_strategy == 'MAXDET':
        selection_method = 'maxdet'
        kernel_transforms=[]
        sel_with_train = False
    elif args.kernel_strategy == 'MAXDIAG':
        selection_method = 'maxdiag'
        kernel_transforms=[]
        sel_with_train = False
    #elif args.kernel_strategy == 'BAIT':
    #    selection_method = 'bait'
    #    kernel_transforms=[]
    #    sel_with_train = False
    elif args.kernel_strategy == 'FW':
        selection_method = 'fw'
        kernel_transforms=[]
        sel_with_train = False
    elif args.kernel_strategy == 'LCMD':
        selection_method = 'lcmd'
        kernel_transforms=[('rp', [512])] 
        sel_with_train = True
    elif args.kernel_strategy == 'D-OptimalDesign':
        selection_method = 'maxdet'
        kernel_transforms=[]
        sel_with_train = True
    elif args.kernel_strategy == 'DIR':
        selection_method = 'dir'
        kernel_transforms=[]
        sel_with_train = True
else:
    if args.base_kernel != 'linear':
        args.wb_exp_name += '_' + args.base_kernel
    args.wb_exp_name += '_' + args.strategy_name


save_dir = os.path.join(SAVE_DIR, args.wb_exp_name)
os.makedirs(save_dir, exist_ok=True)
    
params = {
    'weight_bias_track': args.wandb,
    'wb_proj_name': args.wb_proj_name,
    'wb_exp_name': args.wb_exp_name,
    'hidden_size': args.hidden_size,
    'uncertainty' : args.uncertainty, 
    'uncertainty_reg' : args.uncertainty_reg,
    'direction_lambda' : args.direction_lambda,
    'device': args.device,
    'epoch_per_cycle': args.epoch_per_cycle,
    'retrain': args.retrain,
    'simple_loss': args.simple_loss
}

if args.num_dim != 100:
    args.wb_exp_name += '_'  + str(args.num_dim)
if args.dim_reduce != 'NA':
    args.wb_exp_name += '_'  + str(args.dim_reduce)
if args.normalize_method != 'NA':
    args.wb_exp_name += '_'  + str(args.normalize_method)


if args.hvg_kernel:
    adata = dataset.pert_data.adata
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))
    gene_hvg_idx = np.array([geneid2idx[i] for i in adata.var.sort_values('dispersions_norm')[::-1][:args.hvg_num].index.values])
    args.wb_exp_name += '_hvg' + str(args.hvg_num)
else:
    gene_hvg_idx = None

if args.valid_perts:
    args.wb_exp_name += '_val_pert'

if args.wandb:
    import wandb
    wandb.login(host='https://genentech.wandb.io', key=os.environ.get('WANDB_API_KEY'))
    wandb.init(project=args.wb_proj_name, name=args.wb_exp_name) 
    wandb.config.update(params)

net = Net(params, device, dataset.pert_data, args.model_name, save_dir, args.fix_evaluation)                   # load network

if args.use_prior:
    #if args.dataset_name != 'replogle_k562_essential_1000hvg+pert_in_gene':
    #    raise ValueError('prior currently imnplemented for this dataset only...')

    import pickle

    def load_kernel(kernel_name):
        kernel_path = KERNEL_PATHS.get(
            args.dataset_name,
            KERNEL_PATHS['replogle_k562_essential_1000hvg'],
        )
        if not os.path.exists(kernel_path + kernel_name):
            raise ValueError('Kernel does not exist')
        with open(kernel_path + kernel_name + '/pert_list.pkl', 'rb') as f:
            pert_list = pickle.load(f)
        with open(kernel_path + kernel_name + '/kernel.pkl', 'rb') as f:
            kernel_npy = pickle.load(f)
        with open(kernel_path + kernel_name + '/feat.pkl', 'rb') as f:
            feat = pickle.load(f)
        return pert_list, kernel_npy, feat

    if args.use_single_prior:
        kernel_list = [args.single_prior]
    else:
        if args.dataset_name == 'replogle_k562_gw_1000hvg':
            kernel_list = ['pops_kernel', 'esm_kernel', 
                       'biogpt_kernel', 'node2vec_kernel', 'ops_A549_kernel',
                       'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel']
        elif args.dataset_name == 'replogle_rpe1_essential_1000hvg':
            kernel_list = ['pops_kernel', 'k562_kernel', 'esm_kernel', 
                       'biogpt_kernel', 'node2vec_kernel', 'ops_A549_kernel',
                       'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel']
        else:
            kernel_list = ['pops_kernel', 'rpe1_kernel', 'esm_kernel', 
                       'biogpt_kernel', 'node2vec_kernel', 'ops_A549_kernel',
                       'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel']

    if args.drop_prior:
        before = list(kernel_list)
        kernel_list = [k for k in kernel_list if k != args.drop_prior]
        print(f'Dropped prior {args.drop_prior}: {before} -> {kernel_list}')
        if not kernel_list:
            raise ValueError(f'--drop_prior {args.drop_prior} left an empty kernel list')

    if args.duplicate_prior:
        expanded = []
        for kernel_name in kernel_list:
            if kernel_name == args.duplicate_prior:
                expanded.extend([kernel_name] * max(1, args.duplicate_copies))
            else:
                expanded.append(kernel_name)
        kernel_list = expanded

    def load_or_make_perm(n):
        perm_dir = os.path.join(DATA_ROOT, 'idea3')
        os.makedirs(perm_dir, exist_ok=True)
        path = os.path.join(perm_dir, f'perm_seed{int(args.corrupt_seed)}_n{int(n)}.npy')
        if os.path.isfile(path):
            perm = np.load(path)
            if int(perm.shape[0]) != int(n):
                raise ValueError(f'Permutation {path} has len {perm.shape[0]}, expected {n}')
            return perm
        rng = np.random.default_rng(int(args.corrupt_seed))
        perm = rng.permutation(int(n))
        np.save(path, perm)
        print(f'Saved fixed corruption permutation to {path}')
        return perm

    def maybe_corrupt_kernel(kernel_name, kernel_npy):
        if not (args.corrupt_prior and kernel_name == args.corrupt_prior and args.corrupt_mode == 'permute'):
            return kernel_npy
        lam = float(args.corrupt_lambda)
        if lam <= 0:
            print(f'Corrupt {kernel_name}: λ=0, kernel unchanged')
            return kernel_npy
        perm = load_or_make_perm(kernel_npy.shape[0])
        k_perm = kernel_npy[np.ix_(perm, perm)]
        mixed = k_perm if lam >= 1 else (1.0 - lam) * kernel_npy + lam * k_perm
        print(
            f'Corrupt {kernel_name}: K_λ=(1-λ)K+λPKP^T  λ={lam} '
            f'seed={int(args.corrupt_seed)} perm[:8]={perm[:8].tolist()}'
        )
        return mixed

    prior_kernel_list, prior_feat_list = [],[] 
    
    for i in kernel_list:
        _, k, f = load_kernel(i)
        k = maybe_corrupt_kernel(i, k)
        if args.kernel_normalize_feat:
            print('normalizing feature and then compute kernels!')
            from sklearn.preprocessing import StandardScaler    
            normalizer = StandardScaler()
            f = normalizer.fit_transform(f)
            k = np.dot(f, f.T)
            prior_kernel_list.append(k)
        else:
            prior_kernel_list.append(k)
        prior_feat_list.append(f)

    pert_list, true_gold, truth_feat = load_kernel('ground_truth_delta')
    pert_list = [i.split('+')[0] for i in pert_list]
    #print(pert_list)
    #print(prior_kernel_list)
    if args.strategy_name == 'kernel_based_active_learning':
        strategy = get_strategy(args.strategy_name)(dataset, net, selection_method, args.base_kernel, 
                                                    kernel_transforms, device, sel_with_train, 
                                                    args.reduce_latent_feat_dim_via_pca, use_prior_only = args.use_prior_only, 
                                                    integrate_mode = args.integrate_mode, normalize_mode = args.normalize_mode, 
                                                    prior_kernel_list = prior_kernel_list, prior_kernel_pert_list = pert_list, 
                                                    train_gold = true_gold, normalize_kernel = args.normalize_kernel,
                                                    normalize_method = args.normalize_method, add_ctrl = add_ctrl, 
                                                    prior_feat_list = prior_feat_list, gene_hvg_idx = gene_hvg_idx, lamb = args.lamb,
                                                    prior_kernel_names = [k.replace('_kernel', '') for k in kernel_list],
                                                    selection_log = args.selection_log,
                                                    selection_log_dir = args.selection_log_dir,
                                                    run_id = args.run, seed = args.seed,
                                                    model_weight = args.model_weight,
                                                    weight_schedule = args.weight_schedule,
                                                    detect_drop = args.detect_drop,
                                                    detect_tau = args.detect_tau,
                                                    detect_target = args.detect_target,
                                                    detect_min_round = args.detect_min_round)
    elif args.strategy_name in ['KMeansSampling', 'MaxDist', 'TypiClust']:
        strategy = get_strategy(args.strategy_name)(dataset, net, args.base_kernel, use_prior_only = args.use_prior_only, 
                                                    integrate_mode = args.integrate_mode, normalize_mode = args.normalize_mode, 
                                                    prior_kernel_list = prior_kernel_list, prior_kernel_pert_list = pert_list, 
                                                    train_gold = true_gold, train_feat = truth_feat, prior_feat_list = prior_feat_list,
                                                    use_kernel_for_kmeans = args.use_kernel_for_kmeans, 
                                                    add_ctrl = add_ctrl, mode = args.cluster_mode, dim_reduce = args.dim_reduce, 
                                                    num_dim = args.num_dim, normalize_method = args.normalize_method, gene_hvg_idx = gene_hvg_idx)
else:
    if args.strategy_name == 'kernel_based_active_learning':
        strategy = get_strategy(args.strategy_name)(dataset, net, selection_method, args.base_kernel, 
                                                    kernel_transforms, device, sel_with_train, 
                                                    args.reduce_latent_feat_dim_via_pca,
                                                    normalize_kernel = args.normalize_kernel,
                                                    normalize_method = args.normalize_method,
                                                    add_ctrl = add_ctrl, gene_hvg_idx = gene_hvg_idx, lamb = args.lamb)
    elif args.strategy_name in ['KMeansSampling', 'KCenterGreedy', 'KMeansUncertainty', 'MaxDist', 'TypiClust']:
        strategy = get_strategy(args.strategy_name)(dataset, net, 
                                                    args.base_kernel, 
                                                    add_ctrl = add_ctrl,
                                                    mode = args.cluster_mode, 
                                                    dim_reduce = args.dim_reduce, 
                                                    num_dim = args.num_dim, 
                                                    normalize_method = args.normalize_method, 
                                                    gene_hvg_idx = gene_hvg_idx)  # load strategy
    else:
        strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

# start experiment
if args.labeled_genes_file:
    with open(args.labeled_genes_file) as f:
        gene_names = [line.strip() for line in f if line.strip()]
    init_idx = dataset.initialize_from_genes(gene_names)
    args.n_init_labeled = int(len(init_idx))
else:
    init_idx = dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

metrics = ['pearson_delta', 
            'frac_opposite_direction_top20_non_dropout',
            'mse_non_dropout',
            'mse_top20_de_non_dropout',
            'pearson_delta_top20_de_non_dropout', 'mse_4_non_dropout', 'mse_4_top20_de_non_dropout']

round_metrics = []
DUMP_NS = {int(x) for x in args.dump_n_labeled.split(',') if x.strip()} if args.dump_state_dir else set()


def dump_common_state(n_labeled, round_idx, pearson_before=None):
    if not args.dump_state_dir or int(n_labeled) not in DUMP_NS:
        return
    state_dir = os.path.join(args.dump_state_dir, f'n{int(n_labeled)}')
    os.makedirs(state_dir, exist_ok=True)
    labeled_idxs = np.where(dataset.labeled_idxs)[0]
    pool_idxs = np.where(~dataset.labeled_idxs)[0]
    labeled_genes = [str(g) for g in dataset.pert_train[labeled_idxs]]
    pool_genes = [str(g) for g in dataset.pert_train[pool_idxs]]
    with open(os.path.join(state_dir, 'labeled_genes.txt'), 'w') as f:
        f.write('\n'.join(labeled_genes) + '\n')
    with open(os.path.join(state_dir, 'pool_genes.txt'), 'w') as f:
        f.write('\n'.join(pool_genes) + '\n')
    ckpt_dir = os.path.join(state_dir, 'gears_checkpoint')
    strategy.net.save_checkpoint(ckpt_dir)
    meta = {
        'source_wb_exp_name': args.wb_exp_name,
        'run': args.run,
        'seed': args.seed,
        'round': int(round_idx),
        'n_labeled': int(n_labeled),
        'pearson_before': pearson_before,
        'dataset_name': args.dataset_name,
        'integrate_mode': args.integrate_mode,
        'normalize_mode': args.normalize_mode,
        'model_weight': args.model_weight,
        'weight_schedule': args.weight_schedule,
        'n_labeled_genes': len(labeled_genes),
        'n_pool_genes': len(pool_genes),
        'checkpoint': ckpt_dir,
        'phase0_commit': 'a06d119',
    }
    import json as _json
    with open(os.path.join(state_dir, 'state_meta.json'), 'w') as f:
        _json.dump(meta, f, indent=2)
    if pearson_before is not None:
        with open(os.path.join(state_dir, 'state_metrics.json'), 'w') as f:
            _json.dump({'n_labeled': int(n_labeled), 'pearson_before': float(pearson_before)}, f, indent=2)
    print(f'Dumped common state to {state_dir}')


def record_round_metrics(round_idx, n_labeled, eval_out):
    row = {
        'run': args.run,
        'seed': args.seed,
        'round': round_idx,
        'n_labeled': n_labeled,
    }
    for m in metrics:
        vals = [j[m] for i, j in eval_out.items() if m in j]
        if vals:
            row[m] = float(np.mean(vals))
    round_metrics.append(row)
    if 'pearson_delta' in row:
        print(f"Round {round_idx} pearson delta: {row['pearson_delta']}")
        if hasattr(strategy, 'last_pearson'):
            strategy.last_pearson = row['pearson_delta']
    return row.get('pearson_delta')


# round 0 accuracy
print("Round 0")
if args.load_checkpoint:
    print(f'Loading GEARS checkpoint from {args.load_checkpoint} (skip initial train)')
    strategy.net.load_checkpoint(args.load_checkpoint)
elif args.batch_exp:
    genes_available_per_round = {}
    genes_available_per_round[0] = dataset.pert_train[init_idx]
    all_batch_idx = dataset.pert_data.all_batch_idx
    np.random.seed(args.seed)
    np.random.shuffle(all_batch_idx)
    batch_idx_round = {}
    rounds = args.n_round
    rounds += 1 # initialized round
    num_batches_per_run = int(len(all_batch_idx) / rounds)
    for round in range(rounds):
        batch_idx_round[round] = all_batch_idx[num_batches_per_run * round : num_batches_per_run * (round+1)]
    strategy.train({'batch_idx': batch_idx_round, 'genes_available_per_round': genes_available_per_round})  
else:
    strategy.train()

res, out = strategy.eval(test_data)

if args.wandb:
    for m in metrics:
        wandb.log({'test_round_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

p0 = record_round_metrics(0, args.n_init_labeled, out)
if args.load_checkpoint:
    dump_metrics = os.path.join(os.path.dirname(args.load_checkpoint.rstrip('/')), 'state_metrics.json')
    if os.path.isfile(dump_metrics):
        import json as _json
        dumped_p = _json.loads(open(dump_metrics).read()).get('pearson_before')
        print(
            f'Fork P_before dump={dumped_p} eval_after_load={p0} '
            f'(official P_before uses dump; eval is sanity only)'
        )
        if dumped_p is not None:
            if p0 is not None and abs(float(p0) - float(dumped_p)) > 1e-4:
                print(f'WARNING: eval_after_load differs from dump P_before by {float(p0) - float(dumped_p)}')
            round_metrics[-1]['pearson_delta'] = float(dumped_p)
            round_metrics[-1]['pearson_delta_eval_after_load'] = p0
            p0 = float(dumped_p)
            if hasattr(strategy, 'last_pearson'):
                strategy.last_pearson = p0
    print(f'Fork checkpoint_path: {args.load_checkpoint}')
    print(f'Fork training_seed: np={args.seed} torch_run={args.run}')
dump_common_state(args.n_init_labeled, 0, p0)


round2query = {}

for rd in range(1, args.n_round+1):
    print(f"Round {rd}")

    # query
    if args.strategy_name == 'EssentialSampling':
        query_idxs = strategy.query(args.n_query, args.save_kernel, args.wb_exp_name + '_round' + str(rd), essential_gene)
    elif args.strategy_name == 'EssentialWeighted':
        if args.dataset_name == 'replogle_k562_gw_1000hvg':
            ess_root = ESSENTIAL_GENE_PATHS['gw']
        else:
            ess_root = ESSENTIAL_GENE_PATHS['default']
        gene2ess = pickle.load(open(os.path.join(ess_root, args.essential_assay + '_essential/gene2ess.pkl'), 'rb'))
        query_idxs = strategy.query(args.n_query, args.save_kernel, args.wb_exp_name + '_round' + str(rd), gene2ess)
    else:
        if args.valid_perts:
            valid_perts = strategy.net.gears_model.valid_perts
            query_idxs = strategy.query(args.n_query, args.save_kernel, args.wb_exp_name + '_round' + str(rd), valid_perts, round = rd)
        else:
            query_idxs = strategy.query(args.n_query, args.save_kernel, args.wb_exp_name + '_round' + str(rd), round = rd)
    try:
        round2query[rd] = dataset.pert_train[query_idxs]
    except:
        print('Querying not saved...')
    print('Querying ' + str(len(query_idxs)) + ' new perturbations!')
    # update labels
    strategy.update(query_idxs)


    if args.batch_exp:
        genes_available_per_round[rd] = dataset.pert_train[query_idxs]
        strategy.train({'batch_idx': batch_idx_round, 'genes_available_per_round': genes_available_per_round})
    else:
        strategy.train()

    # calculate accuracy
    res, out = strategy.eval(test_data)

    if args.wandb:
        for m in metrics:
            wandb.log({'test_round_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

    n_lab = args.n_init_labeled + rd * args.n_query
    p = record_round_metrics(rd, n_lab, out)
    dump_common_state(n_lab, rd, p)


import pickle
import pandas as pd

result_dir = resolve_result_dir(args)
os.makedirs(result_dir, exist_ok=True)
result_base = os.path.join(result_dir, args.wb_exp_name)
with open(result_base + '.pkl', 'wb') as f:
    pickle.dump(round2query, f)

metrics_csv = result_base + '_metrics.csv'
metrics_pkl = result_base + '_metrics.pkl'
pd.DataFrame(round_metrics).to_csv(metrics_csv, index=False)
with open(metrics_pkl, 'wb') as f:
    pickle.dump(round_metrics, f)
print(f"Saved query results to {result_base}.pkl")
print(f"Saved round metrics to {metrics_csv}")

if args.load_checkpoint:
    import json as _json

    def _genes(obj):
        if obj is None:
            return []
        if hasattr(obj, 'tolist'):
            obj = obj.tolist()
        return [str(x) for x in obj]

    p_before = None
    p_after = None
    p_eval = None
    by_round = {int(r['round']): r for r in round_metrics}
    if 0 in by_round:
        p_before = by_round[0].get('pearson_delta')
        p_eval = by_round[0].get('pearson_delta_eval_after_load')
    if 1 in by_round:
        p_after = by_round[1].get('pearson_delta')
    selected = _genes(round2query.get(1))
    dump_metrics = os.path.join(os.path.dirname(args.load_checkpoint.rstrip('/')), 'state_metrics.json')
    dump_p = None
    if os.path.isfile(dump_metrics):
        dump_p = _json.loads(open(dump_metrics).read()).get('pearson_before')
        if dump_p is not None:
            p_before = float(dump_p)
    record = {
        'base_run': int(args.run),
        'base_n_labeled': int(args.n_init_labeled),
        'w_model': float(args.model_weight),
        'P_before': p_before,
        'P_before_eval_after_load': p_eval,
        'P_after': p_after,
        'delta_pearson': (None if p_before is None or p_after is None else float(p_after) - float(p_before)),
        'selected_genes': selected,
        'n_selected': len(selected),
        'training_seed': int(args.seed),
        'training_run': int(args.run),
        'checkpoint_path': args.load_checkpoint,
        'labeled_genes_file': args.labeled_genes_file,
        'wb_exp_name': args.wb_exp_name,
    }
    record_path = result_base + '_fork_record.json'
    with open(record_path, 'w') as f:
        _json.dump(record, f, indent=2)
    print(f'Saved fork record to {record_path}')