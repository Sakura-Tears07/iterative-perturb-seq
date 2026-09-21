"""Shared utilities for IterPert prior-kernel diagnostics.

Loads the precomputed knowledge kernels that IterPert uses for the
replogle_k562_essential_1000hvg dataset, plus the GEARS 'active' split,
and exposes them on a common perturbation axis (no '+ctrl' suffix).
"""
import os
import pickle
import numpy as np

DATA_ROOT = '/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data'
DATASET = 'replogle_k562_essential_1000hvg'
KERNEL_DIR = os.path.join(DATA_ROOT, DATASET + '_kernels', 'knowledge_kernels_1k')
SPLIT_FILE = os.path.join(DATA_ROOT, DATASET, 'splits',
                          DATASET + '_active_1_0.75.pkl')
CACHE_DIR = '/home/lihaoran/ai4s/analysis/cache'

# The 8 prior kernels used in iterpert.py for this dataset, in the same order.
PRIOR_NAMES = ['pops_kernel', 'rpe1_kernel', 'esm_kernel', 'biogpt_kernel',
               'node2vec_kernel', 'ops_A549_kernel', 'ops_HeLa_HPLM_kernel',
               'ops_HeLa_DMEM_kernel']


def _load(name):
    with open(os.path.join(KERNEL_DIR, name, 'kernel.pkl'), 'rb') as f:
        k = np.asarray(pickle.load(f), dtype=np.float64)
    with open(os.path.join(KERNEL_DIR, name, 'pert_list.pkl'), 'rb') as f:
        p = [str(x).split('+')[0] for x in pickle.load(f)]
    with open(os.path.join(KERNEL_DIR, name, 'feat.pkl'), 'rb') as f:
        feat = np.asarray(pickle.load(f), dtype=np.float64)
    return k, p, feat


def load_all():
    """Return dict with kernels, truth, feats aligned on the ground-truth pert axis."""
    truth_k, ref_perts, truth_feat = _load('ground_truth_delta')
    idx = {p: i for i, p in enumerate(ref_perts)}
    assert len(idx) == len(ref_perts), 'duplicate perturbations on truth axis'

    priors, priors_feat, missing = {}, {}, {}
    for n in PRIOR_NAMES:
        k, p, feat = _load(n)
        pos = np.array([idx[q] for q in p])
        K = np.full((len(ref_perts), len(ref_perts)), np.nan)
        K[np.ix_(pos, pos)] = k
        F = np.full((len(ref_perts), feat.shape[1]), np.nan)
        F[pos] = feat
        priors[n] = K
        priors_feat[n] = F
        missing[n] = len(ref_perts) - len(p)

    # the GEARS model kernel artifact (not a prior); used as an embedding space
    gears_k, gears_p, _ = _load('gears_kernel')
    gpos = np.array([idx[q] for q in gears_p])
    gears = np.full((len(ref_perts), len(ref_perts)), np.nan)
    gears[np.ix_(gpos, gpos)] = gears_k

    split = pickle.load(open(SPLIT_FILE, 'rb'))
    train_perts = [str(x).split('+')[0] for x in split['train']]
    test_perts = [str(x).split('+')[0] for x in split['test']]
    train_idx = np.array(sorted({idx[p] for p in train_perts if p in idx}))
    test_idx = np.array(sorted({idx[p] for p in test_perts if p in idx}))
    return dict(perts=ref_perts, truth=truth_k, truth_feat=truth_feat,
                priors=priors, priors_feat=priors_feat, missing=missing,
                gears=gears, train_idx=train_idx, test_idx=test_idx)


def normalize_kernel(K, mode='max'):
    """Mirror of iterpert/bmdal/utils.normalize_kernel for mode='max'.

    Required so that kernels with wildly different scales (pops has values up
    to 3.7e6, esm up to 1.2e2) do not let one prior dominate a plain mean.
    """
    K = np.array(K, dtype=np.float64)
    if mode == 'max':
        off = K[~np.eye(K.shape[0], dtype=bool)]
        return K / off.max()
    if mode == 'diag':
        d = np.diag(K).copy()
        d[d == 0] = 1.0
        s = np.sqrt(np.abs(d))
        return K / np.outer(s, s)
    raise ValueError(mode)


def kernel_alignment(A, B):
    """Cosine similarity between two (sub-)kernel matrices, Frobenius inner product."""
    a = np.asarray(A, dtype=np.float64).ravel()
    b = np.asarray(B, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def submatrix(K, idx):
    return K[np.ix_(idx, idx)]


def sample_idx(pool, n, rng):
    return rng.choice(pool, size=min(n, len(pool)), replace=False)


def subset_fusion(priors, base_kernel, idx, weights=None, names=None):
    """Fuse prior kernels + base kernel on index subset `idx`."""
    names = names if names is not None else list(priors.keys())
    ks = [submatrix(priors[n], idx) for n in names]
    if base_kernel is not None and weights is not None and len(weights) == len(ks) + 1:
        ks = ks + [submatrix(base_kernel, idx)]
    if weights is None:
        weights = np.ones(len(ks)) / len(ks)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    out = np.zeros_like(ks[0])
    for w, K in zip(weights, ks):
        out += w * K
    return out


def knn_predict_deltas(K_query_pool, Y_pool, k=5):
    """Predict delta vectors of query rows by kernel-weighted kNN over pool.

    K_query_pool: (n_query, n_pool) similarity
    Y_pool:       (n_pool, n_genes) ground-truth deltas
    Returns (n_query, n_genes) predictions.
    """
    S = np.asarray(K_query_pool, dtype=np.float64).copy()
    np.fill_diagonal(S[:, :], 0.0) if S.shape[0] == S.shape[1] else None
    k = min(k, S.shape[1])
    out = np.zeros((S.shape[0], Y_pool.shape[1]))
    for i in range(S.shape[0]):
        row = S[i]
        top = np.argpartition(-row, k - 1)[:k]
        w = row[top]
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        out[i] = w @ Y_pool[top]
    return out


def row_corr(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den
