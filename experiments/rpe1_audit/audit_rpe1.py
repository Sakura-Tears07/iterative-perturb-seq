#!/usr/bin/env python3
"""Gate-1/Gate-2 audit for the cross-cell-line (RPE1 target) task.

Documents what is available, what must be built, and the leakage rule, BEFORE any
model training. Re-runnable; writes audit_rpe1.json next to the script.

Usage: iterpert_env/bin/python experiments/rpe1_audit/audit_rpe1.py
"""
import collections, hashlib, json, os, sys
import numpy as np
import anndata as ad

RPE1 = '/data/lhr/ai4s/iterpert/scratch/rpe1/replogle_rpe1_essential/perturb_processed.h5ad'
K562 = ('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
        'replogle_k562_essential_1000hvg/perturb_processed.h5ad')
K562_KERNEL_AXIS = ('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/'
                    'replogle_k562_essential_1000hvg_kernels/knowledge_kernels_1k/'
                    'ground_truth_delta/pert_list.pkl')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_rpe1.json')


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(chunk), b''):
            h.update(c)
    return h.hexdigest()


def main():
    a = ad.read_h5ad(RPE1, backed='r')
    k = ad.read_h5ad(K562, backed='r')
    cond = a.obs['condition'].astype(str).values
    ctrl_mask = a.obs['control'].astype(bool).values
    cnt = collections.Counter(cond)
    ctrl_conds = {c for c in cnt if ctrl_mask[cond == c].all()}
    per = {c: v for c, v in cnt.items() if c not in ctrl_conds}
    v = np.array(list(per.values()))
    rg = set(map(str, a.var['gene_name'].values))
    kg = set(map(str, k.var['gene_name'].values))
    import pickle
    kaxis = {str(x).split('+')[0] for x in pickle.load(open(K562_KERNEL_AXIS, 'rb'))}
    rperms = {c.split('+')[0] for c in per}
    out = dict(
        rpe1=dict(path=RPE1, n_cells=int(a.n_obs), n_genes=int(a.n_vars),
                  n_perturbations=len(per), control_cells=int(ctrl_mask.sum()),
                  cells_per_pert=dict(min=int(v.min()), p25=float(np.percentile(v, 25)),
                                      median=float(np.median(v)), p75=float(np.percentile(v, 75)),
                                      max=int(v.max()), mean=float(v.mean())),
                  capacity={str(t): int((v >= t).sum()) for t in [25, 50, 100, 143, 200, 300]}),
        k562=dict(n_cells=int(k.n_obs), n_genes=int(k.n_vars),
                  kernel_axis=len(kaxis)),
        shared=dict(gene_intersection=len(kg & rg),
                    gene_intersection_frac_of_k562=len(kg & rg) / len(kg),
                    perturbation_overlap=len(kaxis & rperms),
                    perturbation_overlap_frac_of_rpe1=len(kaxis & rperms) / len(rperms)),
        prior_provenance={
            'rpe1_kernel': 'RPE1 NTC-centred pseudobulk (paper App. A.1) -> TARGET-INTERNAL for an RPE1 target: MUST BE EXCLUDED',
            'k562_kernel': 'K562 NTC-centred pseudobulk -> the external transfer prior for RPE1; NOT published, must be built locally from the local K562 data',
            'pops_kernel': 'public scRNA atlas (PoPS) gene embeddings - cell-line independent',
            'esm_kernel': 'ESM protein-LM gene embeddings - cell-line independent',
            'biogpt_kernel': 'BioGPT literature gene embeddings - cell-line independent',
            'node2vec_kernel': 'PPI network node2vec embeddings - cell-line independent',
            'ops_*_kernel': 'optical pooled screen CellProfiler perturbation embeddings - different assays/cell lines',
            'ground_truth_delta': 'the target response kernel itself (RPE1) - scorer/upper-bound only, never a prior',
        },
        source_files_for_kernel_build={
            'esm': 'knowledge_kernels/esm_emb/gene2esm.pkl',
            'biogpt': 'knowledge_kernels/biogpt_emb/gene2biogpt.pkl',
            'node2vec': 'knowledge_kernels/gears_emb/gene2gears_node2vec.pkl',
            'pops': 'knowledge_kernels/pops_emb/gene2pops_all.pkl',
            'ops': 'OPS embeddings (gene2ops)',
            'archive': 'Google Drive id 16p9sQYkhpM-PBAcNWQdjL9pxDZ46krQX (reachable: HTTP 303)',
        },
        leakage_rule=['exclude rpe1_kernel for an RPE1 target',
                      'ground_truth_delta never enters any policy observation',
                      'K562-derived priors are allowed (external, different cell line)'],
    )
    json.dump(out, open(OUT, 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
