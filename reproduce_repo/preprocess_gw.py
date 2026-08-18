#!/usr/bin/env python3
"""Preprocess genome-wide K562 data for Fig.6."""
import os
from pathlib import Path

import scanpy as sc
from scipy import sparse
from gears import PertData

DATA_ROOT = Path(os.environ.get("ITERPERT_DATA_ROOT", "/data/zy/iterpert"))
RAW_PATH = DATA_ROOT / "cache" / "downloads" / "raw" / "ReplogleWeissman2022_K562_gwps.h5ad"
GEARS_DATA_ROOT = DATA_ROOT / "datasets" / "gears_data"
DATASET_NAME = "replogle_k562_gw_1000hvg"
OUTPUT_DIR = GEARS_DATA_ROOT / DATASET_NAME

print("=" * 70)
print("Genome-wide (Fig.6) preprocessing")
print("=" * 70)
print(f"Raw:    {RAW_PATH}")
print(f"Output: {OUTPUT_DIR}")

if not RAW_PATH.exists():
    raise FileNotFoundError(RAW_PATH)

GEARS_DATA_ROOT.mkdir(parents=True, exist_ok=True)

print("\nLoading raw h5ad (may take several minutes)...")
adata = sc.read_h5ad(RAW_PATH)
print(adata)

# GEARS expects condition / cell_type / gene_name columns
if "condition" not in adata.obs.columns and "perturbation" in adata.obs.columns:
    adata.obs = adata.obs.rename(columns={"perturbation": "condition"})
if "cell_type" not in adata.obs.columns and "cell_line" in adata.obs.columns:
    adata.obs = adata.obs.rename(columns={"cell_line": "cell_type"})
adata.obs["condition"] = adata.obs["condition"].astype(str).apply(
    lambda x: x + "+ctrl" if x != "control" else "ctrl"
)
if "gene_name" not in adata.var.columns:
    if "ensembl_id" in adata.var.columns:
        adata.var["gene_name"] = adata.var["ensembl_id"].astype(str)
    else:
        adata.var["gene_name"] = adata.var.index.astype(str)

if adata.n_vars > 1200:
    print("\nSubsetting to 1000 HVG...")
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True, flavor="seurat_v3")

required_uns = [
    "rank_genes_groups_cov_all", "top_non_dropout_de_20",
    "non_dropout_gene_idx", "non_zeros_gene_idx", "top_non_zero_de_20",
]
skip_calc_de = all(k in adata.uns for k in required_uns)
print(f"skip_calc_de={skip_calc_de}")

pert_data = PertData(data_path=str(GEARS_DATA_ROOT))
pert_data.new_data_process(
    dataset_name=DATASET_NAME,
    adata=adata,
    skip_calc_de=skip_calc_de,
)

for path in [OUTPUT_DIR / "perturb_processed.h5ad", OUTPUT_DIR / "data_pyg" / "cell_graphs.pkl"]:
    if not path.exists():
        raise RuntimeError(f"Missing: {path}")
    print(f"OK: {path} ({path.stat().st_size / 1e9:.2f} GB)")

check = PertData(data_path=str(GEARS_DATA_ROOT))
check.load(data_path=str(OUTPUT_DIR))
print(f"Perturbation groups: {len(check.dataset_processed)}")
print("GW PREPROCESSING SUCCESS")
