import os
from pathlib import Path

import scanpy as sc
from scipy import sparse

from gears import PertData


# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path(
    os.environ.get("ITERPERT_DATA_ROOT", "/data/zy/iterpert")
)

RAW_PATH = (
    DATA_ROOT
    / "cache"
    / "downloads"
    / "raw"
    / "replogle_k562_essential_1000hvg.h5ad"
)

GEARS_DATA_ROOT = (
    DATA_ROOT
    / "datasets"
    / "gears_data"
)

DATASET_NAME = "replogle_k562_essential_1000hvg"

OUTPUT_DIR = GEARS_DATA_ROOT / DATASET_NAME


# ============================================================
# 1. Load raw h5ad
# ============================================================

print("=" * 70)
print("IterPert Essential preprocessing")
print("=" * 70)

print(f"Raw file:     {RAW_PATH}")
print(f"GEARS root:   {GEARS_DATA_ROOT}")
print(f"Output dir:   {OUTPUT_DIR}")

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Raw h5ad not found: {RAW_PATH}")

GEARS_DATA_ROOT.mkdir(parents=True, exist_ok=True)

print("\n[1/5] Loading raw h5ad...")
adata = sc.read_h5ad(RAW_PATH)

print(adata)
print("X type:", type(adata.X))
print("shape:", adata.shape)


# ============================================================
# 2. Basic schema validation
# ============================================================

print("\n[2/5] Checking GEARS-required schema...")

required_obs = ["condition", "cell_type"]
required_var = ["gene_name"]

for col in required_obs:
    if col not in adata.obs.columns:
        raise ValueError(f"Missing required obs column: {col}")

for col in required_var:
    if col not in adata.var.columns:
        raise ValueError(f"Missing required var column: {col}")

if "ctrl" not in set(adata.obs["condition"].astype(str)):
    raise ValueError("No 'ctrl' condition found.")

print("condition: OK")
print("cell_type: OK")
print("gene_name: OK")
print("ctrl: OK")

if "batch" in adata.obs.columns:
    print(
        "batch: OK",
        f"({adata.obs['batch'].nunique()} unique batches)"
    )

if not sparse.issparse(adata.X):
    print(
        "WARNING: adata.X is dense. "
        "GEARS create_cell_graph_dataset() expects rows supporting "
        ".toarray(); sparse X is preferred."
    )
else:
    print("X sparse format: OK")


# ============================================================
# 3. Decide whether DE processing can be skipped
# ============================================================

print("\n[3/5] Checking existing GEARS DE metadata...")

required_uns = [
    "rank_genes_groups_cov_all",
    "top_non_dropout_de_20",
    "non_dropout_gene_idx",
    "non_zeros_gene_idx",
    "top_non_zero_de_20",
]

for key in required_uns:
    print(f"{key}: {'FOUND' if key in adata.uns else 'MISSING'}")

has_precomputed_de = all(
    key in adata.uns
    for key in required_uns
)

if has_precomputed_de:
    skip_calc_de = True
    print(
        "\nAll required DE/dropout metadata already exists."
        "\n=> Using skip_calc_de=True"
    )
else:
    skip_calc_de = False
    print(
        "\nSome GEARS metadata is missing."
        "\n=> Using skip_calc_de=False"
        "\n=> GEARS will recompute DE/dropout information."
    )


# ============================================================
# 4. Run original GEARS preprocessing
# ============================================================

print("\n[4/5] Running PertData.new_data_process()...")

pert_data = PertData(
    data_path=str(GEARS_DATA_ROOT)
)

pert_data.new_data_process(
    dataset_name=DATASET_NAME,
    adata=adata,
    skip_calc_de=skip_calc_de,
)

processed_h5ad = OUTPUT_DIR / "perturb_processed.h5ad"
pyg_file = OUTPUT_DIR / "data_pyg" / "cell_graphs.pkl"

print("\nGenerated files:")

for path in [processed_h5ad, pyg_file]:
    if not path.exists():
        raise RuntimeError(f"Expected output missing: {path}")

    size_gb = path.stat().st_size / (1024 ** 3)
    print(f"  {path}")
    print(f"    size = {size_gb:.3f} GB")


# ============================================================
# 5. Reload exactly as run.py will
# ============================================================

print("\n[5/5] Validating PertData.load()...")

check_data = PertData(
    data_path=str(GEARS_DATA_ROOT)
)

check_data.load(
    data_path=str(OUTPUT_DIR)
)

print("\nValidation summary")
print("-" * 70)

print("dataset_name:", check_data.dataset_name)
print("adata shape:", check_data.adata.shape)

print(
    "conditions:",
    check_data.adata.obs["condition"].nunique()
)

print(
    "control cells:",
    int(
        (
            check_data.adata.obs["condition"].astype(str)
            == "ctrl"
        ).sum()
    )
)

print(
    "PyG perturbation groups:",
    len(check_data.dataset_processed)
)

print(
    "pert graph genes:",
    len(check_data.pert_names)
)

print("\nSample perturbations:")
print(
    list(check_data.dataset_processed.keys())[:10]
)

print("\n" + "=" * 70)
print("ESSENTIAL PREPROCESSING SUCCESS")
print("=" * 70)
