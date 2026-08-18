"""Path configuration for demo notebooks.

Uses the same data root as reproduce_repo/local_paths.py.
Override with environment variable ITERPERT_DATA_ROOT (default: /data/zy/iterpert).
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
DATA_ROOT = os.environ.get("ITERPERT_DATA_ROOT", "/data/zy/iterpert")
DEFAULT_DEVICE = os.environ.get("ITERPERT_DEVICE", "cuda:0")

GEARS_DATA_PATH = os.path.join(DATA_ROOT, "datasets", "gears_data") + os.sep
# PertData 必须在 path 根目录找到 gene2go_all.pkl 等共享文件，不能单独用空目录
DEMO_DATA_PATH = GEARS_DATA_PATH  # 兼容旧引用；实际写入 {path}/{dataset_name}/ 子目录
DEMO_ADAMSON_DATASET = "demo_adamson"

# GEARS 共享文件（PertData 初始化时需要，位于 GEARS_DATA_PATH 根目录）
GEARS_SHARED_FILES = (
    "gene2go_all.pkl",
    "gene2go.pkl",
    "essential_all_data_pert_genes.pkl",
)
GEARS_SHARED_DIRS = ("go_essential_all",)

KERNELS_ESSENTIAL_1K = os.path.join(DATA_ROOT, "knowledge_kernels", "essential_1k") + os.sep
KERNELS_GENOME_WIDE = os.path.join(DATA_ROOT, "knowledge_kernels", "genome_wide") + os.sep

RAW_DATA_DIR = os.path.join(DATA_ROOT, "cache", "downloads", "raw")
GW_RAW_H5AD = os.path.join(RAW_DATA_DIR, "ReplogleWeissman2022_K562_gwps.h5ad")
GW_PROCESSED_H5AD = os.path.join(
    DATA_ROOT, "datasets", "processed", "ReplogleWeissman2022_K562_gwps_processed_hvg1000.h5ad"
)

EMBEDDINGS_DIR = os.path.join(DATA_ROOT, "knowledge_kernels", "embeddings") + os.sep
ESM_EMB = os.path.join(EMBEDDINGS_DIR, "esm_emb", "gene2esm.pkl")
BIOGPT_EMB = os.path.join(EMBEDDINGS_DIR, "biogpt_emb", "gene2biogpt.pkl")
NODE2VEC_EMB = os.path.join(EMBEDDINGS_DIR, "gears_emb", "gene2gears_node2vec.pkl")
POPS_EMB = os.path.join(EMBEDDINGS_DIR, "pops_emb", "gene2pops_all.pkl")

PROFILE_AGG_DIR = os.path.join(DATA_ROOT, "external", "Profile_Aggregation", "outputs") + os.sep
OPS_A549_CSV = os.path.join(
    PROFILE_AGG_DIR,
    "20200805_A549_WG_Screen_guide_normalized_feature_select_median_merged_ALLBATCHES___CP186___ALLWELLS_gene_aggregated.csv",
)
OPS_HELA_HPLM_CSV = os.path.join(
    PROFILE_AGG_DIR,
    "20210422_6W_CP257_guide_normalized_feature_select_median_merged_ALLBATCHES___HPLM___ALLWELLS_gene_aggregated.csv",
)
OPS_HELA_DMEM_CSV = os.path.join(
    PROFILE_AGG_DIR,
    "20210422_6W_CP257_guide_normalized_feature_select_median_merged_ALLBATCHES___DMEM___ALLWELLS_gene_aggregated.csv",
)

DEMO_RESULTS_PATH = str(REPO_ROOT / "results" / "demo") + os.sep

ESSENTIAL_DATASET = "replogle_k562_essential_1000hvg"
GW_DATASET = "replogle_k562_gw_1000hvg"
ESSENTIAL_DATA_PATH = os.path.join(GEARS_DATA_PATH, ESSENTIAL_DATASET)
GW_DATA_PATH = os.path.join(GEARS_DATA_PATH, GW_DATASET)
ADAMSON_H5AD = os.path.join(GEARS_DATA_PATH, "adamson", "perturb_processed.h5ad")


def resolve_gears_h5ad(dataset_name: str) -> str:
    """Return perturb_processed.h5ad path (handles nested zip extract layout)."""
    flat = os.path.join(GEARS_DATA_PATH, dataset_name, "perturb_processed.h5ad")
    nested = os.path.join(GEARS_DATA_PATH, dataset_name, dataset_name, "perturb_processed.h5ad")
    if os.path.exists(flat):
        return flat
    if os.path.exists(nested):
        return nested
    return flat


def ensure_gears_layout(dataset_name: str) -> str:
    """Symlink nested zip layout to flat path expected by GEARS/IterPert."""
    flat = os.path.join(GEARS_DATA_PATH, dataset_name, "perturb_processed.h5ad")
    nested = os.path.join(GEARS_DATA_PATH, dataset_name, dataset_name, "perturb_processed.h5ad")
    if os.path.exists(flat):
        return flat
    if not os.path.exists(nested):
        raise FileNotFoundError(f"找不到 {dataset_name} 数据: {nested}")
    os.symlink(os.path.relpath(nested, os.path.dirname(flat)), flat)
    return flat


def ensure_gears_shared_files(path: str = GEARS_DATA_PATH) -> None:
    """Verify shared GEARS files exist locally (avoid dataverse download)."""
    missing = [f for f in GEARS_SHARED_FILES if not os.path.exists(os.path.join(path, f))]
    missing += [d for d in GEARS_SHARED_DIRS if not os.path.exists(os.path.join(path, d))]
    if missing:
        raise FileNotFoundError(
            "GEARS 共享文件缺失，PertData 会尝试从网络下载:\n  "
            + "\n  ".join(missing)
        )


def setup_notebook_paths():
    """Add repo roots to sys.path so iterpert, paths, and bmdal_reg import correctly."""
    for path in (DEMO_DIR, REPO_ROOT, REPO_ROOT / "reproduce_repo"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def iterpert_kernel_path(dataset_name=ESSENTIAL_DATASET):
    """Path where the iterpert API expects preprocessed knowledge kernels."""
    return os.path.join(GEARS_DATA_PATH, f"{dataset_name}_kernels", "knowledge_kernels_1k") + os.sep


def ensure_kernel_symlink(dataset_name=ESSENTIAL_DATASET):
    """Link preprocessed kernels to the path expected by iterpert.initialize_active_learning_strategy."""
    link = iterpert_kernel_path(dataset_name).rstrip(os.sep)
    if "gw" in dataset_name:
        target = KERNELS_GENOME_WIDE.rstrip(os.sep)
    else:
        target = KERNELS_ESSENTIAL_1K.rstrip(os.sep)

    link_parent = os.path.dirname(link)
    os.makedirs(link_parent, exist_ok=True)

    if os.path.islink(link):
        if os.path.realpath(link) == os.path.realpath(target):
            return link + os.sep
        os.remove(link)
    elif os.path.exists(link):
        return link + os.sep

    os.symlink(target, link)
    return link + os.sep


def print_paths():
    """Print key paths for debugging."""
    keys = [
        "DATA_ROOT", "GEARS_DATA_PATH", "KERNELS_ESSENTIAL_1K",
        "KERNELS_GENOME_WIDE", "DEMO_RESULTS_PATH", "GW_RAW_H5AD",
        "DEFAULT_DEVICE",
    ]
    for key in keys:
        print(f"{key}: {globals()[key]}")


def check_environment():
    """Validate conda env and repo dependencies before running notebooks."""
    import sys

    py = sys.version_info
    if py.major != 3 or py.minor != 8:
        print(f"WARN: 推荐 Python 3.8（当前 {py.major}.{py.minor}）")

    setup_notebook_paths()
    missing = []
    for mod in ("torch", "scanpy", "torch_geometric", "pandas", "tqdm"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    try:
        import bmdal_reg  # noqa: F401
    except ImportError:
        missing.append("bmdal_reg (via reproduce_repo/)")

    if missing:
        raise ImportError(f"缺少依赖: {', '.join(missing)}")

    import torch
    print(f"Python {py.major}.{py.minor} | torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()} | 默认设备: {DEFAULT_DEVICE}")
    print("环境检查通过")


def check_essential_ready():
    """Check files needed for data_tutorial / train_tutorial."""
    ensure_gears_shared_files()
    required = [
        os.path.join(GEARS_DATA_PATH, ESSENTIAL_DATASET, "perturb_processed.h5ad"),
        os.path.join(GEARS_DATA_PATH, ESSENTIAL_DATASET, "data_pyg", "cell_graphs.pkl"),
        os.path.join(KERNELS_ESSENTIAL_1K, "pops_kernel", "kernel.pkl"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Essential 1K 数据未就绪:\n  " + "\n  ".join(missing))
    ensure_kernel_symlink(ESSENTIAL_DATASET)
    os.makedirs(DEMO_RESULTS_PATH, exist_ok=True)
    print("Essential 1K 数据就绪")


def check_gw_ready(strict: bool = True) -> dict:
    """Check GW prerequisites. Returns status dict; raises if strict and not ready."""
    status = {
        "gw_preprocess": os.path.exists(os.path.join(GW_DATA_PATH, "perturb_processed.h5ad")),
        "gene_id_h5ad": os.path.exists(GW_PROCESSED_H5AD),
        "embeddings": os.path.exists(ESM_EMB),
    }
    if all(status.values()):
        print("Genome-wide 前置数据就绪")
        return status

    lines = ["Genome-wide 前置数据未就绪:"]
    if not status["gene_id_h5ad"]:
        lines.append(f"  - gene_id 中间文件: {GW_PROCESSED_H5AD}")
        lines.append("    快速修复: python demo/prepare_gw_prerequisites.py --gene-id-only")
    if not status["gw_preprocess"]:
        lines.append(f"  - GW 预处理: {GW_DATA_PATH}/perturb_processed.h5ad")
        lines.append("    耗时修复: python demo/prepare_gw_prerequisites.py --full-preprocess")
    if not status["embeddings"]:
        lines.append(f"  - ESM embedding: {ESM_EMB}")
        lines.append("    手动下载: https://drive.google.com/file/d/16p9sQYkhpM-PBAcNWQdjL9pxDZ46krQX/view")
        lines.append(f"    解压到: {EMBEDDINGS_DIR}")

    msg = "\n".join(lines)
    if strict:
        raise FileNotFoundError(msg)
    print(msg)
    return status
