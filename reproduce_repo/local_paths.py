"""Local path configuration for ITERpert reproduction."""
import os
from pathlib import Path

DATA_ROOT = os.environ.get("ITERPERT_DATA_ROOT", "/data/zy/iterpert")
REPO_ROOT = Path(__file__).resolve().parents[1]

GEARS_DATA_PATH = os.path.join(DATA_ROOT, "datasets", "gears_data") + os.sep
SAVE_DIR = os.path.join(DATA_ROOT, "checkpoints")
RES_DIR = str(REPO_ROOT / "results")
GRADIENT_KERNEL_DIR = os.path.join(DATA_ROOT, "cache", "gradient_kernel")


def resolve_result_dir(args) -> str:
    """Return experiment-specific subdirectory under results/."""
    root = Path(RES_DIR)

    if getattr(args, "use_single_prior", False):
        prior = getattr(args, "single_prior", "unknown_prior")
        return str(root / "fig4c" / "single_prior" / prior / "runs")

    dataset = getattr(args, "dataset_name", "")
    if "essential" in dataset:
        if getattr(args, "epoch_per_cycle", 20) == 1:
            return str(root / "fig4" / "essential_1k" / "smoke" / "runs")

        if (
            getattr(args, "use_prior", False)
            and getattr(args, "kernel_strategy", None) == "Core-Set"
            and getattr(args, "base_kernel", None) == "diff_effect"
        ):
            return str(root / "fig4" / "essential_1k" / "iterpert" / "runs")

        strategy = getattr(args, "strategy_name", "")
        if strategy == "kernel_based_active_learning":
            slug = getattr(args, "kernel_strategy", "unknown").lower().replace("-", "_")
            return str(root / "fig4" / "essential_1k" / "baselines" / slug / "runs")
        if strategy == "TypiClust":
            return str(root / "fig4" / "essential_1k" / "baselines" / "typiclust" / "runs")
        if strategy == "KMeansSampling":
            return str(root / "fig4" / "essential_1k" / "baselines" / "kmeans" / "runs")

    if "gw" in dataset:
        return str(root / "fig6" / "genome_wide" / "runs")

    return str(root / "misc")

KERNEL_PATHS = {
    "replogle_k562_essential_1000hvg": os.path.join(
        DATA_ROOT, "knowledge_kernels", "essential_1k") + os.sep,
    "replogle_k562_essential_1000hvg+pert_in_gene": os.path.join(
        DATA_ROOT, "knowledge_kernels", "essential_1k") + os.sep,
    "replogle_k562_gw_1000hvg": os.path.join(
        DATA_ROOT, "knowledge_kernels", "genome_wide") + os.sep,
    "replogle_rpe1_essential_1000hvg": os.path.join(
        DATA_ROOT, "knowledge_kernels", "essential_1k") + os.sep,
}

CUSTOM_TEST_SPLIT = os.path.join(
    GEARS_DATA_PATH,
    "replogle_k562_essential_1000hvg+pert_in_gene",
    "splits",
    "replogle_k562_essential_1000hvg+pert_in_gene_active_1_0.75.pkl",
)

ESSENTIAL_GENE_PATHS = {
    "gw": os.path.join(DATA_ROOT, "knowledge_kernels", "genome_wide"),
    "default": os.path.join(DATA_ROOT, "knowledge_kernels", "essential_1k"),
}
