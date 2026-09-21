"""Local path configuration for ITERPert reproduction.

Environment variables (all optional; defaults target the local machine):

    ITERPERT_DATA_ROOT          base data dir
    ITERPERT_GEARS_DATA_PATH    GEARS processed data (defaults to <root>/perturb_seq_data/gears_data)
    ITERPERT_ESS1K_KERNEL_DIR   essential-1k knowledge kernels (defaults to <gears>/replogle_k562_essential_1000hvg_kernels/knowledge_kernels_1k)
    ITERPERT_GW_KERNEL_DIR      genome-wide knowledge kernels (fig6; not available locally)
    ITERPERT_CHECKPOINT_DIR     model checkpoint dir
"""
import os
from pathlib import Path

DATA_ROOT = os.environ.get("ITERPERT_DATA_ROOT", "/data/lhr/ai4s/iterpert/scratch")
REPO_ROOT = Path(__file__).resolve().parents[1]

GEARS_DATA_PATH = os.environ.get(
    "ITERPERT_GEARS_DATA_PATH",
    os.path.join(DATA_ROOT, "perturb_seq_data", "gears_data"),
) + os.sep
SAVE_DIR = os.environ.get(
    "ITERPERT_CHECKPOINT_DIR", os.path.join(DATA_ROOT, "checkpoints"))
RES_DIR = str(REPO_ROOT / "results")
GRADIENT_KERNEL_DIR = os.path.join(DATA_ROOT, "knowledge_kernels", "gradient_kernel")

ESS1K_KERNEL_DIR = os.environ.get(
    "ITERPERT_ESS1K_KERNEL_DIR",
    os.path.join(GEARS_DATA_PATH,
                 "replogle_k562_essential_1000hvg_kernels", "knowledge_kernels_1k"),
) + os.sep

GW_KERNEL_DIR = os.environ.get(
    "ITERPERT_GW_KERNEL_DIR",
    os.path.join(DATA_ROOT, "knowledge_kernels_gw"),
) + os.sep


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
    "replogle_k562_essential_1000hvg": ESS1K_KERNEL_DIR,
    "replogle_k562_essential_1000hvg+pert_in_gene": ESS1K_KERNEL_DIR,
    "replogle_k562_gw_1000hvg": GW_KERNEL_DIR,
    "replogle_rpe1_essential_1000hvg": ESS1K_KERNEL_DIR,
}

CUSTOM_TEST_SPLIT = os.path.join(
    GEARS_DATA_PATH,
    "replogle_k562_essential_1000hvg+pert_in_gene",
    "splits",
    "replogle_k562_essential_1000hvg+pert_in_gene_active_1_0.75.pkl",
)

ESSENTIAL_GENE_PATHS = {
    "gw": GW_KERNEL_DIR,
    "default": ESS1K_KERNEL_DIR,
}
