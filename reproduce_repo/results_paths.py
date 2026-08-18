"""Shared path helpers for results/ directory layout."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIG4 = RESULTS / "fig4"
FIG4C = RESULTS / "fig4c"
COMPARISON = FIG4 / "comparison"
SCRIPTS = FIG4 / "_scripts"

METHOD_SLUGS = {
    "IterPert": ("essential_1k", "iterpert"),
    "Random": ("essential_1k", "baselines/random"),
    "Core-Set": ("essential_1k", "baselines/core_set"),
    "BALD": ("essential_1k", "baselines/bald"),
    "LCMD": ("essential_1k", "baselines/lcmd"),
    "BADGE": ("essential_1k", "baselines/badge"),
    "ACS-FW": ("essential_1k", "baselines/acs_fw"),
    "BatchBALD": ("essential_1k", "baselines/batchbald"),
    "TypiClust": ("essential_1k", "baselines/typiclust"),
    "KMeans": ("essential_1k", "baselines/kmeans"),
}

LEGACY_SLUG = {
    "iterpert": "iterpert",
    "random": "random",
    "coreset": "core_set",
    "bald": "bald",
    "lcmd": "lcmd",
    "badge": "badge",
    "acs_fw": "acs_fw",
    "batchbald": "batchbald",
    "typiclust": "typiclust",
    "kmeans": "kmeans",
}


def method_root(method: str) -> Path:
    rel = METHOD_SLUGS[method][1] if method in METHOD_SLUGS else method
    prefix = "essential_1k" if method != "IterPert" else "essential_1k"
    if method in METHOD_SLUGS:
        return FIG4 / METHOD_SLUGS[method][0] / METHOD_SLUGS[method][1]
    return FIG4 / "essential_1k" / rel


def method_runs(method: str) -> Path:
    return method_root(method) / "runs"


def method_tables(method: str) -> Path:
    return method_root(method) / "tables"


def method_figures(method: str) -> Path:
    return method_root(method) / "figures"


def method_manifests(method: str) -> Path:
    return method_root(method) / "manifests"


def legacy_slug_to_method_dir(slug: str) -> Path:
    name = LEGACY_SLUG.get(slug, slug)
    if name == "iterpert":
        return FIG4 / "essential_1k" / "iterpert"
    return FIG4 / "essential_1k" / "baselines" / name


def ensure_method_dirs(method: str) -> None:
    for p in (method_runs(method), method_tables(method), method_figures(method), method_manifests(method)):
        p.mkdir(parents=True, exist_ok=True)


def ensure_comparison_dirs() -> None:
    for p in (COMPARISON / "tables", COMPARISON / "figures", COMPARISON / "manifests"):
        p.mkdir(parents=True, exist_ok=True)
