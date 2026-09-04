"""Shared utilities for IterPert offline analysis."""
from __future__ import annotations

import glob
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
ANALYSIS = RESULTS / "analysis"
DATA_ROOT = Path("/data/zy/iterpert")
KERNEL_ROOT = DATA_ROOT / "knowledge_kernels" / "essential_1k"

METHODS = {
    "IterPert": {
        "pattern": "*priormean_new_max*Core-Set_diff_effect_metrics.csv",
        "filter": lambda f: (
            "100_5_100" in f
            and "single_" not in f
            and "_mw" not in f
            and "_sched_" not in f
            and "dup" not in f
        ),
    },
    "Random": {
        "pattern": "*Random_cross_gene_out_metrics.csv",
        "filter": lambda f: "100_5_100" in f,
    },
    "Core-Set": {
        "pattern": "*Core-Set_cross_gene_out_metrics.csv",
        "filter": lambda f: "100_5_100" in f,
    },
    "BADGE": {
        "pattern": "*BADGE_cross_gene_out_metrics.csv",
        "filter": lambda f: "100_5_100" in f,
    },
    "TypiClust": {
        "pattern": "*TypiClust_metrics.csv",
        "filter": lambda f: "100_5_100" in f,
    },
    "BALD": {
        "pattern": "*BALD_cross_gene_out_metrics.csv",
        "filter": lambda f: "100_5_100" in f and "BatchBALD" not in f,
    },
    "LCMD": {
        "pattern": "*LCMD_cross_gene_out_metrics.csv",
        "filter": lambda f: "100_5_100" in f,
    },
    "ACS-FW": {
        "pattern": "*ACS-FW_cross_gene_out_metrics.csv",
        "filter": lambda f: "100_5_100" in f,
    },
    "BatchBALD": {
        "pattern": "*BatchBALD_cross_gene_out_metrics.csv",
        "filter": lambda f: "100_5_100" in f,
    },
    "KMeans": {
        "pattern": "*KMeansSampling_metrics.csv",
        "filter": lambda f: "100_5_100" in f,
    },
}

ESSENTIAL_PRIORS = [
    "pops_kernel",
    "rpe1_kernel",
    "esm_kernel",
    "biogpt_kernel",
    "node2vec_kernel",
    "ops_A549_kernel",
    "ops_HeLa_HPLM_kernel",
    "ops_HeLa_DMEM_kernel",
]


def find_method_files(spec: dict) -> list[str]:
    files = sorted(glob.glob(str(RESULTS / "**" / spec["pattern"]), recursive=True))
    return [f for f in files if spec["filter"](f)]


def load_method_long(name: str, spec: dict | None = None) -> pd.DataFrame:
    spec = spec or METHODS[name]
    files = find_method_files(spec)
    if not files:
        raise FileNotFoundError(f"No result files found for method={name}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["method"] = name
    return df


def pivot_runs(df: pd.DataFrame, value_col: str = "pearson_delta") -> pd.DataFrame:
    p = df.pivot_table(index="run", columns="n_labeled", values=value_col, aggfunc="first")
    p.columns = [int(c) for c in p.columns]
    return p.sort_index()


def trapezoid_nalc(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    if len(x_arr) < 2:
        return float("nan")
    return float(np.trapz(y_arr, x_arr) / (x_arr[-1] - x_arr[0]))


def compute_round_gains(pivot: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(pivot.columns)
    rows = []
    for run in pivot.index:
        prev = None
        for col in cols:
            val = pivot.loc[run, col]
            gain = float(val - prev) if prev is not None else float(val)
            rows.append({"run": run, "n_labeled": col, "pearson_delta": val, "round_gain": gain})
            prev = val
    return pd.DataFrame(rows)


def threshold_crossing(pivot: pd.DataFrame, threshold: float) -> pd.Series:
    cols = sorted(pivot.columns)

    def first_cross(row: pd.Series) -> float:
        for col in cols:
            if row[col] >= threshold:
                return float(col)
        return float("nan")

    return pivot.apply(first_cross, axis=1)


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    boot = np.empty(n_boot, dtype=float)
    n = len(diffs)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(diffs[idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return mean_diff, float(lo), float(hi)


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def kernel_alignment_uncentered(K1: np.ndarray, K2: np.ndarray) -> float:
    denom = np.linalg.norm(K1, "fro") * np.linalg.norm(K2, "fro")
    if denom == 0:
        return float("nan")
    return float(np.trace(K1 @ K2) / denom)


def kernel_cka(K1: np.ndarray, K2: np.ndarray, zero_diagonal: bool = False) -> float:
    K1 = np.asarray(K1, dtype=float).copy()
    K2 = np.asarray(K2, dtype=float).copy()
    if zero_diagonal:
        np.fill_diagonal(K1, 0.0)
        np.fill_diagonal(K2, 0.0)
    K1c = center_kernel(K1)
    K2c = center_kernel(K2)
    denom = np.linalg.norm(K1c, "fro") * np.linalg.norm(K2c, "fro")
    if denom == 0:
        return float("nan")
    return float(np.sum(K1c * K2c) / denom)


def offdiag_spearman(K1: np.ndarray, K2: np.ndarray) -> float:
    from scipy.stats import spearmanr

    n = K1.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(spearmanr(K1[iu], K2[iu]).correlation)


def load_prior_kernel(name: str) -> tuple[list[str], np.ndarray]:
    root = KERNEL_ROOT / name
    with open(root / "pert_list.pkl", "rb") as f:
        pert_list = pickle.load(f)
    with open(root / "kernel.pkl", "rb") as f:
        kernel = pickle.load(f)
    pert_list = [p.split("+")[0] for p in pert_list]
    return pert_list, np.asarray(kernel)


def normalize_gene_name(name: str) -> str:
    return name.split("+")[0]


def ensure_analysis_dir(name: str) -> Path:
    out = ANALYSIS / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
