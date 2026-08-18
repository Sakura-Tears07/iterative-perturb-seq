#!/usr/bin/env python3
"""P0.2: Prior kernel similarity matrix with multiple alignment definitions."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    ESSENTIAL_PRIORS,
    ensure_analysis_dir,
    kernel_alignment_uncentered,
    kernel_cka,
    load_prior_kernel,
    offdiag_spearman,
)

METRICS = [
    ("uncentered_alignment", kernel_alignment_uncentered),
    ("centered_cka", lambda a, b: kernel_cka(a, b, zero_diagonal=False)),
    ("centered_cka_zero_diag", lambda a, b: kernel_cka(a, b, zero_diagonal=True)),
    ("offdiag_spearman", offdiag_spearman),
]


def align_kernels(priors: dict[str, np.ndarray], pert_list: list[str]) -> dict[str, np.ndarray]:
    index = {p: i for i, p in enumerate(pert_list)}
    aligned = {}
    for name, kernel in priors.items():
        idx = [index[p] for p in pert_list]
        aligned[name] = kernel[np.ix_(idx, idx)]
    return aligned


def build_pairwise_table(aligned: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(aligned.keys())
    rows = []
    for i, a in enumerate(names):
        for b in names[i:]:
            row = {"prior_a": a, "prior_b": b}
            for metric_name, fn in METRICS:
                row[metric_name] = fn(aligned[a], aligned[b])
            rows.append(row)
    return pd.DataFrame(rows)


def matrix_from_pairs(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    names = sorted(set(df["prior_a"]) | set(df["prior_b"]))
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for _, row in df.iterrows():
        mat.loc[row["prior_a"], row["prior_b"]] = row[metric]
        mat.loc[row["prior_b"], row["prior_a"]] = row[metric]
    return mat


def plot_heatmap(mat: pd.DataFrame, title: str, outpath: Path) -> None:
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(8, 6))
    short = [n.replace("_kernel", "") for n in mat.index]
    sns.heatmap(mat, xticklabels=short, yticklabels=short, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster(mat: pd.DataFrame, title: str, outpath: Path) -> None:
    dist = 1 - mat.to_numpy()
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    short = [n.replace("_kernel", "") for n in mat.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    from scipy.cluster.hierarchy import dendrogram

    dendrogram(Z, labels=short, ax=ax, leaf_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def rank_stability(pairwise: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m, _ in METRICS if m != "offdiag_spearman"]
    rows = []
    priors = sorted(set(pairwise["prior_a"]))
    for metric in metrics:
        sub = pairwise[pairwise["prior_a"] != pairwise["prior_b"]]
        for prior in priors:
            vals = []
            for other in priors:
                if other == prior:
                    continue
                row = sub[((sub["prior_a"] == prior) & (sub["prior_b"] == other)) | ((sub["prior_a"] == other) & (sub["prior_b"] == prior))]
                if len(row):
                    vals.append(row.iloc[0][metric])
            rows.append({"prior": prior, "metric": metric, "mean_similarity_to_others": np.mean(vals)})
    return pd.DataFrame(rows)


def main() -> None:
    outdir = ensure_analysis_dir("e0_baseline")

    priors = {}
    ref_pert = None
    for name in ESSENTIAL_PRIORS:
        pert_list, kernel = load_prior_kernel(name)
        priors[name] = kernel
        ref_pert = pert_list if ref_pert is None else ref_pert

    aligned = align_kernels(priors, ref_pert)
    pairwise = build_pairwise_table(aligned)
    pairwise.to_csv(outdir / "prior_pairwise_similarity.csv", index=False)

    for metric, _ in METRICS:
        mat = matrix_from_pairs(pairwise, metric)
        mat.to_csv(outdir / f"prior_similarity_{metric}.csv")
        plot_heatmap(mat, f"Prior similarity ({metric})", outdir / f"prior_similarity_{metric}.png")

    centered = matrix_from_pairs(pairwise, "centered_cka")
    plot_cluster(centered, "Prior clustering (1 - centered CKA)", outdir / "prior_cluster_dendrogram.png")

    stability = rank_stability(pairwise)
    stability.to_csv(outdir / "prior_mean_similarity.csv", index=False)

    print("Saved prior similarity analysis to", outdir)
    print("\nHighest-redundancy pairs (centered CKA):")
    sub = pairwise[pairwise["prior_a"] != pairwise["prior_b"]].sort_values("centered_cka", ascending=False)
    print(sub.head(8)[["prior_a", "prior_b", "centered_cka", "uncentered_alignment"]].to_string(index=False))


if __name__ == "__main__":
    main()
