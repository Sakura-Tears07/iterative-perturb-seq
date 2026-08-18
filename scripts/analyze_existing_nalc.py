#!/usr/bin/env python3
"""P0.1: Recompute learning curves with nALC, paired differences, and threshold crossing."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    METHODS,
    compute_round_gains,
    ensure_analysis_dir,
    load_method_long,
    paired_bootstrap_ci,
    pivot_runs,
    threshold_crossing,
    trapezoid_nalc,
)

PRIMARY_COMPARISONS = [
    ("IterPert", "BADGE"),
    ("IterPert", "Random"),
    ("IterPert", "Core-Set"),
    ("IterPert", "TypiClust"),
]

THRESHOLDS = [0.25, 0.27]
BUDGETS = [200, 300, 400, 600]


def build_method_summary(all_pivots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, pivot in all_pivots.items():
        nalc = trapezoid_nalc(pivot.columns, pivot.mean(axis=0))
        for col in pivot.columns:
            rows.append({
                "method": method,
                "n_labeled": int(col),
                "mean": pivot[col].mean(),
                "std": pivot[col].std(ddof=1),
                "min": pivot[col].min(),
                "max": pivot[col].max(),
                "count": pivot[col].notna().sum(),
            })
        rows.append({
            "method": method,
            "n_labeled": "nALC",
            "mean": nalc,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "count": len(pivot),
        })
    return pd.DataFrame(rows)


def build_nalc_table(all_pivots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, pivot in all_pivots.items():
        per_run = {run: trapezoid_nalc(pivot.columns, pivot.loc[run]) for run in pivot.index}
        rows.append({
            "method": method,
            "nalc_mean": np.mean(list(per_run.values())),
            "nalc_std": np.std(list(per_run.values()), ddof=1),
            **{f"run_{run}": val for run, val in per_run.items()},
        })
    return pd.DataFrame(rows)


def build_paired_differences(all_pivots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method_a, method_b in PRIMARY_COMPARISONS:
        pivot_a = all_pivots[method_a]
        pivot_b = all_pivots[method_b]
        common_runs = sorted(set(pivot_a.index) & set(pivot_b.index))
        for budget in BUDGETS:
            if budget not in pivot_a.columns or budget not in pivot_b.columns:
                continue
            a = pivot_a.loc[common_runs, budget].to_numpy(dtype=float)
            b = pivot_b.loc[common_runs, budget].to_numpy(dtype=float)
            mean_diff, lo, hi = paired_bootstrap_ci(a, b)
            rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "comparison": f"{method_a}_minus_{method_b}",
                "budget": budget,
                "mean_difference": mean_diff,
                "std_difference": float(np.std(a - b, ddof=1)),
                "paired_bootstrap_ci_low": lo,
                "paired_bootstrap_ci_high": hi,
                "win_rate_across_seeds": float(np.mean(a > b)),
                "n_pairs": len(common_runs),
            })

        nalc_a = np.array([trapezoid_nalc(pivot_a.columns, pivot_a.loc[r]) for r in common_runs])
        nalc_b = np.array([trapezoid_nalc(pivot_b.columns, pivot_b.loc[r]) for r in common_runs])
        mean_diff, lo, hi = paired_bootstrap_ci(nalc_a, nalc_b)
        rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "comparison": f"{method_a}_minus_{method_b}",
            "budget": "nALC",
            "mean_difference": mean_diff,
            "std_difference": float(np.std(nalc_a - nalc_b, ddof=1)),
            "paired_bootstrap_ci_low": lo,
            "paired_bootstrap_ci_high": hi,
            "win_rate_across_seeds": float(np.mean(nalc_a > nalc_b)),
            "n_pairs": len(common_runs),
        })
    return pd.DataFrame(rows)


def build_threshold_table(all_pivots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, pivot in all_pivots.items():
        for threshold in THRESHOLDS:
            crossing = threshold_crossing(pivot, threshold)
            rows.append({
                "method": method,
                "threshold": threshold,
                "mean_n_labeled": crossing.mean(),
                "std_n_labeled": crossing.std(ddof=1),
                "fraction_reached": float(crossing.notna().mean()),
                **{f"run_{run}": crossing.loc[run] for run in crossing.index},
            })
    return pd.DataFrame(rows)


def build_round_gain_summary(all_pivots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, pivot in all_pivots.items():
        gains = compute_round_gains(pivot)
        for n_labeled, sub in gains.groupby("n_labeled"):
            rows.append({
                "method": method,
                "n_labeled": int(n_labeled),
                "round_gain_mean": sub["round_gain"].mean(),
                "round_gain_std": sub["round_gain"].std(ddof=1),
            })
    return pd.DataFrame(rows)


def plot_learning_curves(all_pivots: dict[str, pd.DataFrame], outpath: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    highlight = {"IterPert": "#d62728", "BADGE": "#e377c2", "Random": "#7f7f7f", "Core-Set": "#bcbd22", "TypiClust": "#2ca02c"}
    for method in ["IterPert", "BADGE", "TypiClust", "Random", "Core-Set"]:
        if method not in all_pivots:
            continue
        pivot = all_pivots[method]
        x = sorted(pivot.columns)
        mean = pivot[x].mean(axis=0)
        std = pivot[x].std(axis=0)
        color = highlight.get(method, None)
        lw = 2.5 if method == "IterPert" else 1.8
        ax.plot(x, mean, "o-", label=method, color=color, linewidth=lw)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("Number of labeled perturbations")
    ax.set_ylabel("Pearson delta")
    ax.set_title("E0 scientific summary: learning speed vs final performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    outdir = ensure_analysis_dir("e0_baseline")
    all_pivots: dict[str, pd.DataFrame] = {}
    availability = {}

    for name in METHODS:
        try:
            df = load_method_long(name)
            all_pivots[name] = pivot_runs(df)
            availability[name] = len(find_files := __import__("analysis_utils").find_method_files(METHODS[name]))
        except FileNotFoundError:
            availability[name] = 0

    summary = build_method_summary(all_pivots)
    paired = build_paired_differences(all_pivots)
    nalc = build_nalc_table(all_pivots)
    thresholds = build_threshold_table(all_pivots)
    round_gains = build_round_gain_summary(all_pivots)

    summary.to_csv(outdir / "e0_method_summary.csv", index=False)
    paired.to_csv(outdir / "e0_paired_differences.csv", index=False)
    nalc.to_csv(outdir / "e0_nalc.csv", index=False)
    thresholds.to_csv(outdir / "e0_threshold_crossing.csv", index=False)
    round_gains.to_csv(outdir / "e0_round_gains.csv", index=False)
    pd.DataFrame([{"method": k, "n_runs_found": v} for k, v in availability.items()]).to_csv(
        outdir / "e0_availability.csv", index=False
    )

    plot_learning_curves(all_pivots, outdir / "learning_curves_primary.png")

    print("Saved E0 analysis to", outdir)
    print("\nKey paired differences (IterPert - BADGE):")
    sub = paired[(paired["method_a"] == "IterPert") & (paired["method_b"] == "BADGE")]
    for _, row in sub.iterrows():
        print(
            f"  budget={row['budget']:>4}: diff={row['mean_difference']:+.4f} "
            f"[{row['paired_bootstrap_ci_low']:+.4f}, {row['paired_bootstrap_ci_high']:+.4f}] "
            f"win_rate={row['win_rate_across_seeds']:.2f}"
        )

    print("\nnALC ranking:")
    print(nalc.sort_values("nalc_mean", ascending=False)[["method", "nalc_mean", "nalc_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
