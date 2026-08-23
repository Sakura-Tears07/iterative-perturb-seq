#!/usr/bin/env python3
"""Aggregate Fig.4c single-prior ablation results."""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from results_paths import FIG4C, RESULTS

PRIORS = [
    "rpe1_kernel",
    "ops_A549_kernel",
    "pops_kernel",
    "esm_kernel",
    "biogpt_kernel",
    "node2vec_kernel",
    "ops_HeLa_HPLM_kernel",
    "ops_HeLa_DMEM_kernel",
]

OUT = FIG4C / "comparison"
RUNS_GLOB = "GEARS_*single_{prior}_run*_Core-Set_diff_effect_metrics.csv"


def load_prior(prior: str) -> tuple[pd.DataFrame, list[str]]:
    pattern = str(FIG4C / "single_prior" / prior / "runs" / RUNS_GLOB.format(prior=prior.replace("_kernel", "")))
    # glob with prior slug in path
    files = sorted(glob.glob(str(FIG4C / "single_prior" / prior / "runs" / f"*single_{prior}_run*_metrics.csv")))
    if not files:
        raise FileNotFoundError(f"No metrics for {prior}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["prior"] = prior
    return df, files


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("n_labeled")["pearson_delta"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    all_rows = []
    at600 = []
    for prior in PRIORS:
        try:
            df, files = load_prior(prior)
        except FileNotFoundError:
            print(f"SKIP {prior}: no data")
            continue
        if len(files) != 5:
            print(f"WARN {prior}: expected 5 runs, got {len(files)}")
        summ = summarize(df)
        summ["prior"] = prior
        all_rows.append(summ)
        sub = df[df["n_labeled"] == 600]
        if not sub.empty:
            at600.append({"prior": prior, "mean": sub["pearson_delta"].mean(), "std": sub["pearson_delta"].std(), "n": len(files)})

    if not all_rows:
        raise RuntimeError("No Fig.4c data found")

    long_df = pd.concat(all_rows, ignore_index=True)
    long_df.to_csv(OUT / "tables" / "all_priors_long.csv", index=False)

    summary600 = pd.DataFrame(at600).sort_values("mean", ascending=False)
    summary600.to_csv(OUT / "tables" / "all_priors_at600.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    for prior in long_df["prior"].unique():
        sub = long_df[long_df["prior"] == prior]
        plt.plot(sub["n_labeled"], sub["mean"], marker="o", label=prior.replace("_kernel", ""))
        plt.fill_between(sub["n_labeled"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.15)
    plt.xlabel("Labeled perturbations")
    plt.ylabel("Pearson delta")
    plt.title("Fig.4c — Single Prior Ablation")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "all_priors.png", dpi=150)
    plt.close()

    print("Saved to:", OUT)
    for _, r in summary600.iterrows():
        print(f"  {r['prior']:22s}  {r['mean']:.4f} ± {r['std']:.4f}  (n={int(r['n'])})")


if __name__ == "__main__":
    main()
