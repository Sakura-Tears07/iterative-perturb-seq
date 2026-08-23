#!/usr/bin/env python3
"""Aggregate Fig.4 Essential 1K results and plot learning curves."""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from results_paths import (
    COMPARISON,
    METHOD_SLUGS,
    RESULTS,
    ensure_comparison_dirs,
    ensure_method_dirs,
    method_figures,
    method_manifests,
    method_tables,
)

COMMIT = "52011a935e8e9e9a14b7d92a99c6c014d46e6db5"

METHODS = {
    "IterPert": {
        "pattern": "*priormean_new_max*Core-Set_diff_effect_metrics.csv",
        "filter": lambda f: (
            "100_5_100" in f
            and "single_" not in f
            and "/essential_1k/iterpert/runs/" in f
            and "dup" not in f
            and "alignment" not in f
            and "prior_only" not in f
        ),
        "color": "#d62728",
    },
    "Random": {"pattern": "*Random_cross_gene_out_metrics.csv", "filter": lambda f: "100_5_100" in f, "color": "#7f7f7f"},
    "Core-Set": {"pattern": "*Core-Set_cross_gene_out_metrics.csv", "filter": lambda f: "100_5_100" in f, "color": "#bcbd22"},
    "BALD": {"pattern": "*BALD_cross_gene_out_metrics.csv", "filter": lambda f: "100_5_100" in f and "BatchBALD" not in f, "color": "#17becf"},
    "LCMD": {"pattern": "*LCMD_cross_gene_out_metrics.csv", "filter": lambda f: "100_5_100" in f, "color": "#9467bd"},
    "BADGE": {"pattern": "*BADGE_cross_gene_out_metrics.csv", "filter": lambda f: "100_5_100" in f, "color": "#e377c2"},
    "ACS-FW": {"pattern": "*ACS-FW_cross_gene_out_metrics.csv", "filter": lambda f: "100_5_100" in f, "color": "#8c564b"},
    "BatchBALD": {"pattern": "*BatchBALD_cross_gene_out_metrics.csv", "filter": lambda f: "100_5_100" in f, "color": "#1f77b4"},
    "TypiClust": {"pattern": "*TypiClust_metrics.csv", "filter": lambda f: "100_5_100" in f, "color": "#2ca02c"},
    "KMeans": {"pattern": "*KMeansSampling_metrics.csv", "filter": lambda f: "100_5_100" in f, "color": "#ff7f0e"},
}


def find_files(spec: dict) -> list[str]:
    files = sorted(glob.glob(str(RESULTS / "**" / spec["pattern"]), recursive=True))
    return [f for f in files if spec["filter"](f)]


def load_method(name: str, spec: dict) -> tuple[pd.DataFrame, list[str]]:
    files = find_files(spec)
    if not files:
        raise FileNotFoundError(f"No files for {name}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["method"] = name
    return df, files


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("n_labeled")["pearson_delta"]
        .agg(mean="mean", std="std", min="min", max="max", count="count")
        .reset_index()
    )


def pivot_runs(df: pd.DataFrame) -> pd.DataFrame:
    p = df.pivot_table(index="run", columns="n_labeled", values="pearson_delta")
    p.columns = [int(c) for c in p.columns]
    return p.sort_index()


def save_method_outputs(name: str, df: pd.DataFrame) -> None:
    ensure_method_dirs(name)
    summarize(df).to_csv(method_tables(name) / "summary.csv", index=False)
    pivot_runs(df).to_csv(method_tables(name) / "pivot.csv", index=False)


def build_master_table(all_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, summary in all_summaries.items():
        for _, r in summary.iterrows():
            rows.append({
                "method": method,
                "n_labeled": int(r["n_labeled"]),
                "mean": r["mean"], "std": r["std"],
                "min": r["min"], "max": r["max"],
                "count": int(r["count"]),
            })
    return pd.DataFrame(rows)


def build_comparison_wide(all_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    labels = sorted({int(r["n_labeled"]) for s in all_summaries.values() for _, r in s.iterrows()})
    out: dict = {"n_labeled": labels}
    for method, summary in all_summaries.items():
        key = method.lower().replace("-", "").replace(" ", "_")
        lookup = summary.set_index("n_labeled")
        out[f"{key}_mean"] = [lookup.loc[n, "mean"] if n in lookup.index else float("nan") for n in labels]
        out[f"{key}_std"] = [lookup.loc[n, "std"] if n in lookup.index else float("nan") for n in labels]
    return pd.DataFrame(out)


def plot_fig4(master: pd.DataFrame, outpath: Path, methods: list[str], title: str, highlight: str | None = "IterPert") -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        sub = master[master["method"] == method].sort_values("n_labeled")
        spec = METHODS[method]
        lw = 2.8 if method == highlight else 1.8
        zorder = 10 if method == highlight else 2
        ax.plot(sub["n_labeled"], sub["mean"], "o-", label=method, color=spec["color"],
                linewidth=lw, markersize=5 if method == highlight else 4, zorder=zorder)
        ax.fill_between(sub["n_labeled"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                        color=spec["color"], alpha=0.15 if method == highlight else 0.08,
                        linewidth=0, zorder=zorder - 1)
    ax.set_xlabel("Number of labeled perturbations")
    ax.set_ylabel("Pearson correlation (delta)")
    ax.set_title(title)
    ax.set_xticks([100, 200, 300, 400, 500, 600])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9)
    ax.set_ylim(0.08, 0.32)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_manifest(all_summaries: dict[str, pd.DataFrame], file_lists: dict[str, list[str]]) -> None:
    lines = [
        "Fig.4 Essential 1K — Complete Reproduction Summary",
        "=" * 52,
        f"Commit:  {COMMIT}",
        "Dataset: replogle_k562_essential_1000hvg",
        "Metric:  pearson_delta (test set, fix_evaluation=True)",
        "Runs:    seed=1, run=1..10, epoch=20, init=100, query=100, rounds=5",
        "", "@600 ranking (mean ± std):",
    ]
    ranked = sorted(
        ((m, all_summaries[m].loc[all_summaries[m]["n_labeled"] == 600, "mean"].values[0],
          all_summaries[m].loc[all_summaries[m]["n_labeled"] == 600, "std"].values[0])
         for m in all_summaries),
        key=lambda x: -x[1],
    )
    for i, (method, mean, std) in enumerate(ranked, 1):
        lines.append(f"  {i:2d}. {method:12s}  {mean:.4f} ± {std:.4f}  (n={len(file_lists[method])})")
    lines.extend(["", "Status: ALL 10 METHODS COMPLETE"])
    (COMPARISON / "manifests" / "all_methods_manifest.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_comparison_dirs()
    all_summaries: dict[str, pd.DataFrame] = {}
    file_lists: dict[str, list[str]] = {}

    for name, spec in METHODS.items():
        df, files = load_method(name, spec)
        if len(files) != 10:
            raise RuntimeError(f"{name}: expected 10 runs, got {len(files)}")
        file_lists[name] = files
        all_summaries[name] = summarize(df)
        save_method_outputs(name, df)

    master_long = build_master_table(all_summaries)
    tables = COMPARISON / "tables"
    figures = COMPARISON / "figures"
    master_long.to_csv(tables / "all_methods_long.csv", index=False)
    build_comparison_wide(all_summaries).to_csv(tables / "all_methods_comparison.csv", index=False)
    at600 = master_long[master_long["n_labeled"] == 600].sort_values("mean", ascending=False)
    at600[["method", "mean", "std", "min", "max"]].to_csv(tables / "all_methods_summary.csv", index=False)

    all_names = list(METHODS.keys())
    plot_fig4(master_long, figures / "all_methods.png", all_names, "Fig.4 Essential 1K — All Methods")
    plot_fig4(master_long, figures / "main_baselines.png", ["IterPert", "Random", "LCMD", "BADGE"],
              "Fig.4 — IterPert vs Key Baselines")
    plot_fig4(master_long, figures / "all_baselines.png", [m for m in all_names if m != "IterPert"],
              "Fig.4 — Baselines (no IterPert)", highlight=None)

    write_manifest(all_summaries, file_lists)

    print("Saved to:", COMPARISON)
    for _, r in at600.iterrows():
        print(f"  {r['method']:12s}  {r['mean']:.4f} ± {r['std']:.4f}")


if __name__ == "__main__":
    main()
