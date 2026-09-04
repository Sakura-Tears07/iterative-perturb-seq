#!/usr/bin/env python3
"""Does prior kernel alignment predict Fig.4c single-prior utility?"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS,
    ensure_analysis_dir,
    load_prior_kernel,
    kernel_alignment_uncentered,
    trapezoid_nalc,
)

ALIGN_CSV = RESULTS / "analysis" / "idea2_alignment" / "mean_alignment_by_round.csv"
FIG4C_LONG = RESULTS / "fig4c" / "comparison" / "tables" / "all_priors_long.csv"
FIG4C_600 = RESULTS / "fig4c" / "comparison" / "tables" / "all_priors_at600.csv"

# Fig.4c prior name → alignment column name
PRIOR_MAP = {
    "rpe1_kernel": "rpe1",
    "pops_kernel": "pops",
    "esm_kernel": "esm",
    "biogpt_kernel": "biogpt",
    "node2vec_kernel": "node2vec",
    "ops_A549_kernel": "ops_A549",
    "ops_HeLa_HPLM_kernel": "ops_HeLa_HPLM",
    "ops_HeLa_DMEM_kernel": "ops_HeLa_DMEM",
}


def prior_nalc(long_df: pd.DataFrame) -> pd.Series:
    rows = {}
    for prior, g in long_df.groupby("prior"):
        g = g.sort_values("n_labeled")
        rows[prior] = trapezoid_nalc(g["n_labeled"], g["mean"])
    return pd.Series(rows, name="nALC")


def main() -> None:
    out = ensure_analysis_dir("idea2_alignment_vs_utility")
    if not FIG4C_LONG.exists():
        print(f"Missing {FIG4C_LONG}")
        return
    long_df = pd.read_csv(FIG4C_LONG)
    util = prior_nalc(long_df).rename("utility_nALC")
    at600 = None
    if FIG4C_600.exists():
        at600 = pd.read_csv(FIG4C_600).set_index("prior")["mean"].rename("utility_P600")

    # Alignment from Phase-0 sweep logs (labeled-set K[S,S] vs gold)
    align_frames = []
    if ALIGN_CSV.exists():
        align = pd.read_csv(ALIGN_CSV).set_index("n_labeled")
        for prior, col in PRIOR_MAP.items():
            if col not in align.columns:
                continue
            for n_lab, row in align.iterrows():
                align_frames.append({
                    "prior": prior,
                    "kernel": col,
                    "n_labeled_align": int(n_lab),
                    "alignment": float(row[col]),
                    "source": "sweep_logs",
                })

    # Also: static prior–gold alignment on full kernel (offline, no GEARS)
    try:
        gold_names, gold_K = load_prior_kernel("ground_truth_delta")
        name_to_i = {g: i for i, g in enumerate(gold_names)}
        for prior, col in PRIOR_MAP.items():
            try:
                names, K = load_prior_kernel(prior)
            except Exception as exc:
                print("skip kernel", prior, exc)
                continue
            # Align on intersection order of gold
            idx = [name_to_i[n.split("+")[0] if "+" in n else n] for n in names
                   if (n.split("+")[0] if "+" in n else n) in name_to_i]
            # Better: reindex both by shared gene names
            pnames = [n.split("+")[0] for n in names]
            gnames = [n.split("+")[0] for n in gold_names]
            shared = [g for g in pnames if g in set(gnames)]
            if len(shared) < 50:
                continue
            pi = [pnames.index(g) for g in shared]
            gi = [gnames.index(g) for g in shared]
            Kp = K[np.ix_(pi, pi)]
            Kg = gold_K[np.ix_(gi, gi)]
            a = kernel_alignment_uncentered(Kp, Kg)
            align_frames.append({
                "prior": prior,
                "kernel": col,
                "n_labeled_align": -1,
                "alignment": float(a),
                "source": "full_kernel_vs_gold",
            })
    except Exception as exc:
        print("full-kernel alignment skipped:", exc)

    align_df = pd.DataFrame(align_frames)
    if align_df.empty:
        print("No alignment rows")
        return
    align_df.to_csv(out / "prior_alignments.csv", index=False)

    util_df = util.reset_index().rename(columns={"index": "prior"})
    if at600 is not None:
        util_df = util_df.merge(at600.reset_index(), on="prior", how="left")

    merged = align_df.merge(util_df, on="prior", how="inner")
    merged.to_csv(out / "alignment_with_utility.csv", index=False)

    corr_rows = []
    for (source, n_lab), sub in merged.groupby(["source", "n_labeled_align"]):
        if len(sub) < 4:
            continue
        for util_col in ["utility_nALC", "utility_P600"]:
            if util_col not in sub.columns or sub[util_col].isna().all():
                continue
            rho, p = spearmanr(sub["alignment"], sub[util_col])
            corr_rows.append({
                "alignment_source": source,
                "n_labeled_align": int(n_lab),
                "utility": util_col,
                "spearman": float(rho),
                "pvalue": float(p),
                "n_priors": int(len(sub)),
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out / "spearman_alignment_vs_utility.csv", index=False)

    # Scatter: alignment@100 vs nALC (most relevant early-trust story)
    sub = merged[(merged["source"] == "sweep_logs") & (merged["n_labeled_align"] == 100)]
    if sub.empty:
        sub = merged[merged["source"] == "full_kernel_vs_gold"]
    if not sub.empty and "utility_nALC" in sub.columns:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(sub["alignment"], sub["utility_nALC"], s=50)
        for r in sub.itertuples():
            ax.annotate(r.kernel, (r.alignment, r.utility_nALC), fontsize=8,
                        textcoords="offset points", xytext=(4, 4))
        rho, p = spearmanr(sub["alignment"], sub["utility_nALC"])
        ax.set_xlabel("kernel alignment vs gold")
        ax.set_ylabel("Fig.4c single-prior nALC")
        ax.set_title(f"Alignment vs prior utility (Spearman ρ={rho:.2f}, p={p:.3f})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "alignment_vs_nalc.png", dpi=160)
        plt.close(fig)

    print(corr_df.to_string(index=False) if not corr_df.empty else "no correlations")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
