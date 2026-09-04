#!/usr/bin/env python3
"""Extract round-wise kernel alignment from Idea 2 sweep logs / round_state JSON.

Alignment is kernel_alignment(K[S,S], gold) on the labeled set at query time,
so n_labeled is 100, 200, 300, 400, 500 (not 600). The model kernel is built
after GEARS trained on S — model values are optimistic.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import DATA_ROOT, ensure_analysis_dir

REPO = Path(__file__).resolve().parents[1]
JSON_DIR = REPO / "results" / "round_states"
LOG_DIR = DATA_ROOT / "logs" / "idea2_weight_sweep"
ALIGN_RE = re.compile(r"observed alignments K\[S,S\] vs gold:\s*(\{.*\})")
MW_RE = re.compile(r"_mw([0-9.]+)_")
ROUND_RE = re.compile(r"round(\d+)")
HIGHLIGHT = ["model", "rpe1", "ops_HeLa_DMEM", "ops_HeLa_HPLM"]
N_INIT = 100
N_QUERY = 100


def n_labeled_at_query(round_id: int) -> int:
    return N_INIT + (int(round_id) - 1) * N_QUERY


def rows_from_mapping(alignments: dict, *, model_weight, run, round_id, source) -> list[dict]:
    priors = [v for k, v in alignments.items() if k != "model"]
    mean_prior = float(np.mean(priors)) if priors else float("nan")
    a_model = float(alignments.get("model", float("nan")))
    rows = []
    for kernel, value in alignments.items():
        rows.append({
            "source": source,
            "run": int(run) if run is not None else 1,
            "model_weight": None if model_weight is None else float(model_weight),
            "round": int(round_id),
            "n_labeled": n_labeled_at_query(round_id),
            "kernel": kernel,
            "alignment": float(value),
            "mean_prior_alignment": mean_prior,
            "model_over_prior": a_model / mean_prior if mean_prior else float("nan"),
        })
    return rows


def load_json_states() -> list[dict]:
    rows = []
    if not JSON_DIR.exists():
        return rows
    for path in sorted(JSON_DIR.glob("*_round_state.json")):
        data = json.loads(path.read_text())
        alignments = data.get("alignments") or {}
        if not alignments:
            continue
        m = MW_RE.search(path.name)
        r = ROUND_RE.search(path.name)
        rows.extend(rows_from_mapping(
            alignments,
            model_weight=float(m.group(1)) if m else data.get("model_weight"),
            run=data.get("run_id", 1),
            round_id=int(r.group(1) if r else data.get("round", 1)),
            source=path.name,
        ))
    return rows


def load_logs() -> list[dict]:
    rows = []
    if not LOG_DIR.exists():
        return rows
    for path in sorted(LOG_DIR.glob("mw*_run*.log")):
        m = re.search(r"mw([0-9.]+)_run(\d+)", path.name)
        if not m:
            continue
        mw, run = float(m.group(1)), int(m.group(2))
        text = path.read_text(errors="replace")
        hits = ALIGN_RE.findall(text)
        for i, blob in enumerate(hits, start=1):
            alignments = ast.literal_eval(blob)
            rows.extend(rows_from_mapping(
                alignments,
                model_weight=mw,
                run=run,
                round_id=i,
                source=path.name,
            ))
    return rows


def pick_rows(json_rows: list[dict], log_rows: list[dict]) -> pd.DataFrame:
    # Prefer JSON (structured); fill missing (weight, round, kernel) from logs.
    frames = []
    if json_rows:
        frames.append(pd.DataFrame(json_rows).assign(from_json=True))
    if log_rows:
        frames.append(pd.DataFrame(log_rows).assign(from_json=False))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(
        subset=["model_weight", "run", "round", "kernel"], keep="first"
    )
    return df.sort_values(["model_weight", "run", "round", "kernel"])


def plot_alignment(df: pd.DataFrame, out: Path) -> None:
    mean = (
        df.groupby(["n_labeled", "kernel"])["alignment"]
        .agg(["mean", "std"])
        .reset_index()
    )
    kernels = list(dict.fromkeys(
        HIGHLIGHT + [k for k in sorted(df["kernel"].unique()) if k not in HIGHLIGHT]
    ))
    colors = {
        "model": "#111111",
        "rpe1": "#d62728",
        "ops_HeLa_DMEM": "#1f77b4",
        "ops_HeLa_HPLM": "#2ca02c",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for ax, title, kernel_list in (
        (axes[0], "All kernels (model included)", kernels),
        (axes[1], "Priors only", [k for k in kernels if k != "model"]),
    ):
        for kernel in kernel_list:
            sub = mean[mean["kernel"] == kernel]
            if sub.empty:
                continue
            lw = 2.4 if kernel in HIGHLIGHT else 1.1
            alpha = 1.0 if kernel in HIGHLIGHT else 0.55
            color = colors.get(kernel)
            ax.plot(
                sub["n_labeled"], sub["mean"],
                label=kernel, linewidth=lw, alpha=alpha, color=color,
                marker="o", markersize=4,
            )
            if kernel in HIGHLIGHT and sub["std"].notna().any():
                ax.fill_between(
                    sub["n_labeled"],
                    sub["mean"] - sub["std"].fillna(0),
                    sub["mean"] + sub["std"].fillna(0),
                    color=color or "C0",
                    alpha=0.12,
                    linewidth=0,
                )
        ax.set_xlabel("n labeled (at query)")
        ax.set_title(title)
        ax.set_xticks([100, 200, 300, 400, 500])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False, ncol=2)
    axes[0].set_ylabel("kernel alignment vs gold on S")
    fig.suptitle(
        "Phase 0 weight sweep, seed 1 — alignment of K[S,S] vs observed gold\n"
        "Model kernel is fit on S; high model alignment is expected and optimistic",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    out = ensure_analysis_dir("idea2_alignment")
    df = pick_rows(load_json_states(), load_logs())
    if df.empty:
        print("No alignment records. Need sweep logs or results/round_states/*.json")
        return
    csv_path = out / "roundwise_alignment.csv"
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby(["n_labeled", "kernel"], as_index=False)["alignment"]
        .mean()
        .pivot(index="n_labeled", columns="kernel", values="alignment")
        .sort_index()
    )
    summary.to_csv(out / "mean_alignment_by_round.csv")

    ratio_rows = []
    for n_lab, sub in df.groupby("n_labeled"):
        model = sub.loc[sub["kernel"] == "model", "alignment"].mean()
        prior = sub.loc[sub["kernel"] != "model", "alignment"].mean()
        ratio_rows.append({
            "n_labeled": int(n_lab),
            "mean_model_alignment": float(model),
            "mean_prior_alignment": float(prior),
            "r_t": float(model / prior) if prior else float("nan"),
        })
    ratio_df = pd.DataFrame(ratio_rows).sort_values("n_labeled")
    ratio_df.to_csv(out / "alignment_ratio.csv", index=False)

    png = out / "roundwise_alignment.png"
    plot_alignment(df, png)

    print(summary.round(3).to_string())
    print()
    print(ratio_df.round(3).to_string(index=False))
    print()
    print(
        "Caveat: model alignment is K[S,S] after training on S. "
        "Hypothesis 'early priors > model' would need model < RPE1/OPS at n=100."
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
