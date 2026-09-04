#!/usr/bin/env python3
"""Summarize the 5-seed schedule experiment (works with a partial matrix)."""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import RESULTS, ensure_analysis_dir, trapezoid_nalc

TAG_SPECS = {
    "paper": {
        "glob": str(RESULTS / "fig4" / "essential_1k" / "iterpert" / "runs" / "*priormean_new_max*_metrics.csv"),
        "ok": lambda p: (
            "_mw" not in p and "_sched_" not in p and "single_" not in p and "dup" not in p
        ),
    },
    "fixed_0.5": {
        "glob": str(RESULTS / "**" / "*_mw0.5_*_metrics.csv"),
        "ok": lambda p: True,
    },
    "fixed_0.75": {
        "glob": str(RESULTS / "**" / "*_mw0.75_*_metrics.csv"),
        "ok": lambda p: True,
    },
    "early_prior": {
        "glob": str(RESULTS / "**" / "*_sched_early_prior_*_metrics.csv"),
        "ok": lambda p: True,
    },
    "early_model": {
        "glob": str(RESULTS / "**" / "*_sched_early_model_*_metrics.csv"),
        "ok": lambda p: True,
    },
}
RUN_RE = re.compile(r"_run(\d+)_")


def load_tag(tag: str) -> pd.DataFrame:
    spec = TAG_SPECS[tag]
    files = [p for p in glob.glob(spec["glob"], recursive=True) if spec["ok"](p)]
    frames = []
    for path in sorted(files):
        m = RUN_RE.search(Path(path).name)
        df = pd.read_csv(path)
        df["method"] = tag
        df["source"] = Path(path).name
        if "run" not in df.columns and m:
            df["run"] = int(m.group(1))
        if tag == "paper":
            df = df[df["run"].isin({1, 2, 3, 4, 5})]
            if df.empty:
                continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    out = ensure_analysis_dir("idea2_schedule")
    frames = []
    for tag in TAG_SPECS:
        df = load_tag(tag)
        n_runs = df["run"].nunique() if not df.empty and "run" in df.columns else 0
        print(f"{tag:12s}  files/runs={n_runs}")
        if not df.empty:
            frames.append(df)
    if not frames:
        print("No schedule metrics yet.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "all_runs.csv", index=False)

    mean = (
        all_df.groupby(["method", "n_labeled"])["pearson_delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["method", "n_labeled"])
    )
    mean.to_csv(out / "mean_by_round.csv", index=False)

    nalc_rows = []
    for (method, run), sub in all_df.groupby(["method", "run"]):
        sub = sub.sort_values("n_labeled")
        nalc_rows.append({
            "method": method,
            "run": int(run),
            "nALC": trapezoid_nalc(sub["n_labeled"], sub["pearson_delta"]),
            "pearson_600": float(sub.loc[sub["n_labeled"].idxmax(), "pearson_delta"]),
        })
    nalc = pd.DataFrame(nalc_rows)
    nalc.to_csv(out / "per_run_nalc.csv", index=False)
    nalc.groupby("method", as_index=False)[["nALC", "pearson_600"]].mean().to_csv(
        out / "mean_nalc.csv", index=False
    )
    print()
    print(mean.to_string(index=False))
    print()
    print(nalc.groupby("method")[["nALC", "pearson_600"]].agg(["mean", "count"]).to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
