#!/usr/bin/env python3
"""Idea 2: summarize model-weight sweep metrics (best w_model by round)."""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import RESULTS, ensure_analysis_dir
MW_RE = re.compile(r"_mw([0-9.]+)_")


def main() -> None:
    OUT = ensure_analysis_dir("idea2_weight_sweep")
    files = sorted(glob.glob(str(RESULTS / "**" / "*_mw*_metrics.csv"), recursive=True))
    if not files:
        print("No *_mw*_metrics.csv found. Run configs/experiments/idea2_weight_sweep/run_sweep.sh")
        return
    frames = []
    for path in files:
        m = MW_RE.search(Path(path).name)
        if not m:
            continue
        df = pd.read_csv(path)
        df["model_weight"] = float(m.group(1))
        df["source"] = Path(path).name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT / "all_runs.csv", index=False)

    by_round = (
        all_df.groupby(["n_labeled", "model_weight"], as_index=False)["pearson_delta"]
        .mean()
        .sort_values(["n_labeled", "model_weight"])
    )
    by_round.to_csv(OUT / "mean_by_round_weight.csv", index=False)

    winners = []
    for n_lab, sub in by_round.groupby("n_labeled"):
        best = sub.loc[sub["pearson_delta"].idxmax()]
        winners.append({
            "n_labeled": int(n_lab),
            "best_model_weight": float(best["model_weight"]),
            "best_pearson": float(best["pearson_delta"]),
        })
    win_df = pd.DataFrame(winners)
    win_df.to_csv(OUT / "best_weight_by_round.csv", index=False)
    print(win_df.to_string(index=False))
    print()
    print("n=100 is init evaluation (no acquisition). Paper equal-mean is w_model ≈ 1/9.")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
