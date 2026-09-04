#!/usr/bin/env python3
"""Verify a Fig.4 Essential 1K run has complete metrics (6 budget points)."""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "fig4" / "essential_1k"

METHOD_MAP = {
    "BADGE": ("baselines/badge", "*BADGE*metrics.csv"),
    "LCMD": ("baselines/lcmd", "*LCMD*metrics.csv"),
    "IterPert": ("iterpert", "*Core-Set_diff_effect_metrics.csv"),
}

EXPECTED = [100, 200, 300, 400, 500, 600]


def find_metrics(method: str, run: int) -> Path | None:
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method}")
    sub, pattern = METHOD_MAP[method]
    root = RESULTS / sub / "runs"
    files = sorted(root.glob(pattern))
    for f in files:
        m = re.search(rf"run{run}[^0-9]", f.name)
        if m:
            return f
    return None


def verify(path: Path) -> tuple[bool, str]:
    df = pd.read_csv(path)
    if "n_labeled" not in df.columns or "pearson_delta" not in df.columns:
        return False, "missing columns"
    budgets = sorted(df["n_labeled"].unique())
    missing = [b for b in EXPECTED if b not in budgets]
    if missing:
        return False, f"missing budgets {missing} (have {budgets})"
    dup = df.duplicated(subset=["round", "n_labeled"]).any()
    if dup:
        return False, "duplicate round/budget rows"
    if len(budgets) != 6:
        return False, f"expected 6 budgets, got {len(budgets)}"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--run", type=int, required=True)
    args = parser.parse_args()

    path = find_metrics(args.method, args.run)
    if path is None:
        print(f"FAIL: no metrics file for {args.method} run={args.run}", file=sys.stderr)
        sys.exit(1)

    ok, msg = verify(path)
    if ok:
        print(f"OK: {path.name} ({msg})")
        sys.exit(0)
    print(f"FAIL: {path.name} — {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
