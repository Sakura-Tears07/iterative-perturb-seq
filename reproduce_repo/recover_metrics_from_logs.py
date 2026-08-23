#!/usr/bin/env python3
"""Recover *_metrics.csv from run logs when result files were lost."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from local_paths import REPO_ROOT, resolve_result_dir
from reorganize_results import classify

LOG_ROOT = Path("/data/zy/iterpert/logs")
ROUND_RE = re.compile(r"Round (\d+) pearson delta: ([0-9.eE+-]+)")
SAVED_RE = re.compile(r"Saved round metrics to (.+?_metrics\.csv)")


class Args:
    """Minimal namespace for resolve_result_dir / classify."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def infer_args_from_log(log_path: Path, wb_exp_name: str) -> Args:
    text = log_path.read_text(errors="replace")
    kwargs = {
        "dataset_name": "replogle_k562_essential_1000hvg",
        "epoch_per_cycle": 20,
        "use_single_prior": "single_" in wb_exp_name,
        "single_prior": "",
        "use_prior": "prior" in wb_exp_name,
        "kernel_strategy": "Core-Set",
        "base_kernel": "diff_effect" if "diff_effect" in wb_exp_name else "cross_gene_out",
        "strategy_name": "kernel_based_active_learning",
    }
    if kwargs["use_single_prior"]:
        m = re.search(r"single_([a-zA-Z0-9_]+)_run", wb_exp_name)
        if m:
            prior = m.group(1)
            kwargs["single_prior"] = prior if prior.endswith("_kernel") else prior + "_kernel"
    if "TypiClust" in wb_exp_name:
        kwargs["strategy_name"] = "TypiClust"
    elif "KMeansSampling" in wb_exp_name:
        kwargs["strategy_name"] = "KMeansSampling"
    elif "Random" in wb_exp_name and "priormean" not in wb_exp_name:
        kwargs["kernel_strategy"] = "Random"
    elif "_BADGE_" in wb_exp_name or wb_exp_name.endswith("_BADGE_cross_gene_out"):
        kwargs["kernel_strategy"] = "BADGE"
    elif "BatchBALD" in wb_exp_name:
        kwargs["kernel_strategy"] = "BatchBALD"
    elif "ACS-FW" in wb_exp_name:
        kwargs["kernel_strategy"] = "ACS-FW"
    elif "LCMD" in wb_exp_name:
        kwargs["kernel_strategy"] = "LCMD"
    elif "BALD" in wb_exp_name and "BatchBALD" not in wb_exp_name:
        kwargs["kernel_strategy"] = "BALD"
    elif "Core-Set" in wb_exp_name and not kwargs["use_single_prior"]:
        if kwargs["use_prior"]:
            kwargs["kernel_strategy"] = "Core-Set"
        else:
            kwargs["kernel_strategy"] = "Core-Set"
    if "Random_cross_gene" in wb_exp_name:
        kwargs["kernel_strategy"] = "Random"
    m = re.search(r"_run(\d+)", wb_exp_name)
    run = int(m.group(1)) if m else int(log_path.stem.replace("run", ""))
    kwargs["run"] = run
    return Args(**kwargs)


def parse_log(log_path: Path) -> tuple[str, list[dict]] | None:
    text = log_path.read_text(errors="replace")
    saved = SAVED_RE.search(text)
    if not saved:
        return None
    csv_name = Path(saved.group(1)).name
    wb_exp_name = csv_name.replace("_metrics.csv", "")

    rows = []
    for round_idx, val in ROUND_RE.findall(text):
        n_labeled = 100 + int(round_idx) * 100
        rows.append(
            {
                "run": infer_args_from_log(log_path, wb_exp_name).run,
                "seed": 1,
                "round": int(round_idx),
                "n_labeled": n_labeled,
                "pearson_delta": float(val),
            }
        )
    if len(rows) < 6:
        return None
    return wb_exp_name, rows


def target_path(wb_exp_name: str, log_path: Path) -> Path:
    args = infer_args_from_log(log_path, wb_exp_name)
    result_dir = Path(resolve_result_dir(args))
    return result_dir / f"{wb_exp_name}_metrics.csv"


def recover_glob(pattern: str) -> int:
    recovered = 0
    for log_path in sorted(LOG_ROOT.glob(pattern)):
        parsed = parse_log(log_path)
        if not parsed:
            continue
        wb_exp_name, rows = parsed
        dest = target_path(wb_exp_name, log_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dest, index=False)
        recovered += 1
        print(f"  {log_path.name} -> {dest.relative_to(REPO_ROOT)}")
    return recovered


def main() -> None:
    total = 0
    print("Recovering Fig.4 Essential 1K...")
    total += recover_glob("fig4/iterpert/run*.log")
    total += recover_glob("fig4/baselines/*/run*.log")
    print("Recovering Fig.4c single prior...")
    total += recover_glob("fig4c/*/run*.log")
    print(f"Recovered {total} metrics files")
    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
