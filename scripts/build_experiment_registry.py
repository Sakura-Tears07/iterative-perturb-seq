#!/usr/bin/env python3
"""Build experiment_registry.csv from existing results manifests and run files."""
from __future__ import annotations

import glob
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import METHODS, RESULTS, find_method_files

REGISTRY = RESULTS / "experiment_registry.csv"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(RESULTS.parents[0]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def infer_config(filename: str) -> dict:
    name = Path(filename).name
    cfg = {
        "init_size": 100 if "100_5_100" in name else None,
        "query_size": 100 if "100_5_100" in name else None,
        "rounds": 5 if "100_5_100" in name else None,
        "seed": None,
        "run_id": None,
    }
    m = re.search(r"run(\d+)", name)
    if m:
        cfg["run_id"] = int(m.group(1))
        cfg["seed"] = 1
    return cfg


def build_registry() -> pd.DataFrame:
    commit = git_commit()
    rows = []
    exp_id = 0
    for method, spec in METHODS.items():
        for metrics_path in find_method_files(spec):
            metrics_path = Path(metrics_path)
            cfg = infer_config(metrics_path.name)
            pkl = metrics_path.with_name(metrics_path.name.replace("_metrics.csv", ".pkl"))
            log_guess = Path("/data/zy/iterpert/logs/fig4") / method.lower().replace("-", "_")
            exp_id += 1
            rows.append({
                "experiment_id": f"fig4_{method.lower()}_{cfg['run_id'] or exp_id}",
                "method": method,
                "base_kernel": "diff_effect" if "diff_effect" in metrics_path.name else "cross_gene_out",
                "prior_list": "all_essential" if method == "IterPert" else "",
                "fusion": "mean_new_max" if method == "IterPert" else "",
                "selection_rule": "Core-Set_prior" if method == "IterPert" else method,
                "init_size": cfg["init_size"],
                "query_size": cfg["query_size"],
                "rounds": cfg["rounds"],
                "seed": cfg["seed"],
                "run_id": cfg["run_id"],
                "status": "complete",
                "metrics_path": str(metrics_path),
                "log_path": str(log_guess / f"run{cfg['run_id']}.log") if cfg["run_id"] else "",
                "selection_pkl": str(pkl) if pkl.exists() else "",
                "commit": commit,
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = build_registry()
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REGISTRY, index=False)
    print(f"Wrote {len(df)} rows to {REGISTRY}")


if __name__ == "__main__":
    main()
