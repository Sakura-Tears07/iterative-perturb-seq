#!/usr/bin/env python3
"""Idea 3c — choose Observed-KA τ on the frozen calibration split only.

    python scripts/analyze_hard_reject_threshold.py

Does not open GPU. Does not retune on validation/holdout. Writes frozen_tau.json.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import REPO_ROOT, ensure_analysis_dir

PROTOCOL_SRC = REPO_ROOT / "configs" / "experiments" / "idea3_intervention" / "PROTOCOL.md"
DET_LONG = REPO_ROOT / "results" / "analysis" / "idea3_detection" / "detector_long.csv"

CAL_PERMS = (20260826, 20260827, 20260828)
VAL_PERM = 20260829
HOLDOUT_PERM = 20260830
GPU_TEST_PERM = 20260831
SIGNAL = "obs"
FPR_PRIMARY = 0.10
FPR_FALLBACK = 0.20


def rates(df: pd.DataFrame, tau: float) -> dict[str, float]:
    clean = df[df["lambda"] == 0.0]["z"].dropna()
    bad = df[df["lambda"] == 1.0]["z"].dropna()
    mid = df[df["lambda"] == 0.5]["z"].dropna()
    return {
        "n_clean": int(len(clean)),
        "n_bad": int(len(bad)),
        "n_mid": int(len(mid)),
        "FPR_clean": float((clean < tau).mean()) if len(clean) else float("nan"),
        "TPR_bad": float((bad < tau).mean()) if len(bad) else float("nan"),
        "reject_mid": float((mid < tau).mean()) if len(mid) else float("nan"),
        "keep_clean": float((clean >= tau).mean()) if len(clean) else float("nan"),
    }


def pick_tau(cal: pd.DataFrame) -> tuple[float, float, str]:
    zs = np.sort(cal["z"].dropna().unique())
    grid = np.unique(np.concatenate([
        zs,
        np.linspace(float(zs.min()), float(zs.max()), 401),
    ]))
    for budget, name in ((FPR_PRIMARY, "FPR<=0.10"), (FPR_FALLBACK, "FPR<=0.20 fallback")):
        feasible = []
        for tau in grid:
            r = rates(cal, float(tau))
            if r["FPR_clean"] <= budget + 1e-12:
                feasible.append((float(tau), r["TPR_bad"], r["FPR_clean"]))
        if feasible:
            # most sensitive inside budget = largest τ
            tau_star, tpr, fpr = max(feasible, key=lambda t: (t[0], t[1]))
            return tau_star, budget, name
    return float("nan"), float("nan"), "infeasible"


def split_obs() -> pd.DataFrame:
    df = pd.read_csv(DET_LONG)
    df = df[df["signal"] == SIGNAL].copy()
    if df.empty:
        raise FileNotFoundError(f"No obs rows in {DET_LONG}")
    return df


def write_plots(cal: pd.DataFrame, tau: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    colors = {0.0: "#1f77b4", 0.5: "#ff7f0e", 1.0: "#d62728"}
    labels = {0.0: "λ=0 clean", 0.5: "λ=0.5 (not used for τ)", 1.0: "λ=1 bad"}
    for lam in (0.0, 0.5, 1.0):
        ys = cal.loc[cal["lambda"] == lam, "z"].dropna()
        ax.hist(ys, bins=18, alpha=0.45, color=colors[lam], label=labels[lam], density=True)
    if np.isfinite(tau):
        ax.axvline(tau, color="k", ls="--", lw=1.4, label=f"τ*={tau:.3f} (cal only)")
    ax.set_xlabel("Observed KA  z")
    ax.set_ylabel("density")
    ax.set_title("Calibration perms only — do not retune on val/holdout/GPU")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "obs_ka_cal_hist.png", dpi=160)
    plt.close(fig)


def main() -> None:
    out = ensure_analysis_dir("idea3_intervention")
    shutil.copyfile(PROTOCOL_SRC, out / "PROTOCOL.md")
    df = split_obs()
    cal = df[df["perm_seed"].isin(CAL_PERMS)]
    val = df[df["perm_seed"] == VAL_PERM]
    hold = df[df["perm_seed"] == HOLDOUT_PERM]

    tau, budget, rule = pick_tau(cal)
    cal_r = rates(cal, tau) if np.isfinite(tau) else {}
    val_r = rates(val, tau) if np.isfinite(tau) else {}
    hold_r = rates(hold, tau) if np.isfinite(tau) else {}

    frozen = {
        "signal": SIGNAL,
        "action": "drop RPE1 only",
        "detect_from_round": 1,
        "tau": None if not np.isfinite(tau) else float(tau),
        "fpr_budget": None if not np.isfinite(budget) else float(budget),
        "rule": rule,
        "cal_perms": list(CAL_PERMS),
        "val_perm": VAL_PERM,
        "offline_holdout_perm": HOLDOUT_PERM,
        "gpu_test_perm": GPU_TEST_PERM,
        "calibration": cal_r,
        "validation_do_not_retune": val_r,
        "offline_holdout_do_not_retune": hold_r,
        "lambda_0.5_not_used_for_tau": True,
        "launch_gpu": bool(np.isfinite(tau)),
    }
    (out / "frozen_tau.json").write_text(json.dumps(frozen, indent=2) + "\n")

    lines = [
        PROTOCOL_SRC.read_text().strip(),
        "",
        "--- live τ (calibration only) ---",
        "",
        f"rule: {rule}",
        f"τ* = {tau:.6f}" if np.isfinite(tau) else "τ* infeasible — do not launch GPU",
        "",
        "Calibration   " + json.dumps(cal_r),
        "Validation    " + json.dumps(val_r) + "  (do not retune)",
        "Offline hold  " + json.dumps(hold_r) + "  (do not retune)",
        "",
    ]
    if np.isfinite(tau):
        lines.append(f"GPU test perm frozen: {GPU_TEST_PERM}. Launch Detected vs Equal vs Oracle after this file exists.")
    else:
        lines.append("I-Gate not opened: no conservative τ on calibration.")
    (out / "THRESHOLD.md").write_text("\n".join(lines) + "\n")

    if np.isfinite(tau):
        write_plots(cal, tau, out)

    print(json.dumps(frozen, indent=2))
    print(f"Wrote {out / 'frozen_tau.json'}")


if __name__ == "__main__":
    main()
