#!/usr/bin/env python3
"""Audit alignment integrate_mode for leakage and alignment-definition sensitivity."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduce_repo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_utils import (
    ESSENTIAL_PRIORS,
    ensure_analysis_dir,
    kernel_alignment_uncentered,
    kernel_cka,
    load_prior_kernel,
    save_json,
)
from bmdal_reg.bmdal.algorithms import kernel_alignment

REPO = Path(__file__).resolve().parents[1] / "reproduce_repo"


def audit_code_paths() -> dict:
    kal_src = inspect.getsource(kernel_alignment)
    alg_src = (REPO / "bmdal_reg/bmdal/algorithms.py").read_text()
    kbal_src = (REPO / "query_strategies/kernel_based_active_learning.py").read_text()

    findings = []

    # 1) train_gold slicing for non-learn modes
    if "train_gold = self.train_gold[reindex_train][:, reindex_train]" in kbal_src:
        findings.append({
            "check": "train_gold_labeled_only",
            "status": "PASS",
            "detail": "Non-learn modes slice train_gold to labeled perturbations only before fusion.",
        })
    else:
        findings.append({"check": "train_gold_labeled_only", "status": "FAIL", "detail": "Expected labeled-only train_gold slice not found."})

    # 2) alignment uses labeled block of kernels
    if "kernel_sub = [k[-len(self.features['train']):,-len(self.features['train']):]" in alg_src:
        findings.append({
            "check": "alignment_kernel_sub_labeled_block",
            "status": "PASS",
            "detail": "Alignment compares kernels on the labeled block only.",
        })

    # 3) learn mode uses full train_gold at inference - potential leak
    if "infer_dataset = CustomDataset([i for i in kernel_all], self.train_gold)" in alg_src:
        findings.append({
            "check": "learn_mode_full_train_gold_inference",
            "status": "WARN",
            "detail": "integrate_mode=learn uses full precomputed train_gold during inference weight prediction.",
        })

    # 4) alignment formula
    if "np.trace(K1 @ K2)" in kal_src:
        findings.append({
            "check": "alignment_formula",
            "status": "WARN",
            "detail": "Uses uncentered Frobenius cosine (trace / fro norms), diagonal-dominated for similarity kernels.",
        })

    return {"findings": findings, "kernel_alignment_source": kal_src.strip()}


def simulate_alignment_rankings(n_labeled: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    _, truth = load_prior_kernel("ground_truth_delta")
    n = truth.shape[0]
    labeled_idx = rng.choice(n, size=n_labeled, replace=False)

    obs = truth[np.ix_(labeled_idx, labeled_idx)]

    rows = []
    for prior_name in ESSENTIAL_PRIORS + ["ground_truth_delta"]:
        _, K = load_prior_kernel(prior_name)
        K_lab = K[np.ix_(labeled_idx, labeled_idx)]
        rows.append({
            "prior": prior_name.replace("_kernel", ""),
            "uncentered": kernel_alignment_uncentered(K_lab, obs),
            "centered_cka": kernel_cka(K_lab, obs, zero_diagonal=False),
            "centered_cka_zero_diag": kernel_cka(K_lab, obs, zero_diagonal=True),
            "code_kernel_alignment": float(kernel_alignment(K_lab, obs)),
        })

    df = pd.DataFrame(rows)
    for col in ["uncentered", "centered_cka", "centered_cka_zero_diag", "code_kernel_alignment"]:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="average")
    return df


def rank_agreement(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["uncentered", "centered_cka", "centered_cka_zero_diag", "code_kernel_alignment"]
    from scipy.stats import spearmanr

    rows = []
    for i, a in enumerate(metrics):
        for b in metrics[i + 1 :]:
            rho = spearmanr(df[a], df[b]).correlation
            rows.append({"metric_a": a, "metric_b": b, "spearman_rank_corr": float(rho)})
    return pd.DataFrame(rows)


def main() -> None:
    outdir = ensure_analysis_dir("e0_baseline")
    code_audit = audit_code_paths()
    save_json(outdir / "alignment_code_audit.json", code_audit)

    sim_rows = []
    for n_lab in [10, 20, 50, 100]:
        df = simulate_alignment_rankings(n_labeled=n_lab, seed=1)
        df["n_labeled_sim"] = n_lab
        sim_rows.append(df)
    sim = pd.concat(sim_rows, ignore_index=True)
    sim.to_csv(outdir / "alignment_metric_comparison.csv", index=False)

    agreement = rank_agreement(sim[sim["n_labeled_sim"] == 100])
    agreement.to_csv(outdir / "alignment_rank_agreement.csv", index=False)

    print("Alignment audit saved to", outdir)
    print("\nCode findings:")
    for f in code_audit["findings"]:
        print(f"  [{f['status']}] {f['check']}: {f['detail']}")

    print("\nPrior ranking under n_labeled=100 (centered CKA vs code alignment):")
    sub = sim[sim["n_labeled_sim"] == 100].sort_values("centered_cka", ascending=False)
    print(sub[["prior", "centered_cka", "code_kernel_alignment", "rank_centered_cka", "rank_code_kernel_alignment"]].to_string(index=False))


if __name__ == "__main__":
    main()
