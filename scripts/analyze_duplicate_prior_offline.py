#!/usr/bin/env python3
"""Zero-GPU duplicate-prior check: fused kernel and top-100 selection shift."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduce_repo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_utils import ESSENTIAL_PRIORS, ensure_analysis_dir, load_prior_kernel
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduce_repo" / "bmdal_reg"))
from bmdal.algorithms import normalize_kernel

DATA_ROOT = Path("/data/zy/iterpert")
ROUND_STATE_DIR = DATA_ROOT / "round_states"


def get_index(A, B):
    index_dict_A = {val: i for i, val in enumerate(A)}
    return np.array([index_dict_A[b] for b in B])


def fuse_mean_new_max(kernel_all: list[np.ndarray]) -> np.ndarray:
    weights = [1 / len(kernel_all)] * len(kernel_all)
    k_agg = np.zeros_like(kernel_all[0])
    for K, w in zip(kernel_all, weights):
        k_agg += w * K
    return np.maximum.reduce([k_agg] + kernel_all)


def maxdist_topn(pool_block: np.ndarray, train_block: np.ndarray, n: int = 100) -> np.ndarray:
    """Greedy max-min on fused kernel without torch."""
    n_pool = pool_block.shape[0]
    n_train = train_block.shape[0]
    full = np.block([[train_block, pool_block[:, :n_pool]], [pool_block[:n_pool, :].T, pool_block]])
    # full is [train+pool, train+pool] but pool_block is pool x (train+pool) reindex - simplify:
    # Use square [pool+train] matrix
    K = np.block([[train_block, pool_block[:, :n_train]], [pool_block[:n_train, :].T, pool_block[:n_pool, :n_pool]]])
    # Actually pool_block from reindex is (pool+train) x (pool+train)
    K = pool_block
    n_total = K.shape[0]
    n_pool_only = n_total - n_train

    selected = list(range(n_train, n_total))  # start with train indices treated as anchors
    min_d = np.full(n_pool_only, np.inf)
    chosen = []

    for _ in range(min(n, n_pool_only)):
        # pool indices 0..n_pool_only-1 correspond to rows n_train..n_total-1
        scores = min_d.copy()
        best = int(np.argmax(scores))
        chosen.append(best)
        idx = n_train + best
        diag = np.diag(K)
        for p in range(n_pool_only):
            j = n_train + p
            if p in chosen:
                continue
            sq = diag[idx] + diag[j] - 2 * K[idx, j]
            min_d[p] = min(min_d[p], sq)
        min_d[best] = -np.inf
    return np.array(chosen, dtype=int)


def greedy_top100(K: np.ndarray, n_train: int, batch: int = 100) -> np.ndarray:
    n = K.shape[0]
    pool_idx = np.arange(n_train, n)
    min_sq = np.full(len(pool_idx), np.inf)
    selected = []
    diag = np.diag(K)

    # initialize with train anchors
    for t in range(n_train):
        j = t
        for pi, p in enumerate(pool_idx):
            sq = diag[j] + diag[p] - 2 * K[j, p]
            min_sq[pi] = min(min_sq[pi], sq)

    for _ in range(batch):
        pi = int(np.argmax(min_sq))
        p = pool_idx[pi]
        selected.append(int(p - n_train))
        min_sq[pi] = -np.inf
        for pj, pp in enumerate(pool_idx):
            if min_sq[pj] <= -np.inf:
                continue
            sq = diag[p] + diag[pp] - 2 * K[p, pp]
            min_sq[pj] = min(min_sq[pj], sq)
    return np.array(selected)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def load_round_state(run: int, round_id: int) -> dict | None:
    path = ROUND_STATE_DIR / f"run{run}_round{round_id}_state.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def build_kernels_from_state(state: dict, duplicate_prior: str, n_copies: int) -> tuple[list[np.ndarray], list[str], int]:
    pert_list = state["pert_list"]
    pool_genes = state["pool_genes"]
    train_genes = state["train_genes"]
    reindex = get_index(pert_list, pool_genes + train_genes)
    reindex_train = get_index(pert_list, train_genes)
    n_train = len(train_genes)

    kernel_all = []
    names = []
    for prior_name in ESSENTIAL_PRIORS:
        _, K = load_prior_kernel(prior_name)
        K = normalize_kernel(K[np.ix_(reindex, reindex)], mode="diag")
        copies = n_copies if prior_name == duplicate_prior else 1
        for _ in range(copies):
            kernel_all.append(K)
            names.append(prior_name.replace("_kernel", ""))

    base_k = normalize_kernel(state["base_kernel"], mode="diag")
    kernel_all.append(base_k)
    names.append("model")
    return kernel_all, names, n_train


def offline_from_saved_state(state_path: Path, duplicate_prior: str, copies_list: list[int]) -> pd.DataFrame:
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    rows = []
    baseline_sel = None
    for n_copies in copies_list:
        kernel_all, names, n_train = build_kernels_from_state(state, duplicate_prior, n_copies)
        k_fused = fuse_mean_new_max(kernel_all)
        sel = greedy_top100(k_fused, n_train, batch=100)
        eff_w = {}
        counts = {}
        for nm in names:
            counts[nm] = counts.get(nm, 0) + 1
        for nm, c in counts.items():
            eff_w[nm] = c / len(names)
        row = {
            "n_copies": n_copies,
            "duplicate_prior": duplicate_prior,
            "fused_frobenius_norm": float(np.linalg.norm(k_fused, "fro")),
            "top100_jaccard_vs_1copy": float("nan"),
            "effective_weight_duplicated": eff_w.get(duplicate_prior.replace("_kernel", ""), 0.0),
            "top100_preview": sel[:10].tolist(),
        }
        if n_copies == 1:
            baseline_sel = sel
            row["top100_jaccard_vs_1copy"] = 1.0
        else:
            row["top100_jaccard_vs_1copy"] = jaccard(sel, baseline_sel)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="", help="Path to round state pkl")
    parser.add_argument("--duplicate-prior", type=str, default="rpe1_kernel")
    parser.add_argument("--copies", type=str, default="1,2,4,8")
    args = parser.parse_args()

    outdir = ensure_analysis_dir("e2_duplicate_prior")
    copies_list = [int(x) for x in args.copies.split(",")]

    if args.state:
        df = offline_from_saved_state(Path(args.state), args.duplicate_prior, copies_list)
        df.to_csv(outdir / "duplicate_prior_offline_check.csv", index=False)
        print(df.to_string(index=False))
        return

    print("No round state provided.")
    print("After the first logged pilot run, pass e.g.:")
    print("  --state /data/zy/iterpert/round_states/run1_round1_state.pkl")
    print("\nWriting template config only.")
    template = {
        "duplicate_prior": args.duplicate_prior,
        "copies_list": copies_list,
        "checks": ["fused_kernel_diff", "top100_jaccard", "effective_weight"],
        "go_signal": "top100_jaccard < 0.8 or nALC shift in full runs",
    }
    (outdir / "duplicate_prior_offline_template.json").write_text(json.dumps(template, indent=2))


if __name__ == "__main__":
    main()
