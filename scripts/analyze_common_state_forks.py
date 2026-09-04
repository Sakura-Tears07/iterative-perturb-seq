#!/usr/bin/env python3
"""Common-state fork analysis protocol (frozen before looking at results).

Do NOT pick a "best weight" first. Primary evidence is:

1. Nine ΔP(w | S) curves — look for shape that repeats across runs.
2. Contrast C(S) = mean(ΔP(.75), ΔP(1)) − mean(ΔP(0), ΔP(.25)).
3. Spearman(w, ΔP) as an auxiliary (misses mid-weight optima).
4. Pairwise Jaccard of selected gene sets (did w actually change the action?).

Official P_before comes from dump state_metrics.json, not from a re-eval
after the branch loads the checkpoint.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import DATA_ROOT, RESULTS, ensure_analysis_dir

STATE_ROOT = DATA_ROOT / "common_states"
LOG_DIR = DATA_ROOT / "logs" / "idea2_fork"
RUNS = (1, 2, 3)
NS = (100, 300, 500)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
STAGE = {100: "early", 300: "middle", 500: "late"}
JACCARD_PAIRS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (0.0, 1.0))
FORK_RE = re.compile(r"_mw([0-9.]+)_fork_n(\d+)_run(\d+)_")


def canon_gene(name: str) -> str:
    g = str(name).strip()
    if g.endswith("+ctrl"):
        g = g[:-5]
    return g


def wkey(w: float) -> float:
    return round(float(w), 4)


def parse_name(name: str):
    m = FORK_RE.search(name)
    if not m:
        return None
    return int(m.group(3)), int(m.group(2)), wkey(m.group(1))


def load_labeled(run: int, n: int) -> set[str]:
    path = STATE_ROOT / f"paper_run{run}" / f"n{n}" / "labeled_genes.txt"
    if not path.is_file():
        return set()
    return {canon_gene(line) for line in path.read_text().splitlines() if line.strip()}


def dump_p_before(run: int, n: int) -> float | None:
    path = STATE_ROOT / f"paper_run{run}" / f"n{n}" / "state_metrics.json"
    if not path.is_file():
        return None
    return float(json.loads(path.read_text())["pearson_before"])


def dump_checkpoint(run: int, n: int) -> str:
    return str(STATE_ROOT / f"paper_run{run}" / f"n{n}" / "gears_checkpoint")


def load_selected_from_pkl(pkl_path: Path) -> list[str]:
    if not pkl_path.is_file():
        return []
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    genes = obj.get(1, []) if isinstance(obj, dict) else []
    if hasattr(genes, "tolist"):
        genes = genes.tolist()
    return [canon_gene(g) for g in genes]


def load_selected_from_csv(csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    col = "selected_gene_name" if "selected_gene_name" in df.columns else None
    if col is None:
        return []
    return [canon_gene(g) for g in df[col].tolist()]


def empty_grid() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        for n in NS:
            p0 = dump_p_before(run, n)
            for w in WEIGHTS:
                rows.append({
                    "run": run,
                    "state_n": n,
                    "stage": STAGE[n],
                    "w": w,
                    "P_before": p0,
                    "P_after": np.nan,
                    "delta_pearson": np.nan,
                    "P_before_eval_after_load": np.nan,
                    "n_selected": np.nan,
                    "n_overlap_labeled": np.nan,
                    "training_seed": 1,
                    "training_run": run,
                    "checkpoint_path": dump_checkpoint(run, n),
                    "source": "",
                    "complete": False,
                })
    return pd.DataFrame(rows)


def attach_results(grid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = grid.copy()
    selected_map: dict[tuple[int, int, float], list[str]] = {}
    files = sorted(glob.glob(str(RESULTS / "idea2" / "forks" / "runs" / "*_fork*_metrics.csv")))
    by_key = {(int(r.run), int(r.state_n), wkey(r.w)): i for i, r in df.iterrows()}
    sel_dir = RESULTS / "selection_logs"

    for path in files:
        parsed = parse_name(Path(path).name)
        if not parsed:
            print("skip unparsed", Path(path).name)
            continue
        run, n, w = parsed
        key = (run, n, w)
        if key not in by_key:
            print("skip unexpected", key, Path(path).name)
            continue
        i = by_key[key]
        metrics = pd.read_csv(path)
        after = metrics.loc[metrics["round"] == 1, "pearson_delta"]
        if after.empty:
            continue
        p1 = float(after.iloc[0])
        p0 = dump_p_before(run, n)
        eval0 = metrics.loc[metrics["round"] == 0, "pearson_delta_eval_after_load"]
        p_eval = float(eval0.iloc[0]) if not eval0.empty and pd.notna(eval0.iloc[0]) else np.nan

        stem = Path(path).name.replace("_metrics.csv", "")
        record_path = Path(path).with_name(stem + "_fork_record.json")
        pkl_path = Path(path).with_name(stem + ".pkl")
        selected: list[str] = []
        if record_path.is_file():
            rec = json.loads(record_path.read_text())
            selected = [canon_gene(g) for g in rec.get("selected_genes", [])]
            if rec.get("P_before") is not None:
                p0 = float(rec["P_before"])
            if rec.get("P_after") is not None:
                p1 = float(rec["P_after"])
            if rec.get("P_before_eval_after_load") is not None:
                p_eval = rec["P_before_eval_after_load"]
        if not selected:
            selected = load_selected_from_pkl(pkl_path)
        if not selected:
            selected = load_selected_from_csv(sel_dir / f"{stem}_round1_selected_only.csv")

        labeled = load_labeled(run, n)
        overlap = len(set(selected) & labeled) if selected else np.nan
        df.at[i, "P_before"] = p0
        df.at[i, "P_after"] = p1
        df.at[i, "delta_pearson"] = (np.nan if p0 is None else p1 - float(p0))
        df.at[i, "P_before_eval_after_load"] = p_eval
        df.at[i, "n_selected"] = len(selected) if selected else np.nan
        df.at[i, "n_overlap_labeled"] = overlap
        df.at[i, "source"] = Path(path).name
        df.at[i, "complete"] = True
        selected_map[key] = selected
    return df, selected_map


def contrast_c(deltas: dict[float, float]) -> float:
    hi = 0.5 * (deltas[0.75] + deltas[1.0])
    lo = 0.5 * (deltas[0.0] + deltas[0.25])
    return hi - lo


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def sanity_for_state(df: pd.DataFrame, run: int, n: int) -> dict:
    sub = df[(df["run"] == run) & (df["state_n"] == n)].copy()
    issues = []
    if len(sub) != 5:
        issues.append(f"expected 5 weights, got {len(sub)}")
    complete = bool(sub["complete"].all()) if len(sub) else False
    p_befores = [float(x) for x in sub["P_before"] if pd.notna(x)]
    p_evals = [float(x) for x in sub["P_before_eval_after_load"] if pd.notna(x)]
    ckpts = set(sub["checkpoint_path"].astype(str))
    n_sel = sub["n_selected"].tolist()
    overlap = sub["n_overlap_labeled"].tolist()

    p_before_ok = len(set(np.round(p_befores, 12))) <= 1 and len(p_befores) == 5
    if not p_before_ok:
        issues.append(f"P_before not identical: {p_befores}")
    eval_ok = True
    if len(p_evals) == 5:
        eval_ok = (max(p_evals) - min(p_evals)) < 1e-6
        if not eval_ok:
            issues.append(f"eval_after_load not identical: {p_evals}")
        if p_befores and abs(p_evals[0] - p_befores[0]) > 1e-4:
            issues.append(
                f"eval_after_load {p_evals[0]} != dump P_before {p_befores[0]}"
            )
    elif complete:
        issues.append("missing eval_after_load on some branches")
        eval_ok = False
    n_sel_ok = all(pd.notna(x) and int(x) == 100 for x in n_sel) if complete else False
    if complete and not n_sel_ok:
        issues.append(f"n_selected != 100: {n_sel}")
    overlap_ok = all(pd.notna(x) and int(x) == 0 for x in overlap) if complete else False
    if complete and not overlap_ok:
        issues.append(f"selected ∩ labeled_before != ∅: {overlap}")
    ckpt_ok = len(ckpts) == 1
    if not ckpt_ok:
        issues.append(f"checkpoint_path not unique: {ckpts}")
    return {
        "run": run,
        "state_n": n,
        "complete": complete,
        "p_before_identical": p_before_ok,
        "eval_after_load_identical": eval_ok,
        "n_selected_100": n_sel_ok,
        "no_overlap_labeled": overlap_ok,
        "same_checkpoint": ckpt_ok,
        "ok": complete and p_before_ok and eval_ok and n_sel_ok and overlap_ok and ckpt_ok,
        "issues": "; ".join(issues),
        "P_before": p_befores[0] if p_befores else np.nan,
    }


def write_response_curves(df: pd.DataFrame, out: Path) -> None:
    done = df[df["complete"]].copy()
    if done.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.5), sharex=True, sharey=True)
    for i, run in enumerate(RUNS):
        for j, n in enumerate(NS):
            ax = axes[j, i]
            sub = done[(done["run"] == run) & (done["state_n"] == n)].sort_values("w")
            if sub.empty:
                ax.set_title(f"run{run} n={n} (pending)")
                ax.grid(True, alpha=0.3)
                continue
            ax.plot(sub["w"], sub["delta_pearson"], "o-", color="#1f77b4", lw=2)
            ax.axhline(0.0, color="0.6", lw=0.8)
            ax.set_title(f"run{run}  n={n}")
            ax.set_xticks(list(WEIGHTS))
            ax.grid(True, alpha=0.3)
            if j == 2:
                ax.set_xlabel(r"$w_{\mathrm{model}}$")
            if i == 0:
                ax.set_ylabel(r"$\Delta P(w\mid S)$")
    fig.suptitle("Do not inspect argmax(w). Look for shape repeating across runs.")
    fig.tight_layout()
    fig.savefig(out / "response_curves_9panel.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    stage = (
        done.groupby(["state_n", "w"])["delta_pearson"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    for n, label, color in (
        (100, "n=100", "#1f77b4"),
        (300, "n=300", "#ff7f0e"),
        (500, "n=500", "#d62728"),
    ):
        sub = stage[stage["state_n"] == n]
        if sub.empty:
            continue
        ax.plot(sub["w"], sub["mean"], "o-", label=label, color=color, lw=2)
        if sub["count"].min() > 1:
            ax.fill_between(
                sub["w"],
                sub["mean"] - sub["std"].fillna(0),
                sub["mean"] + sub["std"].fillna(0),
                color=color,
                alpha=0.15,
                linewidth=0,
            )
    ax.set_xlabel(r"$w_{\mathrm{model}}$")
    ax.set_ylabel(r"mean $\Delta$Pearson across runs")
    ax.set_title("Stage-average response (std band). Gate: bands must not fully cover.")
    ax.set_xticks(list(WEIGHTS))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "stage_weight_curves.png", dpi=160)
    plt.close(fig)


def gate_text(c_df: pd.DataFrame, n_complete: int) -> str:
    lines = [
        "FROZEN GATE — do not override from a single 0.01 spike.",
        "",
        "CONTINUE dynamic fusion only if MOST runs satisfy:",
        "  n=100: C < 0  (prior-heavy / low w favored)",
        "  n=500: C > 0  (model-heavy / high w favored)",
        "AND stage-average curves are visually distinct (std does not swallow the shift).",
        "",
        "STOP Idea 2 if run directions conflict, e.g.",
        "  run1: prior→model, run2: model→prior, run3: mid-weight best.",
        "Verdict in that case: No reproducible state-dependent weight preference.",
        "",
    ]
    if n_complete < 45:
        lines.append(f"INCOMPLETE: {n_complete}/45 forks. Do not judge the hypothesis yet.")
        return "\n".join(lines)
    if c_df.empty:
        lines.append("No C(S) table; cannot judge.")
        return "\n".join(lines)
    pivot = c_df.pivot(index="state_n", columns="run", values="C")
    n100 = pivot.loc[100] if 100 in pivot.index else pd.Series(dtype=float)
    n500 = pivot.loc[500] if 500 in pivot.index else pd.Series(dtype=float)
    n100_neg = int((n100 < 0).sum())
    n500_pos = int((n500 > 0).sum())
    monotone = []
    for run in RUNS:
        if run not in pivot.columns:
            continue
        c100, c300, c500 = pivot.loc[100, run], pivot.loc[300, run], pivot.loc[500, run]
        monotone.append(bool(c100 < c300 < c500))
    lines.append(f"n=100 C<0: {n100_neg}/3 runs")
    lines.append(f"n=500 C>0: {n500_pos}/3 runs")
    lines.append(f"C_100 < C_300 < C_500: {sum(monotone)}/3 runs")
    if n100_neg >= 2 and n500_pos >= 2:
        lines.append("TENTATIVE: majority stage contrast matches hypothesis. Inspect 9-panel + std bands.")
    else:
        lines.append("STOP: No reproducible state-dependent weight preference. Idea 2 closed.")
        lines.append("Next: prior reliability / corruption pilot, λ ∈ {0, 0.5, 1}.")
    return "\n".join(lines)


def run_sanity(df: pd.DataFrame, run: int, n: int) -> int:
    report = sanity_for_state(df, run, n)
    print(json.dumps(report, indent=2, default=str))
    if not report["complete"]:
        print(f"\nrun{run}/n{n} not complete. First-wave sanity only — do not judge hypothesis.")
        return 0
    if not report["ok"]:
        print("\nSTOP: common-state fork is not a common state (or selection is invalid).")
        return 1
    print("\nPipeline sanity OK. Do not interpret ΔP / C(S) from a single state.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    out = ensure_analysis_dir("idea2_fork")
    grid = empty_grid()
    grid.to_csv(out / "table_45_template.csv", index=False)
    df, selected_map = attach_results(grid)

    if args.sanity_only:
        sys.exit(run_sanity(df, args.run, args.n))

    keep_cols = [c for c in df.columns if not c.startswith("_")]
    table = df[keep_cols].sort_values(["run", "state_n", "w"])
    table.to_csv(out / "table_45.csv", index=False)

    sanity_rows = [sanity_for_state(df, run, n) for run in RUNS for n in NS]
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(out / "sanity_by_state.csv", index=False)

    done = df[df["complete"]].copy()
    n_complete = int(done.shape[0])

    c_rows = []
    rho_rows = []
    for (run, n), sub in done.groupby(["run", "state_n"]):
        m = {wkey(r.w): float(r.delta_pearson) for r in sub.itertuples()}
        if all(w in m for w in (0.0, 0.25, 0.75, 1.0)):
            c_rows.append({
                "run": int(run),
                "state_n": int(n),
                "stage": STAGE[int(n)],
                "C": contrast_c(m),
                "mean_low_w": 0.5 * (m[0.0] + m[0.25]),
                "mean_high_w": 0.5 * (m[0.75] + m[1.0]),
                "interpretation": "model-heavy" if contrast_c(m) > 0 else "prior-heavy",
            })
        if len(sub) >= 3:
            rho, p = spearmanr(sub["w"], sub["delta_pearson"])
            rho_rows.append({
                "run": int(run),
                "state_n": int(n),
                "stage": STAGE[int(n)],
                "spearman_w_vs_delta": float(rho),
                "pvalue": float(p),
            })
    c_df = pd.DataFrame(c_rows)
    if not c_df.empty:
        c_df.to_csv(out / "contrast_C_by_state.csv", index=False)
        c_pivot = c_df.pivot(index="state_n", columns="run", values="C")
        c_pivot.to_csv(out / "contrast_C_stage_by_run.csv")
    rho_df = pd.DataFrame(rho_rows)
    if not rho_df.empty:
        rho_df.to_csv(out / "spearman_by_state.csv", index=False)

    jac_rows = []
    for run in RUNS:
        for n in NS:
            sets = {
                w: set(genes)
                for (r, nn, w), genes in selected_map.items()
                if r == run and nn == n and genes
            }
            for wa, wb in JACCARD_PAIRS:
                if wa in sets and wb in sets:
                    jac_rows.append({
                        "run": int(run),
                        "state_n": int(n),
                        "w_a": wa,
                        "w_b": wb,
                        "jaccard": jaccard(sets[wa], sets[wb]),
                        "n_intersect": len(sets[wa] & sets[wb]),
                        "n_union": len(sets[wa] | sets[wb]),
                    })
    jac_df = pd.DataFrame(jac_rows)
    if not jac_df.empty:
        jac_df.to_csv(out / "selection_jaccard.csv", index=False)

    write_response_curves(df, out)
    verdict = gate_text(c_df, n_complete)
    (out / "GATE.md").write_text(verdict + "\n")

    print(f"complete {n_complete}/45")
    print()
    print(table[["run", "state_n", "w", "P_before", "P_after", "delta_pearson"]].to_string(index=False))
    print()
    print(sanity_df.to_string(index=False))
    if not c_df.empty:
        print()
        print(c_df.to_string(index=False))
    if not jac_df.empty:
        print()
        print(jac_df.to_string(index=False))
    print()
    print(verdict)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
