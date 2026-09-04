#!/usr/bin/env python3
"""Idea 3 corruption pilot — frozen analysis protocol.

Primary evidence is same-seed paired differences, not cross-seed means
and not argmax @600.

  D_full(λ, r) = nALC(full clean, r) − nALC(full λ, r)
  D_two(λ, r)  = nALC(two-source clean, r) − nALC(two-source λ, r)
  H(λ, r)      = nALC(oracle-drop, r) − nALC(full λ, r)

H compares oracle-drop to *corrupted full*, not to clean IterPert.

Do not require P_0 > P_.5 > P_1. The λ=1 sign is the gate; mid-λ
non-monotonicity is allowed (geometry / diversity, not a bug).

Do not judge A/B/C until runs 1–3 exist for every new cell.
"""
from __future__ import annotations

import glob
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import DATA_ROOT, RESULTS, ensure_analysis_dir, trapezoid_nalc

ROUND_STATES = DATA_ROOT / "round_states"
REPO_SEL = RESULTS / "selection_logs"
RUN_RE = re.compile(r"_run(\d+)_")
LAM_RE = re.compile(r"_corrupt_permute_rpe1_l([0-9.]+)_")
RUNS = (1, 2, 3)
BUDGETS = (200, 300, 400, 600)
NEW_CELLS = ("two_source_l0.5", "two_source_l1", "full_l0.5", "full_l1", "oracle_drop")

PROTOCOL = """\
FROZEN PROTOCOL (do not retune after looking at curves)

Primary table (same seed, nALC):
  run | D_two(0.5) | D_two(1) | D_full(0.5) | D_full(1) | H(1)

D > 0 means corruption hurt relative to the clean sibling.
H(1) = nALC(oracle-drop) − nALC(full λ=1).
Oracle is compared to corrupted full, not to clean IterPert.

Early-budget damage, same pairing:
  D_n(λ, r) = P_n(clean, r) − P_n(corrupt, r)
  n ∈ {200, 300, 400, 600}

Do NOT require λ-monotonicity. λ=0.5 beating clean is allowed.
The gate asks whether λ=1 is a *reproducible* injury.

Mechanism chain, per round:
  λ → RPE1 alignment(S_t) → J(S_λ, S_clean) → ΔP
  Three interpretations:
    1. alignment ↓, genes change, Pearson ↓  → bad prior misleads acquisition
    2. genes change, Pearson ~unchanged       → many near-equivalent batches
    3. alignment ↓, genes barely change       → RPE1 diluted by the other priors

Gate (majority = ≥2 of 3 runs). Judge only when all new cells have 3 runs.

  A  full λ=1 damaged in most seeds AND H(1)>0 in most seeds
     → reliability estimation has headroom.
     A/B intermediate also counts as continue: early (@200/@300) damaged
     in most seeds even if @600 recovers. Experimental efficiency is the goal.

  B  two-source λ=1 damaged in most seeds, full λ=1 not
     → averaging is the robustness; next corrupt a *fraction* of priors.

  C  two-source λ=1 has no stable degradation
     → stop this line. Do not invent a reliability method.

No adaptive drop / reweight / LLM planner until A or B.
If A: next question is detection (can observables separate λ=0 vs 1),
not a weighting formula. Gate ≠ publication-ready; 3 seeds decide
whether to continue, not whether to write a method claim.
Do not design from a single-run @600 spike.
"""


def canon(g: str) -> str:
    g = str(g).strip()
    return g[:-5] if g.endswith("+ctrl") else g


def lam_of(r) -> float:
    return float(getattr(r, "lambda"))


def nalc_and_at(df: pd.DataFrame) -> dict:
    sub = df.sort_values("n_labeled")
    row = {"nALC": trapezoid_nalc(sub["n_labeled"], sub["pearson_delta"])}
    for b in BUDGETS:
        hit = sub.loc[sub["n_labeled"] == b, "pearson_delta"]
        row[f"P{b}"] = float(hit.iloc[0]) if len(hit) else np.nan
    return row


def load_csv(path: Path, method: str, lam, setting: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    m = RUN_RE.search(path.name)
    if "run" not in df.columns and m:
        df["run"] = int(m.group(1))
    df["method"] = method
    df["lambda"] = lam
    df["setting"] = setting
    df["source"] = path.name
    return df


def collect() -> pd.DataFrame:
    frames = []
    paper = RESULTS / "fig4" / "essential_1k" / "iterpert" / "runs"
    for path in sorted(paper.glob("*priormean_new_max_run*_metrics.csv")):
        name = path.name
        if any(x in name for x in ("_mw", "_sched_", "single_", "dup", "corrupt", "drop_")):
            continue
        df = load_csv(path, "full_clean", 0.0, "full")
        df = df[df["run"].isin(set(RUNS))]
        if not df.empty:
            frames.append(df)

    rpe1 = RESULTS / "fig4c" / "single_prior" / "rpe1_kernel" / "runs"
    for path in sorted(rpe1.glob("*single_rpe1_kernel_run*_metrics.csv")):
        df = load_csv(path, "two_source_clean", 0.0, "two_source")
        df = df[df["run"].isin(set(RUNS))]
        if not df.empty:
            frames.append(df)

    mw1 = list(paper.glob("*_mw1.0_run*_metrics.csv"))
    mw1 += list((RESULTS / "idea2" / "runs").glob("*_mw1.0_run*_metrics.csv"))
    mw1 += list((RESULTS / "idea3" / "pilot" / "runs").glob("*_mw1.0_run*_metrics.csv"))
    seen = set()
    for path in mw1:
        df = load_csv(path, "model_only", np.nan, "model_only")
        key = int(df["run"].iloc[0])
        if key in seen:
            continue
        seen.add(key)
        frames.append(df)

    idea3 = RESULTS / "idea3" / "pilot" / "runs"
    for path in sorted(idea3.glob("*_metrics.csv")):
        name = path.name
        if "_drop_rpe1_" in name:
            frames.append(load_csv(path, "oracle_drop", np.nan, "full"))
            continue
        if "_mw1.0_" in name:
            continue
        lam_m = LAM_RE.search(name)
        if not lam_m:
            continue
        lam = float(lam_m.group(1))
        if "single_rpe1" in name:
            frames.append(load_csv(path, f"two_source_l{lam:g}", lam, "two_source"))
        else:
            frames.append(load_csv(path, f"full_l{lam:g}", lam, "full"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def per_run_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, run), sub in long_df.groupby(["method", "run"]):
        stats = nalc_and_at(sub)
        stats.update({
            "method": method,
            "run": int(run),
            "setting": sub["setting"].iloc[0],
            "lambda": sub["lambda"].iloc[0],
        })
        rows.append(stats)
    return pd.DataFrame(rows).sort_values(["setting", "lambda", "method", "run"])


def lookup(per_run: pd.DataFrame, method: str, run: int) -> pd.Series | None:
    hit = per_run[(per_run["method"] == method) & (per_run["run"] == run)]
    if hit.empty:
        return None
    return hit.iloc[0]


def paired_primary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in RUNS:
        row = {"run": run}
        two0 = lookup(per_run, "two_source_clean", run)
        full0 = lookup(per_run, "full_clean", run)
        drop = lookup(per_run, "oracle_drop", run)
        for lam, key in ((0.5, "0.5"), (1.0, "1")):
            two = lookup(per_run, f"two_source_l{lam:g}", run)
            full = lookup(per_run, f"full_l{lam:g}", run)
            row[f"D_two({key})"] = (
                np.nan if two0 is None or two is None
                else float(two0["nALC"]) - float(two["nALC"])
            )
            row[f"D_full({key})"] = (
                np.nan if full0 is None or full is None
                else float(full0["nALC"]) - float(full["nALC"])
            )
        full1 = lookup(per_run, "full_l1", run)
        row["H(1)"] = (
            np.nan if drop is None or full1 is None
            else float(drop["nALC"]) - float(full1["nALC"])
        )
        d_full1 = row["D_full(1)"]
        h1 = row["H(1)"]
        row["RF(1)"] = (
            np.nan if pd.isna(d_full1) or pd.isna(h1) or d_full1 == 0
            else float(h1) / float(d_full1)
        )
        rows.append(row)
    cols = ["run", "D_two(0.5)", "D_two(1)", "D_full(0.5)", "D_full(1)", "H(1)", "RF(1)"]
    return pd.DataFrame(rows)[cols]


def paired_budget(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = (
        ("two_source", "two_source_clean", "two_source_l0.5", 0.5),
        ("two_source", "two_source_clean", "two_source_l1", 1.0),
        ("full", "full_clean", "full_l0.5", 0.5),
        ("full", "full_clean", "full_l1", 1.0),
    )
    for run in RUNS:
        for setting, clean_m, corrupt_m, lam in specs:
            clean = lookup(per_run, clean_m, run)
            corrupt = lookup(per_run, corrupt_m, run)
            if clean is None or corrupt is None:
                continue
            rec = {
                "run": run,
                "setting": setting,
                "lambda": lam,
                "D_nALC": float(clean["nALC"]) - float(corrupt["nALC"]),
            }
            for b in BUDGETS:
                rec[f"D{b}"] = float(clean[f"P{b}"]) - float(corrupt[f"P{b}"])
            rows.append(rec)
        drop = lookup(per_run, "oracle_drop", run)
        full1 = lookup(per_run, "full_l1", run)
        if drop is not None and full1 is not None:
            rec = {
                "run": run,
                "setting": "oracle_vs_full_l1",
                "lambda": 1.0,
                "D_nALC": float(drop["nALC"]) - float(full1["nALC"]),
            }
            for b in BUDGETS:
                rec[f"D{b}"] = float(drop[f"P{b}"]) - float(full1[f"P{b}"])
            rec["note"] = "H, not D: oracle-drop minus corrupted full"
            rows.append(rec)
    return pd.DataFrame(rows)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def load_selected(stem_glob: str, round_id: int) -> set[str] | None:
    name = f"{stem_glob}_round{round_id}_selected_only"
    for root in (ROUND_STATES, REPO_SEL):
        for ext in (".pkl", ".parquet", ".csv"):
            matches = list(Path(root).glob(name + ext))
            if not matches:
                continue
            path = matches[0]
            if ext == ".csv":
                df = pd.read_csv(path)
            elif ext == ".parquet":
                df = pd.read_parquet(path)
            else:
                obj = pickle.load(open(path, "rb"))
                df = obj if isinstance(obj, pd.DataFrame) else None
            if df is None or "selected_gene_name" not in getattr(df, "columns", []):
                continue
            return {canon(g) for g in df["selected_gene_name"].tolist()}
    return None


def paper_stem(run: int) -> str:
    return (
        "GEARS_100_5_100_256_1_rt_simple_loss_v2_ess_1k_fix_eval_"
        f"priormean_new_max_run{run}_Core-Set_diff_effect"
    )


def two_source_clean_stem(run: int) -> str:
    return (
        "GEARS_100_5_100_256_1_rt_simple_loss_v2_ess_1k_fix_eval_"
        f"priormean_new_max_single_rpe1_kernel_run{run}_Core-Set_diff_effect"
    )


def idea3_stems() -> dict[tuple[str, int], str]:
    out = {}
    for path in glob.glob(str(RESULTS / "idea3" / "pilot" / "runs" / "*_metrics.csv")):
        stem = Path(path).name.replace("_metrics.csv", "")
        m = RUN_RE.search(stem)
        if not m:
            continue
        run = int(m.group(1))
        if "_drop_rpe1_" in stem:
            out[("oracle_drop", run)] = stem
        elif "single_rpe1" in stem and "_l0.5_" in stem:
            out[("two_source_l0.5", run)] = stem
        elif "single_rpe1" in stem and "_l1_cs" in stem:
            out[("two_source_l1", run)] = stem
        elif "_l0.5_" in stem:
            out[("full_l0.5", run)] = stem
        elif "_l1_cs" in stem:
            out[("full_l1", run)] = stem
    return out


def rpe1_alignment() -> pd.DataFrame:
    rows = []
    for path in sorted(ROUND_STATES.glob("*_state.pkl")):
        name = path.name
        if "rpe1" not in name and "priormean_new_max" not in name:
            continue
        try:
            data = pickle.load(open(path, "rb"))
        except Exception:
            continue
        aligns = data.get("alignments") or {}
        if "rpe1" not in aligns:
            continue
        m = RUN_RE.search(name)
        rd = re.search(r"_round(\d+)_", name)
        lam_m = LAM_RE.search(name)
        if "single_rpe1" in name:
            setting = "two_source"
            method = "two_source_clean" if not lam_m else f"two_source_l{float(lam_m.group(1)):g}"
        elif "_drop_" in name:
            continue
        else:
            setting = "full"
            method = "full_clean" if not lam_m else f"full_l{float(lam_m.group(1)):g}"
        rows.append({
            "setting": setting,
            "method": method,
            "lambda": 0.0 if not lam_m else float(lam_m.group(1)),
            "run": int(m.group(1)) if m else np.nan,
            "round": int(rd.group(1)) if rd else np.nan,
            "rpe1_alignment": float(aligns["rpe1"]),
            "model_alignment": aligns.get("model"),
            "source": name,
        })
    return pd.DataFrame(rows)


def pearson_after_round(long_df: pd.DataFrame, method: str, run: int, round_id: int) -> float:
    n = 100 + int(round_id) * 100
    hit = long_df[
        (long_df["method"] == method)
        & (long_df["run"] == run)
        & (long_df["n_labeled"] == n)
    ]
    if hit.empty:
        return np.nan
    return float(hit["pearson_delta"].iloc[0])


def mechanism_chain(long_df: pd.DataFrame, aln: pd.DataFrame) -> pd.DataFrame:
    stems = idea3_stems()
    pairs = [
        ("two_source", "two_source_l0.5", two_source_clean_stem, "two_source_clean", 0.5),
        ("two_source", "two_source_l1", two_source_clean_stem, "two_source_clean", 1.0),
        ("full", "full_l0.5", paper_stem, "full_clean", 0.5),
        ("full", "full_l1", paper_stem, "full_clean", 1.0),
    ]
    rows = []
    for setting, method, clean_fn, clean_method, lam in pairs:
        for run in RUNS:
            corrupt_stem = stems.get((method, run))
            if not corrupt_stem:
                continue
            clean_stem = clean_fn(run)
            for rd in range(1, 6):
                a = load_selected(corrupt_stem, rd)
                b = load_selected(clean_stem, rd)
                jac = jaccard(a, b) if a is not None and b is not None else np.nan
                aln_hit = pd.DataFrame()
                if not aln.empty:
                    aln_hit = aln[
                        (aln["method"] == method)
                        & (aln["run"] == run)
                        & (aln["round"] == rd)
                    ]
                p_c = pearson_after_round(long_df, method, run, rd)
                p_0 = pearson_after_round(long_df, clean_method, run, rd)
                rows.append({
                    "setting": setting,
                    "lambda": lam,
                    "run": run,
                    "round": rd,
                    "n_labeled_before": 100 + (rd - 1) * 100,
                    "n_labeled_after": 100 + rd * 100,
                    "rpe1_alignment": (
                        float(aln_hit["rpe1_alignment"].iloc[0]) if len(aln_hit) else np.nan
                    ),
                    "jaccard_vs_clean": jac,
                    "P_after": p_c,
                    "P_clean_after": p_0,
                    "delta_P_vs_clean": (p_0 - p_c) if pd.notna(p_c) and pd.notna(p_0) else np.nan,
                })
    return pd.DataFrame(rows)


def classify_mechanism(chain: pd.DataFrame) -> pd.DataFrame:
    """Coarse per-(setting, λ) story. Not a gate input."""
    rows = []
    if chain.empty:
        return pd.DataFrame()
    for (setting, lam), sub in chain.groupby(["setting", "lambda"]):
        jac = sub["jaccard_vs_clean"].dropna()
        d_p = sub["delta_P_vs_clean"].dropna()
        aln = sub["rpe1_alignment"].dropna()
        mean_j = float(jac.mean()) if len(jac) else np.nan
        mean_dp = float(d_p.mean()) if len(d_p) else np.nan
        mean_a = float(aln.mean()) if len(aln) else np.nan
        genes_moved = pd.notna(mean_j) and mean_j < 0.5
        perf_down = pd.notna(mean_dp) and mean_dp > 0.01
        # alignment drop is relative; without clean sibling KA we only note level
        if genes_moved and perf_down:
            story = "mislead: selection changes and Pearson drops"
        elif genes_moved and pd.notna(mean_dp) and mean_dp <= 0.01:
            story = "equivalent batches: selection changes, Pearson ~flat"
        elif (not genes_moved) and pd.notna(mean_j):
            story = "diluted: selection barely moves (RPE1 weakly used)"
        else:
            story = "incomplete logs"
        rows.append({
            "setting": setting,
            "lambda": lam,
            "mean_jaccard": mean_j,
            "mean_delta_P": mean_dp,
            "mean_rpe1_alignment": mean_a,
            "story": story,
        })
    return pd.DataFrame(rows)


def n_pos(series: pd.Series) -> tuple[int, int]:
    s = series.dropna()
    return int((s > 0).sum()), int(len(s))


def majority_pos(series: pd.Series) -> bool:
    k, n = n_pos(series)
    return n >= 3 and k >= 2


def gate_text(primary: pd.DataFrame, budget: pd.DataFrame, per_run: pd.DataFrame) -> str:
    lines = [PROTOCOL.strip(), "", "--- live numbers ---", ""]
    counts = per_run.groupby("method")["run"].nunique().to_dict()
    lines.append("runs per method: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    missing = [m for m in NEW_CELLS if counts.get(m, 0) < 3]
    complete = not missing

    if not primary.empty:
        lines.append("")
        lines.append(primary.to_string(index=False))

    if missing:
        lines.append("")
        lines.append("INCOMPLETE: " + ", ".join(missing) + ". Do not judge A/B/C.")
        lines.append("Partial paired table is for pipeline sanity only.")
        n_have = int(primary[["D_two(1)", "D_full(1)", "H(1)"]].notna().all(axis=1).sum())
        if n_have >= 2:
            lines.append(
                f"{n_have}/3 seeds already filled; if run3 does not reverse signs, "
                "this is likely Gate A — still not a verdict."
            )
        return "\n".join(lines)

    d_two_1 = primary["D_two(1)"]
    d_full_1 = primary["D_full(1)"]
    h1 = primary["H(1)"]
    d_two_05 = primary["D_two(0.5)"]
    lines.append("")
    for name, s in (
        ("D_two(1)>0", d_two_1),
        ("D_full(1)>0", d_full_1),
        ("H(1)>0", h1),
        ("D_two(0.5)>0 (not a gate; monotonicity not required)", d_two_05),
    ):
        k, n = n_pos(s)
        lines.append(f"  {name}: {k}/{n} runs")

    early = budget[(budget["setting"] == "full") & (budget["lambda"] == 1.0)]
    early200 = early["D200"] if not early.empty else pd.Series(dtype=float)
    early300 = early["D300"] if not early.empty else pd.Series(dtype=float)
    late600 = early["D600"] if not early.empty else pd.Series(dtype=float)
    if not early.empty:
        k200, n200 = n_pos(early200)
        k300, n300 = n_pos(early300)
        k600, n600 = n_pos(late600)
        lines.append(f"  D_full_200(1)>0: {k200}/{n200}")
        lines.append(f"  D_full_300(1)>0: {k300}/{n300}")
        lines.append(f"  D_full_600(1)>0: {k600}/{n600}")

    two_hurt = majority_pos(d_two_1)
    full_hurt = majority_pos(d_full_1)
    oracle_helps = majority_pos(h1)
    early_hurt = (not early.empty) and (majority_pos(early200) or majority_pos(early300))
    late_ok = (not early.empty) and (not majority_pos(late600))

    lines.append("")
    if full_hurt and oracle_helps:
        lines.append("GATE A: full λ=1 damaged in most seeds, oracle-drop recovers vs corrupted full.")
        lines.append("Next is detection: can observables separate clean vs corrupted RPE1?")
        lines.append("Do NOT start adaptive weighting yet. 3-seed majority ≠ a paper claim.")
        if int((h1 < 0).sum()) == 1 or int((d_full_1 < 0).sum()) == 1:
            lines.append("One seed reversed: still Gate A by majority, but expand seeds before methods.")
    elif early_hurt and oracle_helps and late_ok:
        lines.append("GATE A/B intermediate (counts as continue):")
        lines.append("full λ=1 hurts early (@200/@300) in most seeds; @600 mostly recovers.")
        lines.append("That is a real experimental-efficiency cost. Reliability still worth studying.")
    elif two_hurt and not full_hurt:
        lines.append("GATE B: two-source λ=1 damaged; full averaging absorbs one bad prior.")
        lines.append("Next: increase the fraction of corrupted priors, not adaptive weights.")
    elif not two_hurt:
        lines.append("GATE C: two-source λ=1 has no stable degradation. Stop this line.")
        lines.append("Do not invent a reliability method for AutoExplore.")
    else:
        lines.append("GATE mixed: inspect the paired table. Do not start adaptive dropping.")

    if (d_two_05.dropna() < 0).sum() >= 2:
        lines.append("")
        lines.append("Note: D_two(0.5)<0 in ≥2 runs. Mid-λ can beat clean; not a protocol violation.")

    if "RF(1)" in primary.columns:
        rf = primary["RF(1)"].dropna()
        if len(rf):
            lines.append("")
            lines.append("Recovery fraction RF(1)=H(1)/D_full(1)  (layer 2; not a gate input):")
            for _, r in primary.iterrows():
                lines.append(f"  run{int(r['run'])}: {r['RF(1)']:.3f}" if pd.notna(r["RF(1)"]) else f"  run{int(r['run'])}: NaN")
            lines.append(f"  mean: {float(rf.mean()):.3f}")
    return "\n".join(lines)


def write_plots(long_df: pd.DataFrame, budget: pd.DataFrame, chain: pd.DataFrame, out: Path) -> None:
    palette = {
        "two_source_clean": "#1f77b4",
        "two_source_l0.5": "#ff7f0e",
        "two_source_l1": "#d62728",
        "full_clean": "#1f77b4",
        "full_l0.5": "#ff7f0e",
        "full_l1": "#d62728",
        "oracle_drop": "#2ca02c",
        "model_only": "#7f7f7f",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, setting, title in (
        (axes[0], "two_source", "P3-A  model + RPE1"),
        (axes[1], "full", "P3-B  full IterPert"),
    ):
        sub = long_df[long_df["setting"].isin([setting, "model_only"])]
        if setting == "full":
            sub = long_df[long_df["setting"].isin(["full", "model_only"])]
        mean = sub.groupby(["method", "n_labeled"])["pearson_delta"].mean().reset_index()
        for method, g in mean.groupby("method"):
            if setting == "two_source" and (method.startswith("full") or method == "oracle_drop"):
                continue
            ax.plot(g["n_labeled"], g["pearson_delta"], "o-", label=method,
                    color=palette.get(method), lw=2)
        ax.set_title(title)
        ax.set_xlabel("n labeled")
        ax.set_xticks([100, 200, 300, 400, 500, 600])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Pearson delta")
    fig.suptitle("Learning curves (mean over available runs). Gate uses paired D/H, not these means.")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)

    if not budget.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        xs = [200, 300, 400, 600]
        for setting, lam, label, color in (
            ("two_source", 1.0, "two-source λ=1", "#d62728"),
            ("full", 1.0, "full λ=1", "#1f77b4"),
        ):
            sub = budget[(budget["setting"] == setting) & (budget["lambda"] == lam)]
            if sub.empty:
                continue
            ys = [float(sub[f"D{b}"].mean()) for b in xs]
            ax.plot(xs, ys, "o-", label=label, color=color, lw=2)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel("n labeled")
        ax.set_ylabel("paired D = P_clean − P_corrupt (mean over runs)")
        ax.set_title("Early-budget damage at λ=1 (positive = hurt)")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "early_budget_damage_lambda1.png", dpi=160)
        plt.close(fig)

    if not chain.empty:
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
        for setting, lam, color, name in (
            ("two_source", 1.0, "#d62728", "two λ=1"),
            ("full", 1.0, "#1f77b4", "full λ=1"),
        ):
            sub = chain[(chain["setting"] == setting) & (chain["lambda"] == lam)]
            if sub.empty:
                continue
            g = sub.groupby("round").mean(numeric_only=True)
            axes[0].plot(g.index, g["rpe1_alignment"], "o-", color=color, label=name, lw=2)
            axes[1].plot(g.index, g["jaccard_vs_clean"], "o-", color=color, label=name, lw=2)
            axes[2].plot(g.index, g["delta_P_vs_clean"], "o-", color=color, label=name, lw=2)
        axes[0].set_title("RPE1 alignment on S_t")
        axes[1].set_title("Jaccard vs clean selection")
        axes[2].set_title("ΔP = P_clean − P_corrupt")
        axes[2].axhline(0.0, color="0.6", lw=0.8)
        for ax, ylab in zip(axes, ("KA", "Jaccard", "paired ΔPearson")):
            ax.set_xlabel("round")
            ax.set_ylabel(ylab)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Mechanism chain at λ=1: alignment → selection divergence → Pearson")
        fig.tight_layout()
        fig.savefig(out / "mechanism_chain_lambda1.png", dpi=160)
        plt.close(fig)

        write_mechanism_scatters(chain, out)


def write_mechanism_scatters(chain: pd.DataFrame, out: Path) -> None:
    """Per-(run, round) scatters. Prepared now; interpret after Gate, not before."""
    df = chain.reset_index(drop=True).copy()
    df["selection_divergence"] = 1.0 - df["jaccard_vs_clean"]
    colors = {"two_source": "#d62728", "full": "#1f77b4"}
    markers = {1: "o", 2: "s", 3: "^"}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    rng = np.random.default_rng(0)
    x_lam = df["lambda"].to_numpy(dtype=float) + rng.uniform(-0.04, 0.04, len(df))
    for setting, sub in df.groupby("setting"):
        c = colors.get(setting, "0.4")
        for run, g in sub.groupby("run"):
            m = markers.get(int(run), "o")
            pos = g.index.to_numpy()
            axes[0].scatter(
                x_lam[pos], g["rpe1_alignment"],
                c=c, marker=m, s=28, alpha=0.85, label=f"{setting} r{int(run)}",
            )
            axes[1].scatter(
                g["rpe1_alignment"], g["selection_divergence"],
                c=c, marker=m, s=28, alpha=0.85,
            )
            axes[2].scatter(
                g["selection_divergence"], g["delta_P_vs_clean"],
                c=c, marker=m, s=28, alpha=0.85,
            )
    axes[0].set_xlabel("λ (jittered)")
    axes[0].set_ylabel("RPE1 alignment")
    axes[0].set_xticks([0.5, 1.0])
    axes[1].set_xlabel("RPE1 alignment")
    axes[1].set_ylabel("selection divergence  1−J")
    axes[2].set_xlabel("selection divergence  1−J")
    axes[2].set_ylabel("ΔPearson damage")
    axes[2].axhline(0.0, color="0.6", lw=0.8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Mechanism scatters (observation only until Gate)")
    fig.tight_layout()
    fig.savefig(out / "mechanism_scatters.png", dpi=160)
    plt.close(fig)


def main() -> None:
    out = ensure_analysis_dir("idea3_corruption")
    (out / "PROTOCOL.md").write_text(PROTOCOL)
    long_df = collect()
    if long_df.empty:
        print("No metrics found.")
        (out / "GATE.md").write_text("INCOMPLETE: no metrics.\n")
        return
    long_df.to_csv(out / "all_rounds.csv", index=False)
    per_run = per_run_table(long_df)
    per_run.to_csv(out / "per_run.csv", index=False)

    primary = paired_primary(per_run)
    primary.to_csv(out / "paired_primary.csv", index=False)
    budget = paired_budget(per_run)
    if not budget.empty:
        budget.to_csv(out / "paired_budget_damage.csv", index=False)

    aln = rpe1_alignment()
    if not aln.empty:
        aln.to_csv(out / "rpe1_alignment.csv", index=False)
    chain = mechanism_chain(long_df, aln if not aln.empty else pd.DataFrame())
    if not chain.empty:
        chain.to_csv(out / "mechanism_chain.csv", index=False)
        round_tbl = chain.rename(columns={
            "lambda": "λ",
            "rpe1_alignment": "RPE1 alignment",
            "jaccard_vs_clean": "J(clean, corrupt)",
            "delta_P_vs_clean": "ΔPearson damage",
        })[
            ["run", "round", "setting", "λ", "RPE1 alignment",
             "J(clean, corrupt)", "ΔPearson damage"]
        ]
        round_tbl.to_csv(out / "round_mechanism.csv", index=False)
        stories = classify_mechanism(chain)
        if not stories.empty:
            stories.to_csv(out / "mechanism_stories.csv", index=False)

    write_plots(long_df, budget, chain, out)
    verdict = gate_text(primary, budget, per_run)
    (out / "GATE.md").write_text(verdict + "\n")

    preview_cols = ["run", "D_two(1)", "D_full(1)", "H(1)"]
    print("Gate table (judge from these three columns first):")
    print(primary[preview_cols].to_string(index=False))
    print()
    print(primary.to_string(index=False))
    print()
    if not budget.empty:
        show = budget[budget["setting"].isin(["two_source", "full"])]
        cols = ["run", "setting", "lambda", "D200", "D300", "D400", "D600", "D_nALC"]
        print(show[cols].to_string(index=False))
        print()
    print(verdict)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
