#!/usr/bin/env python3
"""Idea 3c hard-reject I-Gate. Judge only after held-out perm campaigns finish.

    python scripts/analyze_idea3_intervention.py
"""
from __future__ import annotations

import glob
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import DATA_ROOT, RESULTS, REPO_ROOT, ensure_analysis_dir, trapezoid_nalc

PROTOCOL_SRC = REPO_ROOT / "configs" / "experiments" / "idea3_intervention" / "PROTOCOL.md"
TAU_JSON = REPO_ROOT / "results" / "analysis" / "idea3_intervention" / "frozen_tau.json"
RUNS = (1, 2, 3)
RUN_RE = re.compile(r"_run(\d+)_")


def paper_metrics(run: int) -> Path | None:
    hits = list((RESULTS / "fig4").glob(
        f"**/*100_5_100*priormean_new_max_run{run}_Core-Set_diff_effect_metrics.csv"
    ))
    hits = [p for p in hits if "single_" not in p.name and "_mw" not in p.name
            and "_sched_" not in p.name and "_detect_" not in p.name]
    return hits[0] if hits else None


def oracle_metrics(run: int) -> Path | None:
    hits = list((RESULTS / "idea3" / "pilot" / "runs").glob(f"*drop_rpe1_run{run}_*_metrics.csv"))
    return hits[0] if hits else None


def nalc_of(path: Path) -> float:
    df = pd.read_csv(path).sort_values("n_labeled")
    return trapezoid_nalc(df["n_labeled"], df["pearson_delta"])


def classify_intervention(name: str) -> str | None:
    if "_detect_" in name and "_corrupt_" in name:
        return "detected_l1"
    if "_detect_" in name:
        return "detected_clean"
    if "_corrupt_permute_rpe1_l1_" in name and "cs20260831" in name:
        return "equal_l1"
    return None


def collect() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        p = paper_metrics(run)
        o = oracle_metrics(run)
        if p:
            rows.append({"run": run, "method": "equal_clean", "nALC": nalc_of(p), "source": p.name})
        if o:
            rows.append({"run": run, "method": "oracle_drop", "nALC": nalc_of(o), "source": o.name})
    for path in glob.glob(str(RESULTS / "idea3" / "intervention" / "runs" / "*_metrics.csv")):
        name = Path(path).name
        m = RUN_RE.search(name)
        kind = classify_intervention(name)
        if not m or not kind:
            continue
        rows.append({"run": int(m.group(1)), "method": kind, "nALC": nalc_of(Path(path)), "source": name})
    return pd.DataFrame(rows)


def lookup(df: pd.DataFrame, method: str, run: int) -> float:
    hit = df[(df["method"] == method) & (df["run"] == run)]
    if hit.empty:
        return float("nan")
    return float(hit["nALC"].iloc[0])


def paired(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in RUNS:
        eq_c = lookup(df, "equal_clean", run)
        det_c = lookup(df, "detected_clean", run)
        eq_1 = lookup(df, "equal_l1", run)
        det_1 = lookup(df, "detected_l1", run)
        ora = lookup(df, "oracle_drop", run)
        rec = det_1 - eq_1
        head = ora - eq_1
        eta = rec / head if np.isfinite(rec) and np.isfinite(head) and head != 0 else np.nan
        rows.append({
            "run": run,
            "R_recover": rec,
            "eta": eta,
            "C_clean": eq_c - det_c,
            "nALC_equal_clean": eq_c,
            "nALC_detected_clean": det_c,
            "nALC_equal_l1": eq_1,
            "nALC_detected_l1": det_1,
            "nALC_oracle": ora,
        })
    return pd.DataFrame(rows)


def igate(primary: pd.DataFrame) -> str:
    need = primary.dropna(subset=["R_recover", "C_clean"])
    if len(need) < 3:
        missing = []
        for run in RUNS:
            row = primary[primary["run"] == run].iloc[0]
            if not np.isfinite(row["R_recover"]):
                missing.append(f"run{run} corrupt pair")
            if not np.isfinite(row["C_clean"]):
                missing.append(f"run{run} clean pair")
        return "INCOMPLETE: " + ", ".join(missing) + ". Do not judge I-Gate."
    r_pos = int((primary["R_recover"] > 0).sum())
    c_pos = int((primary["C_clean"] > 0).sum())
    eta_pos = int((primary["eta"] > 0).sum())
    mean_c = float(primary["C_clean"].mean())
    mean_r = float(primary["R_recover"].mean())
    lines = [
        f"R_recover>0: {r_pos}/3",
        f"C_clean>0 (detected worse than clean equal): {c_pos}/3  mean C={mean_c:+.4f}",
        f"eta>0: {eta_pos}/3  mean R={mean_r:+.4f}",
    ]
    recover_ok = r_pos >= 2
    eta_ok = eta_pos >= 2
    clean_hurt = c_pos >= 2 and mean_c > 0.005
    if recover_ok and eta_ok and not clean_hurt:
        lines.append("I-GATE A: hard reject recovers on λ=1 without a stable clean cost.")
        lines.append("Later: detector ablations. No softmax/LLM yet.")
    elif (not recover_ok) and not clean_hurt:
        lines.append("I-GATE B: detection fires but recovery is unstable. Soft downweight is allowed later; no LLM.")
    else:
        lines.append("I-GATE C: clean cost too large or no recovery. Stop the intervention method line.")
    return "\n".join(lines)


def main() -> None:
    out = ensure_analysis_dir("idea3_intervention")
    shutil.copyfile(PROTOCOL_SRC, out / "PROTOCOL.md")
    long_df = collect()
    if not long_df.empty:
        long_df.to_csv(out / "per_run_nalc.csv", index=False)
    primary = paired(long_df) if not long_df.empty else pd.DataFrame()
    if not primary.empty:
        primary.to_csv(out / "paired_intervention.csv", index=False)
    tau = json.loads(TAU_JSON.read_text()) if TAU_JSON.is_file() else {}
    verdict = igate(primary) if not primary.empty else "INCOMPLETE: no metrics."
    (out / "IGATE.md").write_text(
        PROTOCOL_SRC.read_text() + "\n\n--- live I-Gate ---\n\n"
        + f"frozen tau={tau.get('tau')} test_perm={tau.get('gpu_test_perm')}\n\n"
        + (primary.to_string(index=False) + "\n\n" if not primary.empty else "")
        + verdict + "\n"
    )
    if not primary.empty:
        print(primary.to_string(index=False))
        print()
    print(verdict)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
