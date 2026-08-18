#!/usr/bin/env python3
"""Move flat results/ files into organized subdirectories."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Files to keep at top level or move to aggregated/
AGGREGATED_NAMES = {
    "aggregate_fig4_results.py",
}

AGGREGATED_PATTERNS = [
    "*_fig4_summary.csv",
    "*_fig4_pivot.csv",
    "*_fig4_manifest.txt",
    "fig4_*",
    "fig4_three_way*",
    "coreset_fig4*",
    "random_fig4*",
    "bald_fig4*",
    "lcmd_fig4*",
    "iterpert_fig4*",
    "badge_fig4*",
    "acs_fw_fig4*",
    "batchbald_fig4*",
    "typiclust_fig4*",
    "kmeans_fig4*",
]


def classify(filename: str) -> Path | None:
    """Return target directory relative to RESULTS, or None to skip."""
    if filename in AGGREGATED_NAMES:
        return RESULTS / "fig4" / "aggregated" / "_scripts" if filename.endswith(".py") else RESULTS / "fig4" / "aggregated"

    for pat in AGGREGATED_PATTERNS:
        if Path(filename).match(pat):
            return RESULTS / "fig4" / "aggregated"

    if not filename.startswith("GEARS_"):
        return RESULTS / "misc"

    # Smoke tests (1 epoch)
    if "_100_1_100_" in filename or "_epo1_" in filename:
        return RESULTS / "fig4" / "essential_1k" / "smoke" / "runs"

    # Fig 4c single prior
    m = re.search(r"single_([a-zA-Z0-9_]+)_run\d+", filename)
    if m:
        prior = m.group(1)
        if not prior.endswith("_kernel"):
            prior = prior + "_kernel"
        return RESULTS / "fig4c" / "single_prior" / prior / "runs"

    # IterPert full (prior + diff_effect, not single)
    if "priormean_new_max" in filename and "Core-Set_diff_effect" in filename:
        return RESULTS / "fig4" / "essential_1k" / "iterpert" / "runs"

    baseline_map = [
        ("BatchBALD", "batchbald"),
        ("ACS-FW", "acs_fw"),
        ("BADGE", "badge"),
        ("LCMD", "lcmd"),
        ("Random", "random"),
        ("Core-Set", "core_set"),
        ("BALD", "bald"),
        ("TypiClust", "typiclust"),
        ("KMeansSampling", "kmeans"),
    ]
    for token, slug in baseline_map:
        if token in filename:
            return RESULTS / "fig4" / "essential_1k" / "baselines" / slug / "runs"

    return RESULTS / "misc"


def main() -> None:
    moved = 0
    skipped = 0
    for item in sorted(RESULTS.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            # already organized subdirs
            if item.name in ("fig4", "fig4c", "fig6", "misc", "analysis", "demo"):
                skipped += 1
            continue

        if item.name == "README.md":
            skipped += 1
            continue

        target_dir = classify(item.name)
        if target_dir is None:
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / item.name
        if dest.exists():
            print(f"SKIP (exists): {item.name}")
            skipped += 1
            continue

        shutil.move(str(item), str(dest))
        moved += 1

    print(f"Done: moved={moved}, skipped={skipped}")
    print("Structure:")
    for p in sorted(RESULTS.rglob("*")):
        if p.is_dir() and p.parent == RESULTS:
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {p.name}/  ({n} files)")


if __name__ == "__main__":
    main()
