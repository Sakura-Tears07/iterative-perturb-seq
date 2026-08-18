#!/usr/bin/env python3
"""Reorganize results/ into runs/, tables/, figures/, manifests/ per category."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from results_paths import (
    COMPARISON,
    FIG4,
    FIG4C,
    LEGACY_SLUG,
    RESULTS,
    SCRIPTS,
    ensure_comparison_dirs,
    legacy_slug_to_method_dir,
)

RUN_GLOB = ("GEARS_*.pkl", "GEARS_*.csv")

# Method dirs that should contain runs/
METHOD_DIRS = [
    FIG4 / "essential_1k" / "iterpert",
    FIG4 / "essential_1k" / "smoke",
    *list((FIG4 / "essential_1k" / "baselines").glob("*")),
    *list((FIG4C / "single_prior").glob("*")),
]


def _move(src: Path, dest: Path, moved: list[str]) -> None:
    if not src.exists() or src.resolve() == dest.resolve():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    shutil.move(str(src), str(dest))
    moved.append(f"{src.relative_to(RESULTS)} -> {dest.relative_to(RESULTS)}")


def move_runs_into_subdirs(moved: list[str]) -> None:
    for method_dir in METHOD_DIRS:
        if not method_dir.is_dir():
            continue
        runs_dir = method_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for pattern in RUN_GLOB:
            for f in method_dir.glob(pattern):
                if f.parent.name == "runs":
                    continue
                _move(f, runs_dir / f.name, moved)


def reorganize_legacy_aggregated(moved: list[str]) -> None:
    old_agg = FIG4 / "aggregated"
    if not old_agg.is_dir():
        return

    ensure_comparison_dirs()

    # Per-method tables / manifests / figures
    for slug, _ in LEGACY_SLUG.items():
        method_dir = legacy_slug_to_method_dir(slug)
        tables = method_dir / "tables"
        figures = method_dir / "figures"
        manifests = method_dir / "manifests"
        for d in (tables, figures, manifests):
            d.mkdir(parents=True, exist_ok=True)

        mapping = [
            (f"{slug}_fig4_summary.csv", tables / "summary.csv"),
            (f"{slug}_fig4_pivot.csv", tables / "pivot.csv"),
            (f"{slug}_fig4_manifest.txt", manifests / "manifest.txt"),
        ]
        for src_name, dest in mapping:
            _move(old_agg / src_name, dest, moved)

        # Method-specific sanity plots
        for png in old_agg.glob(f"{slug}_fig4*.png"):
            _move(png, figures / png.name.replace(f"{slug}_fig4_", ""), moved)

    # IterPert sanity plot special name
    _move(
        old_agg / "iterpert_fig4_sanity.png",
        legacy_slug_to_method_dir("iterpert") / "figures" / "sanity.png",
        moved,
    )

    # Cross-method comparison outputs
    comp_tables = COMPARISON / "tables"
    comp_figures = COMPARISON / "figures"
    comp_manifests = COMPARISON / "manifests"

    table_map = {
        "fig4_all_methods_long.csv": "all_methods_long.csv",
        "fig4_all_methods_comparison.csv": "all_methods_comparison.csv",
        "fig4_all_methods_summary.csv": "all_methods_summary.csv",
        "fig4_three_way_comparison.csv": "three_way_comparison.csv",
    }
    for src, dst in table_map.items():
        _move(old_agg / src, comp_tables / dst, moved)

    figure_map = {
        "fig4_all_methods.png": "all_methods.png",
        "fig4_main_baselines.png": "main_baselines.png",
        "fig4_all_baselines.png": "all_baselines.png",
        "fig4_three_way_sanity.png": "three_way_sanity.png",
    }
    for src, dst in figure_map.items():
        _move(old_agg / src, comp_figures / dst, moved)

    manifest_map = {
        "fig4_all_methods_manifest.txt": "all_methods_manifest.txt",
        "fig4_three_way_manifest.txt": "three_way_manifest.txt",
    }
    for src, dst in manifest_map.items():
        _move(old_agg / src, comp_manifests / dst, moved)

    # Scripts & readme
    scripts_src = old_agg / "_scripts"
    if scripts_src.is_dir():
        SCRIPTS.mkdir(parents=True, exist_ok=True)
        for f in scripts_src.iterdir():
            _move(f, SCRIPTS / f.name, moved)
        try:
            scripts_src.rmdir()
        except OSError:
            pass

    _move(old_agg / "README.md", RESULTS / "README.md", moved)

    # Remove empty aggregated/
    try:
        if old_agg.is_dir() and not any(old_agg.iterdir()):
            old_agg.rmdir()
    except OSError:
        pass


def reorganize_misc(moved: list[str]) -> None:
    misc = RESULTS / "misc"
    if not misc.is_dir():
        return
    comp_tables = COMPARISON / "tables"
    comp_tables.mkdir(parents=True, exist_ok=True)
    for f in misc.glob("*.csv"):
        _move(f, comp_tables / f.name, moved)


def main() -> None:
    moved: list[str] = []
    move_runs_into_subdirs(moved)
    reorganize_legacy_aggregated(moved)
    reorganize_misc(moved)

    if moved:
        print(f"Reorganized {len(moved)} items")
        for line in moved[:20]:
            print(f"  {line}")
        if len(moved) > 20:
            print(f"  ... and {len(moved) - 20} more")
    else:
        print("Nothing to reorganize (structure already up to date).")
        print("If flat GEARS_* files remain in results/, run reproduce_repo/reorganize_results.py first.")

    print("\nStructure:")
    for label, path in [
        ("comparison/tables", COMPARISON / "tables"),
        ("comparison/figures", COMPARISON / "figures"),
        ("iterpert/runs", FIG4 / "essential_1k/iterpert/runs"),
        ("iterpert/tables", FIG4 / "essential_1k/iterpert/tables"),
    ]:
        n = len(list(path.glob("*"))) if path.exists() else 0
        print(f"  {label}: {n} files")


if __name__ == "__main__":
    main()
