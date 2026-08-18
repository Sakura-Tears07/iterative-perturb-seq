#!/usr/bin/env python3
"""Prepare Genome-wide prerequisites for knowledge_kernels_process.ipynb."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))
from paths import (  # noqa: E402
    DATA_ROOT,
    EMBEDDINGS_DIR,
    ESM_EMB,
    GEARS_DATA_PATH,
    GW_DATA_PATH,
    GW_PROCESSED_H5AD,
    GW_RAW_H5AD,
    REPO_ROOT,
    ensure_gears_shared_files,
)

EMBEDDINGS_GDRIVE = (
    "https://drive.google.com/file/d/16p9sQYkhpM-PBAcNWQdjL9pxDZ46krQX/view?usp=drive_link"
)


def prepare_gene_id_h5ad(force: bool = False) -> Path:
    """Build lightweight h5ad with obs[condition, gene_id] from raw GW file."""
    out = Path(GW_PROCESSED_H5AD)
    if out.exists() and not force:
        print(f"SKIP gene_id file exists: {out}")
        return out

    raw = Path(GW_RAW_H5AD)
    if not raw.exists():
        raise FileNotFoundError(f"Raw GW h5ad missing: {raw}")

    print(f"Reading obs from {raw} (backed mode)...")
    adata = sc.read_h5ad(raw, backed="r")
    obs = adata.obs
    sub = obs.loc[obs["perturbation"] != "control", ["perturbation", "gene_id"]].copy()
    sub = sub.drop_duplicates(subset=["perturbation"])
    sub["condition"] = sub["perturbation"].astype(str) + "+ctrl"
    sub = sub[["condition", "gene_id"]].reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    mini = ad.AnnData(X=np.zeros((len(sub), 1), dtype=np.float32), obs=sub)
    mini.write_h5ad(out)
    print(f"OK gene_id mapping: {out} ({len(sub)} perturbations)")
    return out


def run_gw_preprocess(force: bool = False) -> None:
    """Run full GEARS preprocessing (slow, ~1-3 hours, needs scikit-misc)."""
    gw_h5ad = Path(GW_DATA_PATH) / "perturb_processed.h5ad"
    if gw_h5ad.exists() and not force:
        print(f"SKIP GW preprocess exists: {gw_h5ad}")
        return

    try:
        import skmisc  # noqa: F401
    except ImportError:
        print("Installing scikit-misc...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-misc", "-q"])

    script = REPO_ROOT / "reproduce_repo" / "preprocess_gw.py"
    env = os.environ.copy()
    env["ITERPERT_DATA_ROOT"] = DATA_ROOT
    env["PYTHONPATH"] = str(REPO_ROOT / "reproduce_repo") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Running {script} (this may take 1-3 hours)...")
    subprocess.check_call([sys.executable, str(script)], env=env, cwd=str(script.parent))


def check_embeddings() -> bool:
    expected = [
        ESM_EMB,
        os.path.join(EMBEDDINGS_DIR, "biogpt_emb", "gene2biogpt.pkl"),
        os.path.join(EMBEDDINGS_DIR, "pops_emb", "gene2pops_all.pkl"),
        os.path.join(EMBEDDINGS_DIR, "gears_emb", "gene2gears_node2vec.pkl"),
    ]
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        print("\nEmbedding 文件缺失（需手动下载）:")
        for p in missing:
            print(f"  - {p}")
        print(f"\n下载链接: {EMBEDDINGS_GDRIVE}")
        print(f"解压到: {EMBEDDINGS_DIR}")
        print("期望目录结构:")
        print("  embeddings/esm_emb/gene2esm.pkl")
        print("  embeddings/biogpt_emb/gene2biogpt.pkl")
        print("  embeddings/pops_emb/gene2pops_all.pkl")
        print("  embeddings/gears_emb/gene2gears_node2vec.pkl")
        return False
    print("OK embeddings found")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GW notebook prerequisites")
    parser.add_argument("--gene-id-only", action="store_true", help="Only build gene_id h5ad (fast)")
    parser.add_argument("--full-preprocess", action="store_true", help="Also run full GW GEARS preprocess")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    ensure_gears_shared_files()
    prepare_gene_id_h5ad(force=args.force)

    if args.full_preprocess and not args.gene_id_only:
        run_gw_preprocess(force=args.force)

    gw_ok = (Path(GW_DATA_PATH) / "perturb_processed.h5ad").exists()
    emb_ok = check_embeddings()

    print("\n=== Status ===")
    print(f"  gene_id h5ad:  OK")
    print(f"  GW preprocess: {'OK' if gw_ok else 'MISSING (run with --full-preprocess)'}")
    print(f"  embeddings:    {'OK' if emb_ok else 'MISSING (manual download)'}")
    if not gw_ok or not emb_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
