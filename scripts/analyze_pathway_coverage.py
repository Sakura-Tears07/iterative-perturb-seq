#!/usr/bin/env python3
"""P0.3: Pathway / GO coverage analysis from existing selection pkls."""
from __future__ import annotations

import glob
import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import METHODS, ensure_analysis_dir, find_method_files, normalize_gene_name

GO_PATH = Path("/data/zy/iterpert/datasets/gears_data/gene2go_all.pkl")
ANALYSIS_METHODS = ["IterPert", "Random", "Core-Set", "BADGE", "TypiClust"]


def load_gene2go() -> dict[str, set[str]]:
    with open(GO_PATH, "rb") as f:
        raw = pickle.load(f)
    return {gene: set(terms) for gene, terms in raw.items()}


def load_selection_pkls(method: str) -> dict[int, list[str]]:
    slug_map = {
        "IterPert": "iterpert",
        "Random": "baselines/random",
        "Core-Set": "baselines/core_set",
        "BADGE": "baselines/badge",
        "TypiClust": "baselines/typiclust",
    }
    root = Path(__file__).resolve().parents[1] / "results/fig4/essential_1k" / slug_map[method]
    files = sorted((root / "runs").glob("*.pkl"))
    files = [f for f in files if "metrics" not in f.name]
    merged: dict[int, list[str]] = {r: [] for r in range(1, 6)}
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        for rd, genes in d.items():
            merged[rd].extend([normalize_gene_name(g) for g in genes])
    return merged


def go_entropy(genes: list[str], gene2go: dict[str, set[str]]) -> float:
    term_counts = Counter()
    for g in genes:
        for term in gene2go.get(g, ()):
            term_counts[term] += 1
    if not term_counts:
        return float("nan")
    probs = np.array(list(term_counts.values()), dtype=float)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def pathway_repeat_rate(genes: list[str], gene2go: dict[str, set[str]]) -> float:
    primary_terms = []
    for g in genes:
        terms = gene2go.get(g, set())
        if terms:
            primary_terms.append(sorted(terms)[0])
    if not primary_terms:
        return float("nan")
    counts = Counter(primary_terms)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(primary_terms)


def cumulative_go_terms(genes_by_round: dict[int, list[str]], gene2go: dict[str, set[str]]) -> pd.DataFrame:
    seen_genes: set[str] = set()
    seen_terms: set[str] = set()
    rows = []
    for rd in sorted(genes_by_round):
        new_genes = [g for g in genes_by_round[rd] if g not in seen_genes]
        seen_genes.update(new_genes)
        new_terms = set()
        for g in new_genes:
            new_terms.update(gene2go.get(g, ()))
        n_new_terms = len(new_terms - seen_terms)
        seen_terms.update(new_terms)
        rows.append({
            "round": rd,
            "n_new_genes": len(new_genes),
            "n_new_go_terms": n_new_terms,
            "cumulative_go_terms": len(seen_terms),
            "go_entropy_batch": go_entropy(new_genes, gene2go),
            "pathway_repeat_rate_batch": pathway_repeat_rate(new_genes, gene2go),
        })
    return pd.DataFrame(rows)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def cross_method_jaccard(all_selections: dict[str, dict[int, list[str]]]) -> pd.DataFrame:
    rows = []
    methods = list(all_selections.keys())
    for rd in range(1, 6):
        sets = {m: set(all_selections[m].get(rd, [])) for m in methods}
        for i, a in enumerate(methods):
            for b in methods[i + 1 :]:
                rows.append({"round": rd, "method_a": a, "method_b": b, "jaccard": jaccard(sets[a], sets[b])})
    return pd.DataFrame(rows)


def bio_similarity_to_prev(genes_by_round: dict[int, list[str]], gene2go: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    prev_terms: set[str] = set()
    for rd in sorted(genes_by_round):
        terms = set()
        for g in genes_by_round[rd]:
            terms.update(gene2go.get(g, ()))
        rows.append({
            "round": rd,
            "jaccard_go_with_prev": jaccard(terms, prev_terms) if prev_terms else float("nan"),
            "n_go_terms": len(terms),
        })
        prev_terms = terms
    return pd.DataFrame(rows)


def plot_cumulative_go(summary: pd.DataFrame, outpath: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, sub in summary.groupby("method"):
        ax.plot(sub["round"], sub["cumulative_go_terms"], "o-", label=method)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative GO terms covered")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    outdir = ensure_analysis_dir("e0_baseline")
    gene2go = load_gene2go()

    per_method_rows = []
    all_selections: dict[str, dict[int, list[str]]] = {}
    for method in ANALYSIS_METHODS:
        sel = load_selection_pkls(method)
        if not sel:
            print(f"Skipping {method}: no selection pkls found")
            continue
        all_selections[method] = sel
        cum = cumulative_go_terms(sel, gene2go)
        cum["method"] = method
        per_method_rows.append(cum)
        bio_similarity_to_prev(sel, gene2go).to_csv(outdir / f"pathway_bio_similarity_{method.lower()}.csv", index=False)

    if not per_method_rows:
        raise RuntimeError("No selection pkls found for pathway analysis")

    summary = pd.concat(per_method_rows, ignore_index=True)
    summary.to_csv(outdir / "pathway_coverage_by_round.csv", index=False)
    cross_method_jaccard(all_selections).to_csv(outdir / "pathway_selection_jaccard.csv", index=False)
    plot_cumulative_go(summary, outdir / "pathway_cumulative_go_terms.png")

    print("Saved pathway coverage analysis to", outdir)
    print(summary.groupby("method")[["n_new_go_terms", "go_entropy_batch"]].mean().round(3))


if __name__ == "__main__":
    main()
