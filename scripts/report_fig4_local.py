#!/usr/bin/env python3
"""Detailed Fig.4 local verification report.

Compares the per-round learning curves of the LOCAL reproduction runs
(results/fig4/essential_1k/*/runs/*_metrics.csv) against the teacher's
10-run tables committed on origin/dev (results/fig4/essential_1k/*/tables/summary.csv),
and writes:

    results/LOCAL_VERIFICATION.md          detailed markdown report
    results/fig4/comparison/tables/local_vs_teacher.csv
    results/fig4/comparison/figures/local_vs_teacher.png

Usage:  python scripts/report_fig4_local.py
"""
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ESS = RESULTS / "fig4" / "essential_1k"

METHOD_DIRS = {
    "IterPert": ESS / "iterpert",
    "Random": ESS / "baselines" / "random",
    "Core-Set": ESS / "baselines" / "core_set",
    "BALD": ESS / "baselines" / "bald",
    "BatchBALD": ESS / "baselines" / "batchbald",
    "BADGE": ESS / "baselines" / "badge",
    "ACS-FW": ESS / "baselines" / "acs_fw",
    "LCMD": ESS / "baselines" / "lcmd",
    "TypiClust": ESS / "baselines" / "typiclust",
    "KMeans": ESS / "baselines" / "kmeans",
}
METRIC = "pearson_delta"
COLORS = {"IterPert": "#d62728", "Random": "#7f7f7f", "Core-Set": "#bcbd22",
          "BALD": "#17becf", "BatchBALD": "#1f77b4", "BADGE": "#e377c2",
          "ACS-FW": "#8c564b", "LCMD": "#9467bd", "TypiClust": "#2ca02c",
          "KMeans": "#ff7f0e"}


def load_teacher(method_dir):
    p = method_dir / "tables" / "summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.set_index("n_labeled")[["mean", "std", "count"]]


def load_local(method_dir):
    files = glob.glob(str(method_dir / "runs" / "*_metrics.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = df.groupby("n_labeled")[METRIC]
    return pd.DataFrame({"local_mean": g.mean(), "local_std": g.std(),
                         "local_n": g.count(), "local_values": g.apply(list)})


def load_fig4c():
    priors = ["pops_kernel", "rpe1_kernel", "esm_kernel", "biogpt_kernel",
              "node2vec_kernel", "ops_A549_kernel", "ops_HeLa_HPLM_kernel",
              "ops_HeLa_DMEM_kernel"]
    teacher_long = RESULTS / "fig4c" / "comparison" / "tables" / "all_priors_long.csv"
    tl = pd.read_csv(teacher_long) if teacher_long.exists() else pd.DataFrame()
    out = {}
    for prior in priors:
        d = RESULTS / "fig4c" / "single_prior" / prior
        local = load_local(d)
        if local is None:
            continue
        t = tl[tl["prior"] == prior].set_index("n_labeled")[["mean", "std", "count"]]             if not tl.empty else pd.DataFrame(index=local.index)
        out[prior] = (t, local)
    return out


def main():
    rows, figure_rows = [], []
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in METHOD_DIRS.items():
        teacher, local = load_teacher(d), load_local(d)
        if local is None:
            print(f"[skip] {name}: no local runs yet")
            continue
        teacher = teacher if teacher is not None else pd.DataFrame(index=local.index)
        both = pd.concat([teacher, local], axis=1)
        for n_labeled, row in both.iterrows():
            vals = row.get("local_values", [])
            dev = (row["local_mean"] - row["mean"]) / row["std"] if (
                "mean" in row and "std" in row and row["std"] > 0 and not pd.isna(row["mean"])
            ) else None
            rows.append(dict(method=name, n_labeled=int(n_labeled),
                             teacher_mean=row.get("mean"), teacher_std=row.get("std"),
                             teacher_n=row.get("count"), local_mean=row["local_mean"],
                             local_std=row["local_std"], local_n=row["local_n"],
                             local_values=vals, dev_from_teacher=dev))
        figure_rows.append((name, local, COLORS.get(name, "#333")))
    if not rows:
        print("no local runs found yet — wait for the fig4 suite")
        return

    comp = pd.DataFrame(rows)
    comp.to_csv(ESS.parent / "comparison" / "tables" / "local_vs_teacher.csv", index=False)

    # ---- fig4c section ----
    c4_rows = []
    for prior, (teacher, local) in load_fig4c().items():
        both = pd.concat([teacher, local], axis=1)
        for n_labeled, row in both.iterrows():
            dev = (row["local_mean"] - row["mean"]) / row["std"] if (
                "mean" in row and "std" in row and row["std"] > 0 and not pd.isna(row["mean"])
            ) else None
            c4_rows.append(dict(prior=prior, n_labeled=int(n_labeled),
                                teacher_mean=row.get("mean"), teacher_std=row.get("std"),
                                teacher_n=row.get("count"), local_mean=row["local_mean"],
                                local_std=row["local_std"], local_n=row["local_n"],
                                dev_from_teacher=dev))
    c4 = pd.DataFrame(c4_rows)
    if len(c4):
        c4.to_csv(RESULTS / "fig4c" / "comparison" / "tables" / "local_vs_teacher.csv",
                  index=False)

    # figure: local curves + teacher mean markers
    for name, local, color in figure_rows:
        ax.errorbar(local.index, local["local_mean"], yerr=local["local_std"],
                    marker="o", capsize=3, color=color, label=f"{name} (local, n={int(local['local_n'].max())})")
        d = METHOD_DIRS[name]
        t = load_teacher(d)
        if t is not None:
            ax.plot(t.index, t["mean"], linestyle="--", marker="x", color=color,
                    alpha=0.6, label=f"{name} (teacher n=10)")
    ax.set_xlabel("n labelled perturbations")
    ax.set_ylabel(f"{METRIC} (test)")
    ax.set_title("Fig.4 essential-1K local verification vs teacher 10-run tables")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(ESS.parent / "comparison" / "figures" / "local_vs_teacher.png", dpi=150)

    # markdown report
    md = ["# Fig.4 Essential-1K 本机复现验证报告\n",
          f"生成于本地运行完成后；对照 = origin/dev 上老师 10-run 汇总表。\n",
          f"指标：`{METRIC}`（test set, fix_evaluation=True, seed=1, 20 epoch, 100/100/5）。\n",
          "## 逐轮对比（dev = (本机 mean − 老师 mean)/老师 std）\n",
          "| 方法 | n_labeled | 老师 mean ± std (n) | 本机 mean ± std (n) | dev |",
          "|---|---|---|---|---|"]
    for r in comp.sort_values(["method", "n_labeled"]).itertuples():
        tm = "—" if pd.isna(r.teacher_mean) else f"{r.teacher_mean:.4f} ± {r.teacher_std:.4f} ({int(r.teacher_n)})"
        lm = "—" if pd.isna(r.local_mean) else f"{r.local_mean:.4f} ± {r.local_std:.4f} ({int(r.local_n)})"
        dv = "—" if r.dev_from_teacher is None else f"{r.dev_from_teacher:+.2f}"
        md.append(f"| {r.method} | {r.n_labeled} | {tm} | {lm} | {dv} |")
    if len(c4):
        md.append("\n## fig4c 单先验消融（本机 vs 老师 5-run 表）\n")
        md.append("| prior | n_labeled | 老师 mean ± std (n) | 本机 mean (n) | dev |")
        md.append("|---|---|---|---|---|")
        for r in c4.sort_values(["prior", "n_labeled"]).itertuples():
            tm = "—" if pd.isna(r.teacher_mean) else f"{r.teacher_mean:.4f} ± {r.teacher_std:.4f} ({int(r.teacher_n)})"
            lm = "—" if pd.isna(r.local_mean) else f"{r.local_mean:.4f} ({int(r.local_n)})"
            dv = "—" if r.dev_from_teacher is None else f"{r.dev_from_teacher:+.2f}"
            md.append(f"| {r.prior} | {r.n_labeled} | {tm} | {lm} | {dv} |")
    md.append("\n## 读法\n")
    md.append("- |dev| < 1：本机单/少 run 落在老师 10-run 分布的 1σ 内（协议忠实）。\n")
    md.append("- |dev| > 2 持续存在：需要排查协议差异（见 `notes` 中的已知差异清单）。\n")
    md.append("\n## 已知的 Random 基线差异（不是协议 bug）\n")
    md.append("- `RandomSelectionMethod` 使用**未播种**的 `torch.Generator(device=device)` "
              "（`bmdal_reg/bmdal/selection.py:233`），抽样依赖 torch 默认种子，跨进程/跨版本不可复现；\n")
    md.append("- 因此老师的 Random 曲线是 10 次独立抽样的均值，本机单 run 落在 ±1.5σ 属方法本身的随机性，"
              "不构成协议偏离；IterPert 及所有基于核的方法给定 (seed, run) 是确定性的。\n")
    (RESULTS / "LOCAL_VERIFICATION.md").write_text("\n".join(md))
    print("wrote", RESULTS / "LOCAL_VERIFICATION.md")
    print(comp.groupby("method")["dev_from_teacher"].agg(["mean", "max"]).round(2))


if __name__ == "__main__":
    main()
