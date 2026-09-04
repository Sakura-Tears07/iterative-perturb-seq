# Results Directory Layout

```
results/
├── fig4/                          # 论文 Fig.4 Essential 1K
│   ├── essential_1k/
│   │   ├── iterpert/runs/         # 10× metrics.csv（主实验）
│   │   ├── baselines/{method}/runs/
│   │   └── */tables/              # 每方法 summary/pivot
│   └── comparison/                # 跨方法汇总（tables + figures）
├── fig4c/                         # 论文 Fig.4c 单 prior 消融
│   ├── single_prior/{prior}/runs/ # 每个 prior 5× metrics.csv
│   └── comparison/                # 8 prior 对比曲线
├── analysis/                      # E0 / Idea 2 / Idea 3 汇总表与门控结论（入库）
├── idea2/                         # weight sweep / schedule / fork 的 metrics.csv
├── idea3/                         # corruption / intervention 的 metrics.csv
└── README.md
```

`analysis/`、`idea2/`、`idea3/` 的 csv / md / png / json 跟 `thoughts` 走。`.pkl`、`selection_logs/`、`round_states/` 和 Fig.4/4c 的 raw `runs/` 仍不入库。

Line B:

```bash
python scripts/analyze_weight_sweep.py
python scripts/analyze_existing_nalc.py
```

## 维护命令

```bash
# 归类 flat GEARS_* → fig4/fig4c/.../runs/
python reproduce_repo/reorganize_results.py

# 整理二级目录结构
python reproduce_repo/reorganize_results_structure.py

# 汇总 Fig.4 / Fig.4c
python reproduce_repo/aggregate_fig4_results.py
python reproduce_repo/aggregate_fig4c_results.py
```

**注意**：`analysis/` 不属于论文 Fig.4 主实验，不要与 `fig4/essential_1k/` 混放。
