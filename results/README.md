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
├── pilot/                         # thoughts 分支 P1 机制实验（非论文主复现）
│   ├── iterpert_runs/             # alignment/dup/prior_only 等 smoke
│   └── selection_logs/            # --selection_log 诊断 CSV
├── analysis/                      # E0 离线分析（thoughts 分支脚本产出）
└── README.md
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

**注意**：`pilot/` 和 `analysis/` 不属于论文 Fig.4 主实验，不要与 `fig4/essential_1k/` 混放。
