# Results Directory Layout

```
results/
├── fig4/
│   ├── essential_1k/
│   │   ├── iterpert/
│   │   │   ├── runs/        # 原始 run 输出 (*.pkl, *_metrics.csv)
│   │   │   ├── tables/      # summary.csv, pivot.csv
│   │   │   ├── figures/     # 方法专属图
│   │   │   └── manifests/   # 封存说明
│   │   ├── baselines/
│   │   │   ├── random/      # 同上结构
│   │   │   ├── badge/
│   │   │   └── ...
│   │   └── smoke/runs/
│   ├── comparison/          # 跨方法对比
│   │   ├── tables/          # all_methods_summary.csv 等
│   │   ├── figures/         # all_methods.png 等
│   │   └── manifests/
│   └── _scripts/            # 维护脚本 wrapper
├── fig4c/
│   └── single_prior/
│       └── {prior}/
│           └── runs/
├── fig6/genome_wide/runs/   # Fig.6 (future)
└── misc/
```

## 维护命令

从仓库根目录运行（推荐）：

```bash
# 1. 归类 results/ 根目录下的 flat GEARS_* 文件 → fig4/fig4c/.../runs/
python reproduce_repo/reorganize_results.py

# 2. 整理 runs/tables/comparison 二级结构（已整理时会提示 up to date）
python reproduce_repo/reorganize_results_structure.py

# 3. 重新生成 Fig.4 汇总表和对比图
python reproduce_repo/aggregate_fig4_results.py
```

也可从 `results/fig4/_scripts/` 运行同名 wrapper。

**注意**：步骤 1 处理根目录散落的 `GEARS_*`；步骤 2 不会移动 flat 文件。若根目录仍有 `GEARS_*`，请先跑步骤 1。

新实验输出路径由 `local_paths.resolve_result_dir()` 自动写入 `{category}/runs/`。
