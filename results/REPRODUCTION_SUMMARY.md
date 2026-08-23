# ITERpert 本地复现汇总

**Commit**: `52011a935e8e9e9a14b7d92a99c6c014d46e6db5`  
**数据根**: `/data/zy/iterpert`  
**完成日期**: 2026-08-23

## Fig.4 Essential 1K（10 方法 × 10 runs）— 完成

| 排名 | 方法 | @600 Pearson δ |
|------|------|----------------|
| 1 | **IterPert** | 0.2845 ± 0.0121 |
| 2 | BADGE | 0.2770 ± 0.0126 |
| 3 | LCMD | 0.2764 ± 0.0111 |
| 4 | ACS-FW | 0.2747 ± 0.0159 |
| 5 | TypiClust | 0.2747 ± 0.0126 |
| 6 | KMeans | 0.2746 ± 0.0187 |
| 7 | Random | 0.2620 ± 0.0139 |
| 8 | Core-Set | 0.2512 ± 0.0257 |
| 9 | BatchBALD | 0.2500 ± 0.0158 |
| 10 | BALD | 0.2487 ± 0.0136 |

- 原始 metrics: `results/fig4/essential_1k/*/runs/`（未纳入 git）
- 汇总: `results/fig4/comparison/`

## Fig.4c 单 Prior 消融（8 prior × 5 runs）— 完成

| 排名 | Prior | @600 Pearson δ |
|------|-------|----------------|
| 1 | ops_HeLa_DMEM | 0.2989 ± 0.0064 |
| 2 | ops_HeLa_HPLM | 0.2952 ± 0.0118 |
| 3 | rpe1 | 0.2904 ± 0.0091 |
| 4 | ops_A549 | 0.2782 ± 0.0140 |
| 5 | pops | 0.2669 ± 0.0033 |
| 6 | biogpt | 0.2531 ± 0.0200 |
| 7 | esm | 0.2510 ± 0.0164 |
| 8 | node2vec | 0.1991 ± 0.0258 |

- 原始 metrics: `results/fig4c/single_prior/*/runs/`
- 汇总: `results/fig4c/comparison/`

## 未完成

| 实验 | 状态 | 阻塞 |
|------|------|------|
| Fig.6 GW 主动学习 | 未开始 | genome_wide embedding 未下载 |
| Demo notebooks | 部分完成 | knowledge_kernels_process 可选 |

## 维护命令

```bash
cd reproduce_repo
python aggregate_fig4_results.py
python aggregate_fig4c_results.py
```
