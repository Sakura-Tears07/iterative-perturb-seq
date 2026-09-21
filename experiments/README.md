# EXP-1：自适应先验加权探索

这个分支在 `dev`（干净复现）之上，加一个实验性研究问题：

> **先验失效/质量异质时，"在已观测扰动上学融合权重"能不能赢过等权融合？**

## 代码改动（相对 dev）

| 文件 | 改动 |
|---|---|
| `iterpert/bmdal/algorithms.py` | 新增 3 个**只用已标注子集**拟合权重的融合算子（见下） |
| `reproduce_repo/bmdal_reg/bmdal/algorithms.py` | 同一批算子的 bmdal_reg 副本（run.py 路径） |
| `reproduce_repo/run.py` | `--integrate_mode` 增加 `softmax_align / best_align / ridge_lab` |
| `iterpert/iterpert.py` | API 包装器：`integrate_mode` 可配置、每轮指标落盘 `*_metrics.json`、TypiClust 分支顺序 bug 的 opt-in 修复（`fix_typiclust_branch`） |
| `experiments/` | 本目录：离线诊断与闭环代理脚本 + 关键结果 JSON |

## 新融合算子

- `softmax_align`：softmax(对齐/0.1) 加权——保留弱先验，对比论文 `alignment` 的硬归一化
- `best_align`：one-hot 选对齐最高的核（"固定挑最强先验"上界）
- `ridge_lab`：非负岭回归拟合真值子核，向等权收缩

三者都只使用 `train_gold`（已观测扰动），可在线部署。

## 用法

```bash
# 真实闭环（API 路径）
CUDA_VISIBLE_DEVICES=0 python experiments/real_loop.py \
    --strategy IterPert --mode ridge_lab --tag exp1_ridge \
    --n_init 20 --n_query 5 --n_round 3 --epochs 10

# 或 reproduce_repo 路径
cd reproduce_repo
python run.py --use_prior --kernel_strategy Core-Set --base_kernel diff_effect \
    --integrate_mode ridge_lab --normalize_mode max ...
```

## 已有结论（离线，详见 experiments/results/*.json 与仓库外笔记）

- 预算扫描 13 个点：最好与最差方法的 spread 最大仅 0.016（论文预算下 0.0041），
  最优方法身份在 align/ridge/shrink/oracle/random 间随机跳动 → **决策价值 ≈ 0**。
- oracle（偷看已观测真值选权重）也只赢等权 +0.003。
- 真实 GEARS 小预算 6 方法对照：单 seed 下各方法差异小于 seed 间噪声（0.03–0.05）。
- 结论：**不建议继续投入自适应权重**；瓶颈不在"估不准权重"，而在"调权重没有价值"。
  保留本分支作为该负面结论的可复现记录。
