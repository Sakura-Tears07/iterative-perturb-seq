# EXP-2：面向 AutoExplore §3.2 旗舰基准的延伸计划

`auto_explore.pdf`（老师初稿，未发表）§3.2 把 **Perturb-seq 筛选设计** 作为旗舰基准，
其循环结构 = IterPert 闭环 + LLM Research Planner。本分支从 `dev`（干净复现）出发，
目标是：**补完该基准在这台机器上能跑的实验，并在此之上提升表现。**

本分支当前是**计划与脚手架**；代码随实验推进逐步加入（每个里程碑单独 commit）。

## 0. 本地资源盘点（已确认）

| 资源 | 状态 |
|---|---|
| K562 essential-1K GEARS 数据 | ✅ `/data/lhr/.../perturb_seq_data/gears_data/` |
| 8 个先验核 + ground_truth_delta + gears_kernel | ✅（knowledge_kernels_1k，2042×2042 对齐） |
| fig4 主实验复现（dev 分支） | ✅ 老师 10 方法×10runs 表 + 本机 13-run 验证套件运行中 |
| fig6 基因组尺度 | ❌ 缺 gw 核与数据（与老师结论一致，跳过） |
| RPE1 数据集（跨细胞系迁移实验） | ❌ 本机缺失，`knowledge_kernels_rpe1/` 为空 |
| 本地 LLM 权重 | ✅ `/data/model/qwen3-8b`（完整 5 shard）、`Qwen3-32B`、`Qwen2.5-1.5B-Instruct`、`SmolLM2-360M`、CodeLlama 系列 |
| LLM 推理栈 | ✅ `sglang_pc` conda env；GPU 被 GEARS 占满时用 Qwen2.5-1.5B/SmolLM2 迭代 |

## 1. 初稿的预期结果 ↔ 已有的实测证据

初稿 §3.2 声称：*"AutoExplore 在目标上下文观测少、先验有用但不完美时优势最大；
多个先验冲突、目标细胞系不同于先验细胞系、或目标同时包含预测与发现覆盖时，
应优于 IterPert 类方法。"*

此前离线实验（`exp1` 分支 `experiments/`）已给出必须面对的事实：

1. **预测指标上没有决策价值**：预算扫描 13 个点，最好与最差策略的 spread ≤0.016；
   oracle（偷看真值选权重）也只赢等权 +0.003；真实 GEARS 上 seed 噪声 0.03–0.05 > 方法差。
2. **先验失效场景下自适应权重也无增益**（noise2/noise4 场景），但"固定挑最强先验"
   （rpe1，跨细胞系核）在预算充裕时稳健小幅领先 —— 支持"固定非等权"而非"在线重学"。
3. **成本/深度维度有价值**：36.8% 扰动 <100 细胞；50 细胞 r=0.82 → 100 细胞 r=0.92，
   边际收益递减 —— 这是唯一被证明有非零决策价值的变量。

→ 结论：**在"预测精度"这个单一目标上打 LLM planner 是死路**。
初稿里真正可能有价值的差异化，是 §3.2 的另外两个维度：
**发现型指标**（pathway 覆盖、regulator 命中、多样性）和
**测量投入分配**（多保真：每个扰动测多少细胞）。planner 的战场应放在这里。

## 2. 实验里程碑（按序推进）

### M1：fig4 复现基线收口（dev，进行中）
- 13-run 本机验证（10 方法 run1 + IterPert run2/3 + Random run2）；
- 与老师 10×10 汇总表逐轮对比 → 确认本机协议忠实；
- 产出 `results/LOCAL_VERIFICATION.md`（逐轮 mean±std、双侧差异、与老师表的偏差）。

### M2：预测目标的预算曲线 + 发现型指标（复用 dev 数据）
- 用 fig4 已产生的选择集计算**发现型指标**（GO/pathway 覆盖、DE 基因恢复、模块多样性），
  检查"预测持平"的方法在发现维度是否分化（初稿 claim 3 的直接检验）；
- 在 20/50/100 初始预算上补跑 Random/IterPert/固定-rpe1 三方法（每方法 ≥5 seeds，
  依据：seed 噪声 0.03–0.05）。

### M3：多保真分配（初稿 §3.2 的"readout depth"设置）
- 复用 `exp1:experiments/stage3_cell_budget.py` 的实测 r(d) 曲线，构造不同细胞深度的
  低保真观测（σ² ∝ 1/cells）；
- 固定总细胞预算 C，比较 breadth（600×100）/ depth（300×200）/ adaptive
  （25 细胞初筛→重点扰动追加到 200）；
- 指标：预测 + 发现，按"每单位细胞预算的信息增益"归一。

### M4：LLM planner（最后才做）
- 先做"决策价值检验"：把 planner 可观察统计量喂给线性探针，预测 M2/M3 里
  **真正有差异**的指标；探针学不到 → LLM 也学不到（stage4 已对预测指标做过，结论为"学得到但没价值"）。
- 若有价值：`sglang` 起 Qwen3-8B（或 Qwen2.5-1.5B 快速迭代），动作空间 =
  {本轮信哪组先验、多样性 vs 不确定性占比、给哪些扰动加测深}；
  对照 = 同工具无 planner（**初稿钦定的最重要消融**）+ prompt-only LLM + 固定调度。
- 训练顺序参照 AutoLLMResearch：多策略生成轨迹 → 按真实评估筛选 → 监督蒸馏 → 再考虑 RL。
- 必须 ≥5 seeds 成对比较 + 报 seed 方差（否则不可信）。

### M5（可选）：跨细胞系迁移
- 需要先取 RPE1 essential 数据（dataverse）与对应核；本机当前缺失。
- 若不可得：用"K562 核 → K562 扰动子空间位移"的人工迁移场景替代。

## 3. 不做什么

- 不在纯预测指标上继续投入自适应融合权重（exp1 已记录负面结论）；
- 不引用 `auto_explore.pdf` 为已发表工作（未收录任何索引）；
- 不把 LLM 的自我评估当作奖励来源（初稿 §2 也要求奖励来自可验证结果）。

## 4. 环境

```bash
export ITERPERT_DATA_ROOT=/data/lhr/ai4s/iterpert/scratch
conda activate iterpert_env      # 复现/GEARS
conda activate sglang_pc         # LLM 推理（Qwen3-8B 等）
# LLM 权重: /data/model/qwen3-8b  (8B, 完整)
```
