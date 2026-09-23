# B 阶段第一步：RPE1（跨细胞系）数据可用性与实验协议审计

**范围**：只做数据可用性与协议准备，**未启动任何模型训练**。审计脚本
`experiments/rpe1_audit/audit_rpe1.py`（可重跑），结果 `audit_rpe1.json`。

---

## 闸门 1：RPE1 原始响应 / 对照 / 基因映射 / 候选容量 —— **已确认可获得**

数据来源：Dataverse「GEARS」数据集，`replogle_rpe1_essential.zip`
（file id **7458694**，0.67 GB，无限制，published 2023-10-13）。

**已实际下载并校验**：
- 路径：`/data/lhr/ai4s/iterpert/scratch/rpe1/replogle_rpe1_essential/perturb_processed.h5ad`
- **MD5 = `169370a8da6093470f122172ab30f7c5`（与 Dataverse 元数据完全一致）**
- 包内结构与我们的 K562 数据**同构**（同名 `perturb_processed.h5ad`），且条件命名一致（`GENE+ctrl`）

| 指标 | RPE1 | 对照：K562（在用） |
|---|---|---|
| 细胞 × 基因 | 162,733 × 5,000 | 310,385 × 1,000 HVG |
| 对照条件 | 独立 `ctrl` 条件，**11,485** 细胞 | `ctrl`，10,691 |
| 扰动数 | **1,543** | 2,057 |
| 细胞/扰动 | min 13 · p25 43 · **median 66** · p75 105 · max 3,458 | median 121 |
| ≥25 细胞 | 1,504（97.5%） | — |
| **≥50 细胞** | **1,035（67.1%）** | — |
| **≥100 细胞** | **426（27.6%）** | ~63% |
| ≥143 细胞（cap100+30%参考所需） | 218（14.1%） | — |
| 与 K562 扰动轴重叠 | **1,302 / 1,543 = 84.4%** | — |
| 与 K562 基因轴交集 | **662 / 1,000 = 66.2%** | — |

→ **闸门 1 通过**，但容量比 K562 小得多，这直接约束了闸门 3 的口径。

## 闸门 2：先验来源清单与泄漏规则 —— **规则已定，核包必须自建**

**关键事实**：论文附录 A.1 写明 `rpe1_kernel` = **RPE1 的 NTC-centered pseudobulk**。
因此对 **RPE1 作为目标**的任务：

- ❌ **必须剔除 `rpe1_kernel`**（它是目标自身的响应，属泄漏）；
- ✅ 外部迁移先验是 **`k562_kernel`**（K562 pseudobulk），与论文在 K562 任务里用 RPE1 先验完全镜像；
- ❌ `ground_truth_delta`（RPE1 目标响应）只用于评分，绝不进入任何策略观测。

| 先验 | 来源 | 对 RPE1 目标是否可用 |
|---|---|---|
| rpe1_kernel | RPE1 pseudobulk | **不可用（目标内部）** |
| **k562_kernel** | K562 pseudobulk | **可用（外部迁移先验）— 需自建** |
| pops_kernel | 公共 scRNA 图谱（PoPS） | 可用（基因级，与细胞系无关） |
| esm_kernel | ESM 蛋白语言模型 | 可用（基因级） |
| biogpt_kernel | BioGPT 文献嵌入 | 可用（基因级） |
| node2vec_kernel | PPI 网络 node2vec | 可用（基因级） |
| ops_A549 / ops_HeLa_* | 光学混合筛选 CellProfiler | 可用（不同测定/细胞系） |

**核包可得性**：Dataverse 上**不存在** RPE1-target 的 `knowledge_kernels` 包
（搜索 `knowledge_kernels` 命中 0；我们手上的包是 K562-target 那一份）。
→ 需要**本地自建 RPE1 轴的核**，所需素材与脚本：

- 基因级嵌入归档：Google Drive id `16p9sQYkhpM-PBAcNWQdjL9pxDZ46krQX`
  （**实测可达**：HTTP 303）；notebook 指明文件为
  `esm_emb/gene2esm.pkl`、`biogpt_emb/gene2biogpt.pkl`、
  `gears_emb/gene2gears_node2vec.pkl`、`pops_emb/gene2pops_all.pkl`（+ OPS `gene2ops`）
- 目标真值核 `ground_truth_delta`(RPE1)：由本地 RPE1 adata 计算（NTC-centered pseudobulk）
- `k562_kernel`：由**本地 K562** adata 计算，投影到与 RPE1 的共同基因空间（交集 662 基因）
- 构建流程参考 `demo/knowledge_kernels_process.ipynb`（作者给新数据集建核的示例）

## 闸门 3：在看目标结果之前锁定预算口径

容量约束下的两个可行口径（**必须二选一并在开跑前写死**）：

| 口径 | 候选人群 | 规模 | 备注 |
|---|---|---|---|
| **A（推荐）每扰动 cap=50 细胞** | RPE1 扰动且 ≥50 细胞 | **1,035** | 支撑 100 初始 + 3×100 查询（400 次选择）仍有余量 |
| B 每扰动 cap=100 细胞 | RPE1 扰动且 ≥100 细胞 | 426 | 400 次选择后候选池几乎耗尽，只能做 1–2 轮查询 |

**推荐 A**，并同时写死：
- **人群声明**：所有结论限于"RPE1 essential 中 ≥50 细胞的扰动"，不外推到低容量扰动；
- **等账单规则**（吸取 exp3 教训）：不按"扰动数 × cap"记账，而按
  `Σ min(可用细胞数, cap)` **实际揭示细胞**记账；若各臂账单不等价，
  则统一用"补足到同一总细胞数"的固定规则（与 exp3 相同做法），并在报告中给出每臂实际账单；
- 对照细胞 11,485 **单独计费**，不计入扰动预算；`ctrl` 从候选池移除；
- **测试/目标扰动不设上限**，只用于评分。

## 闸门 4：实验计划（沿用同一套四臂，不同时换终点与算法）

- **终点不变**：固定预算下**未见扰动**的预测质量（pearson_delta 等 4 指标）；
- **四臂不变**：`static_prior`（RPE1 轴的 8 个外部先验，**不含 rpe1_kernel**）、
  `iterpert_frozen`（第 0 轮后冻结选样模型）、`iterpert_full`（每轮更新模型核）、
  `random`（独立查询种子）；
- **规模**：先做**一个预算设置**（cap=50）、**多轮查询**（建议 100 初始 + 3×100）、
  **3 次配对重复**的小规模实验；确认方向后再扩重复或扫描细胞数；
- **验收标准同前**：更新选样是否优于 `static_prior` **且** 优于 `frozen`；
  仅优于 `random` 不足以证明反馈值得学习；
- 预测器与选样口径完全沿用 exp3 的实现（严格只用已购买细胞）。

## 尚待完成的具体工作（按顺序）

1. 下载基因级嵌入归档（Drive id `16p9sQYkhpM-PBAcNWQdjL9pxDZ46krQX`），
   **核对 RPE1 基因符号的覆盖率**（1,543 个扰动基因中有多少能查到嵌入）；
2. 用 `new_data_process` 把 RPE1 处理成 GEARS 数据集（决定 HVG：RPE1 自身 1000 HVG，
   或与 K562 交集的 662 基因——**这个选择会改变先验质量，需在报告中声明**）；
3. 自建 RPE1 轴核包：`ground_truth_delta`(RPE1) + `k562_kernel` + 基因级先验（esm/biogpt/node2vec/pops/ops），
   **明确排除 `rpe1_kernel`**；
4. 用同一套 `MFBudgetEnv`/四臂代码跑一个小规模配对实验（cap=50）。

## 风险与不确定项

- **嵌入覆盖率未验证**：RPE1 的 5,000 基因与嵌入字典的基因符号覆盖率未知，需下载归档后核对；
- **HVG 选择影响可比性**：RPE1 用 5,000 基因、K562 用 1,000 HVG，共同空间只有 662 基因，
  这可能削弱 `k562_kernel` 的迁移信号；
- **容量限制**：cap=100 时候选仅 426 个，无法支撑长战役；
- **先验银行与 K562 设置不对称**：剔除了 rpe1_kernel 后，RPE1 任务的先验集合
  与论文 K562 任务并非严格镜像（论文 K562 有 8 个先验，RPE1 任务将是 7 个 + k562_kernel）；
- **先验冲突只是假设**：不能预设反馈一定获益；这一步的价值是**第一次直接检验初稿的迁移主张**。
