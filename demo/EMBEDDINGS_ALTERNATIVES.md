# Embedding 替代方案（无法访问 Google Drive 时）

Google Drive 链接是作者私有分享，**无权限 + 服务器网络不通** 都会导致下载失败。这不影响 Essential 1K 实验（kernel 已在 `knowledge_kernels/essential_1k/`）。

## 当前状态（2026-08-18）

| 项目 | 状态 |
|------|------|
| GW 预处理 `replogle_k562_gw_1000hvg/` | ✅ 完成（h5ad 8.3G + pyg 17G） |
| gene_id 中间文件 | ✅ 完成 |
| Google Drive embedding | ❌ 不可用 |

---

## 方案 A：先不学 GW notebook（推荐）

Essential 1K 的 demo 已足够理解核心代码：

- `data_tutorial.ipynb` → 数据流
- `train_tutorial.ipynb` → 主动学习 + **已有 prior kernels**
- `knowledge_kernels/essential_1k/` → 作者预计算好的 kernel，无需 raw embedding

Fig.4 / Fig.4c 复现也不依赖 Google Drive embedding。

---

## 方案 B：GW notebook 跑「不需要 embedding」的部分

`knowledge_kernels_process.ipynb` 前半段（`ground_truth_delta` kernel）只需 GW adata，**不需要** embedding 文件。Restart kernel 后跑到 `save_kernel('ground_truth_delta', ...)` 即可理解 kernel 构建逻辑。

---

## 方案 C：从公开源重新生成 embedding

论文中各 modality 的原始来源：

| Embedding | 公开来源 | 难度 |
|-----------|---------|------|
| ESM | [HuggingFace ESM-2](https://huggingface.co/facebook/esm2_t48_15B_UR50D) 或小模型 `esm2_t6_8M_UR50D` | 高（大模型/需序列） |
| BioGPT | [microsoft/BioGPT-Large](https://huggingface.co/microsoft/BioGPT-Large) | 中 |
| PoPS | [FinucaneLab Dropbox](https://www.dropbox.com/scl/fo/ne7xhxkt4dwhvd52a59ub/...) 或 [Embpy_Data](https://huggingface.co/datasets/theislab/Embpy_Data) | 中 |
| node2vec | 本地 `gears_data/go_essential_all/` PPI 网络自行计算 | 低 |
| OPS (Cell Painting) | [Broad PERISCOPE](https://github.com/broadinstitute/2022_PERISCOPE) | 中 |

国内可试 **HF 镜像**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install huggingface_hub transformers fair-esm
```

生成结果需整理为 notebook 期望的 pickle 格式：
```
knowledge_kernels/embeddings/
├── esm_emb/gene2esm.pkl          # dict: gene_symbol -> np.ndarray
├── biogpt_emb/gene2biogpt.pkl
├── pops_emb/gene2pops_all.pkl
└── gears_emb/gene2gears_node2vec.pkl
```

---

## 方案 D：联系作者

GitHub [Issue #1](https://github.com/Genentech/iterative-perturb-seq/issues/1) 已有人问过 whole-genome kernel 下载。可向作者请求：
- 改传到 Harvard Dataverse / Zenodo
- 或开放 Google Drive 权限

邮箱见论文：lopez.romain@gene.com

---

## 方案 E：请有权限的同事代下

若 Genentech/Broad 同事能访问 Google Drive，下载后：
```bash
scp knowledge_kernels.zip user@server:/data/zy/iterpert/cache/downloads/embeddings/
bash demo/download_embeddings.sh /data/zy/iterpert/cache/downloads/embeddings/knowledge_kernels.zip
```
