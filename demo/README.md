# Demo 教程

三个 notebook 演示 IterPert 核心流程。路径统一由 `demo/paths.py` 管理。

## 1. 选择正确的 Kernel

在 Cursor / Jupyter 里，**必须选择 `iterpert` kernel**（不要用系统 Python 3.12）。

若列表里没有，在终端注册一次：

```bash
conda activate iterpert
python -m ipykernel install --user --name iterpert --display-name "iterpert"
```

然后在 notebook 右上角切换 kernel → **iterpert**。

## 2. 环境变量（可选）

```bash
export ITERPERT_DATA_ROOT=/data/zy/iterpert   # 数据根目录，默认值
export ITERPERT_DEVICE=cuda:0                 # GPU 设备，默认 cuda:0
```

## 3. Notebook 运行顺序

| Notebook | 用途 | 能否直接跑 |
|----------|------|-----------|
| `data_tutorial.ipynb` | 数据加载、自定义 adata | ✅ 可以 |
| `train_tutorial.ipynb` | 完整主动学习循环 | ✅ 可以（demo 用 epoch=1） |
| `knowledge_kernels_process.ipynb` | GW kernel 构建 | ❌ 需三步准备（见下） |

每个 notebook 第一个 cell 会自动运行 `check_environment()` 和 `check_*_ready()`。

### GW notebook 前置三步

```bash
conda activate iterpert
cd /home/zy/workspace/iterative-perturb-seq

# 1. 快速：生成 gene_id 映射文件（~1 分钟）
python demo/prepare_gw_prerequisites.py --gene-id-only

# 2. 慢：GW GEARS 预处理（1-3 小时，需 scikit-misc）
python demo/prepare_gw_prerequisites.py --full-preprocess

# 3. 手动：embedding（服务器通常无法直连 Google Drive）

**本服务器无法访问 Google Drive**，需在本机（开 VPN）下载后上传：

```bash
# 本机（能访问 Google 的电脑）：
pip install gdown
gdown 16p9sQYkhpM-PBAcNWQdjL9pxDZ46krQX -O knowledge_kernels.zip

# 上传到服务器：
scp knowledge_kernels.zip zy@<服务器>:/data/zy/iterpert/cache/downloads/embeddings/

# 服务器上解压：
bash demo/download_embeddings.sh /data/zy/iterpert/cache/downloads/embeddings/knowledge_kernels.zip
```

链接：https://drive.google.com/file/d/16p9sQYkhpM-PBAcNWQdjL9pxDZ46krQX/view
```

## 4. 依赖说明

`iterpert` conda 环境里 **没有** pip 安装的 `bmdal_reg`，notebook 通过 `setup_notebook_paths()` 自动加载 `reproduce_repo/bmdal_reg/`。

`knowledge_kernels_process.ipynb` 额外需要：

```bash
pip install biothings_client   # gene ID 映射
pip install scikit-misc        # GW 预处理用（preprocess_gw.py）
```

## 5. 数据目录

```
/data/zy/iterpert/
├── datasets/gears_data/
│   ├── gene2go_all.pkl              # PertData 共享文件（必须在 path 根目录）
│   ├── replogle_k562_essential_1000hvg/
│   ├── adamson/                     # 原始 adamson（只读参考）
│   └── demo_adamson/                # 自定义 adata 处理输出
├── knowledge_kernels/essential_1k/
└── cache/downloads/raw/
```

**注意**：自定义 adata 时 `path` 必须是 `GEARS_DATA_PATH`，不能指向空目录；用 `dataset_name="demo_adamson"` 区分输出。

## 6. 代码对应关系

| Notebook 步骤 | 源码文件 |
|--------------|---------|
| `initialize_data()` | `iterpert/data_pert.py`, `iterpert/gears/pertdata.py` |
| `initialize_model()` | `iterpert/nets_pert.py`, `iterpert/gears/gears.py` |
| `initialize_active_learning_strategy()` | `iterpert/iterpert.py`, `iterpert/query_strategies/` |
| `start()` | `iterpert/iterpert.py` |

复现脚本在 `reproduce_repo/run.py`，与 demo 共用数据根目录但 API 不同。
