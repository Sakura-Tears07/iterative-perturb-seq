# Branch Layout

This fork uses three branches with distinct roles.

| Branch | Base | Purpose |
|--------|------|---------|
| `master` | Upstream IterPert | Original paper code. Do not add local reproduction or experiment changes. |
| `dev` | `master` | **Line A — reproduction.** Configurable paths, Fig.4/4c metrics export, results layout. Does **not** change core AL logic in `bmdal_reg/`. |
| `thoughts` | `dev` | **Line B — round-dependent fusion weights.** `--model_weight` / `--weight_schedule` on top of paper `mean_new`. |

Fig.6 GW reproduction is **deferred** (author embeddings unavailable). It does not block Line B.

## Workflow

```bash
# Line A — paper reproduction
git checkout dev
cd reproduce_repo
python run.py ...   # results/fig4, results/fig4c

# Line B — fusion-weight experiments
git checkout thoughts
PARALLEL=1 bash configs/experiments/idea2_weight_sweep/run_sweep.sh
python scripts/analyze_weight_sweep.py
```

Current question and results: [`RESEARCH.md`](RESEARCH.md).

## Environment

```bash
export ITERPERT_DATA_ROOT=/data/zy/iterpert
```

See `reproduce_repo/local_paths.py` and `results/README.md` on `dev`.
