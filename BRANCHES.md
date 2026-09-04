# Branch Layout

This fork uses three branches with distinct roles.

| Branch | Base | Purpose |
|--------|------|---------|
| `master` | Upstream IterPert | Original paper code. Do not add local reproduction or experiment changes. |
| `dev` | `master` | **Line A — reproduction.** Configurable paths, Fig.4/4c metrics export, results layout. Does **not** change core AL logic in `bmdal_reg/`. |
| `thoughts` | `dev` | **Line B.** Idea 3 Gate A + D-Gate A. **Now Idea 3c:** Observed-KA hard reject vs equal vs oracle (held-out perm). |

Fig.6 GW reproduction is **deferred** (author embeddings unavailable). It does not block Line B.

## Workflow

```bash
# Line A — paper reproduction
git checkout dev
cd reproduce_repo
python run.py ...   # results/fig4, results/fig4c

# Line B — Idea 3c hard reject
git checkout thoughts
python scripts/analyze_hard_reject_threshold.py
PARALLEL=1 bash configs/experiments/idea3_intervention/run_hard_reject.sh
python scripts/analyze_idea3_intervention.py
```

Current question: [`RESEARCH.md`](RESEARCH.md).

## Environment

```bash
export ITERPERT_DATA_ROOT=/data/zy/iterpert
```

See `reproduce_repo/local_paths.py` and `results/README.md` on `dev`.
