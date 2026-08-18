# Branch Layout

This fork uses three branches with distinct roles.

| Branch | Base | Purpose |
|--------|------|---------|
| `master` | Upstream IterPert | Original paper code; do not add local reproduction or experiment changes here. |
| `dev` | `master` | **Local reproducibility**: configurable paths, Fig.4 metrics export, results directory layout, preprocessing helpers. Does **not** change core active-learning logic in `bmdal_reg/`. |
| `thoughts` | `dev` | **Experimental extensions**: selection logging, offline analysis scripts, adaptive-fusion pilots. Safe to iterate; not required for baseline reproduction. |

## Typical workflow

```bash
# Reproduce Fig.4 Essential 1K (paths + metrics only)
git checkout dev
cd reproduce_repo
python run.py ...   # outputs under results/fig4/...

# Run mechanism / diagnostic experiments
git checkout thoughts
python scripts/analyze_existing_nalc.py
bash configs/experiments/p1_pilot/run_pilot.sh
```

## Environment

Set data root before running (default `/data/zy/iterpert`):

```bash
export ITERPERT_DATA_ROOT=/data/zy/iterpert
```

See `reproduce_repo/local_paths.py` and `results/README.md` on `dev`.
