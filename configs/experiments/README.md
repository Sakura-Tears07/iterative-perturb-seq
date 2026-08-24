# Experiment configs (`thoughts`)

| Directory | What it runs |
|-----------|----------------|
| `idea2_weight_sweep/` | \(w_{\text{model}} \in \{0,0.25,0.5,0.75,1.0\}\), then analyze best weight by round |

```bash
PARALLEL=1 bash configs/experiments/idea2_weight_sweep/run_sweep.sh
python scripts/analyze_weight_sweep.py
```

`run.py` uses `--device cuda:0` inside `CUDA_VISIBLE_DEVICES`. See [`RESEARCH.md`](../../RESEARCH.md).
