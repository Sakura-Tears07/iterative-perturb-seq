# Model / prior weight sweep

Paper IterPert uses equal-weight `mean_new` (8 priors + model ⇒ \(w_{\text{model}}\approx 1/9\)). This sweep asks whether that weight should stay fixed.

```bash
# 5 GPUs, one weight per card
PARALLEL=1 bash configs/experiments/idea2_weight_sweep/run_sweep.sh

# serial
CUDA_VISIBLE_DEVICES=4 bash configs/experiments/idea2_weight_sweep/run_sweep.sh

python scripts/analyze_weight_sweep.py
```

`w_model ∈ {0, 0.25, 0.5, 0.75, 1.0}`; remaining mass is split equally across priors.

Seed 1 result: early rounds (200–300) prefer \(w \le 0.5\); later rounds prefer \(0.75\); \(w=1\) is bad early. Full table in [`RESEARCH.md`](../../../RESEARCH.md).

Optional follow-up (same flags, not yet run):

| `--weight_schedule` | Round 1 → 5 \(w_{\text{model}}\) |
|---------------------|----------------------------------|
| `fixed`             | `--model_weight` |
| `early_prior`       | 0.2, 0.3, 0.5, 0.7, 0.8 |
| `early_model`       | 0.8, 0.7, 0.5, 0.3, 0.2 |
