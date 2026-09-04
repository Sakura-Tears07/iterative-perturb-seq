# 5-seed schedule

Compare paper equal-mean against two fixed weights and two **opposite** round schedules. Early-Model is the negative control.

| tag | flags |
|-----|--------|
| `paper` | Fig.4 IterPert runs 1–5 (reused, not re-run) |
| `fixed_0.5` | `--model_weight 0.5` |
| `fixed_0.75` | `--model_weight 0.75` |
| `early_prior` | `--weight_schedule early_prior` |
| `early_model` | `--weight_schedule early_model` |

Fusion code: Phase 0 commit `a06d119`. GPU 0 is often occupied; default cards are 1–4.

```bash
PARALLEL=1 bash configs/experiments/idea2_schedule/run_schedule.sh
python scripts/analyze_schedule.py
```

Seed 1 of Fixed-0.5 / Fixed-0.75 is reused from the weight sweep. Completing the matrix is 18 full campaigns.
