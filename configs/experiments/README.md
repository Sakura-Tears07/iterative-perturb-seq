# Experiment configs (`thoughts`)

Phase 0 fusion freeze: commit `a06d119`.

| Directory | Status |
|-----------|--------|
| `idea2_weight_sweep/` | Phase 0 diagnostic — done |
| `idea2_schedule/` | 5-seed schedule — **negative, frozen** |
| `idea2_fork/` | Common-state fork — **negative, frozen** |
| `idea3_corruption/` | Gate A **passed** — RPE1 λ=1 hurts two-source and full fusion |
| `idea3_detection/` | D-Gate A — same-state detector audit |
| `idea3_intervention/` | **Current:** Observed-KA hard reject, held-out perm `20260831` |
| `p1_pilot/` | Superseded by `idea3_corruption/` |

```bash
PARALLEL=1 bash configs/experiments/idea3_corruption/run_pilot.sh
python scripts/analyze_idea3_corruption.py
python scripts/analyze_prior_reliability_detection.py
python scripts/analyze_hard_reject_threshold.py
PARALLEL=1 bash configs/experiments/idea3_intervention/run_hard_reject.sh
python scripts/analyze_idea3_intervention.py
```
