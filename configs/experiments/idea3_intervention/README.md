# Idea 3c — Detected hard reject (first GPU batch)

Binary reject of RPE1 when Observed KA on \(S_t\) is below the frozen \(\tau\). Protocol: [`PROTOCOL.md`](PROTOCOL.md).

```bash
python scripts/analyze_hard_reject_threshold.py   # calibration only; writes frozen_tau.json
PARALLEL=1 bash configs/experiments/idea3_intervention/run_hard_reject.sh
python scripts/analyze_idea3_intervention.py
```

Nine new campaigns: detected on clean, detected on λ=1 (perm `20260831`), equal λ=1 (same perm). Fig.4 clean equal and oracle-drop runs 1–3 are reused.

Do not retune \(\tau\) after looking at nALC.
