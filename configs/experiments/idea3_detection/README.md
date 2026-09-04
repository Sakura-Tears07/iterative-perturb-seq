# Idea 3b — Same-state detector audit (CPU)

Corruption Gate A is done. This stage asks whether unreliability is **observable** from legal inference-time information.

```bash
python scripts/analyze_prior_reliability_detection.py
```

No GPU. No classifier. No extra λ. Protocol is frozen in [`PROTOCOL.md`](PROTOCOL.md) **before** looking at scores.

Outputs: `results/analysis/idea3_detection/`.
