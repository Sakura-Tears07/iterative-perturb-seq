# Research program (`thoughts`)

Current question: **should IterPert use the same model/prior fusion weight every round?**

Paper default is equal-weight mean over 8 priors + 1 model kernel, so \(w_{\text{model}} \approx 1/9\). That number never changes.

---

## Line A (already done)

Fig.4 / Fig.4c / E0 are the baseline. IterPert's advantage is **early-budget learning speed**, not a large gap at 600 genes. Fig.6 GW is deferred.

## Line B (this week)

Sweep a **fixed** \(w_{\text{model}}\) for the whole campaign:

```text
w_model ∈ {0, 0.25, 0.5, 0.75, 1.0}
remaining mass split equally across the 8 priors
```

1 seed (`run=1`). Not a new algorithm — a diagnostic.

```bash
PARALLEL=1 bash configs/experiments/idea2_weight_sweep/run_sweep.sh
python scripts/analyze_weight_sweep.py
```

### Result (Pearson Δ, seed 1)

| n labeled | w=0 | w=0.25 | w=0.5 | w=0.75 | w=1.0 | best w |
|----------:|----:|-------:|------:|-------:|------:|-------:|
| 100 (init, no AL) | 0.107 | 0.103 | 0.107 | 0.110 | 0.108 | noise |
| **200** | 0.232 | 0.232 | **0.232** | 0.220 | **0.176** | 0–0.5 |
| **300** | 0.269 | 0.269 | **0.275** | 0.258 | 0.229 | 0.5 |
| 400 | 0.256 | 0.253 | 0.259 | **0.303** | 0.210 | 0.75 |
| 500 | 0.269 | 0.271 | 0.264 | **0.287** | 0.241 | 0.75 |
| 600 | 0.285 | 0.273 | 0.283 | **0.290** | 0.276 | 0.75 |

Fig.4 IterPert, 10-run mean: 200 → 0.234, 400 → 0.273, 600 → 0.284.

### What this says

1. **Priors matter early.** \(w=1\) (model kernel only) collapses at 200 (0.176 vs 0.232). Throwing priors away is not a late-stage-only mistake.
2. **Early (200–300) wants a modest model weight.** \(w \in \{0, 0.25, 0.5\}\) are tied; \(w=0.75\) already hurts at 200. Paper equal-mean (\(\approx 0.11\)) sits in the good region.
3. **Later rounds prefer more model.** From 400 onward \(w=0.75\) wins. The 400-gene 0.303 is **above** IterPert's 10-run max (0.293) — treat as a possible lucky trajectory until more seeds exist.
4. **The pattern is not monotone \(0 \to 1\).** It is: trust priors (or a balanced mix) first, then raise \(w_{\text{model}}\), but never all the way to 1.

That is enough to justify a **round-dependent schedule**, not enough to claim a new method.

### Next (one experiment)

Run the schedule the table suggests, against the two best fixed weights, with more than one seed:

| condition | what it is |
|-----------|------------|
| paper equal mean | `--model_weight -1` (default) |
| fixed 0.5 | `--model_weight 0.5` |
| fixed 0.75 | `--model_weight 0.75` |
| `early_prior` | `--weight_schedule early_prior` (0.2 → 0.8) |

Decision: if the schedule beats the best *fixed* \(w\) on nALC **and** the 400-gene spike of \(w=0.75\) replicates, keep going. If not, stop — the sweep was 1-seed noise.

Do not add a new selection rule, alignment fusion, or planner until this is resolved.
