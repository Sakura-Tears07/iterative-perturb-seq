# Research program (`thoughts`)

## Decision (2026-08-26): Idea 2 sealed

Equal fusion is hard to beat in the **clean** setting. Hand schedules failed. Common-state forks failed.

**Negative result (clean):** fusion weight *does* change which genes are acquired (\(J(w=0,w=1)\approx 0.03\)), but there is **no reproducible stage-dependent preference**. Round / labeled-set size is the wrong axis for setting model-vs-prior weight.

| Evidence | Result |
|----------|--------|
| 5-seed schedules | Paper equal nALC **0.252** best; Early-Prior 0.245 worst |
| Alignment vs weight | labeled-set KA stays high; not a reliability signal for \(w_t\) |
| 45 common-state forks | \(C_{100}<0\) in 3/3; \(C_{500}>0\) in **0/3**; all nine \(C(S)<0\) |

Details: `results/analysis/idea2_schedule/`, `results/analysis/idea2_fork/`. Fusion math remains freeze **`a06d119`**. Do not add schedules, extra seeds, or more \(w\) values.

---

## Now: Idea 3 — prior reliability under corruption

### Decision (2026-08-26): Gate A passed

Complete semantic corruption of a strong RPE1 prior reproducibly degrades both two-source and full multimodal IterPert. Oracle removal recovers part of the lost nALC in 2/3 seeds.

| run | \(D_{\text{two}}(1)\) | \(D_{\text{full}}(1)\) | \(H(1)\) | \(H/D_{\text{full}}\) |
| --- | ---: | ---: | ---: | ---: |
| 1 | +0.0229 | +0.0178 | +0.0030 | 0.17 |
| 2 | +0.0289 | +0.0295 | +0.0186 | 0.63 |
| 3 | +0.0281 | +0.0165 | **−0.0027** | −0.16 |

- \(D_{\text{two}}(1)>0\): **3/3**
- \(D_{\text{full}}(1)>0\): **3/3** — averaging does not fully absorb one bad prior
- \(H(1)>0\): **2/3** — majority still A; run3 reverse is small on nALC

Damage direction is stable. Oracle recovery is heterogeneous. Gate A means continue research; **3 seeds are not enough for a method claim** — expand seeds before writing one.

## Now: Idea 3b — can unreliability be detected from observed feedback?

### Decision (2026-08-26): D-Gate A passed (CPU same-state audit)

At least one frozen signal separates clean vs λ=1 RPE1 on the same experimental state, across runs and five permutation seeds.

| signal | D-Gate | \(\Delta z_{0,1}>0\) | \(\Delta z_{0,0.5}>0\) |
| --- | --- | ---: | ---: |
| Observed KA | **A** | 15/15, all 5 perms | 15/15 |
| Prior–model KA | **A** | 12/12 (n=500 has no saved Gram) | 12/12 |
| Local 10-NN | **A** | 15/15, all 5 perms | 15/15 |
| Cross-prior consensus on \(S_t\) | **B** | 12/15 (fails only at n=100) | 0/15 |
| Consensus on full kernel | **B** | 15/15 | 0/15 |

Primary-seed median \(z\): observed KA clean / .5 / 1 ≈ **0.55 / 0.48 / 0.22**. Graded, little overlap at λ=1.

Caveats: n=100 labeled set is identical across the three dump runs (`--seed 1`); all five perms are the same *mechanism* (gene shuffle). Same-state detection is not yet a deployed drop rule. λ=0.5 is detectable while campaign nALC barely moved — do **not** treat every KA drop as a reason to reject.

**Next (not now):** hard reject \(z_i<\tau\) with threshold fit on some perm seeds and test on a held-out seed. No softmax. No GPU campaigns. No classifier.

Corruption Gate A is closed. Current question:

> When heterogeneous biological priors contain unreliable evidence, can we identify it using only information already obtained in the campaign?

**Not** extra \(\lambda\). **Not** softmax / adaptive drop. **Not** remaining model-only GPU jobs. **Not** a classifier on 15 states.

Same-state CPU audit (like Idea 2 forks): 3 Paper runs \(\times\) \(\{100,200,300,400,500\}\) labeled sets. On each \(S_t\), score clean / \(\lambda=0.5\) / \(\lambda=1\) RPE1. Then repeat on 5 permutation seeds.

Four frozen signals: observed KA, prior–model KA, cross-prior consensus, local 10-NN consistency. Protocol: `configs/experiments/idea3_detection/PROTOCOL.md`. Parser: `python scripts/analyze_prior_reliability_detection.py`.

Mechanism (revised; not a monotone causal chain):

\[
\text{prior corruption}
\rightarrow
\{\text{observed biological inconsistency},\ \text{acquisition trajectory shift}\}
\rightarrow
\text{campaign-level performance drop}.
\]

Inconsistency has a visible signal. Trajectory shift has run1 Jaccard; run2/3 selection logs were incomplete. Performance drop is 3/3. Detector asks whether representation disagrees with observed biology, not whether selection changed.

### Pilot design (RPE1 only)

RPE1 is a strong, interpretable related-cell Perturb-seq prior. If scrambling it does nothing, a large corruption sweep is not worth it.

\[
K_\lambda=(1-\lambda)K_{\mathrm{RPE1}}+\lambda P K_{\mathrm{RPE1}}P^\top
\]

\(P\) is a **fixed** gene permutation (`corruption_seed=20260826`). \(\lambda\in\{0,0.5,1\}\). This breaks **gene correspondence**, not matrix spectrum.

Campaign seeds 1–3 only. Do not mix a second corruption realization into the pilot.

### Methods

| Tag | What |
|-----|------|
| P3-A two-source | model + \(K_\lambda\), 50/50 |
| P3-B full | original 8 priors + model; only RPE1 replaced by \(K_\lambda\) |
| Oracle-drop | delete RPE1, equal-fuse the rest + model (**diagnostic, not a method**) |
| Model-only | no priors (floor) |

Oracle-drop splits two stories. Compare it to **corrupted full**, not to clean IterPert:

\[
H(1,r)=\mathrm{nALC}(\text{oracle-drop},r)-\mathrm{nALC}(\text{full }\lambda=1,r)
\]

If \(H>0\), deleting the bad prior recovers damage. That is the headroom for detection. Beating clean full is not the question (dropping RPE1 also changes the number of kernels in the average).

### Frozen analysis

Parser: `python scripts/analyze_idea3_corruption.py`

**Primary evidence is same-seed pairing**, not cross-seed means, not argmax @600:

| run | \(D_{\text{two}}(0.5)\) | \(D_{\text{two}}(1)\) | \(D_{\text{full}}(0.5)\) | \(D_{\text{full}}(1)\) | \(H(1)\) |
| --- | ---: | ---: | ---: | ---: | ---: |

\(D>0\) = corruption hurt that seed. Do **not** require \(\lambda\)-monotonicity. Run1 already had two-source \(\lambda=0.5\) beat clean; if that repeats it can be geometry/diversity, not a bug. The gate asks whether **\(\lambda=1\)** is a reproducible injury.

Also report early-budget paired damage \(D_n=P_n^{\text{clean}}-P_n^{\text{corrupt}}\) at \(n\in\{200,300,400,600\}\). A shrinking \(D_n\) is the intended IterPert story: bad prior hurts most when target evidence is scarce.

Mechanism (campaign, not detector): corruption can change observed consistency and/or the acquisition path; those can produce campaign-level damage. Selection change alone is not damage (Idea 2).

### Gate (after 3 runs; majority = ≥2/3)

- **A** — `full λ=1` damaged in most seeds **and** `H(1)>0` vs corrupted full in most seeds. Then study **detection** (can observables separate clean vs corrupted RPE1?). Still no reweight formula, still no LLM planner.
- **A/B intermediate (continue)** — early @200/@300 damaged in most seeds even if @600 recovers. Experimental efficiency is the objective; extra rounds to recover from a bad prior are real cost.
- **B** — two-source λ=1 damaged, full λ=1 not. Averaging is the robustness; next increase the *fraction* of corrupted priors.
- **C** — even two-source λ=1 has no stable degradation. Stop this line. Do not invent a reliability method.

**Gate A is the frozen majority verdict.** Next is Idea 3b detection (`configs/experiments/idea3_detection/`), not reweighting. Do not add λ=.75/.9. Do not spend GPU on remaining model-only before the detector diagnostic.

### Reuse vs new GPU

Reuse Fig.4 IterPert runs 1–3 (full \(\lambda=0\)), Fig.4c RPE1 runs 1–3 (two-source \(\lambda=0\)), and existing model-only run 1. New: \(\lambda\in\{0.5,1\}\) on both settings, oracle-drop, model-only runs 2–3 (**17 campaigns**).

```bash
PARALLEL=1 bash configs/experiments/idea3_corruption/run_pilot.sh
python scripts/analyze_idea3_corruption.py
```

## Now: Idea 3c — does hard reject recover the campaign?

### Threshold (frozen, calibration perms only)

Observed KA, drop RPE1 if \(z<\tau\). \(\tau^\star=0.451245\) under FPR\le10% (actual cal FPR=0, TPR=1). Validation and offline holdout were **not** used to move \(\tau\). GPU test permutation is **`20260831`**.

Do not retune \(\tau\) on nALC.

Parser: `python scripts/analyze_hard_reject_threshold.py` then `PARALLEL=1 bash configs/experiments/idea3_intervention/run_hard_reject.sh`.

**Not this week:** LLM planner, RL, TypiClust fusion, more Idea-2 weights.

