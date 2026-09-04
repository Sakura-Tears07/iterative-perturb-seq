# Idea 3c — Hard-reject intervention (frozen before choosing τ)

Do not tune τ on test campaigns. Do not soft-weight. Do not combine detectors. Do not use LLM.

## Question

After detecting that a prior is unreliable, does the simplest intervention — binary reject — recover campaign performance without hurting clean fusion?

## Detector (v1)

**Observed KA only.** Prior–model KA and local 10-NN are reserved for later detector ablations.

\[
z_{\mathrm{RPE1}}(t)=\mathrm{KA}(K_{\mathrm{RPE1}}[S_t,S_t], K_y[S_t,S_t])
\]

Action is **RPE1-only** (this experiment only corrupts RPE1; a general all-prior filter is a later ablation):

```text
if round >= 1 and z_RPE1 < τ:  drop RPE1 from this round's equal fusion
else:                          keep RPE1
```

Detect from the first acquisition (n=100). Remaining kernels (model + 7 other priors) stay equal-mean (`mean_new`). No continuous weights.

## Held-out split (frozen)

Permutation seeds already used in the D-Gate audit must not be the GPU test seed.

| Role | Permutation seeds | What it is used for |
| --- | --- | --- |
| Calibration | `20260826`, `20260827`, `20260828` | choose τ |
| Validation | `20260829` | report TPR/FPR; **do not retune** |
| Offline holdout | `20260830` | second TPR/FPR check; **do not retune** |
| GPU test | `20260831` | never used in D-Gate or τ selection |

Same-state scores come from Paper dump runs 1–3 (Idea 3b). Campaign **performance** for I-Gate uses new GPU jobs with perm `20260831`. Do not report recovery on the Gate A λ=1 campaigns (perm `20260826`).

Clean equal fusion (Fig.4 runs 1–3) and oracle-drop runs 1–3 may be reused: they do not depend on the permutation.

## How τ is chosen (calibration only)

Observations: each `(run, n_labeled, perm)` on the 15 frozen states, signal=`obs`.

\[
\mathrm{TPR}_{bad}=P(z<\tau\mid\lambda=1),\qquad
\mathrm{FPR}_{clean}=P(z<\tau\mid\lambda=0)
\]

Constraint, then sensitivity:

1. Primary: \(\mathrm{FPR}_{clean}\le 0.10\) on **calibration** perms.
2. \(\tau^\star=\max\{\tau: \mathrm{FPR}_{cal}(\tau)\le 0.10\}\) (most sensitive detector inside the FPR budget).
3. If that set is empty, fallback FPR budget **0.20**, same rule.
4. If still empty: **do not launch GPU**. Intervention is not feasible as a conservative safety detector.

λ=0.5 is **not** used to pick τ. Report \(P(z<\tau\mid\lambda=0.5)\) as a diagnostic (moderate-misspecification reject rate).

Do not maximize accuracy. Prefer missing some bad priors over mass-deleting clean RPE1.

After τ is written to `frozen_tau.json`, it is frozen. Validation/holdout FPR may be worse; do not move τ.

## First GPU batch (only after τ is frozen)

| Method | clean RPE1 | λ=1, perm `20260831` |
| --- | --- | --- |
| Equal fusion | reuse Fig.4 runs 1–3 | **new** runs 1–3 |
| Oracle drop | reuse oracle-drop runs 1–3 | same reuse (perm-independent) |
| Detected hard reject | **new** runs 1–3 | **new** runs 1–3 |

Nine new GPU campaigns. Detected must run on **both** clean and corrupt.

Paired by campaign `run`.

## Metrics (I-Gate inputs)

Robustness (λ=1, perm `20260831`):

\[
R_{\mathrm{recover}}(r)=\mathrm{nALC}_{detected}(r)-\mathrm{nALC}_{equal\ corrupt}(r)
\]

\[
\eta(r)=\frac{\mathrm{nALC}_{detected}(r)-\mathrm{nALC}_{equal\ corrupt}(r)}{\mathrm{nALC}_{oracle}(r)-\mathrm{nALC}_{equal\ corrupt}(r)}
\]

Clean cost (λ=0):

\[
C_{\mathrm{clean}}(r)=\mathrm{nALC}_{equal\ clean}(r)-\mathrm{nALC}_{detected\ clean}(r)
\]

## I-Gate (judge only on the GPU test perm / these runs)

**I-Gate A — hard reject works**

- \(R_{\mathrm{recover}}>0\) in ≥2/3 test runs
- \(C_{\mathrm{clean}}\) has no stable loss: not \(C>0\) in ≥2/3, and mean \(C\) is small vs the corrupt gain
- \(\eta\) is positive in ≥2/3 (moves toward oracle; η may exceed 1)

Then a later ablation may compare other detectors / soft downweight. Still no LLM.

**I-Gate B — detection works, intervention unstable**

Detector fires, but \(R_{\mathrm{recover}}\) is mixed (not ≥2/3) while clean cost stays small.

Then study how to *use* reliability (soft downweight), not a planner.

**I-Gate C — clean cost too large or no recovery**

\(C_{\mathrm{clean}}>0\) in ≥2/3 **or** no stable \(R_{\mathrm{recover}}\).

Stop the intervention method line. Keep detection as a mechanism result.

## Not this batch

Extra λ, softmax fusion, combined detectors, campaign runs 4–7, model-only, LLM planner.


--- live I-Gate ---

frozen tau=0.4512449421097154 test_perm=20260831

 run  R_recover       eta  C_clean  nALC_equal_clean  nALC_detected_clean  nALC_equal_l1  nALC_detected_l1  nALC_oracle
   1   0.001671 -0.216370 0.004618          0.248812             0.244194       0.241657          0.243327     0.233936
   2  -0.004177 -3.207658 0.000092          0.255887             0.255795       0.243632          0.239456     0.244934
   3  -0.007097  8.931067 0.024347          0.249711             0.225365       0.231306          0.224210     0.230512

R_recover>0: 1/3
C_clean>0 (detected worse than clean equal): 3/3  mean C=+0.0097
eta>0: 1/3  mean R=-0.0032
I-GATE C: clean cost too large or no recovery. Stop the intervention method line.
