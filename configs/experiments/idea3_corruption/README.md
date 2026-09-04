# Idea 3 — Prior reliability under RPE1 corruption (pilot)

Question: does IterPert equal fusion stay robust when a *strong* prior’s gene identity is scrambled?

```text
K_λ = (1-λ) K_RPE1 + λ P K_RPE1 Pᵀ
P fixed by corruption_seed=20260826
λ ∈ {0, 0.5, 1}
```

## Layers

- **P3-A** two-source: model + RPE1, 50/50
- **P3-B** full IterPert: 8 priors + model, one of them corrupted
- **Oracle-drop**: delete RPE1 (not a deployable method)
- **Model-only**: no priors

## Reuse vs new GPU

| Cell | GPU |
|------|-----|
| Full λ=0 runs 1–3 | reuse Fig.4 IterPert |
| Model+RPE1 λ=0 runs 1–3 | reuse Fig.4c |
| Model-only run 1 | reuse existing `mw1.0` |
| λ∈{0.5,1} two-source and full, oracle-drop, model-only runs 2–3 | **new** |

```bash
cd /home/zy/workspace/iterative-perturb-seq
export ITERPERT_DATA_ROOT=/data/zy/iterpert
PARALLEL=1 bash configs/experiments/idea3_corruption/run_pilot.sh
python scripts/analyze_idea3_corruption.py
```

Gate (frozen): same-seed \(D\)/\(H\), early-budget \(D_n\), mechanism chain. Majority of 3 runs. No λ-monotonicity requirement. See [`RESEARCH.md`](../../../RESEARCH.md).

Queue order: all Gate cells (two-source / full λ=.5/1, oracle-drop) first; model-only last. Model-only is not required to judge A/B/C.

If Gate A: next is **can we detect** a bad prior, not how to reweight it. Do not add extra λ values before the Gate.
