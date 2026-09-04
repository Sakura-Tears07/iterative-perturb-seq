FROZEN PROTOCOL (do not retune after looking at curves)

Primary table (same seed, nALC):
  run | D_two(0.5) | D_two(1) | D_full(0.5) | D_full(1) | H(1)

D > 0 means corruption hurt relative to the clean sibling.
H(1) = nALC(oracle-drop) − nALC(full λ=1).
Oracle is compared to corrupted full, not to clean IterPert.

Early-budget damage, same pairing:
  D_n(λ, r) = P_n(clean, r) − P_n(corrupt, r)
  n ∈ {200, 300, 400, 600}

Do NOT require λ-monotonicity. λ=0.5 beating clean is allowed.
The gate asks whether λ=1 is a *reproducible* injury.

Mechanism chain, per round:
  λ → RPE1 alignment(S_t) → J(S_λ, S_clean) → ΔP
  Three interpretations:
    1. alignment ↓, genes change, Pearson ↓  → bad prior misleads acquisition
    2. genes change, Pearson ~unchanged       → many near-equivalent batches
    3. alignment ↓, genes barely change       → RPE1 diluted by the other priors

Gate (majority = ≥2 of 3 runs). Judge only when all new cells have 3 runs.

  A  full λ=1 damaged in most seeds AND H(1)>0 in most seeds
     → reliability estimation has headroom.
     A/B intermediate also counts as continue: early (@200/@300) damaged
     in most seeds even if @600 recovers. Experimental efficiency is the goal.

  B  two-source λ=1 damaged in most seeds, full λ=1 not
     → averaging is the robustness; next corrupt a *fraction* of priors.

  C  two-source λ=1 has no stable degradation
     → stop this line. Do not invent a reliability method.

No adaptive drop / reweight / LLM planner until A or B.
If A: next question is detection (can observables separate λ=0 vs 1),
not a weighting formula. Gate ≠ publication-ready; 3 seeds decide
whether to continue, not whether to write a method claim.
Do not design from a single-run @600 spike.
