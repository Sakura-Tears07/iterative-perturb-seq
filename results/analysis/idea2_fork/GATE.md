FROZEN GATE — do not override from a single 0.01 spike.

CONTINUE dynamic fusion only if MOST runs satisfy:
  n=100: C < 0  (prior-heavy / low w favored)
  n=500: C > 0  (model-heavy / high w favored)
AND stage-average curves are visually distinct (std does not swallow the shift).

STOP Idea 2 if run directions conflict, e.g.
  run1: prior→model, run2: model→prior, run3: mid-weight best.
Verdict in that case: No reproducible state-dependent weight preference.

n=100 C<0: 3/3 runs
n=500 C>0: 0/3 runs
C_100 < C_300 < C_500: 1/3 runs
STOP: No reproducible state-dependent weight preference. Idea 2 closed.
Next: prior reliability / corruption pilot, λ ∈ {0, 0.5, 1}.
