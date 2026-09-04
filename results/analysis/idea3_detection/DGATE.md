# Idea 3b — Detector Audit (frozen before looking at results)

Do not add a 5th/6th signal after seeing plots.
Do not train a classifier.
Do not open GPU.
Do not compare clean-trajectory KA to corrupt-trajectory KA.

## Question

Can we tell that RPE1 is unreliable using only information available at that round?

Legal: labeled perturbations \(S_t\), their observed expression, current GEARS checkpoint, prior kernels, pool identities, history.
Illegal as features: hidden-pool expression, ground-truth full kernel on unlabeled genes, the \(\lambda\) label.
\(\lambda\) is only a retrospective evaluation tag.

## Same-state design

Use Paper equal-fusion dump campaigns (`GEARS_100_4_100`, runs 1–3).

| \(n\) | \(S_t\) source | \(K_{\mathrm{model}}[S,S]\) |
| ---: | --- | --- |
| 100, 200, 300, 400 | round-state `train_genes` (query-time) | train block of saved `base_kernel` |
| 500 | `common_states/paper_run{r}/n500/labeled_genes.txt` | not saved (PM = NA) |

\(3\times 5=15\) common states. Each state is scored at \(\lambda\in\{0,0.5,1\}\) on the **same** \(S_t\), same observed \(K_y[S,S]\), same model Gram. The only change is RPE1 reliability.

\[
K_\lambda=(1-\lambda)K_{\mathrm{RPE1}}+\lambda P K_{\mathrm{RPE1}}P^\top
\]

Primary permutation seed `20260826` (campaign seed). Robustness seeds `{20260826,20260827,20260828,20260829,20260830}`. New perms are generated with `np.random.default_rng(seed).permutation(n)` and saved under `$ITERPERT_DATA_ROOT/idea3/`. Do not overwrite `20260826`.

## Four pre-registered signals

Kernel alignment is IterPert's uncentered formula (scale-invariant):

\[
\mathrm{KA}(A,B)=\frac{\mathrm{tr}(AB)}{\|A\|_F\|B\|_F}
\]

| id | signal | uses observed \(y\) on \(S_t\)? | definition |
| --- | --- | --- | --- |
| obs | Observed KA | yes | \(\mathrm{KA}(K_\lambda[S,S], K_y[S,S])\) |
| pm | Prior–model KA | indirect | \(\mathrm{KA}(K_\lambda[S,S], K_{\mathrm{model}}[S,S])\) |
| cons | Cross-prior consensus | no | mean\(_{j\neq\mathrm{RPE1}}\) \(\mathrm{KA}(K_\lambda[S,S], K_j[S,S])\) |
| loc | Local neighbor consistency | yes | mean overlap of each gene's 10-NN in \(K_\lambda[S,S]\) vs \(K_y[S,S]\) (\(k=\min(10,|S|-1)\)) |

`cons_full` is the same consensus detector on the **full** prior kernels (pool identities are known; no hidden \(y\)). It is Detector 3 evaluated at Round-0 scope, not a 5th detector.

Higher \(z\) = more reliable. Paired difference:

\[
\Delta z_{0,1}(S)=z_{\lambda=0}(S)-z_{\lambda=1}(S)
\]

Do not use hidden outcomes. Do not slice \(K_y\) outside \(S_t\).

## What not to do

- No RandomForest / logistic classifier / accuracy%.
- No extra \(\lambda\) values.
- No adaptive drop/reweight/LLM planner until D-Gate A (or B for hard reject only).
- Selection divergence is not a reliability definition (Idea 2: different batches can be near-equivalent).

## D-Gate (majority-style, frozen)

A state counts as positive for a pair of \(\lambda\) if \(\Delta z>0\).

**D-Gate A — observable reliability exists** (must hold for at least one of the four signals):

1. On seed `20260826`, \(\Delta z_{0,1}>0\) in \(\ge 80\%\) of the 15 states.
2. Every campaign run has \(\Delta z_{0,1}>0\) in \(\ge 80\%\) of that run's states (4/5).
3. Every one of the five permutation seeds has \(\Delta z_{0,1}>0\) in \(\ge 80\%\) of states.
4. Signal does not use hidden outcomes.

PM is judged on the 12 states that have a model Gram; 80% of 12 is \(\ge 10/12\). Other signals use 15.

**D-Gate B — only severe-failure detection:** D-Gate A item 1–3 hold for \(\Delta z_{0,1}\), but \(\Delta z_{0,0.5}>0\) in \(<80\%\) of states on the primary seed. Detector can flag complete semantic mismatch, not moderate misspecification. Future method = hard reject, not continuous weighting.

**D-Gate C — no stable observable:** \(\Delta z_{0,1}\) fails the 80% bar on the primary seed, or fails to replicate across permutation seeds. Stop the adaptive reliability line. Oracle headroom is not a license to train a classifier.

If several signals disagree, report each separately. Adaptive work is allowed only if **at least one** pre-registered signal is A (or B for hard-reject only).

## After D-Gate (not this script)

Hard reject \(z_i<\tau\) vs equal fusion / oracle drop / model-only. Threshold fit on some corruption seeds, test on a held-out seed. No in-sample retuning on the three campaign seeds' \(\lambda=1\) cells.

--- live D-Gate ---

Observed KA                  D-Gate A   Δz(0,1)>0 15/15 (1.00)  median=+0.3248   Δz(0,.5)>0 15/15 (1.00)
    run1: 5/5 (1.00)
    run2: 5/5 (1.00)
    run3: 5/5 (1.00)
    perm seeds Δz(0,1): 20260826:15/15  20260827:15/15  20260828:15/15  20260829:15/15  20260830:15/15

Prior–model KA               D-Gate A   Δz(0,1)>0 12/12 (1.00)  median=+0.3303   Δz(0,.5)>0 12/12 (1.00)
    run1: 4/4 (1.00)
    run2: 4/4 (1.00)
    run3: 4/4 (1.00)
    perm seeds Δz(0,1): 20260826:12/12  20260827:12/12  20260828:12/12  20260829:12/12  20260830:12/12

Cross-prior consensus        D-Gate B   Δz(0,1)>0 12/15 (0.80)  median=+0.0778   Δz(0,.5)>0 0/15 (0.00)
    run1: 4/5 (0.80)
    run2: 4/5 (0.80)
    run3: 4/5 (0.80)
    perm seeds Δz(0,1): 20260826:12/15  20260827:12/15  20260828:15/15  20260829:15/15  20260830:12/15

Local 10-NN overlap          D-Gate A   Δz(0,1)>0 15/15 (1.00)  median=+0.1167   Δz(0,.5)>0 15/15 (1.00)
    run1: 5/5 (1.00)
    run2: 5/5 (1.00)
    run3: 5/5 (1.00)
    perm seeds Δz(0,1): 20260826:15/15  20260827:15/15  20260828:15/15  20260829:15/15  20260830:15/15

Consensus (full kernel)      D-Gate B   Δz(0,1)>0 15/15 (1.00)  median=+0.0465   Δz(0,.5)>0 0/15 (0.00)
    run1: 5/5 (1.00)
    run2: 5/5 (1.00)
    run3: 5/5 (1.00)
    perm seeds Δz(0,1): 20260826:15/15  20260827:15/15  20260828:15/15  20260829:15/15  20260830:15/15

OVERALL: D-Gate A — at least one pre-registered signal has observable reliability.
Next (later): hard-reject with held-out corruption seed. No softmax yet.

Sanity (reproduce logged campaign KA on λ=0, same S):
  run1 n=100 obs λ=0 KA=0.451245  logged=0.451245
  run1 n=100 model-vs-gold KA=0.814593  logged=0.814593
