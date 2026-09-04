# Common-state fork (Idea 2 final diagnostic)

From identical Paper equal-mean states, branch only on next-step \(w_{\text{model}}\).

Do not add experiments or retune while the 45 forks run. Do not judge hypothesis from one Pearson spike.

## Layout

```text
/data/zy/iterpert/common_states/paper_run{R}/n{N}/
  labeled_genes.txt
  pool_genes.txt
  gears_checkpoint/{config.pkl,model.pt}
  state_meta.json
  state_metrics.json   ← official P_before
```

## Protocol

```text
same checkpoint → different w → select 100 genes → reveal
→ same retraining seed (--seed 1, --run $run) → P_after
```

`P_before` is the dump value. `eval_after_load` is sanity only.

## Commands

```bash
# Dump already done for runs 1–3, n ∈ {100,300,500}

# 45 one-step forks (failed logs without "Round 1 pearson" are retried)
cd /home/zy/workspace/iterative-perturb-seq
export ITERPERT_DATA_ROOT=/data/zy/iterpert
PARALLEL=1 bash configs/experiments/idea2_fork/run_forks.sh

# After run1/n100 finishes: pipeline sanity only
python scripts/fork_sanity_check.py --run 1 --n 100

# After all 45
python scripts/analyze_common_state_forks.py
```

See [`RESEARCH.md`](../../../RESEARCH.md) for C(S), Jaccard, and the stop/continue gate.
