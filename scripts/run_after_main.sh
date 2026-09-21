#!/usr/bin/env bash
# Chain after the fig4 MAIN scheduler: ablation (prior-only) -> fig4c (8 priors).
set -u
cd /home/lihaoran/ai4s/iterative-perturb-seq
PY=/home/lihaoran/miniconda3/envs/iterpert_env/bin/python
LOG=results/fig4/_scripts/logs/chain_supervisor.log
echo "[$(date '+%F %T')] chain supervisor waiting for main suite..." >> "$LOG"
while pgrep -f "run_fig4_local.py --only main" >/dev/null 2>&1; do sleep 180; done
for GROUP in ablation verif fig4c; do
  echo "[$(date '+%F %T')] launching $GROUP" >> "$LOG"
  setsid nohup "$PY" -u scripts/run_fig4_local.py --only "$GROUP" \
    >> "results/fig4/_scripts/logs/scheduler_${GROUP}.log" 2>&1 < /dev/null &
  CHILD=$!
  # wait for that group's scheduler to finish before starting the next
  while kill -0 $CHILD 2>/dev/null; do sleep 180; done
done
echo "[$(date '+%F %T')] chain complete (main + ablation + fig4c)" >> "$LOG"
