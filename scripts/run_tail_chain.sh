#!/usr/bin/env bash
# Tail chain: wait for the verif scheduler to finish, then ablation -> fig4c.
set -u
cd /home/lihaoran/ai4s/iterative-perturb-seq
PY=/home/lihaoran/miniconda3/envs/iterpert_env/bin/python
LOG=results/fig4/_scripts/logs/tail_chain.log
echo "[$(date '+%F %T')] tail chain waiting for verif..." >> "$LOG"
while pgrep -f "run_fig4_local.py --only verif" >/dev/null 2>&1; do sleep 180; done
for GROUP in ablation fig4c; do
  echo "[$(date '+%F %T')] launching $GROUP" >> "$LOG"
  setsid nohup "$PY" -u scripts/run_fig4_local.py --only "$GROUP" \
    >> "results/fig4/_scripts/logs/scheduler_${GROUP}.log" 2>&1 < /dev/null &
  CHILD=$!
  while kill -0 $CHILD 2>/dev/null; do sleep 180; done
done
echo "[$(date '+%F %T')] tail chain complete" >> "$LOG"
