#!/usr/bin/env bash
# Wait for the fig4 MAIN scheduler to finish, then launch the fig4c single-prior suite.
set -u
cd /home/lihaoran/ai4s/iterative-perturb-seq
PY=/home/lihaoran/miniconda3/envs/iterpert_env/bin/python
LOG=results/fig4/_scripts/logs/fig4c_supervisor.log
echo "[$(date '+%F %T')] fig4c supervisor waiting for main suite..." >> "$LOG"
while pgrep -f "run_fig4_local.py --only main" >/dev/null 2>&1; do sleep 180; done
echo "[$(date '+%F %T')] main suite finished -> launching fig4c" >> "$LOG"
setsid nohup "$PY" -u scripts/run_fig4_local.py --only fig4c \
  >> results/fig4/_scripts/logs/scheduler_fig4c.log 2>&1 < /dev/null &
echo "[$(date '+%F %T')] fig4c scheduler spawned" >> "$LOG"
