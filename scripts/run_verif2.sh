#!/usr/bin/env bash
set -u
cd /home/lihaoran/ai4s/iterative-perturb-seq
PY=/home/lihaoran/miniconda3/envs/iterpert_env/bin/python
LOG=results/fig4/_scripts/logs/verif2_supervisor.log
echo "[$(date '+%F %T')] verif2 supervisor waiting for fig4c..." >> "$LOG"
while pgrep -f "run_fig4_local.py --only fig4c" >/dev/null 2>&1; do sleep 180; done
echo "[$(date '+%F %T')] launching verif2" >> "$LOG"
setsid nohup "$PY" -u scripts/run_fig4_local.py --only verif2 \
  >> "results/fig4/_scripts/logs/scheduler_verif2.log" 2>&1 < /dev/null &
