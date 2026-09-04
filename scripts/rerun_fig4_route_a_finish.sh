#!/usr/bin/env bash
# Finish Route A after parallel BADGE 5-8 already launched:
#   wait BADGE run4 → LCMD run5 → wait 5-8 → postprocess
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
REPRO="$REPO/reproduce_repo"
LOG_ROOT="/data/zy/iterpert/logs/fig4/backfill"
RUN4_LOG="$LOG_ROOT/BADGE_run4.log"
SUMMARY="$LOG_ROOT/route_a_summary.txt"
RUN4_PID="${RUN4_PID:-2807530}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_ROOT/route_a.log"; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iterpert
export ITERPERT_DATA_ROOT="${ITERPERT_DATA_ROOT:-/data/zy/iterpert}"
cd "$REPRO"

wait_pid() {
  local pid=$1 label=$2
  log "Waiting $label PID=$pid ..."
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  log "$label done"
}

verify_metrics() {
  "$REPO/scripts/verify_fig4_run.py" --method "$1" --run "$2"
}

run_lcmd() {
  local attempt log_file="$LOG_ROOT/LCMD_run5.log"
  for ((attempt=1; attempt<=2; attempt++)); do
    log "START LCMD run=5 on GPU=0 (attempt $attempt)"
    CUDA_VISIBLE_DEVICES=0 python -u run.py \
      --seed 1 --run 5 --epoch_per_cycle 20 --device cuda:0 \
      --n_init_labeled 100 --n_query 100 --n_round 5 --batch_size 256 \
      --dataset_name replogle_k562_essential_1000hvg --retrain --model_name GEARS \
      --simple_loss --fix_evaluation \
      --strategy_name kernel_based_active_learning \
      --kernel_strategy LCMD --base_kernel cross_gene_out \
      > "$log_file" 2>&1 && verify_metrics LCMD 5 && return 0
    log "WARN LCMD run=5 attempt $attempt failed"
  done
  return 1
}

# Wait for BADGE run4
if [[ -n "$RUN4_PID" ]] && kill -0 "$RUN4_PID" 2>/dev/null; then
  wait_pid "$RUN4_PID" "BADGE run4"
else
  log "Waiting for BADGE run4 Round 5 pearson in log..."
  while ! grep -q "Round 5 pearson" "$RUN4_LOG" 2>/dev/null; do sleep 30; done
fi
verify_metrics BADGE 4 && echo "OK BADGE run=4" >> "$SUMMARY"

run_lcmd && echo "OK LCMD run=5" >> "$SUMMARY" || echo "FAIL LCMD run=5" >> "$SUMMARY"

for spec in "BADGE:5" "BADGE:6" "BADGE:7" "BADGE:8"; do
  IFS=: read -r method run <<< "$spec"
  logfile="$LOG_ROOT/${method}_run${run}.log"
  while pgrep -f "run.py --seed 1 --run ${run}.*kernel_strategy ${method}" >/dev/null 2>&1; do sleep 30; done
  if verify_metrics "$method" "$run"; then
    echo "OK $method run=$run" >> "$SUMMARY"
    log "OK $method run=$run"
  else
    echo "FAIL $method run=$run" >> "$SUMMARY"
    log "FAIL $method run=$run"
  fi
done

log "Postprocess..."
cd "$REPO"
python reproduce_repo/reorganize_results.py
python reproduce_repo/reorganize_results_structure.py
python reproduce_repo/aggregate_fig4_results.py
[[ -f "$REPO/scripts/analyze_existing_nalc.py" ]] && python "$REPO/scripts/analyze_existing_nalc.py" || true
log "=== Route A finish complete ==="
cat "$SUMMARY" | tee -a "$LOG_ROOT/route_a.log"
