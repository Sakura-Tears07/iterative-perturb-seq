#!/usr/bin/env bash
# Route A: pause fig4c, wait for ops_A549 to finish, parallel backfill on GPUs 1-4.
#
# Assumes BADGE run=4 is already running on GPU 0 (from serial backfill script).
# Launches BADGE 5-8 on GPUs 1-4, then LCMD 5 on GPU 0 after run 4 exits.
#
# Usage:
#   bash scripts/rerun_fig4_route_a.sh
#   SERIAL_PID=2807369 bash scripts/rerun_fig4_route_a.sh   # optional: stop serial script after run4
#
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
REPRO="$REPO/reproduce_repo"
LOG_ROOT="/data/zy/iterpert/logs/fig4/backfill"
FIG4C_LOG="/data/zy/iterpert/logs/fig4c"
LOCK_FILE="$LOG_ROOT/route_a.lock"
SUMMARY="$LOG_ROOT/route_a_summary.txt"
RUN4_LOG="$LOG_ROOT/BADGE_run4.log"
OPS_DIR="$FIG4C_LOG/ops_A549_kernel"
MAX_RETRIES="${MAX_RETRIES:-2}"
POLL_SEC="${POLL_SEC:-30}"

# Auto-detect serial backfill bash if running
SERIAL_PID="${SERIAL_PID:-}"
RUN4_PID="${RUN4_PID:-}"

PARALLEL_JOBS=(
  "1:BADGE:5"
  "2:BADGE:6"
  "3:BADGE:7"
  "4:BADGE:8"
)
LCMD_JOB="0:LCMD:5"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_ROOT/route_a.log"
}

die() {
  log "ERROR: $*"
  exit 1
}

ensure_env() {
  mkdir -p "$LOG_ROOT"
  if [[ -f "$LOCK_FILE" ]]; then
    die "Lock exists ($LOCK_FILE). Another route-a orchestrator may be running."
  fi
  echo $$ > "$LOCK_FILE"
  trap 'rm -f "$LOCK_FILE"' EXIT

  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate iterpert
  export ITERPERT_DATA_ROOT="${ITERPERT_DATA_ROOT:-/data/zy/iterpert}"
}

pause_fig4c() {
  touch "$FIG4C_LOG/PAUSED"
  log "fig4c PAUSED → node2vec will not auto-start after ops_A549"
}

detect_serial_pids() {
  if [[ -z "$SERIAL_PID" ]]; then
    SERIAL_PID=$(pgrep -f "bash scripts/rerun_fig4_gaps_when_gpu_free.sh" | head -1 || true)
  fi
  if [[ -z "$RUN4_PID" ]]; then
    # Pick the longest-running BADGE run=4 process (exclude dataloader workers spawned later).
    RUN4_PID=$(
      ps -eo pid,etimes,cmd \
        | grep "run.py --seed 1 --run 4" \
        | grep "kernel_strategy BADGE" \
        | grep -v grep \
        | sort -k2 -nr \
        | head -1 \
        | awk '{print $1}'
    )
  fi
  [[ -n "$SERIAL_PID" ]] && log "Serial backfill bash PID=$SERIAL_PID"
  [[ -n "$RUN4_PID" ]] && log "BADGE run4 python PID=$RUN4_PID (elapsed=$(ps -o etimes= -p "$RUN4_PID" 2>/dev/null | tr -d ' ')s)"
}

wait_for_ops_a549() {
  # Only need GPUs 1-4 free for parallel wave; do not block on slowest ops run1.
  if [[ "${SKIP_OPS_WAIT:-0}" == "1" ]]; then
    log "SKIP_OPS_WAIT=1 — proceeding without waiting for ops_A549"
    return 0
  fi
  log "Waiting until GPUs 1-4 are idle (ops_A549 runs 2-5 already done)..."
  while true; do
    local done=0 g all_free=1
    for run in 1 2 3 4 5; do
      if grep -q "Round 5 pearson" "$OPS_DIR/run${run}.log" 2>/dev/null; then
        done=$((done + 1))
      fi
    done
    for g in 1 2 3 4; do
      if ! gpu_is_free "$g"; then all_free=0; break; fi
    done
    log "  ops_A549: ${done}/5 done Round 5 | GPUs 1-4 idle=$([[ $all_free -eq 1 ]] && echo yes || echo no)"
    [[ "$all_free" -eq 1 ]] && break
    sleep "$POLL_SEC"
  done
  log "GPUs 1-4 ready for parallel backfill"
  sleep 5
}

gpu_is_free() {
  local idx=$1
  local mem util
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
  [[ "${mem:-99999}" -lt 3500 && "${util:-100}" -lt 15 ]]
}

wait_for_gpus_free() {
  local gpus=(1 2 3 4) g
  log "Waiting for GPUs 1-4 to become idle..."
  while true; do
    local all_free=1
    for g in "${gpus[@]}"; do
      if ! gpu_is_free "$g"; then
        all_free=0
        log "  GPU $g busy (mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')MiB util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$g" | tr -d ' ')%)"
        break
      fi
    done
    [[ "$all_free" -eq 1 ]] && break
    sleep "$POLL_SEC"
  done
  log "GPUs 1-4 idle"
}

verify_metrics() {
  local method=$1 run=$2
  "$REPO/scripts/verify_fig4_run.py" --method "$method" --run "$run"
}

launch_job() {
  local gpu=$1 method=$2 run=$3
  local log_file="$LOG_ROOT/${method}_run${run}.log"
  log "LAUNCH $method run=$run on GPU=$gpu → $log_file"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u run.py \
    --seed 1 --run "$run" --epoch_per_cycle 20 --device cuda:0 \
    --n_init_labeled 100 --n_query 100 --n_round 5 --batch_size 256 \
    --dataset_name replogle_k562_essential_1000hvg --retrain --model_name GEARS \
    --simple_loss --fix_evaluation \
    --strategy_name kernel_based_active_learning \
    --kernel_strategy "$method" --base_kernel cross_gene_out \
    > "$log_file" 2>&1 &
  echo $!
}

run_job_blocking() {
  local gpu=$1 method=$2 run=$3
  local attempt log_file
  log_file="$LOG_ROOT/${method}_run${run}.log"

  for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    log "START $method run=$run (attempt $attempt/$MAX_RETRIES) on GPU=$gpu"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u run.py \
      --seed 1 --run "$run" --epoch_per_cycle 20 --device cuda:0 \
      --n_init_labeled 100 --n_query 100 --n_round 5 --batch_size 256 \
      --dataset_name replogle_k562_essential_1000hvg --retrain --model_name GEARS \
      --simple_loss --fix_evaluation \
      --strategy_name kernel_based_active_learning \
      --kernel_strategy "$method" --base_kernel cross_gene_out \
      > "$log_file" 2>&1
    local ec=$?
    set -e

    if [[ $ec -ne 0 ]]; then
      log "FAIL $method run=$run exit=$ec"
      continue
    fi
    if verify_metrics "$method" "$run"; then
      log "OK $method run=$run"
      echo "OK $method run=$run" >> "$SUMMARY"
      return 0
    fi
    log "WARN $method run=$run metrics incomplete; retrying..."
  done
  echo "FAIL $method run=$run" >> "$SUMMARY"
  return 1
}

stop_serial_after_run4() {
  [[ -n "$SERIAL_PID" ]] || return 0
  if ! kill -0 "$SERIAL_PID" 2>/dev/null; then
    log "Serial backfill bash already stopped (PID=$SERIAL_PID)"
    return 0
  fi
  log "Watcher: stop serial script after BADGE run4 completes (avoid duplicate run5-8)..."
  while true; do
    if grep -q "Round 5 pearson" "$RUN4_LOG" 2>/dev/null; then
      log "BADGE run4 finished — stopping serial bash PID=$SERIAL_PID"
      kill "$SERIAL_PID" 2>/dev/null || true
      sleep 2
      return 0
    fi
    if [[ -n "$RUN4_PID" ]] && ! kill -0 "$RUN4_PID" 2>/dev/null; then
      if grep -q "Round 5 pearson" "$RUN4_LOG" 2>/dev/null; then
        log "BADGE run4 process exited with complete metrics"
      else
        log "WARN: BADGE run4 PID=$RUN4_PID exited before Round 5 pearson; keep waiting on log"
        RUN4_PID=""
        sleep 20
        continue
      fi
      kill "$SERIAL_PID" 2>/dev/null || true
      return 0
    fi
    sleep 20
  done
}

wait_for_pid() {
  local pid=$1 label=$2
  log "Waiting for $label PID=$pid ..."
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  log "$label done"
}

postprocess() {
  log "Reorganizing results..."
  cd "$REPO"
  python reproduce_repo/reorganize_results.py
  python reproduce_repo/reorganize_results_structure.py
  python reproduce_repo/aggregate_fig4_results.py
  if [[ -f "$REPO/scripts/analyze_existing_nalc.py" ]]; then
    log "Refreshing E0 analysis..."
    python "$REPO/scripts/analyze_existing_nalc.py" || log "WARN: analyze_existing_nalc.py failed"
  fi
  log "Postprocess done."
}

main() {
  ensure_env
  cd "$REPRO"
  : > "$SUMMARY"

  pause_fig4c
  detect_serial_pids

  # Stop serial script from launching run5-8 after run4 (run in background)
  stop_serial_after_run4 &
  local watcher_pid=$!

  wait_for_ops_a549
  wait_for_gpus_free

  log "=== Route A parallel wave 1: BADGE 5-8 on GPUs 1-4 ==="
  declare -A job_pids=()
  for spec in "${PARALLEL_JOBS[@]}"; do
    IFS=: read -r gpu method run <<< "$spec"
    job_pids["${method}_${run}"]=$(launch_job "$gpu" "$method" "$run")
    log "  ${method} run=$run PID=${job_pids[${method}_${run}]}"
  done

  # Wait for BADGE run4 to finish on GPU 0, then launch LCMD
  if [[ -n "$RUN4_PID" ]] && kill -0 "$RUN4_PID" 2>/dev/null; then
    wait_for_pid "$RUN4_PID" "BADGE run4"
  else
    log "Waiting for BADGE run4 log to show Round 5 pearson..."
    while ! grep -q "Round 5 pearson" "$RUN4_LOG" 2>/dev/null; do sleep 30; done
  fi
  wait "$watcher_pid" 2>/dev/null || true

  if verify_metrics BADGE 4 2>/dev/null; then
    echo "OK BADGE run=4" >> "$SUMMARY"
    log "OK BADGE run=4 (existing)"
  else
    log "WARN: BADGE run=4 metrics not yet verified"
  fi

  log "=== Route A: LCMD run=5 on GPU 0 ==="
  run_job_blocking 0 LCMD 5 || true

  log "=== Waiting for parallel BADGE 5-8 ==="
  local failed=0
  for spec in "${PARALLEL_JOBS[@]}"; do
    IFS=: read -r gpu method run <<< "$spec"
    local pid="${job_pids[${method}_${run}]}"
    wait_for_pid "$pid" "$method run=$run"
    if verify_metrics "$method" "$run"; then
      echo "OK $method run=$run" >> "$SUMMARY"
      log "OK $method run=$run"
    else
      echo "FAIL $method run=$run" >> "$SUMMARY"
      log "FAIL $method run=$run (metrics incomplete)"
      failed=$((failed + 1))
    fi
  done

  postprocess
  log "=== Route A finished. failed=$failed ==="
  cat "$SUMMARY" | tee -a "$LOG_ROOT/route_a.log"
  [[ "$failed" -eq 0 ]]
}

main "$@"
