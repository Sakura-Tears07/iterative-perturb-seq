#!/usr/bin/env bash
# Backfill incomplete Fig.4 Essential 1K baseline runs (dev-side reproduction).
#
# Targets:
#   BADGE runs 4, 5, 6, 7, 8  (4/6/8 missing; 5/7 have incomplete/corrupt metrics)
#   LCMD  run 5               (missing)
#
# Usage:
#   # Wait until GPU 5 is mostly idle, then run:
#   GPU=5 bash scripts/rerun_fig4_gaps_when_gpu_free.sh
#
#   # Skip waiting and start immediately on GPU 3:
#   GPU=3 SKIP_WAIT=1 bash scripts/rerun_fig4_gaps_when_gpu_free.sh
#
#   # Wait for any single GPU to become free (util < 10%, mem < 4GiB):
#   AUTO_GPU=1 bash scripts/rerun_fig4_gaps_when_gpu_free.sh
#
# Recommended: run on dev branch while fig4c pipeline is not using the same GPU.
#   git checkout dev
#
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
REPRO="$REPO/reproduce_repo"
LOG_ROOT="/data/zy/iterpert/logs/fig4/backfill"
LOCK_FILE="$LOG_ROOT/backfill.lock"
SUMMARY="$LOG_ROOT/backfill_summary.txt"

GPU="${GPU:-}"
AUTO_GPU="${AUTO_GPU:-0}"
SKIP_WAIT="${SKIP_WAIT:-0}"
MAX_RETRIES="${MAX_RETRIES:-2}"
MEM_THRESHOLD_MIB="${MEM_THRESHOLD_MIB:-4096}"
UTIL_THRESHOLD="${UTIL_THRESHOLD:-10}"
POLL_SEC="${POLL_SEC:-60}"

EXPECTED_BUDGETS=(100 200 300 400 500 600)

JOBS=(
  "BADGE:4"
  "BADGE:5"
  "BADGE:6"
  "BADGE:7"
  "BADGE:8"
  "LCMD:5"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_ROOT/backfill.log"
}

die() {
  log "ERROR: $*"
  exit 1
}

ensure_env() {
  mkdir -p "$LOG_ROOT"
  if [[ -f "$LOCK_FILE" ]]; then
    die "Lock exists ($LOCK_FILE). Another backfill may be running. Remove manually if stale."
  fi
  echo $$ > "$LOCK_FILE"
  trap 'rm -f "$LOCK_FILE"' EXIT

  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate iterpert
  export ITERPERT_DATA_ROOT="${ITERPERT_DATA_ROOT:-/data/zy/iterpert}"
  cd "$REPRO"
}

validate_gpu() {
  local n max_idx
  n=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1 | tr -d ' ')
  max_idx=$((n - 1))
  if [[ -z "$GPU" ]]; then
    die "GPU not set. This machine has GPU 0-$max_idx. Example: GPU=0 bash $0"
  fi
  if [[ "$GPU" -lt 0 || "$GPU" -gt "$max_idx" ]]; then
    die "Invalid GPU=$GPU (available: 0-$max_idx). You likely picked a non-existent GPU and training fell back to CPU."
  fi
  log "Using physical GPU $GPU (CUDA_VISIBLE_DEVICES=$GPU → cuda:0 inside run.py)"
}

gpu_mem_used() {
  local idx=$1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' '
}

gpu_util() {
  local idx=$1
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' '
}

gpu_is_idle() {
  local idx=$1
  local mem util
  mem=$(gpu_mem_used "$idx") || return 1
  util=$(gpu_util "$idx") || return 1
  [[ "$mem" -lt "$MEM_THRESHOLD_MIB" && "$util" -lt "$UTIL_THRESHOLD" ]]
}

pick_auto_gpu() {
  local n i
  n=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1 | tr -d ' ')
  for ((i=0; i<n; i++)); do
    if gpu_is_idle "$i"; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

wait_for_gpu() {
  if [[ "$SKIP_WAIT" == "1" ]]; then
    [[ -n "$GPU" ]] || die "Set GPU=<id> or AUTO_GPU=1 when SKIP_WAIT=1"
    log "SKIP_WAIT=1, using GPU=$GPU"
    return 0
  fi

  if [[ "$AUTO_GPU" == "1" ]]; then
    log "Waiting for any idle GPU (mem < ${MEM_THRESHOLD_MIB}MiB, util < ${UTIL_THRESHOLD}%)..."
    while true; do
      if GPU=$(pick_auto_gpu); then
        log "Selected idle GPU=$GPU"
        return 0
      fi
      sleep "$POLL_SEC"
    done
  fi

  [[ -n "$GPU" ]] || die "Set GPU=<id>, or AUTO_GPU=1, or SKIP_WAIT=1 with GPU set"
  log "Waiting for GPU $GPU to become idle..."
  while true; do
    if gpu_is_idle "$GPU"; then
      log "GPU $GPU is idle (mem=$(gpu_mem_used "$GPU")MiB util=$(gpu_util "$GPU")%)"
      return 0
    fi
    log "GPU $GPU busy (mem=$(gpu_mem_used "$GPU")MiB util=$(gpu_util "$GPU")%), sleep ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

verify_metrics() {
  local method=$1 run=$2
  "$REPO/scripts/verify_fig4_run.py" --method "$method" --run "$run"
}

run_one() {
  local method=$1 run=$2
  local attempt log_file metrics_glob
  log_file="$LOG_ROOT/${method}_run${run}.log"

  for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    log "START $method run=$run (attempt $attempt/$MAX_RETRIES) on GPU=$GPU"
    log "  log: $log_file  (tail -f to watch progress)"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" python -u run.py \
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
      log "FAIL $method run=$run exit=$ec (see $log_file)"
      if grep -qiE 'killed|out of memory|OOM|CUDA out of memory' "$log_file"; then
        log "Detected OOM/kill; waiting 120s before retry..."
        sleep 120
      fi
      continue
    fi

    if grep -qx "cpu" "$log_file" 2>/dev/null; then
      log "FAIL $method run=$run: run.py used CPU (check GPU index). See $log_file"
      continue
    fi

    if verify_metrics "$method" "$run"; then
      local pearson
      pearson=$(grep 'Round 5 pearson' "$log_file" | tail -1 || true)
      log "OK $method run=$run $pearson"
      echo "OK $method run=$run" >> "$SUMMARY"
      return 0
    fi

    log "WARN $method run=$run finished but metrics incomplete; retrying..."
  done

  log "GIVE UP $method run=$run after $MAX_RETRIES attempts"
  echo "FAIL $method run=$run" >> "$SUMMARY"
  return 1
}

postprocess() {
  log "Reorganizing results..."
  cd "$REPO"
  python reproduce_repo/reorganize_results.py
  python reproduce_repo/reorganize_results_structure.py
  python reproduce_repo/aggregate_fig4_results.py

  if [[ -f "$REPO/scripts/analyze_existing_nalc.py" ]]; then
    log "Refreshing E0 analysis (thoughts scripts)..."
    python "$REPO/scripts/analyze_existing_nalc.py" || log "WARN: analyze_existing_nalc.py failed"
  fi
  log "Postprocess done. Check results/fig4/comparison and results/analysis/e0_baseline/"
}

main() {
  ensure_env
  : > "$SUMMARY"
  wait_for_gpu
  validate_gpu

  log "=== Fig.4 gap backfill started (GPU=$GPU) ==="
  local failed=0
  for job in "${JOBS[@]}"; do
    IFS=: read -r method run <<< "$job"
    run_one "$method" "$run" || failed=$((failed + 1))
  done

  postprocess
  log "=== Backfill finished: $(( ${#JOBS[@]} - failed ))/${#JOBS[@]} succeeded ==="
  cat "$SUMMARY" | tee -a "$LOG_ROOT/backfill.log"
  [[ "$failed" -eq 0 ]]
}

main "$@"
