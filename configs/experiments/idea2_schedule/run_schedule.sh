#!/usr/bin/env bash
# 5-seed schedule: paper equal (reused) vs fixed 0.5/0.75 vs early_prior vs early_model.
#
#   PARALLEL=1 bash configs/experiments/idea2_schedule/run_schedule.sh
#   GPUS="1 2 3 4" PARALLEL=1 bash configs/experiments/idea2_schedule/run_schedule.sh
#
# Does not re-run paper equal (Fig.4 IterPert) or seed-1 fixed 0.5/0.75 (weight sweep).
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
REPRO="$REPO/reproduce_repo"
PY="${PY:-/home/zy/anaconda3/envs/iterpert/bin/python}"
LOG_DIR="/data/zy/iterpert/logs/idea2_schedule"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-1 2 3 4}"
RUNS="${RUNS:-1 2 3 4 5}"
read -r -a gpu_arr <<< "$GPUS"

mkdir -p "$LOG_DIR"
cd "$REPRO"
export ITERPERT_DATA_ROOT="${ITERPERT_DATA_ROOT:-/data/zy/iterpert}"

COMMON=(
  --device cuda:0
  --dataset_name replogle_k562_essential_1000hvg
  --model_name GEARS
  --n_init_labeled 100
  --n_query 100
  --n_round 5
  --batch_size 256
  --epoch_per_cycle 20
  --retrain --simple_loss --fix_evaluation
  --strategy_name kernel_based_active_learning
  --kernel_strategy Core-Set
  --base_kernel diff_effect
  --use_prior --integrate_mode mean_new --normalize_mode max
  --selection_log
  --seed 1
)

is_done () {
  local log="$1"
  [[ -f "$log" ]] && grep -q "Round 5 pearson" "$log"
}

run_one () {
  local tag="$1" run="$2" gpu="$3"
  shift 3
  local log="$LOG_DIR/${tag}_run${run}.log"
  if is_done "$log"; then
    echo "SKIP $tag run=$run"
    return 0
  fi
  echo "=== $tag run=$run GPU=$gpu ==="
  CUDA_VISIBLE_DEVICES="$gpu" $PY -u run.py "${COMMON[@]}" --run "$run" "$@" \
    > "$log" 2>&1
}

# tag|run|extra args (pipe-separated). Paper equal is Fig.4 — do not launch.
JOBS=()
for run in $RUNS; do
  if [[ "$run" != "1" ]]; then
    JOBS+=("fixed_0.5|$run|--model_weight 0.5")
    JOBS+=("fixed_0.75|$run|--model_weight 0.75")
  fi
  JOBS+=("early_prior|$run|--weight_schedule early_prior")
  JOBS+=("early_model|$run|--weight_schedule early_model")
done

echo "Queue ${#JOBS[@]} jobs. Paper/Equal = Fig.4 IterPert runs $RUNS (not launched)."
echo "Reuse sweep seed 1 for Fixed-0.5 / Fixed-0.75."

if [[ "$PARALLEL" == "1" ]]; then
  n_gpu=${#gpu_arr[@]}
  buckets=()
  for ((i = 0; i < n_gpu; i++)); do
    buckets[i]=""
  done
  for i in "${!JOBS[@]}"; do
    b=$((i % n_gpu))
    buckets[b]+="${JOBS[$i]}"$'\n'
  done
  pids=()
  for i in "${!gpu_arr[@]}"; do
    gpu="${gpu_arr[$i]}"
    (
      while IFS= read -r spec; do
        [[ -z "$spec" ]] && continue
        tag="${spec%%|*}"
        rest="${spec#*|}"
        run="${rest%%|*}"
        extra="${rest#*|}"
        # shellcheck disable=SC2086
        run_one "$tag" "$run" "$gpu" $extra
      done <<< "${buckets[$i]}"
    ) &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  [[ "$fail" -eq 0 ]]
else
  gpu="${CUDA_VISIBLE_DEVICES:-${gpu_arr[0]:-1}}"
  gpu="${gpu%%,*}"
  for spec in "${JOBS[@]}"; do
    tag="${spec%%|*}"
    rest="${spec#*|}"
    run="${rest%%|*}"
    extra="${rest#*|}"
    # shellcheck disable=SC2086
    run_one "$tag" "$run" "$gpu" $extra
  done
fi

echo "Schedule queue finished. python $REPO/scripts/analyze_schedule.py"
