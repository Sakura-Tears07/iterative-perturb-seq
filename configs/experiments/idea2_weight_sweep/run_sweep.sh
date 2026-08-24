#!/usr/bin/env bash
# Idea 2: model-kernel weight sweep. Priors share the remaining mass equally.
#
# Serial (one GPU):
#   CUDA_VISIBLE_DEVICES=4 bash configs/experiments/idea2_weight_sweep/run_sweep.sh
#
# 5-GPU parallel (one weight per card):
#   PARALLEL=1 bash configs/experiments/idea2_weight_sweep/run_sweep.sh
#
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
REPRO="$REPO/reproduce_repo"
PY="${PY:-/home/zy/anaconda3/envs/iterpert/bin/python}"
LOG_DIR="/data/zy/iterpert/logs/idea2_weight_sweep"
WEIGHTS="${WEIGHTS:-0 0.25 0.5 0.75 1.0}"
RUN="${RUN:-1}"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-0 1 2 3 4}"

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
  --seed 1 --run "$RUN"
)

read -r -a weight_arr <<< "$WEIGHTS"
read -r -a gpu_arr <<< "$GPUS"

run_one () {
  local w="$1" gpu="$2"
  local log="$LOG_DIR/mw${w}_run${RUN}.log"
  if [[ -f "$log" ]] && grep -q "Round 5 pearson" "$log"; then
    echo "SKIP model_weight=$w (already complete)"
    return 0
  fi
  echo "=== model_weight=$w GPU=$gpu ==="
  CUDA_VISIBLE_DEVICES="$gpu" $PY -u run.py "${COMMON[@]}" --model_weight "$w" \
    > "$log" 2>&1
}

if [[ "$PARALLEL" == "1" ]]; then
  if [[ ${#weight_arr[@]} -gt ${#gpu_arr[@]} ]]; then
    echo "ERROR: ${#weight_arr[@]} weights but only ${#gpu_arr[@]} GPUs" >&2
    exit 1
  fi
  pids=()
  for i in "${!weight_arr[@]}"; do
    w="${weight_arr[$i]}"
    gpu="${gpu_arr[$i]}"
    run_one "$w" "$gpu" &
    pids+=($!)
  done
  fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "FAIL model_weight=${weight_arr[$i]} (see $LOG_DIR/mw${weight_arr[$i]}_run${RUN}.log)"
      fail=1
    fi
  done
  [[ "$fail" -eq 0 ]]
else
  gpu="${CUDA_VISIBLE_DEVICES:-${gpu_arr[0]}}"
  gpu="${gpu%%,*}"
  for w in "${weight_arr[@]}"; do
    run_one "$w" "$gpu"
  done
fi

echo "Sweep done. python $REPO/scripts/analyze_weight_sweep.py"
