#!/usr/bin/env bash
# Dump Paper equal-mean states at n=100,300,500 for runs 1–3.
# Saves labeled/pool gene lists + GEARS checkpoint for common-state forks.
#
#   PARALLEL=1 bash configs/experiments/idea2_fork/dump_paper_states.sh
#   GPUS="0 1 2" RUNS="1 2 3" PARALLEL=1 bash configs/experiments/idea2_fork/dump_paper_states.sh
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
REPRO="$REPO/reproduce_repo"
PY="${PY:-/home/zy/anaconda3/envs/iterpert/bin/python}"
LOG_DIR="/data/zy/iterpert/logs/idea2_fork_dump"
STATE_ROOT="${STATE_ROOT:-/data/zy/iterpert/common_states}"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-0 1 2}"
RUNS="${RUNS:-1 2 3}"
read -r -a gpu_arr <<< "$GPUS"
read -r -a run_arr <<< "$RUNS"

mkdir -p "$LOG_DIR"
cd "$REPRO"
export ITERPERT_DATA_ROOT="${ITERPERT_DATA_ROOT:-/data/zy/iterpert}"

COMMON=(
  --device cuda:0
  --dataset_name replogle_k562_essential_1000hvg
  --model_name GEARS
  --n_init_labeled 100
  --n_query 100
  --n_round 4
  --batch_size 256
  --epoch_per_cycle 20
  --retrain --simple_loss --fix_evaluation
  --strategy_name kernel_based_active_learning
  --kernel_strategy Core-Set
  --base_kernel diff_effect
  --use_prior --integrate_mode mean_new --normalize_mode max
  --selection_log
  --seed 1
  --dump_n_labeled 100,300,500
)

state_ready () {
  local run="$1"
  local root="$STATE_ROOT/paper_run${run}"
  [[ -f "$root/n100/gears_checkpoint/model.pt" \
  && -f "$root/n300/gears_checkpoint/model.pt" \
  && -f "$root/n500/gears_checkpoint/model.pt" \
  && -f "$root/n100/labeled_genes.txt" \
  && -f "$root/n300/labeled_genes.txt" \
  && -f "$root/n500/labeled_genes.txt" ]]
}

run_one () {
  local run="$1" gpu="$2"
  local log="$LOG_DIR/paper_run${run}.log"
  local dump="$STATE_ROOT/paper_run${run}"
  if state_ready "$run"; then
    echo "SKIP paper_run${run} (states already present)"
    return 0
  fi
  mkdir -p "$dump"
  echo "=== dump paper_run${run} GPU=$gpu → $dump ==="
  CUDA_VISIBLE_DEVICES="$gpu" $PY -u run.py "${COMMON[@]}" \
    --run "$run" --dump_state_dir "$dump" \
    > "$log" 2>&1
}

if [[ "$PARALLEL" == "1" ]]; then
  if [[ ${#run_arr[@]} -gt ${#gpu_arr[@]} ]]; then
    echo "ERROR: more runs than GPUs" >&2
    exit 1
  fi
  pids=()
  for i in "${!run_arr[@]}"; do
    run_one "${run_arr[$i]}" "${gpu_arr[$i]}" &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  [[ "$fail" -eq 0 ]]
else
  gpu="${CUDA_VISIBLE_DEVICES:-${gpu_arr[0]}}"
  gpu="${gpu%%,*}"
  for run in "${run_arr[@]}"; do
    run_one "$run" "$gpu"
  done
fi

echo "Dump done. States under $STATE_ROOT/paper_run{1,2,3}/n{100,300,500}/"
