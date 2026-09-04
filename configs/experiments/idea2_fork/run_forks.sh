#!/usr/bin/env bash
# Common-state forks: identical S_t × w_model ∈ {0,0.25,0.5,0.75,1}.
# Requires dump_paper_states.sh first.
#
# Same (run, n) shares --seed 1 and --run $run so the five weights use the
# same retraining seed. Branches differ only in acquisition weight.
#
#   PARALLEL=1 bash configs/experiments/idea2_fork/run_forks.sh
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
REPRO="$REPO/reproduce_repo"
PY="${PY:-/home/zy/anaconda3/envs/iterpert/bin/python}"
LOG_DIR="/data/zy/iterpert/logs/idea2_fork"
STATE_ROOT="${STATE_ROOT:-/data/zy/iterpert/common_states}"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-0 1 2 3 4}"
RUNS="${RUNS:-1 2 3}"
NS="${NS:-100 300 500}"
WEIGHTS="${WEIGHTS:-0 0.25 0.5 0.75 1.0}"
read -r -a gpu_arr <<< "$GPUS"

mkdir -p "$LOG_DIR"
cd "$REPRO"
export ITERPERT_DATA_ROOT="${ITERPERT_DATA_ROOT:-/data/zy/iterpert}"

COMMON=(
  --device cuda:0
  --dataset_name replogle_k562_essential_1000hvg
  --model_name GEARS
  --n_query 100
  --n_round 1
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
  [[ -f "$log" ]] && grep -q "Round 1 pearson" "$log"
}

run_one () {
  local run="$1" n="$2" w="$3" gpu="$4"
  local state="$STATE_ROOT/paper_run${run}/n${n}"
  local log="$LOG_DIR/fork_run${run}_n${n}_mw${w}.log"
  local labeled="$state/labeled_genes.txt"
  local ckpt="$state/gears_checkpoint"
  if [[ ! -f "$labeled" || ! -f "$ckpt/model.pt" ]]; then
    echo "MISSING state $state" >&2
    return 1
  fi
  if is_done "$log"; then
    echo "SKIP fork run=$run n=$n w=$w"
    return 0
  fi
  echo "=== fork run=$run n=$n w=$w GPU=$gpu ==="
  CUDA_VISIBLE_DEVICES="$gpu" $PY -u run.py "${COMMON[@]}" \
    --run "$run" \
    --labeled_genes_file "$labeled" \
    --load_checkpoint "$ckpt" \
    --model_weight "$w" \
    --n_init_labeled "$n" \
    > "$log" 2>&1
}

JOBS=()
for run in $RUNS; do
  for n in $NS; do
    for w in $WEIGHTS; do
      JOBS+=("${run}|${n}|${w}")
    done
  done
done
echo "Queue ${#JOBS[@]} fork jobs"

if [[ "$PARALLEL" == "1" ]]; then
  n_gpu=${#gpu_arr[@]}
  buckets=()
  for ((i = 0; i < n_gpu; i++)); do buckets[i]=""; done
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
        run="${spec%%|*}"; rest="${spec#*|}"
        n="${rest%%|*}"; w="${rest#*|}"
        run_one "$run" "$n" "$w" "$gpu"
      done <<< "${buckets[$i]}"
    ) &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" -eq 0 ]]
else
  gpu="${CUDA_VISIBLE_DEVICES:-${gpu_arr[0]}}"
  gpu="${gpu%%,*}"
  for spec in "${JOBS[@]}"; do
    run="${spec%%|*}"; rest="${spec#*|}"
    n="${rest%%|*}"; w="${rest#*|}"
    run_one "$run" "$n" "$w" "$gpu"
  done
fi

echo "Forks done. python $REPO/scripts/analyze_common_state_forks.py"
