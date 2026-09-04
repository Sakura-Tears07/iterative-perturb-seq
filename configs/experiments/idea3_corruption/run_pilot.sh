#!/usr/bin/env bash
# Idea 3 RPE1 corruption pilot.
# Reuses clean Full IterPert (Fig.4 runs 1–3), Model+RPE1 (Fig.4c runs 1–3),
# and model-only run 1. Launches only new GPU campaigns.
#
#   PARALLEL=1 bash configs/experiments/idea3_corruption/run_pilot.sh
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
REPRO="$REPO/reproduce_repo"
PY="${PY:-/home/zy/anaconda3/envs/iterpert/bin/python}"
LOG_DIR="/data/zy/iterpert/logs/idea3_pilot"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-0 1 2 3 4}"
RUNS="${RUNS:-1 2 3}"
CORRUPT_SEED="${CORRUPT_SEED:-20260826}"
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
  --result_tag idea3/pilot
  --corrupt_seed "$CORRUPT_SEED"
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

# tag|run|extra — Gate cells first (needed for A/B/C); model-only last (interpretation only).
JOBS=()
for run in $RUNS; do
  JOBS+=("two_source_l0.5|$run|--use_single_prior --single_prior rpe1_kernel --corrupt_prior rpe1_kernel --corrupt_mode permute --corrupt_lambda 0.5")
  JOBS+=("two_source_l1|$run|--use_single_prior --single_prior rpe1_kernel --corrupt_prior rpe1_kernel --corrupt_mode permute --corrupt_lambda 1.0")
  JOBS+=("full_l0.5|$run|--corrupt_prior rpe1_kernel --corrupt_mode permute --corrupt_lambda 0.5")
  JOBS+=("full_l1|$run|--corrupt_prior rpe1_kernel --corrupt_mode permute --corrupt_lambda 1.0")
  JOBS+=("oracle_drop_rpe1|$run|--drop_prior rpe1_kernel")
done
for run in $RUNS; do
  if [[ "$run" != "1" ]]; then
    JOBS+=("model_only|$run|--model_weight 1.0")
  fi
done

echo "Queue ${#JOBS[@]} new Idea-3 jobs (reuse clean full / two-source / model-only run1)."
echo "corruption_seed=$CORRUPT_SEED  permutation file under \$ITERPERT_DATA_ROOT/idea3/"

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
        tag="${spec%%|*}"; rest="${spec#*|}"
        run="${rest%%|*}"; extra="${rest#*|}"
        # shellcheck disable=SC2086
        run_one "$tag" "$run" "$gpu" $extra
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
    tag="${spec%%|*}"; rest="${spec#*|}"
    run="${rest%%|*}"; extra="${rest#*|}"
    # shellcheck disable=SC2086
    run_one "$tag" "$run" "$gpu" $extra
  done
fi

echo "Idea 3 pilot queue finished. python $REPO/scripts/analyze_idea3_corruption.py"
