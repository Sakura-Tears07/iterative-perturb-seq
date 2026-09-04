#!/usr/bin/env bash
# Idea 3c first GPU batch: Detected hard-reject vs Equal (λ=1, held-out perm)
# vs reused Oracle-drop / Fig.4 clean equal.
#
#   PARALLEL=1 bash configs/experiments/idea3_intervention/run_hard_reject.sh
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
REPRO="$REPO/reproduce_repo"
PY="${PY:-/home/zy/anaconda3/envs/iterpert/bin/python}"
LOG_DIR="/data/zy/iterpert/logs/idea3_intervention"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-0 1 2 3 4}"
RUNS="${RUNS:-1 2 3}"
TAU_JSON="${TAU_JSON:-$REPO/results/analysis/idea3_intervention/frozen_tau.json}"
read -r -a gpu_arr <<< "$GPUS"

if [[ ! -f "$TAU_JSON" ]]; then
  echo "ERROR: $TAU_JSON missing. Run python scripts/analyze_hard_reject_threshold.py first." >&2
  exit 1
fi

TAU="$($PY -c "import json; print(json.load(open('$TAU_JSON'))['tau'])")"
TEST_PERM="$($PY -c "import json; print(int(json.load(open('$TAU_JSON'))['gpu_test_perm']))")"
LAUNCH="$($PY -c "import json; print(bool(json.load(open('$TAU_JSON'))['launch_gpu']))")"
if [[ "$LAUNCH" != "True" ]]; then
  echo "ERROR: frozen_tau.json says launch_gpu=false" >&2
  exit 1
fi

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
  --result_tag idea3/intervention
)

DETECT=(
  --detect_drop --detect_target rpe1 --detect_min_round 1 --detect_tau "$TAU"
)
CORRUPT=(
  --corrupt_prior rpe1_kernel --corrupt_mode permute --corrupt_lambda 1.0 --corrupt_seed "$TEST_PERM"
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
  echo "=== $tag run=$run GPU=$gpu tau=$TAU perm=$TEST_PERM ==="
  CUDA_VISIBLE_DEVICES="$gpu" $PY -u run.py "${COMMON[@]}" --run "$run" "$@" \
    > "$log" 2>&1
}

JOBS=()
for run in $RUNS; do
  JOBS+=("detect_clean|$run|${DETECT[*]}")
  JOBS+=("detect_l1|$run|${DETECT[*]} ${CORRUPT[*]}")
  JOBS+=("equal_l1|$run|${CORRUPT[*]}")
done

echo "Queue ${#JOBS[@]} Idea-3c jobs. tau=$TAU  gpu_test_perm=$TEST_PERM"
echo "Reuse Fig.4 clean equal and idea3 oracle-drop runs 1–3. Do not retune tau."

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

echo "Idea 3c queue finished. python $REPO/scripts/analyze_idea3_intervention.py"
