#!/usr/bin/env bash
# P1 pilot matrix: 6 methods x 3 seeds = 18 campaigns
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq/reproduce_repo"
PY="/home/zy/anaconda3/envs/iterpert/bin/python"
cd "$REPO"

COMMON=(
  --dataset_name replogle_k562_essential_1000hvg
  --model_name GEARS
  --n_init_labeled 100
  --n_query 100
  --n_round 5
  --batch_size 256
  --epoch_per_cycle 20
  --retrain
  --simple_loss
  --fix_evaluation
  --strategy_name kernel_based_active_learning
  --kernel_strategy Core-Set
  --base_kernel diff_effect
  --selection_log
)

run_one () {
  local tag="$1"; shift
  local seed="$1"; shift
  local run_id="$1"; shift
  echo "=== $tag seed=$seed run=$run_id ==="
  $PY run.py "${COMMON[@]}" --seed "$seed" --run "$run_id" "$@" \
    2>&1 | tee "/data/zy/iterpert/logs/pilot/${tag}_run${run_id}.log"
}

mkdir -p /data/zy/iterpert/logs/pilot

for run_id in 1 2 3; do
  run_one iterpert_main 1 "$run_id" \
    --use_prior --integrate_mode mean_new --normalize_mode max

  run_one iterpert_alignment 1 "$run_id" \
    --use_prior --integrate_mode alignment --normalize_mode diag

  run_one model_only 1 "$run_id" \
    --use_prior --use_prior_only --integrate_mode mean_new --normalize_mode max

  run_one single_rpe1 1 "$run_id" \
    --use_prior --use_single_prior --single_prior rpe1_kernel \
    --integrate_mode mean_new --normalize_mode max

  run_one dup_rpe1_x4 1 "$run_id" \
    --use_prior --integrate_mode mean_new --normalize_mode max \
    --duplicate_prior rpe1_kernel --duplicate_copies 4

  run_one bad_rpe1_perm 1 "$run_id" \
    --use_prior --integrate_mode mean_new --normalize_mode max \
    --corrupt_prior rpe1_kernel --corrupt_mode permute
done

echo "Pilot complete."
