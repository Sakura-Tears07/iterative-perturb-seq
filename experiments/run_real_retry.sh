#!/usr/bin/env bash
# Run one real closed-loop experiment, retrying on the silent startup crashes we
# hit when several GEARS runs were launched simultaneously (both processes died
# at the first epoch with no traceback; running alone in the same config works).
#
# usage: run_real_retry.sh <gpu> <strategy> <mode> <tag> <seed> [epochs] [n_init] [n_round] [n_query]
set -u
GPU=$1; STRAT=$2; MODE=$3; TAG=$4; SEED=${5:-1}
EPOCHS=${6:-10}; NINIT=${7:-20}; NROUND=${8:-3}; NQUERY=${9:-5}

cd /home/lihaoran/ai4s/analysis
PY=/home/lihaoran/miniconda3/envs/iterpert_env/bin/python
RES=/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/res

for attempt in 1 2 3; do
  if [ -f "$RES/${TAG}_metrics.json" ]; then
    echo "[retry-wrapper] $TAG already has metrics; nothing to do"
    exit 0
  fi
  echo "[retry-wrapper] $TAG attempt $attempt"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONFAULTHANDLER=1 $PY -u real_loop.py \
      --strategy "$STRAT" --mode "$MODE" --tag "$TAG" --seed "$SEED" \
      --epochs "$EPOCHS" --n_init "$NINIT" --n_round "$NROUND" --n_query "$NQUERY" \
      >> "logs/real_${TAG}.log" 2>&1
  rc=$?
  if [ -f "$RES/${TAG}_metrics.json" ]; then
    echo "[retry-wrapper] $TAG DONE on attempt $attempt"
    exit 0
  fi
  echo "[retry-wrapper] $TAG attempt $attempt failed (rc=$rc); backing off"
  sleep 30
done
echo "[retry-wrapper] $TAG FAILED after 3 attempts"
exit 1
