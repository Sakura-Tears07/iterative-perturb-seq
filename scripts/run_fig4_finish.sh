#!/usr/bin/bash
# Finish Fig.4 reproduction: Fig.4c remaining priors (5-GPU parallel) + post-aggregate.
set -euo pipefail

REPO="/home/zy/workspace/iterative-perturb-seq"
BASE="/data/zy/iterpert/logs/fig4c"
LOG="$BASE/fig4c_finish.nohup.log"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iterpert
export ITERPERT_DATA_ROOT=/data/zy/iterpert

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

REMAINING=(node2vec_kernel ops_HeLa_HPLM_kernel ops_HeLa_DMEM_kernel)

log "=== Fig.4 finish pipeline started ==="
for PRIOR in "${REMAINING[@]}"; do
  if grep -q "Fig4c prior=${PRIOR} COMPLETE" "$BASE/${PRIOR}/pipeline.log" 2>/dev/null; then
    n=$(ls "$REPO/results/fig4c/single_prior/${PRIOR}/runs/"*metrics.csv 2>/dev/null | wc -l)
    if [[ "$n" -ge 5 ]]; then
      log "SKIP $PRIOR ($n/5 on disk)"
      continue
    fi
  fi
  log "START $PRIOR (5-GPU parallel)"
  "$BASE/launch_fig4c_prior.sh" "$PRIOR"
  log "DONE $PRIOR"
done

log "=== Post-aggregate ==="
cd "$REPO/reproduce_repo"
python aggregate_fig4_results.py | tee -a "$LOG"
python aggregate_fig4c_results.py | tee -a "$LOG"
python reorganize_results.py | tee -a "$LOG"
log "=== Fig.4 finish pipeline ALL COMPLETE ==="
