#!/usr/bin/env bash
# Download or install IterPert knowledge embeddings for GW kernel notebook.
#
# Usage:
#   # A) 服务器能访问 Google 时：
#   bash demo/download_embeddings.sh
#
#   # B) 本地下载后上传到服务器，再解压：
#   bash demo/download_embeddings.sh /path/to/knowledge_kernels.tar.gz
#
# Google Drive: https://drive.google.com/file/d/16p9sQYkhpM-PBAcNWQdjL9pxDZ46krQX/view

set -euo pipefail

DATA_ROOT="${ITERPERT_DATA_ROOT:-/data/zy/iterpert}"
EMB_DIR="${DATA_ROOT}/knowledge_kernels/embeddings"
CACHE_DIR="${DATA_ROOT}/cache/downloads/embeddings"
GDRIVE_ID="16p9sQYkhpM-PBAcNWQdjL9pxDZ46krQX"

mkdir -p "${EMB_DIR}" "${CACHE_DIR}"

install_and_extract() {
  local archive="$1"
  echo "Extracting ${archive} -> ${EMB_DIR}"
  if [[ "${archive}" == *.tar.gz || "${archive}" == *.tgz ]]; then
    tar -xzf "${archive}" -C "${DATA_ROOT}/knowledge_kernels/"
  elif [[ "${archive}" == *.zip ]]; then
    unzip -o "${archive}" -d "${CACHE_DIR}/extract"
    # common layouts: embeddings/... or knowledge_kernels/...
    if [[ -d "${CACHE_DIR}/extract/embeddings" ]]; then
      cp -a "${CACHE_DIR}/extract/embeddings/." "${EMB_DIR}/"
    elif [[ -d "${CACHE_DIR}/extract/knowledge_kernels/embeddings" ]]; then
      cp -a "${CACHE_DIR}/extract/knowledge_kernels/embeddings/." "${EMB_DIR}/"
    else
      echo "Unknown zip layout. Please extract manually to ${EMB_DIR}"
      exit 1
    fi
  else
    echo "Unsupported archive: ${archive}"
    exit 1
  fi
}

if [[ $# -ge 1 ]]; then
  install_and_extract "$1"
else
  echo "Trying gdown from Google Drive..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate iterpert
  pip install -q gdown
  cd "${CACHE_DIR}"
  gdown "${GDRIVE_ID}" --fuzzy -O knowledge_kernels_archive
  file knowledge_kernels_archive
  if file knowledge_kernels_archive | grep -q "Zip"; then
    mv knowledge_kernels_archive knowledge_kernels.zip
    install_and_extract "${CACHE_DIR}/knowledge_kernels.zip"
  elif file knowledge_kernels_archive | grep -q "gzip"; then
    mv knowledge_kernels_archive knowledge_kernels.tar.gz
    install_and_extract "${CACHE_DIR}/knowledge_kernels.tar.gz"
  else
    echo "Downloaded file format unknown. Inspect ${CACHE_DIR}/knowledge_kernels_archive"
    exit 1
  fi
fi

echo "Checking required files..."
python - <<'PY'
import sys
sys.path.insert(0, "demo")
from paths import ESM_EMB, BIOGPT_EMB, POPS_EMB, NODE2VEC_EMB
import os
paths = [ESM_EMB, BIOGPT_EMB, POPS_EMB, NODE2VEC_EMB]
missing = [p for p in paths if not os.path.exists(p)]
if missing:
    print("Still missing:")
    for p in missing: print(" ", p)
    sys.exit(1)
print("All embedding files OK")
PY
