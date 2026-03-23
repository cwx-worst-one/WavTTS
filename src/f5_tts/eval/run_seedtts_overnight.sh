#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

ENV_PATH="/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts"
if [[ -f "${ENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_PATH}/bin/activate"
else
  echo "[ERROR] cannot find env activate script: ${ENV_PATH}/bin/activate"
  exit 1
fi

export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export MASTER_ADDR="127.0.0.1"
export HF_ENDPOINT="https://hf-mirror.com"
export OMP_NUM_THREADS="1"

# Optional mirrors / proxy (inherit if already set in your shell)
export HTTP_PROXY="${HTTP_PROXY:-}"
export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
export https_proxy="${https_proxy:-${HTTP_PROXY:-}}"

SUMMARY_DIR="${REPO_ROOT}/results/seedtts_eval_summary"
mkdir -p "${SUMMARY_DIR}"
LOG_FILE="${SUMMARY_DIR}/overnight_run_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[INFO] start at $(date)"
  echo "[INFO] repo=${REPO_ROOT}"

  python src/f5_tts/eval/run_seedtts_overnight.py "$@"

  # Always aggregate after run (even if部分失败, 通过脚本本身的现有结果做汇总)
  python src/f5_tts/eval/aggregate_seedtts_results.py

  echo "[INFO] end at $(date)"
} 2>&1 | tee -a "${LOG_FILE}"

echo "[INFO] log saved to ${LOG_FILE}"
