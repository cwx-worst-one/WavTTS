#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =======================
# 你只需要改这几个参数
# =======================
RESULTS_ROOT="${PROJECT_ROOT}/results"
MODELS="F5TTS_v1_Large_wav_x_pred_scale_5_aux_mel_hubert_noise_schedule_0_8_16k,F5TTS_v1_Large_wav_x_pred_scale_aux_mel_hubert_0.5_noise_schedule_0_8_16k,F5TTS_v1_Large_wav_x_pred_scale_aux_mel_hubert_noise_schedule_uniform_16k,F5TTS_v1_Large_wav_x_pred_scale_aux_mel_hubert_noise_schedule_0_8_16k_start_t_0_3,F5TTS_v1_Large_wav_x_pred_scale_aux_mel_noise_schedule_0_8_16k_emilia"
STEPS="200000,400000"
TASKS="seedtts_test_zh,seedtts_test_en"
METRICS="wer,sim,utmos"
GEN_SUBDIR="seed0_euler_nfe32_no_vocoder_ss-1.0_cfg3.0_speed1.0"  # 为空则自动选
OUTPUT_DIR="${PROJECT_ROOT}/results_plots"
DPI=300
PYTHON_BIN="/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/bin/python"

if [[ -z "${MODELS}" ]]; then
  mapfile -t model_dirs < <(
    find "${RESULTS_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name "exp_libritts" -printf '%f\n' | sort
  )

  if [[ ${#model_dirs[@]} -eq 0 ]]; then
    echo "[ERROR] No model directories found under ${RESULTS_ROOT} after excluding exp_libritts." >&2
    exit 1
  fi

  MODELS="$(IFS=,; echo "${model_dirs[*]}")"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_seedtts_metrics_from_results.py" \
  --results-root "${RESULTS_ROOT}" \
  --models "${MODELS}" \
  --steps "${STEPS}" \
  --tasks "${TASKS}" \
  --metrics "${METRICS}" \
  --gen-subdir "${GEN_SUBDIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --dpi "${DPI}"

# bash results_plots/run_plot_seedtts_metrics.sh
