#!/usr/bin/env bash
set -euo pipefail

# Main public training launcher for the current WavTTS training baseline.
# This script intentionally avoids hard-coded personal paths, proxy settings,
# cluster-specific environment variables, and secret tokens.

CONFIG_NAME="${CONFIG_NAME:-F5TTS_v1_Large_wav_x_pred_scale_9_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1}"
DATASET_NAME="${DATASET_NAME:-emilia}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-19200}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./exp/nar_wav_tts}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

EXP_NAME="${EXP_NAME:-${CONFIG_NAME}-${DATASET_NAME}-${NUM_PROCESSES}gpus-${BATCH_SIZE_PER_GPU}sample_per_gpu-${MIXED_PRECISION}}"
OUTDIR="${OUTDIR:-${OUTPUT_ROOT}/${DATASET_NAME}/${EXP_NAME}}"
LOG_DIR="${OUTDIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"
echo "Config: ${CONFIG_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Exp Name: ${EXP_NAME}"
echo "Output dir: ${OUTDIR}"
echo "Master port: ${MASTER_PORT}"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

accelerate launch \
  --main_process_port "${MASTER_PORT}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend no \
  src/f5_tts/train/train.py \
  --config-name "${CONFIG_NAME}.yaml" \
  ++hydra.run.dir="${OUTDIR}" \
  ++ckpts.save_dir="${OUTDIR}/ckpts" \
  ++ckpts.exp_name="${EXP_NAME}" \
  ++datasets.batch_size_per_gpu="${BATCH_SIZE_PER_GPU}" \
  ++datasets.num_workers="${NUM_WORKERS}"


# bash src/f5_tts/train/run_main_train.sh