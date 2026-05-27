#!/usr/bin/env bash
set -euo pipefail

# Main public training launcher for the current WavTTS training baseline.

CONFIG_NAME="WavTTS"
DATASET_NAME="emilia"
NUM_PROCESSES="8"
NUM_MACHINES="1"
MIXED_PRECISION="bf16"
BATCH_SIZE_PER_GPU="19200"
NUM_WORKERS="16"
MASTER_PORT="49500"
OUTPUT_ROOT="./exp/nar_wav_tts"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
# export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

EXP_NAME="${CONFIG_NAME}-${DATASET_NAME}-${NUM_PROCESSES}gpus-${BATCH_SIZE_PER_GPU}sample_per_gpu-${MIXED_PRECISION}"
OUTDIR="${OUTPUT_ROOT}/${DATASET_NAME}/${EXP_NAME}"
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
  src/wavtts/train/train.py \
  --config-name "${CONFIG_NAME}.yaml" \
  ++hydra.run.dir="${OUTDIR}" \
  ++ckpts.save_dir="${OUTDIR}/ckpts" \
  ++ckpts.exp_name="${EXP_NAME}" \
  ++datasets.batch_size_per_gpu="${BATCH_SIZE_PER_GPU}" \
  ++datasets.num_workers="${NUM_WORKERS}"


# bash src/wavtts/train/run_main_train.sh