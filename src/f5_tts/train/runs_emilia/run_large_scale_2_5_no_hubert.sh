#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY="406faa59cf62a3646fa3479a7e133c4cf5a77100"
export WANDB_BASE_URL=https://api.bandw.top
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export HF_ENDPOINT=https://hf-mirror.com
export MASTER_PORT=49099
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN

# accelerate config
num_processes=8     # 1, 8
num_machines=1
mixed_precision=bf16

# training config
batch_size_per_gpu=19200
num_workers=16

DATASET_NAME="emilia"
CONFIG_NAME="F5TTS_v1_Large_wav_x_pred_scale_2_5_aux_mel_noise_schedule_0_8_16k"
EXP_NAME="${CONFIG_NAME}-${DATASET_NAME}-${num_processes}gpus-${batch_size_per_gpu}sample_per_gpu-${mixed_precision}"
OUTDIR="/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/${DATASET_NAME}/${EXP_NAME}"

# log config
LOG_DIR="${OUTDIR}/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a ${LOG_FILE}) 2>&1
echo "Log file: ${LOG_FILE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "Exp Name: ${EXP_NAME}"

accelerate launch --main_process_port ${MASTER_PORT} --num_processes ${num_processes} --num_machines ${num_machines} --mixed_precision ${mixed_precision} --dynamo_backend no src/f5_tts/train/train.py \
    --config-name "${CONFIG_NAME}.yaml" \
    ++hydra.run.dir=${OUTDIR} \
    ++ckpts.save_dir="${OUTDIR}/ckpts" \
    ++ckpts.exp_name=${EXP_NAME} \
    ++datasets.batch_size_per_gpu=${batch_size_per_gpu} \
    ++datasets.num_workers=${num_workers}

# cd /mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav
# bash /mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav/src/f5_tts/train/runs_emilia/run_large_scale_2_5_no_hubert.sh
