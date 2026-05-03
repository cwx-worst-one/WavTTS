#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=disabled
export WANDB_API_KEY="406faa59cf62a3646fa3479a7e133c4cf5a77100"
export HF_ENDPOINT=https://hf-mirror.com
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=46899
export DEBUG_PORT=56789
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN

# accelerate config
num_processes=1     # 1, 8
num_machines=1
mixed_precision=bf16

# training config
batch_size_per_gpu=12800        # 19200, 22400, 25600, 38400, 51200
num_workers=16

CONFIG_NAME="F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_24k_dropout_0_joint_drop_0_1" # F5TTS_v1_Ultra_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k
# EXP_NAME="F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_mean_1_2_std_1_16k-from_noise_schedule_0_8_550k_pretrain-emilia-8gpus-19200sample_per_gpu-bf16"
EXP_NAME="debug_test"
OUTDIR="/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/${EXP_NAME}"
DEBUG_MODE=True     # True, False
DATASET_NAME="LibriTTS_100_360_500"     # LibriTTS_100_360_500, Emilia_ZH_EN, Emilia_ZH_EN_hdfs
tokenizer="char"                        # char, pinyin

# log config
LOG_DIR="${OUTDIR}/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a ${LOG_FILE}) 2>&1
echo "Log file: ${LOG_FILE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "Exp Name: ${EXP_NAME}"

if [ ${DEBUG_MODE} = True ]; then
    python -m debugpy --listen ${MASTER_ADDR}:${DEBUG_PORT} --wait-for-client src/f5_tts/train/train.py \
        --config-name "${CONFIG_NAME}.yaml" \
        ++hydra.run.dir=${OUTDIR} \
        ++ckpts.save_dir="${OUTDIR}/ckpts" \
        ++ckpts.exp_name=${EXP_NAME} \
        ++datasets.batch_size_per_gpu=${batch_size_per_gpu} \
        ++datasets.num_workers=${num_workers} \
        ++datasets.name=${DATASET_NAME} \
        ++model.tokenizer=${tokenizer}
fi

if [ ${DEBUG_MODE} = False ]; then
    accelerate launch --main_process_port ${MASTER_PORT} --num_processes ${num_processes} --num_machines ${num_machines} --mixed_precision ${mixed_precision} --dynamo_backend no src/f5_tts/train/train.py \
        --config-name "${CONFIG_NAME}.yaml" \
        ++hydra.run.dir=${OUTDIR} \
        ++ckpts.save_dir="${OUTDIR}/ckpts" \
        ++ckpts.exp_name=${EXP_NAME} \
        ++datasets.batch_size_per_gpu=${batch_size_per_gpu} \
        ++datasets.num_workers=${num_workers} \
        ++datasets.name=${DATASET_NAME} \
        ++model.tokenizer=${tokenizer}
fi


# bash src/f5_tts/train/debug_train.sh
