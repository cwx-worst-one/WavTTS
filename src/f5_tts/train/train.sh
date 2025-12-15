#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=disabled

CONFIG_NAME="F5TTS_v1_Base_wav"     # F5TTS_v1_Base, F5TTS_v1_Base_wav
EXP_NAME="debug_wav"            # debug_wav, debug_mel
OUTDIR="/inspire/hdd/global_user/chenwenxi-253108120142/exp/f5_tts/${EXP_NAME}"

# training config
batch_size_per_gpu=19200
num_workers=2

# python -m debugpy --listen 5678 --wait-for-client src/f5_tts/train/train.py \
accelerate launch src/f5_tts/train/train.py \
    --config-name "${CONFIG_NAME}.yaml" \
    ++hydra.run.dir=${OUTDIR} \
    ++ckpts.save_dir="${OUTDIR}/ckpts" \
    ++ckpts.exp_name=${EXP_NAME} \
    ++datasets.batch_size_per_gpu=${batch_size_per_gpu} \
    ++datasets.num_workers=${num_workers}


# bash src/f5_tts/train/train.sh