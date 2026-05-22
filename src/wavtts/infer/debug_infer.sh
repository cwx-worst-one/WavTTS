#!/usr/bin/env bash
set -euo pipefail

# Debug inference launcher for the current WavTTS waveform checkpoint.
# Override any variable from the shell, for example:
#   CKPT_FILE=/path/to/model.pt NFE_STEP=16 bash src/wavtts/infer/debug_infer.sh

export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR="127.0.0.1"

CACHE_ROOT="/tmp/wavtts_debug"
mkdir -p "${CACHE_ROOT}/matplotlib" "${CACHE_ROOT}/numba"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba"

MODEL_NAME="WavTTS_scale_8_16k_1000000"
MODEL_CFG="src/wavtts/configs/WavTTS_scale_8_16k.yaml"
CKPT_FILE="/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/emilia/F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1-emilia-8gpus-19200sample_per_gpu-bf16/ckpts/model_1000000.pt"
VOCAB_FILE="data/Emilia_ZH_EN_pinyin/vocab.txt"

REF_AUDIO="tests/test_zh_ref.wav"
REF_TEXT="我拽起裤腿鞋袜未脱就踏进了溪流。"
GEN_TEXT="导航开始，全程二十五分钟，预计需要十二分钟。"
OUTPUT_DIR="tests/debug/${MODEL_NAME}_output"
NFE_STEP="50"
OUTPUT_FILE="model_1000000_nfe_${NFE_STEP}.wav"
DEVICE="cuda"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

# python -m debugpy --listen 127.0.0.1:56789 src/wavtts/infer/infer_cli.py \
.venv/bin/python src/wavtts/infer/infer_cli.py \
    --model_cfg "${MODEL_CFG}" \
    --ckpt_file "${CKPT_FILE}" \
    --ref_audio "${REF_AUDIO}" \
    --ref_text "${REF_TEXT}" \
    --gen_text "${GEN_TEXT}" \
    --nfe_step "${NFE_STEP}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_file "${OUTPUT_FILE}" \
    --vocab_file "${VOCAB_FILE}" \
    --device "${DEVICE}"

# bash src/wavtts/infer/debug_infer.sh
