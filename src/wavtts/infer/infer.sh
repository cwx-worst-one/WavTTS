#!/usr/bin/env bash
set -euo pipefail

# Model Configuration
MODEL_NAME="WavTTS_scale_9_16k"
MODEL_CFG="src/wavtts/configs/WavTTS_scale_9_16k.yaml"
CKPT_FILE="/mnt/hdfs/ssd_hldy/chenwenxi.sylvan/exp/nar_wav_tts/emilia/F5TTS_v1_Large_wav_x_pred_scale_9_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1-emilia-8gpus-19200sample_per_gpu-bf16/ckpts/model_800000.pt"

# Zero-shot TTS Configuration
VOCAB_FILE="infer/examples/vocab.txt"
REF_AUDIO="infer/examples/basic_ref_en.wav"
REF_TEXT="Some call me nature, others call me mother nature."
GEN_TEXT="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring."

# Output Configuration
OUTPUT_DIR="output"
OUTPUT_FILE="infer_cli_basic_new.wav"

# Inference Configuration
NFE_STEP="50"
TIMESTEP_MAPPING="power"
TIMESTEP_POWER="2.0"
SHIFT="3.0"
DEVICE="cuda"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

# python -m debugpy --listen 127.0.0.1:56789 src/wavtts/infer/infer_cli.py \
python src/wavtts/infer/infer_cli.py \
    --model_cfg "${MODEL_CFG}" \
    --ckpt_file "${CKPT_FILE}" \
    --ref_audio "${REF_AUDIO}" \
    --ref_text "${REF_TEXT}" \
    --gen_text "${GEN_TEXT}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_file "${OUTPUT_FILE}" \
    --vocab_file "${VOCAB_FILE}" \
    --nfe_step "${NFE_STEP}" \
    --timestep_mapping "${TIMESTEP_MAPPING}" \
    --timestep_power "${TIMESTEP_POWER}" \
    --shift "${SHIFT}" \
    --device "${DEVICE}"

# bash src/wavtts/infer/infer.sh
