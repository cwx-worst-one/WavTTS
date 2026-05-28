#!/usr/bin/env bash
set -euo pipefail

# Model Configuration
MODEL_NAME="WavTTS"
MODEL_CFG="src/wavtts/configs/WavTTS.yaml"

# Zero-shot TTS Configuration
VOCAB_FILE="infer/examples/vocab.txt"
REF_AUDIO="infer/examples/basic_ref_en.wav"
REF_TEXT="Some call me nature, others call me mother nature."
GEN_TEXT="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall."

# Output Configuration
OUTPUT_DIR="output"
OUTPUT_FILE="infer_cli_basic.wav"

# Inference Configuration
NFE_STEP="50"
TIMESTEP_MAPPING="power"
TIMESTEP_POWER="2.0"
SHIFT="3.0"
DEVICE="cuda"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

python src/wavtts/infer/infer_cli.py \
    --model_cfg "${MODEL_CFG}" \
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
