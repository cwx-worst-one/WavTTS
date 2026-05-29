#!/bin/bash

# WavTTS batch inference examples.
# Use an explicit checkpoint path because WavTTS configs no longer include legacy model fallbacks.

CKPT_PATH="hf://worstchan/WavTTS/model_1200000.pt"
MODEL_NAME=WavTTS
GPU_NUMS=8
WAVLM_CKPT_DIR="<PATH_TO_WAVLM_LARGE_FINETUNE_PTH>"

# e.g. WavTTS, 50 NFE
accelerate launch --num_processes "${GPU_NUMS}" src/wavtts/eval/eval_infer_batch.py -s 0 -n "${MODEL_NAME}" -t "seedtts_test_zh" -c 1200000 -nfe 50 --ckpt_path "${CKPT_PATH}"
accelerate launch --num_processes "${GPU_NUMS}" src/wavtts/eval/eval_infer_batch.py -s 0 -n "${MODEL_NAME}" -t "seedtts_test_en" -c 1200000 -nfe 50 --ckpt_path "${CKPT_PATH}"
accelerate launch --num_processes "${GPU_NUMS}" src/wavtts/eval/eval_infer_batch.py -s 0 -n "${MODEL_NAME}" -t "ls_pc_test_clean" -c 1200000 -nfe 50 -p data/LibriSpeech/test-clean --ckpt_path "${CKPT_PATH}"

# e.g. evaluate WavTTS 50 NFE result on Seed-TTS test-zh
GEN_WAV_DIR=results/WavTTS/1200000/seedtts_test_zh/seed0_nfe50_wav_power2.0_shift3.0_cfg3.0
python src/wavtts/eval/eval_seedtts_testset.py -e wer -l zh --gen_wav_dir "${GEN_WAV_DIR}" --gpu_nums "${GPU_NUMS}"
python src/wavtts/eval/eval_seedtts_testset.py -e sim -l zh --gen_wav_dir "${GEN_WAV_DIR}" --gpu_nums "${GPU_NUMS}" --wavlm_ckpt_dir "$WAVLM_CKPT_DIR"
python src/wavtts/eval/eval_utmos.py --audio_dir "${GEN_WAV_DIR}"
