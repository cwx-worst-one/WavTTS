#!/bin/bash

# WavTTS batch inference examples.
# Use an explicit checkpoint path because WavTTS configs no longer include legacy model fallbacks.

CKPT_PATH=/path/to/wavtts/ckpts/model_1000000.pt
MODEL_NAME=WavTTS

# e.g. WavTTS, 50 NFE
accelerate launch src/wavtts/eval/eval_infer_batch.py -s 0 -n "${MODEL_NAME}" -t "seedtts_test_zh" -c 1000000 -nfe 50 --ckpt_path "${CKPT_PATH}"
accelerate launch src/wavtts/eval/eval_infer_batch.py -s 0 -n "${MODEL_NAME}" -t "seedtts_test_en" -c 1000000 -nfe 50 --ckpt_path "${CKPT_PATH}"
accelerate launch src/wavtts/eval/eval_infer_batch.py -s 0 -n "${MODEL_NAME}" -t "ls_pc_test_clean" -c 1000000 -nfe 50 -p data/LibriSpeech/test-clean --ckpt_path "${CKPT_PATH}"

# e.g. evaluate WavTTS 50 NFE result on Seed-TTS test-zh
GEN_WAV_DIR=results/WavTTS/1000000/seedtts_test_zh/seed0_euler_nfe50_wav_power2.0_shift3.0_cfg3.0_speed1.0_load-fp32_infer-bf16_target_rms0.1
python src/wavtts/eval/eval_seedtts_testset.py -e wer -l zh --gen_wav_dir "${GEN_WAV_DIR}" --gpu_nums 8
python src/wavtts/eval/eval_seedtts_testset.py -e sim -l zh --gen_wav_dir "${GEN_WAV_DIR}" --gpu_nums 8
python src/wavtts/eval/eval_utmos.py --audio_dir "${GEN_WAV_DIR}"
