#!/bin/bash

# WavTTS batch inference examples.
# Use an explicit checkpoint path because WavTTS configs no longer include legacy model fallbacks.

CKPT_PATH=/path/to/wavtts/ckpts/model_1000000.pt
MODEL_NAME=WavTTS_scale_8_16k

# e.g. WavTTS, 16 NFE
accelerate launch src/wavtts/eval/eval_infer_batch.py -s 0 -n "${MODEL_NAME}" -t "seedtts_test_zh" -nfe 16 --ckpt_path "${CKPT_PATH}"
accelerate launch src/wavtts/eval/eval_infer_batch.py -s 0 -n "${MODEL_NAME}" -t "seedtts_test_en" -nfe 16 --ckpt_path "${CKPT_PATH}"
accelerate launch src/wavtts/eval/eval_infer_batch.py -s 0 -n "${MODEL_NAME}" -t "ls_pc_test_clean" -nfe 16 -p data/LibriSpeech/test-clean --ckpt_path "${CKPT_PATH}"

# e.g. evaluate WavTTS 16 NFE result on Seed-TTS test-zh
GEN_WAV_DIR=results/WavTTS_scale_8_16k_1000000/seedtts_test_zh/seed0_euler_nfe16_wav_ss-1_cfg2.0_speed1.0
python src/wavtts/eval/eval_seedtts_testset.py -e wer -l zh --gen_wav_dir "${GEN_WAV_DIR}" --gpu_nums 8
python src/wavtts/eval/eval_seedtts_testset.py -e sim -l zh --gen_wav_dir "${GEN_WAV_DIR}" --gpu_nums 8
python src/wavtts/eval/eval_utmos.py --audio_dir "${GEN_WAV_DIR}"
