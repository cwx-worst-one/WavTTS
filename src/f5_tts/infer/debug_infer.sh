#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export MASTER_ADDR="127.0.0.1"

# model config
model_name="F5TTS_v1_Large_wav_x_pred_scale_aux_mel"
model_dir="/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/F5TTS_v1_Large_wav_x_pred_scale_aux_mel-8gpus-bf16-19200sample_per_gpu"
training_step="550000"
model_cfg="${model_dir}/.hydra/config.yaml"
ckpt_file="${model_dir}/ckpts/model_${training_step}.pt"
vocab_file="/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav/data/LibriTTS_100_360_500_char/vocab.txt"
vocoder_name="no_vocoder"       # vocos, bigvgan, no_vocoder

ref_audio="infer/examples/basic/basic_ref_en.wav"
ref_text="Some call me nature, others call me mother nature."
gen_text="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring."
output_dir="src/f5_tts/infer/debug/${model_name}_output"
nfe_step="32"
output_file="${training_step}_nfe_${nfe_step}.wav"


# python src/f5_tts/infer/infer_cli.py \
python -m debugpy --listen 127.0.0.1:56789 src/f5_tts/infer/infer_cli.py \
    --model_cfg "$model_cfg" \
    --ckpt_file "$ckpt_file" \
    --ref_audio "$ref_audio" \
    --ref_text "$ref_text" \
    --gen_text "$gen_text" \
    --nfe_step "$nfe_step" \
    --output_dir "$output_dir" \
    --output_file "$output_file" \
    --vocab_file "$vocab_file" \
    --vocoder_name "$vocoder_name" \
 

# bash src/f5_tts/infer/debug_infer.sh