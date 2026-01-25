#!/bin/bash

export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=53721
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib64/python3.11/site-packages/nvidia/cublas/lib:/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib64/python3.11/site-packages/nvidia/cudnn/lib
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

eval_metric=("wer sim utmos")  # wer, sim, utmos
# eval_metric=("sim utmos")  # wer, sim, utmos

dataset="libritts_train_clean_100_cross_sentence"  # ls_pc_test_clean, libritts_train_clean_100_cross_sentence, libritts_train_clean_100_same_sentence
ckpt_step=300000
nfe_step=32
output_dir=/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav/results/F5TTS_v1_Small/${ckpt_step}
gen_wav_dir=${output_dir}/${dataset}/seed0_euler_nfe${nfe_step}_vocos_ss-1.0_cfg2.0_speed1.0   # _vocos_ss, _no_vocoder_ss
GPUS="[0,1,2,3]"
# GPUS="[0,1]"
# GPUS="[0]"
LS_TEST_CLEAN_PATH="data/LibriTTS/train-clean-100-cross-sentence"        # data/LibriSpeech/test-clean, data/LibriTTS/train-clean-100-cross-sentence, data/LibriTTS/train-clean-100-same-sentence
LOCAL=""

if [[ " ${eval_metric[@]} " =~ " wer " ]]; then
    python src/f5_tts/eval/eval_librispeech_test_clean.py \
        -e wer -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH" -t "$dataset" $LOCAL
fi

if [[ " ${eval_metric[@]} " =~ " sim " ]]; then
    python src/f5_tts/eval/eval_librispeech_test_clean.py \
        -e sim -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH" -t "$dataset" $LOCAL
fi

if [[ " ${eval_metric[@]} " =~ " utmos " ]]; then
    python src/f5_tts/eval/eval_utmos.py \
        --audio_dir "$gen_wav_dir"
fi

# bash src/f5_tts/eval/eval_ls_pc.sh