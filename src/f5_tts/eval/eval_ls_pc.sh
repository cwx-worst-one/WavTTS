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

dataset="ls_pc_test_clean"
ckpt_step=900000
nfe_step=32
output_dir=/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav/results/F5TTS_v1_Large_wav_x_pred_aux_mel/${ckpt_step}
gen_wav_dir=${output_dir}/${dataset}/seed0_euler_nfe${nfe_step}_no_vocoder_ss-1.0_cfg2.0_speed1.0   # seed0_euler_nfe32_vocos_ss, seed0_euler_nfe32_no_vocoder_ss
GPUS="[0,1]"
LS_TEST_CLEAN_PATH="data/LibriSpeech/test-clean"
LOCAL=""

if [[ " ${eval_metric[@]} " =~ " wer " ]]; then
    python src/f5_tts/eval/eval_librispeech_test_clean.py \
        -e wer -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH" $LOCAL
fi

if [[ " ${eval_metric[@]} " =~ " sim " ]]; then
    python src/f5_tts/eval/eval_librispeech_test_clean.py \
        -e sim -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH" $LOCAL
fi

if [[ " ${eval_metric[@]} " =~ " utmos " ]]; then
    python src/f5_tts/eval/eval_utmos.py \
        --audio_dir "$gen_wav_dir"
fi

# bash src/f5_tts/eval/eval_ls_pc.sh