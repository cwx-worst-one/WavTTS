#!/bin/bash

# 环境变量（可按需修改）
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=53721
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cublas/lib:/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cudnn/lib
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

eval_metric=("wer sim utmos")
# eval_metric=("sim")

lang="zh"  # en, zh, zh_hard
ckpt_step=1200000    # 550000, 700000, 900000, 1000000, 1100000
nfe_step=32
cfg_strength=3.0
output_dir=/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav/results/F5TTS_v1_Large_wav_x_pred_scale_aux_mel_noise_schedule_0_8_16k_emilia/${ckpt_step}
gen_wav_dir=${output_dir}/seedtts_test_${lang}/seed0_euler_nfe${nfe_step}_no_vocoder_ss-1.0_cfg${cfg_strength}_speed1.0   # _vocos_ss, _no_vocoder_ss
GPUS="[0,1,2,3,4,5,6,7]"
# GPUS="[0,1,2,3]"
# GPUS="[0,1]"

LOCAL=""

if [[ " ${eval_metric[@]} " =~ " wer " ]]; then
    python src/f5_tts/eval/eval_seedtts_testset.py \
        -e wer -l "$lang" -g "$gen_wav_dir" -n "$GPUS" $LOCAL
fi

if [[ " ${eval_metric[@]} " =~ " sim " ]]; then
    python src/f5_tts/eval/eval_seedtts_testset.py \
        -e sim -l "$lang" -g "$gen_wav_dir" -n "$GPUS" $LOCAL
fi

if [[ " ${eval_metric[@]} " =~ " utmos " ]]; then
    python src/f5_tts/eval/eval_utmos.py \
        --audio_dir "$gen_wav_dir"
fi

# bash src/f5_tts/eval/eval_seedtts.sh
