#!/bin/bash

# 环境变量（可按需修改）
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${MASTER_PORT_OVERRIDE:-53721}"
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cublas/lib:/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cudnn/lib
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

IFS=' ' read -r -a eval_metric <<< "${EVAL_METRICS_OVERRIDE:-wer sim utmos}"
# eval_metric=("sim")

lang="${LANG_OVERRIDE:-zh}"  # en, zh, zh_hard
ckpt_step="${CKPT_STEP_OVERRIDE:-200000}"    # 550000, 700000, 900000, 1000000, 1100000
nfe_step="${NFE_STEP_OVERRIDE:-32}"
cfg_strength="${CFG_STRENGTH_OVERRIDE:-3.0}"
output_dir="${OUTPUT_DIR_OVERRIDE:-/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav/results/F5TTS_v1_Large_wav_x_pred_scale_aux_mel_hubert_noise_schedule_0_8_16k_audio_convmlp_inout/${ckpt_step}}"
gen_wav_dir="${GEN_WAV_DIR_OVERRIDE:-${output_dir}/seedtts_test_${lang}/seed0_euler_nfe${nfe_step}_no_vocoder_ss-1.0_cfg${cfg_strength}_speed1.0}"   # _vocos_ss, _no_vocoder_ss
GPUS="${GPUS_OVERRIDE:-[0,1,2,3,4,5,6,7]}"
# GPUS="[0,1,2,3]"
# GPUS="[0,1]"

LOCAL=""
case "${LOCAL_OVERRIDE:-}" in
    1|true|TRUE|yes|YES|--local)
        LOCAL="--local"
        ;;
esac

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

# bash src/f5_tts/eval/eval_seedtts_param.sh
