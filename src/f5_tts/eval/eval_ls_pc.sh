#!/bin/bash
set -e

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

eval_metric=("wer sim utmos")  # wer, sim, utmos
# eval_metric=("utmos")  # wer, sim, utmos

dataset="ls_pc_test_clean"  # ls_pc_test_clean, libritts_train_clean_100_cross_sentence, libritts_train_clean_100_same_sentence
ckpt_step=1600000    # 200000, 400000, 550000, 700000, 900000, 1000000, 1100000
seed=0
nfe_step=50         # 32, 50
cfg_strength=3.0
cfg_interval_min=0.0
cfg_interval_max=1.0
timestep_mapping="power"    # uniform, power, sway_sampling, logistic_normal
swaysampling=-1.0
timestep_power=2.0
timestep_logistic_normal_loc=-0.8
timestep_logistic_normal_scale=0.8
shift="7.0"
target_rms=0.1
use_ema=true                # true, false
LOAD_DTYPE="fp32"
INFER_DTYPE="bf16"
MEL_SPEC_TYPE="no_vocoder"
RESULTS_ROOT=/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav_mel_dev/results/F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k
# GPUS="[0,1,2,3,4,5,6,7]"
GPUS="[0,1,2,3]"
# GPUS="[0,1]"
# GPUS="[0]"
LS_TEST_CLEAN_PATH="data/LibriSpeech/test-clean"        # data/LibriSpeech/test-clean, data/LibriTTS/train-clean-100-cross-sentence, data/LibriTTS/train-clean-100-same-sentence
LOCAL=""

output_dir=${RESULTS_ROOT}/${ckpt_step}
gen_wav_subdir=seed${seed}_euler_nfe${nfe_step}_${MEL_SPEC_TYPE}
if [[ "${timestep_mapping}" == "uniform" ]]; then
    gen_wav_subdir+="_uniform"
fi
if [[ "${timestep_mapping}" == "sway_sampling" && "${swaysampling}" != "0" ]]; then
    gen_wav_subdir+="_ss${swaysampling}"
fi
if [[ "${timestep_mapping}" == "power" ]]; then
    gen_wav_subdir+="_power${timestep_power}"
fi
if [[ "${timestep_mapping}" == "logistic_normal" ]]; then
    gen_wav_subdir+="_lnloc${timestep_logistic_normal_loc}_lnscale${timestep_logistic_normal_scale}"
fi
if [[ "${shift}" != "1.0" ]]; then
    gen_wav_subdir+="_shift${shift}"
fi
gen_wav_subdir+="_cfg${cfg_strength}_speed1.0_load-${LOAD_DTYPE}_infer-${INFER_DTYPE}_cfgitv${cfg_interval_min}-${cfg_interval_max}_target_rms${target_rms}"   # _vocos_ss, _no_vocoder_ss, _speed1.0_load-fp32_infer-bf16, _speed1.0

if [[ "${use_ema}" == "false" ]]; then
    gen_wav_subdir+="_no_ema"
fi

gen_wav_dir=${output_dir}/${dataset}/${gen_wav_subdir}
echo "[INFO] evaluating ${gen_wav_dir}"

if [[ ! -d "${gen_wav_dir}" ]]; then
    echo "[ERROR] generated wav dir not found: ${gen_wav_dir}" >&2
    exit 1
fi

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
