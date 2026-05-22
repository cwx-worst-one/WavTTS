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

eval_metric=("wer sim utmos")
# eval_metric=("sim")

# langs=("zh" "en")        # "en" "zh" "zh_hard"
langs=("en")
ckpt_steps=(600000)    # 200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1500000, 1600000
seed=0
nfe_step=50         # 32, 50, 100
ode_method="euler"
cfg_strength=3.0
timestep_mapping="power"    # uniform, power, sway_sampling
swaysampling=-1.0
timestep_power=2.0
shift="2.0"
target_rms=0.1
use_ema=true                # true, false
LOAD_DTYPE="fp32"
INFER_DTYPE="bf16"
RESULTS_ROOT=/mnt/bn/jdy-lq-5/chenwenxi/code/WavTTS_final_release/results/WavTTS_scale_9_16k
GPUS="[0,1,2,3,4,5,6,7]"
# GPUS="[0,1,2,3]"
# GPUS="[0,1]"
# GPUS="[0]"

LOCAL=""

gen_wav_subdir=seed${seed}_${ode_method}_nfe${nfe_step}_wav
if [[ "${timestep_mapping}" == "uniform" ]]; then
    gen_wav_subdir+="_uniform"
fi
if [[ "${timestep_mapping}" == "sway_sampling" && "${swaysampling}" != "0" ]]; then
    gen_wav_subdir+="_ss${swaysampling}"
fi
if [[ "${timestep_mapping}" == "power" ]]; then
    gen_wav_subdir+="_power${timestep_power}"
fi
if [[ "${shift}" != "1.0" ]]; then
    gen_wav_subdir+="_shift${shift}"
fi
gen_wav_subdir+="_cfg${cfg_strength}_speed1.0_load-${LOAD_DTYPE}_infer-${INFER_DTYPE}_target_rms${target_rms}"

if [[ "${use_ema}" == "false" ]]; then
    gen_wav_subdir+="_no_ema"
fi


for ckpt_step in "${ckpt_steps[@]}"; do
    output_dir=${RESULTS_ROOT}/${ckpt_step}
    echo "[INFO] processing ckpt_step=${ckpt_step}"

    for lang in "${langs[@]}"; do
        gen_wav_dir=${output_dir}/seedtts_test_${lang}/${gen_wav_subdir}
        echo "[INFO] evaluating ${gen_wav_dir}"

        if [[ ! -d "${gen_wav_dir}" ]]; then
            echo "[ERROR] generated wav dir not found: ${gen_wav_dir}" >&2
            exit 1
        fi

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
    done
done

# bash src/f5_tts/eval/eval_seedtts.sh
