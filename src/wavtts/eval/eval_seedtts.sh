#!/bin/bash
set -e

# Environment variables; adjust as needed.
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export OMP_NUM_THREADS=1

eval_metric=("wer sim utmos")

langs=("zh" "en")
ckpt_steps=(1200000)
seed=0
nfe_step=50
cfg_strength=3.0
timestep_mapping="power"
timestep_power=2.0
shift="3.0"
RESULTS_ROOT=./results/WavTTS
GPUS="[0,1,2,3,4,5,6,7]"

LOCAL=""
WAVLM_CKPT_DIR="<PATH_TO_WAVLM_LARGE_FINETUNE_PTH>"

gen_wav_subdir=seed${seed}_nfe${nfe_step}_wav
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
gen_wav_subdir+="_cfg${cfg_strength}"


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
            python src/wavtts/eval/eval_seedtts_testset.py \
                -e wer -l "$lang" -g "$gen_wav_dir" -n "$GPUS" $LOCAL
        fi

        if [[ " ${eval_metric[@]} " =~ " sim " ]]; then
            python src/wavtts/eval/eval_seedtts_testset.py \
                -e sim -l "$lang" -g "$gen_wav_dir" -n "$GPUS" $LOCAL --wavlm_ckpt_dir "$WAVLM_CKPT_DIR"
        fi

        if [[ " ${eval_metric[@]} " =~ " utmos " ]]; then
            python src/wavtts/eval/eval_utmos.py \
                --audio_dir "$gen_wav_dir"
        fi
    done
done

# bash src/wavtts/eval/eval_seedtts.sh
