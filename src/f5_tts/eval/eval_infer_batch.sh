#!/bin/bash
set -e
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=53721
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="1"

# Configuration parameters
MODEL_NAME=F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_uniform_16k
RESULT_MODEL_NAME="${MODEL_NAME}"
MEL_SPEC_TYPE="no_vocoder"  # no_vocoder, vocos
# SEEDS=(0 1 2)
SEEDS=(0)
CKPTSTEPS=(400000)  # 200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000
# TASKS=("seedtts_test_zh" "seedtts_test_en" "ls_pc_test_clean")
TASKS=("seedtts_test_zh" "seedtts_test_en")
# TASKS=("seedtts_test_en")
# TASKS=("seedtts_test_zh")
# TASKS=("ls_pc_test_clean")
LS_TEST_CLEAN_PATH="data/LibriSpeech/test-clean"
GPUS="[0,1,2,3,4,5,6,7]"
# GPUS="[0,1,2,3]"
# GPUS="[0]"
TRAIN_GPU_TAG="8gpus"   # 8gpus, 16gpus, 32gpus
OFFLINE_MODE=false       # true, false
CKPT_PATH_DIR=/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/emilia/F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_uniform_16k-emilia-8gpus-19200sample_per_gpu-bf16

cfg_strength=3.0
cfg_interval_min=0.0
cfg_interval_max=1.0
nfe_step=50         # 16, 32, 50
ode_method="euler"  # euler, heun
timestep_mapping="power"   # uniform, sway_sampling, power, logistic_normal
swaysampling=-1
timestep_power=2.0
timestep_logistic_normal_loc=-0.8
timestep_logistic_normal_scale=0.8
shift="7.0"         # 1.0, 7.0
LOAD_DTYPE="fp32"   # bf16, fp16, fp32
INFER_DTYPE="bf16"  # bf16, fp16, fp32

DEBUG=false  # true, false

# Parse arguments
if [ $OFFLINE_MODE = true ]; then
    LOCAL="--local"
else
    LOCAL=""
fi
INFER_ONLY=true  # true, false
while [[ $# -gt 0 ]]; do
    case $1 in
        --infer-only)
            INFER_ONLY=true
            shift
            ;;
        --load-dtype)
            LOAD_DTYPE="$2"
            shift 2
            ;;
        --result-model-name)
            RESULT_MODEL_NAME="$2"
            shift 2
            ;;
        --infer-dtype)
            INFER_DTYPE="$2"
            shift 2
            ;;
        --ode-method)
            ode_method="$2"
            shift 2
            ;;
        --timestep-mapping)
            timestep_mapping="$2"
            shift 2
            ;;
        --timestep-power)
            timestep_power="$2"
            shift 2
            ;;
        --timestep-logistic-normal-loc)
            timestep_logistic_normal_loc="$2"
            shift 2
            ;;
        --timestep-logistic-normal-scale)
            timestep_logistic_normal_scale="$2"
            shift 2
            ;;
        --shift)
            shift="$2"
            shift 2
            ;;
        --train-gpu-tag)
            TRAIN_GPU_TAG="$2"
            shift 2
            ;;
        *)
            echo "======== Unknown parameter: $1"
            exit 1
            ;;
    esac
done

case "$ode_method" in
    euler|heun)
        ;;
    *)
        echo "======== Invalid ode method: ${ode_method}. Expected one of: euler, heun"
        exit 1
        ;;
esac

for dtype_name in "$LOAD_DTYPE" "$INFER_DTYPE"; do
    case "$dtype_name" in
        bf16|fp16|fp32)
            ;;
        *)
            echo "======== Invalid dtype: ${dtype_name}. Expected one of: bf16, fp16, fp32"
            exit 1
            ;;
    esac
done

echo "======== Starting F5-TTS batch evaluation task..."
echo "======== Load dtype: ${LOAD_DTYPE}"
echo "======== Infer dtype: ${INFER_DTYPE}"
echo "======== Result model name: ${RESULT_MODEL_NAME}"
echo "======== Timestep mapping: ${timestep_mapping}"
echo "======== ODE method: ${ode_method}"
echo "======== Train GPU tag: ${TRAIN_GPU_TAG}"
RESULT_EXP_SUFFIX=""
if [ "${TRAIN_GPU_TAG}" != "8gpus" ]; then
    RESULT_EXP_SUFFIX="_${TRAIN_GPU_TAG}"
fi
RESULT_EXPNAME="${RESULT_MODEL_NAME}${RESULT_EXP_SUFFIX}"

if [ "${timestep_mapping}" = "power" ]; then
    echo "======== Timestep power: ${timestep_power}"
fi
if [ "${timestep_mapping}" = "logistic_normal" ]; then
    echo "======== Timestep logistic normal loc: ${timestep_logistic_normal_loc}"
    echo "======== Timestep logistic normal scale: ${timestep_logistic_normal_scale}"
fi
if [ "${shift}" != "1.0" ]; then
    echo "======== Shift: ${shift}"
fi
if [ "$INFER_ONLY" = true ]; then
    echo "======== Mode: Execute infer tasks only"
else
    echo "======== Mode: Execute full pipeline (infer + eval)"
fi


# Function: Execute eval tasks
execute_eval_tasks() {
    local ckptstep=$1
    local seed=$2
    local task_name=$3
    
    local gen_wav_dir="results/${RESULT_EXPNAME}/${ckptstep}/${task_name}/seed${seed}_${ode_method}_nfe${nfe_step}_${MEL_SPEC_TYPE}"
    if [ "${timestep_mapping}" = "uniform" ]; then
        gen_wav_dir+="_uniform"
    fi
    if [ "${timestep_mapping}" = "sway_sampling" ] && [ "${swaysampling}" != "0" ]; then
        gen_wav_dir+="_ss${swaysampling}"
    fi
    if [ "${timestep_mapping}" = "power" ]; then
        gen_wav_dir+="_power${timestep_power}"
    fi
    if [ "${timestep_mapping}" = "logistic_normal" ]; then
        gen_wav_dir+="_lnloc${timestep_logistic_normal_loc}_lnscale${timestep_logistic_normal_scale}"
    fi
    if [ "${shift}" != "1.0" ]; then
        gen_wav_dir+="_shift${shift}"
    fi
    gen_wav_dir+="_cfg${cfg_strength}_speed1.0_load-${LOAD_DTYPE}_infer-${INFER_DTYPE}_cfgitv${cfg_interval_min}-${cfg_interval_max}"
    gen_wav_dir+="_target_rms0.1"
    
    echo ">>>>>>>> Starting eval task: ckptstep=${ckptstep}, seed=${seed}, task=${task_name}, gen_wav_dir=${gen_wav_dir}"
    
    case $task_name in
        "seedtts_test_zh")
            python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l zh -g "$gen_wav_dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l zh -g "$gen_wav_dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
            ;;
        "seedtts_test_en")
            python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l en -g "$gen_wav_dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l en -g "$gen_wav_dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
            ;;
        "ls_pc_test_clean")
            python src/f5_tts/eval/eval_librispeech_test_clean.py -e wer -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH" $LOCAL
            python src/f5_tts/eval/eval_librispeech_test_clean.py -e sim -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH" $LOCAL
            python src/f5_tts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
            ;;
    esac
    
    echo ">>>>>>>> Completed eval task: ckptstep=${ckptstep}, seed=${seed}, task=${task_name}"
}

# Main execution loop
for ckptstep in "${CKPTSTEPS[@]}"; do
    CKPT_PATH="${CKPT_PATH_DIR}/ckpts/model_${ckptstep}.pt"
    if [ ! -f "${CKPT_PATH}" ]; then
        echo "======== Checkpoint not found: ${CKPT_PATH}"
        exit 1
    fi

    echo "======== Processing ckptstep: ${ckptstep}"
    echo "======== Using checkpoint: ${CKPT_PATH}"
    
    for seed in "${SEEDS[@]}"; do
        echo "-------- Processing seed: ${seed}"
        
        # Store eval task PIDs for current seed (if not infer-only mode)
        if [ "$INFER_ONLY" = false ]; then
            declare -a eval_pids
        fi
        
        # Execute each infer task sequentially
        for task in "${TASKS[@]}"; do
            echo ">>>>>>>> Executing infer task: accelerate launch src/f5_tts/eval/eval_infer_batch.py -s ${seed} -n \"${MODEL_NAME}\" -t \"${task}\" -c ${ckptstep} $LOCAL --ckpt_path \"${CKPT_PATH}\" --result_expname \"${RESULT_EXPNAME}\" --odemethod ${ode_method} --cfg_strength ${cfg_strength} --cfg_scale_interval_min ${cfg_interval_min} --cfg_scale_interval_max ${cfg_interval_max} --nfe_step ${nfe_step} --swaysampling ${swaysampling} --timestep_mapping ${timestep_mapping} --timestep_power ${timestep_power} --timestep_logistic_normal_loc ${timestep_logistic_normal_loc} --timestep_logistic_normal_scale ${timestep_logistic_normal_scale} --shift ${shift} --load_dtype ${LOAD_DTYPE} --infer_dtype ${INFER_DTYPE}"
            
            # Execute infer task (foreground execution, wait for completion)
            if [ "$DEBUG" = false ]; then
                accelerate launch --main_process_port ${MASTER_PORT} src/f5_tts/eval/eval_infer_batch.py -s ${seed} -n "${MODEL_NAME}" -t "${task}" -c ${ckptstep} -p "${LS_TEST_CLEAN_PATH}" $LOCAL --ckpt_path "${CKPT_PATH}" --result_expname "${RESULT_EXPNAME}" --odemethod ${ode_method} --nfe_step ${nfe_step} --swaysampling ${swaysampling} --timestep_mapping ${timestep_mapping} --timestep_power ${timestep_power} --timestep_logistic_normal_loc ${timestep_logistic_normal_loc} --timestep_logistic_normal_scale ${timestep_logistic_normal_scale} --shift ${shift} --cfg_strength ${cfg_strength} --cfg_scale_interval_min ${cfg_interval_min} --cfg_scale_interval_max ${cfg_interval_max} --load_dtype ${LOAD_DTYPE} --infer_dtype ${INFER_DTYPE}
            fi

            if [ "$DEBUG" = true ]; then
                python -m debugpy --listen 127.0.0.1:56789 --wait-for-client src/f5_tts/eval/eval_infer_batch.py -s ${seed} -n "${MODEL_NAME}" -t "${task}" -c ${ckptstep} -p "${LS_TEST_CLEAN_PATH}" $LOCAL --ckpt_path "${CKPT_PATH}" --result_expname "${RESULT_EXPNAME}" --odemethod ${ode_method} --nfe_step ${nfe_step} --swaysampling ${swaysampling} --timestep_mapping ${timestep_mapping} --timestep_power ${timestep_power} --timestep_logistic_normal_loc ${timestep_logistic_normal_loc} --timestep_logistic_normal_scale ${timestep_logistic_normal_scale} --shift ${shift} --cfg_strength ${cfg_strength} --cfg_scale_interval_min ${cfg_interval_min} --cfg_scale_interval_max ${cfg_interval_max} --load_dtype ${LOAD_DTYPE} --infer_dtype ${INFER_DTYPE}
            fi
            
            # If not infer-only mode, launch corresponding eval task
            if [ "$INFER_ONLY" = false ]; then
                # Launch corresponding eval task (background execution, non-blocking for next infer)
                execute_eval_tasks $ckptstep $seed $task &
                eval_pids+=($!)
            fi
        done
        
        # If not infer-only mode, wait for all eval tasks of current seed to complete
        if [ "$INFER_ONLY" = false ]; then
            echo ">>>>>>>> All infer tasks for seed ${seed} completed, waiting for corresponding eval tasks to finish..."
            
            for pid in "${eval_pids[@]}"; do
                wait $pid
            done
            
            unset eval_pids  # Clean up array
        fi
        echo "-------- All eval tasks for seed ${seed} completed"
    done
    
    echo "======== Completed ckptstep: ${ckptstep}"
    echo
done

echo "======== All tasks completed!"

# bash src/f5_tts/eval/eval_infer_batch.sh
