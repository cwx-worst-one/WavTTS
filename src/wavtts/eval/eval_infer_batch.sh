#!/bin/bash
set -e
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export MASTER_PORT=53721
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Configuration parameters
MODEL_NAME=WavTTS
RESULT_MODEL_NAME="${MODEL_NAME}"
SEEDS=(0)
CKPTSTEPS=(1200000)
TASKS=("seedtts_test_en" "seedtts_test_zh")
GPUS="[0,1,2,3,4,5,6,7]"
TRAIN_GPU_TAG="8gpus"
CKPT_FILE="hf://worstchan/WavTTS/model_1200000.pt"
WAVLM_CKPT_DIR=""
LS_TEST_CLEAN_PATH="data/LibriSpeech/test-clean"

cfg_strength=3.0
nfe_step=50
timestep_mapping="power"    # power, sway_sampling, uniform
swaysampling=-1
timestep_power=2.0
shift="3.0"                 # 1.0, 3.0

# Parse arguments
INFER_ONLY=true  # true, false
while [[ $# -gt 0 ]]; do
    case $1 in
        --full-eval)
            INFER_ONLY=false
            shift
            ;;
        --infer-only)
            INFER_ONLY=true
            shift
            ;;
        --result-model-name)
            RESULT_MODEL_NAME="$2"
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
        --shift)
            shift="$2"
            shift 2
            ;;
        --train-gpu-tag)
            TRAIN_GPU_TAG="$2"
            shift 2
            ;;
        --wavlm-ckpt-dir)
            WAVLM_CKPT_DIR="$2"
            shift 2
            ;;
        --librispeech-test-clean-path)
            LS_TEST_CLEAN_PATH="$2"
            shift 2
            ;;
        *)
            echo "======== Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "======== Starting WavTTS batch evaluation task..."
echo "======== Result model name: ${RESULT_MODEL_NAME}"
echo "======== Timestep mapping: ${timestep_mapping}"
echo "======== Train GPU tag: ${TRAIN_GPU_TAG}"
if [ -n "${WAVLM_CKPT_DIR}" ]; then
    echo "======== WavLM checkpoint: ${WAVLM_CKPT_DIR}"
fi
RESULT_EXP_SUFFIX=""
if [ "${TRAIN_GPU_TAG}" != "8gpus" ]; then
    RESULT_EXP_SUFFIX="_${TRAIN_GPU_TAG}"
fi
RESULT_EXPNAME="${RESULT_MODEL_NAME}${RESULT_EXP_SUFFIX}"

if [ "${timestep_mapping}" = "power" ]; then
    echo "======== Timestep power: ${timestep_power}"
fi
if [ "${shift}" != "1.0" ]; then
    echo "======== Shift: ${shift}"
fi
if [ "$INFER_ONLY" = true ]; then
    echo "======== Mode: Execute infer tasks only"
else
    echo "======== Mode: Execute full pipeline (infer + eval)"
    if [ -z "${WAVLM_CKPT_DIR}" ]; then
        echo "======== WavLM checkpoint is required for SIM evaluation. Pass --wavlm-ckpt-dir <path>."
        exit 1
    fi
fi


# Function: Execute eval tasks
execute_eval_tasks() {
    local ckptstep=$1
    local seed=$2
    local task_name=$3
    
    local gen_wav_dir="results/${RESULT_EXPNAME}/${ckptstep}/${task_name}/seed${seed}_nfe${nfe_step}_wav"
    if [ "${timestep_mapping}" = "uniform" ]; then
        gen_wav_dir+="_uniform"
    fi
    if [ "${timestep_mapping}" = "sway_sampling" ] && [ "${swaysampling}" != "0" ]; then
        gen_wav_dir+="_ss${swaysampling}"
    fi
    if [ "${timestep_mapping}" = "power" ]; then
        gen_wav_dir+="_power${timestep_power}"
    fi
    if [ "${shift}" != "1.0" ]; then
        gen_wav_dir+="_shift${shift}"
    fi
    gen_wav_dir+="_cfg${cfg_strength}"

    echo ">>>>>>>> Starting eval task: ckptstep=${ckptstep}, seed=${seed}, task=${task_name}, gen_wav_dir=${gen_wav_dir}"
    
    local wavlm_ckpt_args=()
    if [ -n "${WAVLM_CKPT_DIR}" ]; then
        wavlm_ckpt_args=(--wavlm_ckpt_dir "${WAVLM_CKPT_DIR}")
    fi

    case $task_name in
        "seedtts_test_zh")
            python src/wavtts/eval/eval_seedtts_testset.py -e wer -l zh -g "$gen_wav_dir" -n "$GPUS"
            python src/wavtts/eval/eval_seedtts_testset.py -e sim -l zh -g "$gen_wav_dir" -n "$GPUS" "${wavlm_ckpt_args[@]}"
            python src/wavtts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
            ;;
        "seedtts_test_en")
            python src/wavtts/eval/eval_seedtts_testset.py -e wer -l en -g "$gen_wav_dir" -n "$GPUS"
            python src/wavtts/eval/eval_seedtts_testset.py -e sim -l en -g "$gen_wav_dir" -n "$GPUS" "${wavlm_ckpt_args[@]}"
            python src/wavtts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
            ;;
        "ls_pc_test_clean")
            python src/wavtts/eval/eval_librispeech_test_clean.py -e wer -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH"
            python src/wavtts/eval/eval_librispeech_test_clean.py -e sim -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH" "${wavlm_ckpt_args[@]}"
            python src/wavtts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
            ;;
    esac

    echo ">>>>>>>> Completed eval task: ckptstep=${ckptstep}, seed=${seed}, task=${task_name}"
}

# Main execution loop
for ckptstep in "${CKPTSTEPS[@]}"; do
    CKPT_PATH="${CKPT_FILE}"

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
            echo ">>>>>>>> Executing infer task: accelerate launch src/wavtts/eval/eval_infer_batch.py -s ${seed} -n \"${MODEL_NAME}\" -t \"${task}\" -c ${ckptstep} --ckpt_path \"${CKPT_PATH}\" --result_expname \"${RESULT_EXPNAME}\" --cfg_strength ${cfg_strength} --nfe_step ${nfe_step} --swaysampling ${swaysampling} --timestep_mapping ${timestep_mapping} --timestep_power ${timestep_power} --shift ${shift} -p \"${LS_TEST_CLEAN_PATH}\""

            # Execute infer task
            accelerate launch --main_process_port ${MASTER_PORT} src/wavtts/eval/eval_infer_batch.py -s ${seed} -n "${MODEL_NAME}" -t "${task}" -c ${ckptstep} --ckpt_path "${CKPT_PATH}" --result_expname "${RESULT_EXPNAME}" --nfe_step ${nfe_step} --swaysampling ${swaysampling} --timestep_mapping ${timestep_mapping} --timestep_power ${timestep_power} --shift ${shift} --cfg_strength ${cfg_strength} -p "${LS_TEST_CLEAN_PATH}"

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

# bash src/wavtts/eval/eval_infer_batch.sh
