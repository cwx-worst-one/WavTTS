#!/bin/bash
set -e
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=53721
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES="0,1"
# export CUDA_VISIBLE_DEVICES="1"

# Configuration parameters
MODEL_NAME="F5TTS_v1_Large_wav_x_pred_proj_input"
# SEEDS=(0 1 2)
SEEDS=(0)
CKPTSTEPS=(550000)
# TASKS=("seedtts_test_zh" "seedtts_test_en" "ls_pc_test_clean")
# TASKS=("seedtts_test_zh" "seedtts_test_en")
TASKS=("ls_pc_test_clean")
LS_TEST_CLEAN_PATH="data/LibriSpeech/test-clean"
# GPUS="[0,1,2,3,4,5,6,7]"
GPUS="[0,1]"
OFFLINE_MODE=false
CKPT_PATH_DIR=/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/F5TTS_v1_Large_wav_x_pred_proj_input-8gpus-bf16-19200sample_per_gpu-t_eps_0.02-proj_input_768_1024
CKPT_PATH="${CKPT_PATH_DIR}/ckpts/model_${CKPTSTEPS}.pt"

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
        *)
            echo "======== Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "======== Starting F5-TTS batch evaluation task..."
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
    
    local gen_wav_dir="results/${MODEL_NAME}_${ckptstep}/${task_name}/seed${seed}_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0"
    
    echo ">>>>>>>> Starting eval task: ckptstep=${ckptstep}, seed=${seed}, task=${task_name}"
    
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
    echo "======== Processing ckptstep: ${ckptstep}"
    
    for seed in "${SEEDS[@]}"; do
        echo "-------- Processing seed: ${seed}"
        
        # Store eval task PIDs for current seed (if not infer-only mode)
        if [ "$INFER_ONLY" = false ]; then
            declare -a eval_pids
        fi
        
        # Execute each infer task sequentially
        for task in "${TASKS[@]}"; do
            echo ">>>>>>>> Executing infer task: accelerate launch src/f5_tts/eval/eval_infer_batch.py -s ${seed} -n \"${MODEL_NAME}\" -t \"${task}\" -c ${ckptstep} $LOCAL --ckpt_path \"${CKPT_PATH}\""
            
            # Execute infer task (foreground execution, wait for completion)
            if [ "$DEBUG" = false ]; then
                accelerate launch --main_process_port ${MASTER_PORT} src/f5_tts/eval/eval_infer_batch.py -s ${seed} -n "${MODEL_NAME}" -t "${task}" -c ${ckptstep} -p "${LS_TEST_CLEAN_PATH}" $LOCAL --ckpt_path "${CKPT_PATH}"
            fi

            if [ "$DEBUG" = true ]; then
                python -m debugpy --listen 127.0.0.1:56789 --wait-for-client src/f5_tts/eval/eval_infer_batch.py -s ${seed} -n "${MODEL_NAME}" -t "${task}" -c ${ckptstep} -p "${LS_TEST_CLEAN_PATH}" $LOCAL --ckpt_path "${CKPT_PATH}"
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