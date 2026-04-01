#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# ------------------------------
# 直接在这里改默认配置
# 运行时可直接:
#   bash src/f5_tts/eval/run_batch_eval_seedtts_results.sh
# MODELS 留空表示跑 results/ 下除 exp_libritts 之外的全部模型目录
# ------------------------------
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"

MODELS=(
    "F5TTS_v1_Large_wav_x_pred_scale_5_aux_mel_no_hubert_noise_schedule_0_8_16k_fix_mel_loss"
    "F5TTS_v1_Large_wav_x_pred_scale_5_aux_mel_no_hubert_noise_schedule_0_8_16k_mel_only_time_weighted_o1"
    "F5TTS_v1_Large_wav_x_pred_scale_5_aux_mel_no_hubert_noise_schedule_0_8_16k_time_weighted_p1"
    "F5TTS_v1_Large_wav_x_pred_scale_5_aux_mel_no_hubert_noise_schedule_0_8_16k_time_weighted"
    "F5TTS_v1_Large_wav_x_pred_scale_5_aux_mel_no_hubert_noise_schedule_0_8_16k_mel_only_time_weighted_o2"
)

STEPS=(
    200000
)

TASKS=(
    seedtts_test_zh
    seedtts_test_en
)

METRICS=(
    wer
    sim
    utmos
)

EVAL_GPUS="${EVAL_GPUS:-[0,1,2,3,4,5,6,7]}"
MASTER_PORT="${MASTER_PORT:-53721}"
LOCAL_FLAG=0
DRY_RUN_FLAG=0
STRICT_AUDIO_COUNT_FLAG=0
LIST_MODELS_FLAG=0

usage() {
    cat <<EOF
Usage:
  bash src/f5_tts/eval/run_batch_eval_seedtts_results.sh [options]

Options:
  --results-root PATH        Results 根目录，默认: ${RESULTS_ROOT}
  --models CSV               覆盖脚本内的 MODELS 配置，逗号分隔；all 表示全部
  --steps CSV                覆盖脚本内的 STEPS 配置，逗号分隔
  --tasks CSV                覆盖脚本内的 TASKS 配置，逗号分隔
  --metrics CSV              覆盖脚本内的 METRICS 配置，逗号分隔
  --gpus SPEC                GPU 配置，默认 [0,1,2,3,4,5,6,7]
  --master-port PORT         传给 eval_seedtts.sh 的 MASTER_PORT
  --local                    透传 --local 给评测脚本
  --dry-run                  仅打印将执行的项目
  --strict-audio-count       若发现音频条数不完整则返回非 0
  --list-models              列出可选模型目录后退出
  -h, --help                 显示帮助

Examples:
  bash src/f5_tts/eval/run_batch_eval_seedtts_results.sh --list-models
  bash src/f5_tts/eval/run_batch_eval_seedtts_results.sh --models F5TTS_v1_Large_wav_xxx_emilia --steps 200000,400000
  bash src/f5_tts/eval/run_batch_eval_seedtts_results.sh --models all --steps 200000,400000 --dry-run
EOF
}

MODELS_OVERRIDE=""
STEPS_OVERRIDE=""
TASKS_OVERRIDE=""
METRICS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --models)
            MODELS_OVERRIDE="$2"
            shift 2
            ;;
        --steps)
            STEPS_OVERRIDE="$2"
            shift 2
            ;;
        --tasks)
            TASKS_OVERRIDE="$2"
            shift 2
            ;;
        --metrics)
            METRICS_OVERRIDE="$2"
            shift 2
            ;;
        --gpus)
            EVAL_GPUS="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --local)
            LOCAL_FLAG=1
            shift
            ;;
        --dry-run)
            DRY_RUN_FLAG=1
            shift
            ;;
        --strict-audio-count)
            STRICT_AUDIO_COUNT_FLAG=1
            shift
            ;;
        --list-models)
            LIST_MODELS_FLAG=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "${LIST_MODELS_FLAG}" -eq 1 ]]; then
    find "${RESULTS_ROOT}" -maxdepth 1 -mindepth 1 -type d | while read -r path; do
        base=$(basename "${path}")
        if [[ "${base}" != "exp_libritts" ]]; then
            echo "${base}"
        fi
    done | sort
    exit 0
fi

if [[ -n "${MODELS_OVERRIDE}" ]]; then
    if [[ "${MODELS_OVERRIDE}" == "all" ]]; then
        MODELS=()
    else
        IFS=',' read -r -a MODELS <<< "${MODELS_OVERRIDE}"
    fi
fi

if [[ -n "${STEPS_OVERRIDE}" ]]; then
    IFS=',' read -r -a STEPS <<< "${STEPS_OVERRIDE}"
fi

if [[ -n "${TASKS_OVERRIDE}" ]]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_OVERRIDE}"
fi

if [[ -n "${METRICS_OVERRIDE}" ]]; then
    IFS=',' read -r -a METRICS <<< "${METRICS_OVERRIDE}"
fi

cmd=(python src/f5_tts/eval/batch_eval_seedtts_results.py
    --results-root "${RESULTS_ROOT}"
    --eval-gpus "${EVAL_GPUS}"
    --master-port "${MASTER_PORT}")

cmd+=(--steps)
for step in "${STEPS[@]}"; do
    [[ -n "${step}" ]] && cmd+=("${step}")
done

cmd+=(--tasks)
for task in "${TASKS[@]}"; do
    [[ -n "${task}" ]] && cmd+=("${task}")
done

cmd+=(--metrics)
for metric in "${METRICS[@]}"; do
    [[ -n "${metric}" ]] && cmd+=("${metric}")
done

if [[ "${#MODELS[@]}" -gt 0 ]]; then
    cmd+=(--models)
    for model in "${MODELS[@]}"; do
        [[ -n "${model}" ]] && cmd+=("${model}")
    done
fi

if [[ "${LOCAL_FLAG}" -eq 1 ]]; then
    cmd+=(--local)
fi

if [[ "${DRY_RUN_FLAG}" -eq 1 ]]; then
    cmd+=(--dry-run)
fi

if [[ "${STRICT_AUDIO_COUNT_FLAG}" -eq 1 ]]; then
    cmd+=(--strict-audio-count)
fi

echo "[INFO] Running: ${cmd[*]}"
cd "${REPO_ROOT}"
"${cmd[@]}"

# bash src/f5_tts/eval/run_batch_eval_seedtts_results.sh