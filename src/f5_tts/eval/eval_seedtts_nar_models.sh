#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH=/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cublas/lib:/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cudnn/lib
# Evaluate Seed-TTS results from other NAR models.
# Results are expected under /mnt/bn/jdy-lq-5/chenwenxi/data/nar_eval_result.
# Supported layouts:
#   ${NAR_RESULTS_ROOT}/${model}/seedtts_test_${lang}/${infer_subdir}/*.wav
#   ${NAR_RESULTS_ROOT}/${model}/seedtts_${lang}/*.wav
#
# Example:
#   bash src/f5_tts/eval/eval_seedtts_nar_models.sh
#   EVAL_METRICS_OVERRIDE="wer sim" LANGS_OVERRIDE="en" bash src/f5_tts/eval/eval_seedtts_nar_models.sh
#   MODELS_OVERRIDE="ZipVoice maskgct" FORCE_EVAL=1 bash src/f5_tts/eval/eval_seedtts_nar_models.sh

# ==============================================================================
# User config: edit this block to evaluate different models/results.
# ==============================================================================

# Root directory that contains one sub-directory per model.
NAR_RESULTS_ROOT="${NAR_RESULTS_ROOT:-/mnt/bn/jdy-lq-5/chenwenxi/data/nar_eval_result}"

# Models to evaluate, separated by spaces.
# Leave empty to evaluate all model directories under NAR_RESULTS_ROOT.
# Values can be directory names under NAR_RESULTS_ROOT or absolute paths.
# MODELS="${MODELS_OVERRIDE:-}"
MODELS="longcat1b"
LANGS="en zh"
# LANGS="zh"
EVAL_METRICS="wer sim utmos"
# EVAL_METRICS="utmos"
GPUS="[0,1,2,3]"

# Metrics to run, separated by spaces. Supported: wer sim utmos
# EVAL_METRICS="${EVAL_METRICS_OVERRIDE:-wer sim utmos}"

# Languages to evaluate, separated by spaces. Common values: en zh zh_hard
# LANGS="${LANGS_OVERRIDE:-en zh}"

# GPU list passed to eval_seedtts_testset.py -n, e.g. "[0]" or "[0,1,2,3]".
# GPUS="${GPUS_OVERRIDE:-[0,1,2,3]}"

# Set to 1 to re-run a metric even if its result file already exists.
FORCE_EVAL="${FORCE_EVAL:-0}"

# Set to 1 to print commands/tasks without running evaluation.
DRY_RUN="${DRY_RUN:-0}"

# Set to 1 to pass --local to eval_seedtts_testset.py.
LOCAL_EVAL="${LOCAL_OVERRIDE:-0}"

# Set to 1 to fail if a selected model/language has no generated wav directory.
STRICT_MISSING="${STRICT_MISSING:-0}"

# Environment defaults. Usually no need to edit these unless running on a new machine.
MASTER_PORT_VALUE="${MASTER_PORT_OVERRIDE:-${MASTER_PORT:-53721}}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
LD_LIBRARY_PATH_VALUE="${LD_LIBRARY_PATH:-/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cublas/lib:/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cudnn/lib}"
PROXY_VALUE="${HTTP_PROXY:-http://sys-proxy-rd-relay.byted.org:8118}"

# ==============================================================================
# Implementation below. In normal use, only edit the User config block above.
# ==============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# Environment variables (same as eval_seedtts.sh; can be overridden by caller where useful)
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning,ignore::FutureWarning}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT_VALUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_VALUE}"
export HTTP_PROXY="${PROXY_VALUE}"
export http_proxy="${http_proxy:-${PROXY_VALUE}}"
export https_proxy="${https_proxy:-${PROXY_VALUE}}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

IFS=' ' read -r -a eval_metric <<< "${EVAL_METRICS}"
IFS=' ' read -r -a langs <<< "${LANGS}"

LOCAL=""
case "${LOCAL_EVAL}" in
    1|true|TRUE|yes|YES|--local)
        LOCAL="--local"
        ;;
esac

metric_enabled() {
    local needle="$1"
    [[ " ${eval_metric[*]} " =~ " ${needle} " ]]
}

metric_result_file() {
    case "$1" in
        wer) echo "_wer_results.jsonl" ;;
        sim) echo "_sim_results.jsonl" ;;
        utmos) echo "_utmos_results.jsonl" ;;
        *) echo "" ;;
    esac
}

lang_to_dataset_dir() {
    case "$1" in
        en) echo "seedtts_test_en" ;;
        zh) echo "seedtts_test_zh" ;;
        zh_hard) echo "seedtts_test_zh" ;;
        *) echo "seedtts_test_$1" ;;
    esac
}

lang_to_flat_dir() {
    case "$1" in
        en) echo "seedtts_en" ;;
        zh) echo "seedtts_zh" ;;
        zh_hard) echo "seedtts_zh" ;;
        *) echo "seedtts_$1" ;;
    esac
}

collect_gen_dirs() {
    local model_dir="$1"
    local lang="$2"
    local dataset_dir flat_dir task_dir

    dataset_dir=$(lang_to_dataset_dir "${lang}")
    flat_dir=$(lang_to_flat_dir "${lang}")

    task_dir="${model_dir}/${dataset_dir}"
    if [[ -d "${task_dir}" ]]; then
        find -L "${task_dir}" -mindepth 1 -maxdepth 1 -type d | sort
    fi

    task_dir="${model_dir}/${flat_dir}"
    if [[ -d "${task_dir}" ]]; then
        # Some public model results put wavs directly under seedtts_en/seedtts_zh.
        if find -L "${task_dir}" -maxdepth 1 -type f -name '*.wav' -print -quit | grep -q .; then
            echo "${task_dir}"
        fi
        # Also support an extra inference-subdir level if it appears later.
        find -L "${task_dir}" -mindepth 1 -maxdepth 1 -type d | sort
    fi
}

run_metric() {
    local metric="$1"
    local lang="$2"
    local gen_wav_dir="$3"
    local result_file

    result_file=$(metric_result_file "${metric}")
    if [[ "${FORCE_EVAL}" != "1" && -n "${result_file}" && -s "${gen_wav_dir}/${result_file}" ]]; then
        echo "[SKIP] ${metric} exists: ${gen_wav_dir}/${result_file}"
        return 0
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY-RUN] ${metric} lang=${lang} dir=${gen_wav_dir}"
        return 0
    fi

    case "${metric}" in
        wer|sim)
            python src/f5_tts/eval/eval_seedtts_testset.py \
                -e "${metric}" -l "${lang}" -g "${gen_wav_dir}" -n "${GPUS}" ${LOCAL}
            ;;
        utmos)
            python src/f5_tts/eval/eval_utmos.py \
                --audio_dir "${gen_wav_dir}"
            ;;
        *)
            echo "[ERROR] unsupported metric: ${metric}" >&2
            exit 1
            ;;
    esac
}

if [[ ! -d "${NAR_RESULTS_ROOT}" ]]; then
    echo "[ERROR] NAR_RESULTS_ROOT not found: ${NAR_RESULTS_ROOT}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

echo "[INFO] repo root: ${REPO_ROOT}"
echo "[INFO] NAR results root: ${NAR_RESULTS_ROOT}"
echo "[INFO] metrics: ${eval_metric[*]}"
echo "[INFO] langs: ${langs[*]}"
echo "[INFO] gpus: ${GPUS}"

if [[ -n "${MODELS}" ]]; then
    IFS=' ' read -r -a models <<< "${MODELS}"
else
    mapfile -t models < <(find "${NAR_RESULTS_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ "${#models[@]}" -eq 0 ]]; then
    echo "[ERROR] no model directories found under ${NAR_RESULTS_ROOT}" >&2
    exit 1
fi

total_dirs=0
missing_lang_dirs=0

for model in "${models[@]}"; do
    if [[ "${model}" != /* ]]; then
        model="${NAR_RESULTS_ROOT}/${model}"
    fi

    if [[ ! -d "${model}" ]]; then
        echo "[WARN] skip missing model dir: ${model}" >&2
        continue
    fi

    model_name=$(basename "${model}")
    echo "[INFO] ===== model: ${model_name} ====="

    for lang in "${langs[@]}"; do
        mapfile -t gen_wav_dirs < <(collect_gen_dirs "${model}" "${lang}")
        if [[ "${#gen_wav_dirs[@]}" -eq 0 ]]; then
            echo "[WARN] no Seed-TTS ${lang} result dir found for ${model_name}"
            missing_lang_dirs=$((missing_lang_dirs + 1))
            continue
        fi

        for gen_wav_dir in "${gen_wav_dirs[@]}"; do
            wav_count=$(find -L "${gen_wav_dir}" -type f -name '*.wav' | wc -l)
            if [[ "${wav_count}" -eq 0 ]]; then
                echo "[WARN] skip empty wav dir: ${gen_wav_dir}"
                continue
            fi

            total_dirs=$((total_dirs + 1))
            echo "[INFO] evaluating lang=${lang} wavs=${wav_count} dir=${gen_wav_dir}"

            if metric_enabled wer; then
                run_metric wer "${lang}" "${gen_wav_dir}"
            fi
            if metric_enabled sim; then
                run_metric sim "${lang}" "${gen_wav_dir}"
            fi
            if metric_enabled utmos; then
                run_metric utmos "${lang}" "${gen_wav_dir}"
            fi
        done
    done
done

echo "[INFO] done. evaluated candidate dirs: ${total_dirs}"

if [[ "${STRICT_MISSING}" == "1" && "${missing_lang_dirs}" -gt 0 ]]; then
    echo "[ERROR] missing selected language dirs: ${missing_lang_dirs}" >&2
    exit 2
fi

# bash src/f5_tts/eval/eval_seedtts_nar_models.sh
