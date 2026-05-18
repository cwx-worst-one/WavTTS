#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH=/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cublas/lib:/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cudnn/lib
# Evaluate end-to-end (non zero-shot) TTS model results.
# The input is a directory that directly contains generated wav files, e.g.:
#   bash src/f5_tts/eval/eval_e2e_models.sh \
#     /mnt/bn/jdy-lq-5/chenwenxi/data/e2e_eval_result/vits/ls_pc_test_clean_vits_ljs
#
# Only WER and UTMOS are evaluated. SIM is intentionally omitted because these
# end-to-end models are not zero-shot TTS models.

# ==============================================================================
# User-overridable config
# ===============================================================================

# E2E_RESULTS_ROOT=/mnt/bn/jdy-lq-5/chenwenxi/data/e2e_eval_result
# GEN_WAV_DIR="${1:-}"
GEN_WAV_DIR="/mnt/bn/jdy-lq-5/chenwenxi/data/e2e_eval_result/WavTTS/ljspeech_inset_test_9s/1500000/seed0_euler_nfe50_no_vocoder_power2.0_shift7.0_cfg3.0_speed1.0_load-fp32_infer-bf16_cfgitv0.0-1.0_target_rms0.1"

# Metrics to run, separated by spaces. Supported: wer utmos
EVAL_METRICS="wer utmos"

# GPU list passed to WER evaluation, e.g. "[0]" or "[0,1,2,3]".
GPUS="${GPUS_OVERRIDE:-${GPUS:-[0,1,2,3]}}"

# Set to 1 to re-run a metric even if its result file already exists.
FORCE_EVAL="${FORCE_EVAL:-0}"

# Set to 1 to print commands/tasks without running evaluation.
DRY_RUN="${DRY_RUN:-0}"

# Set to 1 to pass --local to eval_librispeech_test_clean.py.
LOCAL_EVAL="${LOCAL_OVERRIDE:-${LOCAL_EVAL:-0}}"

# Paths used by the LibriSpeech-PC eval script.
LS_TEST_CLEAN_PATH="${LS_TEST_CLEAN_PATH:-data/LibriSpeech/test-clean}"

# Environment defaults. Usually no need to edit these unless running on a new machine.
MASTER_PORT_VALUE="${MASTER_PORT_OVERRIDE:-${MASTER_PORT:-53721}}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
LD_LIBRARY_PATH_VALUE="${LD_LIBRARY_PATH:-/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cublas/lib:/mnt/bn/jdy-lq-5/chenwenxi/code/env/f5-tts/lib/python3.11/site-packages/nvidia/cudnn/lib}"
PROXY_VALUE="${HTTP_PROXY:-http://sys-proxy-rd-relay.byted.org:8118}"

# ==============================================================================
# Implementation below. In normal use, only edit the config block above.
# ===============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

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

usage() {
    cat <<USAGE
Usage:
  bash src/f5_tts/eval/eval_e2e_models.sh <generated_wav_dir>

Example:
  bash src/f5_tts/eval/eval_e2e_models.sh /mnt/bn/jdy-lq-5/chenwenxi/data/e2e_eval_result/vits/ls_pc_test_clean_vits_ljs

Environment overrides:
  EVAL_METRICS="wer utmos"      Metrics to run. Supported: wer utmos
  GPUS="[0,1,2,3]"              GPU ids for WER ASR evaluation
  FORCE_EVAL=1                  Recompute existing metric files
  DRY_RUN=1                     Print what would run
  LS_TEST_CLEAN_PATH=...        LibriSpeech test-clean path for ls_pc_test_clean
USAGE
}

if [[ -z "${GEN_WAV_DIR}" || "${GEN_WAV_DIR}" == "-h" || "${GEN_WAV_DIR}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${GEN_WAV_DIR}" != /* ]]; then
    GEN_WAV_DIR="${E2E_RESULTS_ROOT}/${GEN_WAV_DIR}"
fi

if [[ ! -d "${GEN_WAV_DIR}" ]]; then
    echo "[ERROR] generated wav dir not found: ${GEN_WAV_DIR}" >&2
    exit 1
fi

wav_count=$(find -L "${GEN_WAV_DIR}" -maxdepth 1 -type f -name '*.wav' | wc -l)
if [[ "${wav_count}" -eq 0 ]]; then
    echo "[ERROR] no wav files found directly under: ${GEN_WAV_DIR}" >&2
    exit 1
fi

IFS=' ' read -r -a eval_metric <<< "${EVAL_METRICS}"

metric_enabled() {
    local needle="$1"
    [[ " ${eval_metric[*]} " =~ " ${needle} " ]]
}

metric_result_file() {
    case "$1" in
        wer) echo "_wer_results.jsonl" ;;
        utmos) echo "_utmos_results.jsonl" ;;
        *) echo "" ;;
    esac
}

maybe_skip_metric() {
    local metric="$1"
    local result_file
    result_file=$(metric_result_file "${metric}")
    if [[ "${FORCE_EVAL}" != "1" && -n "${result_file}" && -s "${GEN_WAV_DIR}/${result_file}" ]]; then
        echo "[SKIP] ${metric} exists: ${GEN_WAV_DIR}/${result_file}"
        return 0
    fi
    return 1
}

run_or_print() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[DRY-RUN]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

infer_task() {
    local base
    base=$(basename "${GEN_WAV_DIR}")

    # LibriSpeech-PC style directories contain utterance ids like 1188-133604-0001.wav.
    if [[ "${base}" == *"ls_pc_test_clean"* || "${base}" == *"librispeech_test_pc"* ]]; then
        echo "ls_pc_test_clean"
        return 0
    fi

    # LJSpeech inset directories contain utterance ids like LJ001-0016.wav and use
    # data/ljspeech_inset_test_9s.lst. Keep this here so the script can also eval
    # the other e2e_eval_result dirs with 682 LJ*.wav files.
    if [[ "${base}" == *"ljspeech"* || "${base}" == *"F5TTS_ljspeech_eval"* ]]; then
        echo "ljspeech_inset_test_9s"
        return 0
    fi

    # Fallback by wav filename pattern.
    if find -L "${GEN_WAV_DIR}" -maxdepth 1 -type f -name 'LJ*.wav' -print -quit | grep -q .; then
        echo "ljspeech_inset_test_9s"
        return 0
    fi
    if find -L "${GEN_WAV_DIR}" -maxdepth 1 -type f -regextype posix-extended -regex '.*/[0-9]+-[0-9]+-[0-9]+\.wav' -print -quit | grep -q .; then
        echo "ls_pc_test_clean"
        return 0
    fi

    echo "unknown"
}

LOCAL=""
case "${LOCAL_EVAL}" in
    1|true|TRUE|yes|YES|--local)
        LOCAL="--local"
        ;;
esac

run_wer_ljspeech() {
    local metalst="${REPO_ROOT}/data/ljspeech_inset_test_9s.lst"
    local result_path="${GEN_WAV_DIR}/_wer_results.jsonl"

    if [[ ! -f "${metalst}" ]]; then
        echo "[ERROR] LJSpeech metadata not found: ${metalst}" >&2
        exit 1
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY-RUN] python <inline-ljspeech-wer> --gen_wav_dir ${GEN_WAV_DIR} --metalst ${metalst} --gpus ${GPUS}"
        return 0
    fi

    python - "${GEN_WAV_DIR}" "${metalst}" "${GPUS}" "${LOCAL_EVAL}" <<'PY'
import argparse
import ast
import json
import multiprocessing as mp
import os
import sys

import numpy as np

from f5_tts.eval.utils_eval import run_asr_wer


def parse_gpu_nums(gpu_nums_str):
    try:
        if gpu_nums_str.startswith("[") and gpu_nums_str.endswith("]"):
            gpu_list = ast.literal_eval(gpu_nums_str)
            if isinstance(gpu_list, list):
                return gpu_list
        return list(range(int(gpu_nums_str)))
    except (ValueError, SyntaxError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid GPU specification: {gpu_nums_str}. Use a number (e.g., 8) or a list (e.g., [0,1,2,3])"
        ) from exc


def main():
    gen_wav_dir, metalst, gpu_nums, local_eval = sys.argv[1:5]
    gpus = parse_gpu_nums(gpu_nums)
    asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3" if local_eval in {"1", "true", "TRUE", "yes", "YES", "--local"} else ""

    test_set_all = []
    missing = []
    with open(metalst, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            _ref_utt, _ref_dur, _ref_txt, gen_utt, _gen_dur, gen_txt = parts[:6]
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")
            if not os.path.exists(gen_wav):
                missing.append(gen_wav)
                continue
            # prompt_wav is not used by WER; keep a placeholder for run_asr_wer's tuple shape.
            test_set_all.append((gen_wav, "", gen_txt))

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} generated wavs required by {metalst}. First missing files:\n{preview}")
    if not test_set_all:
        raise RuntimeError(f"No eval samples matched between {metalst} and {gen_wav_dir}")

    if len(gpus) == 1:
        test_set = [(gpus[0], test_set_all)]
    else:
        wav_per_job = len(test_set_all) // len(gpus) + 1
        test_set = [(gpus[i], test_set_all[i * wav_per_job : (i + 1) * wav_per_job]) for i in range(len(gpus))]
        test_set = [(rank, subset) for rank, subset in test_set if subset]

    full_results = []
    with mp.Pool(processes=len(test_set)) as pool:
        args = [(rank, "en", subset, asr_ckpt_dir) for rank, subset in test_set]
        results = pool.map(run_asr_wer, args)
        for result in results:
            full_results.extend(result)

    result_path = os.path.join(gen_wav_dir, "_wer_results.jsonl")
    metrics = []
    with open(result_path, "w", encoding="utf-8") as f:
        for line in full_results:
            metrics.append(line["wer"])
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        metric = round(float(np.mean(metrics)), 5)
        f.write(f"\nWER: {metric}\n")

    print(f"\nTotal {len(metrics)} samples")
    print(f"WER: {metric}")
    print(f"WER results saved to {result_path}")


if __name__ == "__main__":
    main()
PY

    echo "[INFO] WER results saved to ${result_path}"
}

cd "${REPO_ROOT}"

task=$(infer_task)

echo "[INFO] repo root: ${REPO_ROOT}"
echo "[INFO] generated wav dir: ${GEN_WAV_DIR}"
echo "[INFO] wav count: ${wav_count}"
echo "[INFO] inferred task: ${task}"
echo "[INFO] metrics: ${eval_metric[*]}"
echo "[INFO] gpus: ${GPUS}"

for metric in "${eval_metric[@]}"; do
    case "${metric}" in
        wer|utmos) ;;
        sim)
            echo "[WARN] skip sim: e2e models are not zero-shot TTS models."
            ;;
        *)
            echo "[ERROR] unsupported metric: ${metric}. Supported: wer utmos" >&2
            exit 1
            ;;
    esac
done

if metric_enabled wer; then
    if ! maybe_skip_metric wer; then
        case "${task}" in
            ls_pc_test_clean)
                run_or_print python src/f5_tts/eval/eval_librispeech_test_clean.py \
                    -e wer -g "${GEN_WAV_DIR}" -n "${GPUS}" -p "${LS_TEST_CLEAN_PATH}" -t ls_pc_test_clean ${LOCAL}
                ;;
            ljspeech_inset_test_9s)
                run_wer_ljspeech
                ;;
            *)
                echo "[ERROR] cannot infer WER metadata for ${GEN_WAV_DIR}" >&2
                echo "[ERROR] supported layouts: ls_pc_test_clean/librispeech_test_pc or ljspeech/F5TTS_ljspeech_eval wav dirs" >&2
                exit 1
                ;;
        esac
    fi
fi

if metric_enabled utmos; then
    if ! maybe_skip_metric utmos; then
        run_or_print python src/f5_tts/eval/eval_utmos.py --audio_dir "${GEN_WAV_DIR}"
    fi
fi

echo "[INFO] done: ${GEN_WAV_DIR}"

# bash src/f5_tts/eval/eval_e2e_models.sh
