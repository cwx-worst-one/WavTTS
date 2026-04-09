#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

# ------------------------------
# 默认复用 run_batch_eval_seedtts_results.sh 中的模型列表，
# 但这里跑的是 infer，不是只做 eval。
# 这些名字是 results / emilia 目录名，不是纯 exp_name。
# 脚本会自动解析：
#   <results_or_emilia_dir_name> -> exp_name + ckpt_path
# ------------------------------
EMILIA_ROOT="${EMILIA_ROOT:-/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/emilia}"

MODELS=(
    "F5TTS_v1_Large_wav_x_pred_scale_5_aux_mel_no_hubert_noise_schedule_0_8_16k_mel_only_time_weighted_o2-emilia-8gpus-19200sample_per_gpu-bf16"
)

STEPS=(
    200000
)

TASKS=(
    seedtts_test_zh
    seedtts_test_en
)

SEED="${SEED:-0}"
NFE_STEP="${NFE_STEP:-32}"
CFG_STRENGTH="${CFG_STRENGTH:-3.0}"
SWAYSAMPLING="${SWAYSAMPLING:--1.0}"
TIMESTEP_MAPPING="${TIMESTEP_MAPPING:-sway_sampling}"
TIMESTEP_POWER="${TIMESTEP_POWER:-}"
SHIFT="${SHIFT:-1.0}"
LOAD_DTYPE="${LOAD_DTYPE:-fp32}"
INFER_DTYPE="${INFER_DTYPE:-bf16}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-54721}"
LOCAL_FLAG=0
DRY_RUN_FLAG=0
FORCE_FLAG=0
LIST_MODELS_FLAG=0
STRICT_AUDIO_COUNT_FLAG=0

usage() {
    cat <<USAGE
Usage:
  bash src/f5_tts/eval/run_batch_infer_seedtts_selected.sh [options]

Options:
  --emilia-root PATH         Emilia 实验根目录，默认: ${EMILIA_ROOT}
  --models CSV              覆盖脚本内的 MODELS，逗号分隔；all 表示扫描 emilia-root 下全部目录
  --steps CSV               覆盖脚本内的 STEPS，逗号分隔
  --tasks CSV               覆盖脚本内的 TASKS，逗号分隔
  --seed N                  推理 seed，默认: ${SEED}
  --nfe-step N              ODE steps，默认: ${NFE_STEP}
  --cfg-strength X          CFG 强度，默认: ${CFG_STRENGTH}
  --swaysampling X          sway sampling，默认: ${SWAYSAMPLING}
  --timestep-mapping NAME   timestep mapping，默认: ${TIMESTEP_MAPPING}
  --timestep-power X        power mapping 参数
  --shift X                 推理采样 shift 参数
  --load-dtype TYPE         加载权重 dtype，默认: ${LOAD_DTYPE}
  --infer-dtype TYPE        推理 dtype，默认: ${INFER_DTYPE}
  --master-port-base PORT   起始端口，每个 combo 自动 +index，默认: ${MASTER_PORT_BASE}
  --local                   透传 --local 给 eval_infer_batch.py
  --force                   即使结果目录已存在，也继续重跑
  --strict-audio-count      若发现结果目录音频数不足，则返回非 0
  --dry-run                 仅打印计划，不实际执行
  --list-models             列出 emilia-root 下可选模型目录后退出
  -h, --help                显示帮助

Examples:
  bash src/f5_tts/eval/run_batch_infer_seedtts_selected.sh --dry-run
  bash src/f5_tts/eval/run_batch_infer_seedtts_selected.sh --models all --steps 200000 --tasks seedtts_test_zh,seedtts_test_en
USAGE
}

MODELS_OVERRIDE=""
STEPS_OVERRIDE=""
TASKS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emilia-root)
            EMILIA_ROOT="$2"
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
        --seed)
            SEED="$2"
            shift 2
            ;;
        --nfe-step)
            NFE_STEP="$2"
            shift 2
            ;;
        --cfg-strength)
            CFG_STRENGTH="$2"
            shift 2
            ;;
        --swaysampling)
            SWAYSAMPLING="$2"
            shift 2
            ;;
        --timestep-mapping)
            TIMESTEP_MAPPING="$2"
            shift 2
            ;;
        --timestep-power)
            TIMESTEP_POWER="$2"
            shift 2
            ;;
        --shift)
            SHIFT="$2"
            shift 2
            ;;
        --load-dtype)
            LOAD_DTYPE="$2"
            shift 2
            ;;
        --infer-dtype)
            INFER_DTYPE="$2"
            shift 2
            ;;
        --master-port-base)
            MASTER_PORT_BASE="$2"
            shift 2
            ;;
        --local)
            LOCAL_FLAG=1
            shift
            ;;
        --force)
            FORCE_FLAG=1
            shift
            ;;
        --strict-audio-count)
            STRICT_AUDIO_COUNT_FLAG=1
            shift
            ;;
        --dry-run)
            DRY_RUN_FLAG=1
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
    find "${EMILIA_ROOT}" -maxdepth 1 -mindepth 1 -type d | while read -r path; do
        basename "${path}"
    done | sort
    exit 0
fi

if [[ -n "${MODELS_OVERRIDE}" ]]; then
    if [[ "${MODELS_OVERRIDE}" == "all" ]]; then
        MODELS=()
        while IFS= read -r line; do
            [[ -n "${line}" ]] && MODELS+=("${line}")
        done < <(find "${EMILIA_ROOT}" -maxdepth 1 -mindepth 1 -type d | sed 's#.*/##' | sort)
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

if [[ ! -d "${EMILIA_ROOT}" ]]; then
    echo "[ERROR] emilia root not found: ${EMILIA_ROOT}" >&2
    exit 1
fi

if [[ "${#MODELS[@]}" -eq 0 ]]; then
    echo "[ERROR] no models selected" >&2
    exit 1
fi

count_expected_wavs() {
    local task="$1"
    local meta_path=""
    case "${task}" in
        seedtts_test_zh)
            meta_path="${REPO_ROOT}/data/seedtts_testset/zh/meta.lst"
            ;;
        seedtts_test_en)
            meta_path="${REPO_ROOT}/data/seedtts_testset/en/meta.lst"
            ;;
        *)
            echo ""
            return 0
            ;;
    esac

    if [[ ! -f "${meta_path}" ]]; then
        echo ""
        return 0
    fi

    awk 'NF {count++} END {print count+0}' "${meta_path}"
}

infer_output_dir() {
    local exp_name="$1"
    local step="$2"
    local task="$3"
    local config_path="$4"
    local mel_spec_type

    mel_spec_type=$(python - <<'PY' "${config_path}"
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
txt = config_path.read_text(encoding='utf-8')
m = re.search(r'^\s*mel_spec_type\s*:\s*([A-Za-z0-9_\-.]+)\b', txt, flags=re.MULTILINE)
if not m:
    raise SystemExit(1)
print(m.group(1).strip().strip('"').strip("'"))
PY
)

    local suffix="seed${SEED}_euler_nfe${NFE_STEP}_${mel_spec_type}"
    if [[ "${TIMESTEP_MAPPING}" == "uniform" ]]; then
        suffix+="_uniform"
    fi
    if [[ "${TIMESTEP_MAPPING}" == "sway_sampling" && "${SWAYSAMPLING}" != "0" ]]; then
        suffix+="_ss${SWAYSAMPLING}"
    fi
    if [[ "${TIMESTEP_MAPPING}" == "power" ]]; then
        suffix+="_power${TIMESTEP_POWER}"
    fi
    if [[ "${SHIFT}" != "1.0" ]]; then
        suffix+="_shift${SHIFT}"
    fi
    suffix+="_cfg${CFG_STRENGTH}_speed1.0_load-${LOAD_DTYPE}_infer-${INFER_DTYPE}"
    suffix+="_cfgitv0.0-1.0_target_rms0.1"

    printf '%s/results/%s/%s/%s/%s' \
        "${REPO_ROOT}" "${exp_name}" "${step}" "${task}" "${suffix}"
}

combo_idx=0
skipped_existing=0
skipped_incomplete=0
ran_count=0

for model_name in "${MODELS[@]}"; do
    [[ -z "${model_name}" ]] && continue

    model_dir="${EMILIA_ROOT}/${model_name}"
    if [[ ! -d "${model_dir}" ]]; then
        echo "[WARN] missing model dir, skip: ${model_dir}"
        continue
    fi

    exp_name="${model_name%%-emilia-*}"
    if [[ -z "${exp_name}" || "${exp_name}" == "${model_name}" ]]; then
        echo "[WARN] cannot parse exp_name from model dir, skip: ${model_name}"
        continue
    fi

    config_path="${REPO_ROOT}/src/f5_tts/configs/${exp_name}.yaml"
    if [[ ! -f "${config_path}" ]]; then
        echo "[WARN] missing config, skip: ${config_path}"
        continue
    fi

    for step in "${STEPS[@]}"; do
        [[ -z "${step}" ]] && continue
        ckpt_path="${model_dir}/ckpts/model_${step}.pt"
        if [[ ! -f "${ckpt_path}" ]]; then
            echo "[WARN] missing ckpt, skip: ${ckpt_path}"
            continue
        fi

        for task in "${TASKS[@]}"; do
            [[ -z "${task}" ]] && continue

            output_dir=$(infer_output_dir "${exp_name}" "${step}" "${task}" "${config_path}")
            expected_wavs=$(count_expected_wavs "${task}")

            if [[ "${FORCE_FLAG}" -eq 0 && -d "${output_dir}" ]]; then
                existing_wavs=$(find "${output_dir}" -type f -name '*.wav' | wc -l | awk '{print $1}')
                if [[ -n "${expected_wavs}" && "${expected_wavs}" != "0" ]]; then
                    if [[ "${existing_wavs}" -eq "${expected_wavs}" ]]; then
                        echo "[SKIP] existing complete infer: ${output_dir} (${existing_wavs}/${expected_wavs})"
                        skipped_existing=$((skipped_existing + 1))
                        continue
                    fi

                    echo "[WARN] existing infer incomplete: ${output_dir} (${existing_wavs}/${expected_wavs})"
                    skipped_incomplete=$((skipped_incomplete + 1))
                    if [[ "${STRICT_AUDIO_COUNT_FLAG}" -eq 1 ]]; then
                        continue
                    fi
                else
                    if [[ "${existing_wavs}" -gt 0 ]]; then
                        echo "[SKIP] existing infer dir with wavs: ${output_dir} (${existing_wavs} wavs)"
                        skipped_existing=$((skipped_existing + 1))
                        continue
                    fi
                fi
            fi

            port=$((MASTER_PORT_BASE + combo_idx))
            combo_idx=$((combo_idx + 1))

            cmd=(
                accelerate launch
                --main_process_port "${port}"
                src/f5_tts/eval/eval_infer_batch.py
                -s "${SEED}"
                -n "${exp_name}"
                -t "${task}"
                -c "${step}"
                --ckpt_path "${ckpt_path}"
                --nfe_step "${NFE_STEP}"
                --swaysampling "${SWAYSAMPLING}"
                --timestep_mapping "${TIMESTEP_MAPPING}"
                --cfg_strength "${CFG_STRENGTH}"
                --load_dtype "${LOAD_DTYPE}"
                --infer_dtype "${INFER_DTYPE}"
            )

            if [[ "${TIMESTEP_MAPPING}" == "power" ]]; then
                cmd+=(--timestep_power "${TIMESTEP_POWER}")
            fi

            cmd+=(--shift "${SHIFT}")

            if [[ "${LOCAL_FLAG}" -eq 1 ]]; then
                cmd+=(--local)
            fi

            echo "[PLAN] model=${model_name} exp=${exp_name} step=${step} task=${task} port=${port}"
            echo "[INFO] output_dir=${output_dir}"
            echo "[CMD] ${cmd[*]}"

            if [[ "${DRY_RUN_FLAG}" -eq 1 ]]; then
                continue
            fi

            (cd "${REPO_ROOT}" && "${cmd[@]}")
            ran_count=$((ran_count + 1))
        done
    done
done

echo "[SUMMARY] ran=${ran_count} skipped_existing=${skipped_existing} skipped_incomplete=${skipped_incomplete}"

if [[ "${STRICT_AUDIO_COUNT_FLAG}" -eq 1 && "${skipped_incomplete}" -gt 0 ]]; then
    exit 2
fi
