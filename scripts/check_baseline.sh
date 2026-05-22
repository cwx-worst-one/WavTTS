#!/usr/bin/env bash
set -euo pipefail

# Lightweight local verification baseline for this repository snapshot.
# Uses the project-local virtualenv by default, as requested.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[baseline] python: ${PYTHON_BIN}"
echo "[baseline] PYTHONPATH: ${PYTHONPATH}"

"${PYTHON_BIN}" tests/test_smoke.py
"${PYTHON_BIN}" tests/test_waveform_dataset_collate.py
"${PYTHON_BIN}" -m compileall -q tests src/f5_tts src/wavtts
bash -n src/f5_tts/train/run_main_train.sh
bash -n src/f5_tts/train/run_train_libritts.sh
bash -n src/wavtts/train/run_main_train.sh
bash -n src/wavtts/train/run_train_libritts.sh
bash -n src/wavtts/infer/debug_infer.sh

echo "[baseline] OK"
