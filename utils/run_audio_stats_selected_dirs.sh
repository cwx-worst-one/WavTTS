#!/usr/bin/env bash
set -euo pipefail

# Run torchaudio stats for a configurable directory list.
# You can add/remove directories in ROOT_DIRS below.

PROJECT_ROOT="/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav_mel_dev"
PY_SCRIPT="$PROJECT_ROOT/utils/audio_stats_torchaudio.py"
OUTPUT_REPORT="$PROJECT_ROOT/doc/audio_stats_details.txt"
RELATIVE_TO="/mnt/bn/jdy-lq-5/chenwenxi/data"

ROOT_DIRS=(
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/EN/EN-B000000"
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/EN/EN-B000001"
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/EN/EN-B000002"
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/EN/EN-B000003"
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/EN/EN-B000004"
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/ZH/ZH-B000000"
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/ZH/ZH-B000001"
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/ZH/ZH-B000002"
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/ZH/ZH-B000003"
  "/mnt/bn/jdy-lq-5/chenwenxi/data/emilia/ZH/ZH-B000004"
)

AUDIO_FIND_EXPR=(
  -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.m4a' -o \
  -iname '*.aac' -o -iname '*.ogg' -o -iname '*.opus' -o -iname '*.wma' -o \
  -iname '*.aiff' -o -iname '*.aif' -o -iname '*.alac'
)

echo "===== Audio stats input directories ====="
echo "Total dirs: ${#ROOT_DIRS[@]}"

total_audio_count=0
for d in "${ROOT_DIRS[@]}"; do
  if [[ -d "$d" ]]; then
    count=$(find "$d" -type f \( "${AUDIO_FIND_EXPR[@]}" \) | wc -l | tr -d ' ')
    echo "[OK] $d"
    echo "     audio files: $count"
    total_audio_count=$((total_audio_count + count))
  else
    echo "[MISS] $d (directory not found)"
    echo "     audio files: 0"
  fi
done

echo "----------------------------------------"
echo "Total audio files (all listed dirs): $total_audio_count"
echo "Output report: $OUTPUT_REPORT"
echo "========================================"

cd "$PROJECT_ROOT"

python "$PY_SCRIPT" \
  "${ROOT_DIRS[@]}" \
  --relative-to "$RELATIVE_TO" \
  --output "$OUTPUT_REPORT"

echo "Done. Report generated at: $OUTPUT_REPORT"

# bash utils/run_audio_stats_selected_dirs.sh
