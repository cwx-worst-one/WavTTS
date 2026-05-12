#!/usr/bin/env python3
"""Apply torchaudio.functional.lowpass_biquad to every audio file in a directory.

Usage:
1. Edit the USER CONFIG section below.
2. Run from repo root:

   python utils/lowpass_biquad_audio_dir.py

The script scans INPUT_DIR recursively, applies lowpass_biquad with each file's
original sample rate, and writes files to OUTPUT_DIR while preserving the
relative directory structure and file extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
import torchaudio
from torchaudio.functional import lowpass_biquad

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ============================= USER CONFIG =============================
# 输入音频目录：脚本会递归处理该目录下所有支持的音频文件
INPUT_DIR_PATH = "/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav_mel_dev/results/F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k/1600000/seedtts_test_en/seed0_euler_nfe50_no_vocoder_power2.0_shift7.0_cfg3.0_speed1.0_load-fp32_infer-bf16_cfgitv0.0-1.0_longcat-prompt-ode_target_rms0.1"

# 输出音频目录：会保留 INPUT_DIR 下的相对目录结构
OUTPUT_DIR_PATH = "/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav_mel_dev/results/F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k/1600000/seedtts_test_en/debug1"

INPUT_DIR = Path(INPUT_DIR_PATH)
OUTPUT_DIR = Path(OUTPUT_DIR_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# lowpass_biquad 参数
CUTOFF_FREQ = 4000.0  # Hz, cutoff frequency
Q = 0.707             # quality factor; 0.707 is a common Butterworth-like value

# 是否覆盖已存在输出文件
OVERWRITE = True

# 保存前是否裁剪到 [-1, 1]，防止滤波后极少量过冲导致保存时削波/报错
CLAMP_OUTPUT = True

# 支持的音频后缀；实际能否读写取决于当前 torchaudio backend/ffmpeg 支持
AUDIO_EXTS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
}
# =======================================================================


def is_relative_to(path: Path, root: Path) -> bool:
    """Path.is_relative_to with resolved paths and a small compatibility wrapper."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def iter_audio_files(input_dir: Path, output_dir: Path, exts: Iterable[str]) -> List[Path]:
    """Collect audio files recursively, skipping OUTPUT_DIR if it is under INPUT_DIR."""
    normalized_exts = {e.lower() for e in exts}
    files: List[Path] = []

    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in normalized_exts:
            continue
        # Avoid re-processing previously generated files when OUTPUT_DIR is inside INPUT_DIR.
        if is_relative_to(path, output_dir):
            continue
        files.append(path)

    return sorted(files)


def progress_iter(items: List[Path]):
    if tqdm is None:
        return items
    return tqdm(items, desc="Lowpass filtering", unit="file")


def process_one(src: Path, input_dir: Path, output_dir: Path) -> Path:
    rel_path = src.relative_to(input_dir)
    dst = output_dir / rel_path

    if dst.exists() and not OVERWRITE:
        return dst

    wav, sample_rate = torchaudio.load(str(src))

    nyquist = sample_rate / 2.0
    if CUTOFF_FREQ <= 0:
        raise ValueError(f"CUTOFF_FREQ must be positive, got {CUTOFF_FREQ}")
    if CUTOFF_FREQ >= nyquist:
        raise ValueError(
            f"CUTOFF_FREQ={CUTOFF_FREQ} must be lower than Nyquist={nyquist} for {src} "
            f"with sample_rate={sample_rate}"
        )

    # torchaudio.load returns shape [channels, time]. lowpass_biquad preserves shape.
    wav_lp = lowpass_biquad(wav, sample_rate=sample_rate, cutoff_freq=CUTOFF_FREQ, Q=Q)

    if CLAMP_OUTPUT:
        wav_lp = torch.clamp(wav_lp, -1.0, 1.0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), wav_lp, sample_rate=sample_rate)
    return dst


def main() -> None:
    input_dir = INPUT_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"INPUT_DIR is not a directory: {input_dir}")

    files = iter_audio_files(input_dir, output_dir, AUDIO_EXTS)
    print(f"INPUT_DIR : {input_dir}")
    print(f"OUTPUT_DIR: {output_dir}")
    print(f"CUTOFF_FREQ={CUTOFF_FREQ} Hz, Q={Q}, OVERWRITE={OVERWRITE}")
    print(f"Found {len(files)} audio files.")

    ok = 0
    skipped = 0
    failed = 0
    failures: List[str] = []

    for src in progress_iter(files):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        if dst.exists() and not OVERWRITE:
            skipped += 1
            continue
        try:
            process_one(src, input_dir, output_dir)
            ok += 1
        except Exception as exc:  # keep going on bad/corrupt/unsupported files
            failed += 1
            msg = f"[ERROR] {src}: {exc}"
            failures.append(msg)
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg)

    print("Done.")
    print(f"  processed: {ok}")
    print(f"  skipped  : {skipped}")
    print(f"  failed   : {failed}")

    if failures:
        report = output_dir / "lowpass_biquad_failures.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Failure report saved to: {report}")


if __name__ == "__main__":
    main()

# python utils/lowpass_biquad_audio_dir.py