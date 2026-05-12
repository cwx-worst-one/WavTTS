#!/usr/bin/env python3
"""Apply noisereduce.reduce_noise to every audio file in a directory.

Usage:
1. Install noisereduce if needed:

   pip install noisereduce

2. Edit the USER CONFIG section below.
3. Run from repo root:

   python utils/noisereduce_audio_dir.py

The script scans INPUT_DIR recursively, denoises each audio file with
noisereduce, and writes files to OUTPUT_DIR while preserving the relative
directory structure and file extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torchaudio

try:
    import noisereduce as nr
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency: noisereduce. Please install it with `pip install noisereduce`."
    ) from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ============================= USER CONFIG =============================
# 输入音频目录：脚本会递归处理该目录下所有支持的音频文件
INPUT_DIR_PATH = "/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav_mel_dev/results/F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k/1600000/seedtts_test_en/seed0_euler_nfe50_no_vocoder_power2.0_shift7.0_cfg3.0_speed1.0_load-fp32_infer-bf16_cfgitv0.0-1.0_target_rms0.1"

# 输出音频目录：会保留 INPUT_DIR 下的相对目录结构
OUTPUT_DIR_PATH = "/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav_mel_dev/results/F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k/1600000/seedtts_test_en/debug3"

INPUT_DIR = Path(INPUT_DIR_PATH)
OUTPUT_DIR = Path(OUTPUT_DIR_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 是否覆盖已存在输出文件
OVERWRITE = True

# 保存前是否裁剪到 [-1, 1]，防止处理后极少量过冲导致保存时削波/报错
CLAMP_OUTPUT = True

# 是否启用 stationary noise reduction。
# - False: 默认的非平稳降噪，更适合随时间变化的噪声
# - True : 平稳降噪，更适合稳定背景噪声
STATIONARY = False

# 降噪强度。1.0 是 noisereduce 默认值；越大通常降噪越强，但也越容易损伤语音。
PROP_DECREASE = 1.0

# 噪声片段配置：
# - USE_NOISE_CLIP=False: 不额外指定 y_noise，让 noisereduce 自动估计噪声
# - USE_NOISE_CLIP=True : 使用每条音频的 [NOISE_START_SEC, NOISE_END_SEC] 作为噪声参考片段
USE_NOISE_CLIP = False
NOISE_START_SEC = 0.0
NOISE_END_SEC = 0.5

# 常用 noisereduce 参数；None 表示使用 noisereduce 默认值
N_FFT: Optional[int] = 1024
HOP_LENGTH: Optional[int] = None
WIN_LENGTH: Optional[int] = None

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
    return tqdm(items, desc="Noise reducing", unit="file")


def build_nr_kwargs() -> dict:
    kwargs = {
        "stationary": STATIONARY,
        "prop_decrease": PROP_DECREASE,
    }
    if N_FFT is not None:
        kwargs["n_fft"] = N_FFT
    if HOP_LENGTH is not None:
        kwargs["hop_length"] = HOP_LENGTH
    if WIN_LENGTH is not None:
        kwargs["win_length"] = WIN_LENGTH
    return kwargs


def get_noise_clip(wav_np: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """Return per-file noise clip with shape [channels, time], or None."""
    if not USE_NOISE_CLIP:
        return None

    start = max(0, int(round(NOISE_START_SEC * sample_rate)))
    end = max(start, int(round(NOISE_END_SEC * sample_rate)))
    end = min(end, wav_np.shape[-1])

    if end <= start:
        raise ValueError(
            f"Invalid noise clip range: {NOISE_START_SEC}-{NOISE_END_SEC}s for audio length "
            f"{wav_np.shape[-1] / sample_rate:.3f}s"
        )

    return wav_np[:, start:end]


def reduce_noise_multichannel(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Denoise torchaudio waveform [channels, time] and return float32 tensor."""
    wav_np = wav.detach().cpu().numpy().astype(np.float32, copy=False)
    y_noise = get_noise_clip(wav_np, sample_rate)
    kwargs = build_nr_kwargs()

    denoised_channels: List[np.ndarray] = []
    for channel_idx in range(wav_np.shape[0]):
        channel_noise = y_noise[channel_idx] if y_noise is not None else None
        denoised = nr.reduce_noise(
            y=wav_np[channel_idx],
            sr=sample_rate,
            y_noise=channel_noise,
            **kwargs,
        )
        denoised_channels.append(np.asarray(denoised, dtype=np.float32))

    denoised_np = np.stack(denoised_channels, axis=0)
    return torch.from_numpy(denoised_np)


def process_one(src: Path, input_dir: Path, output_dir: Path) -> Path:
    rel_path = src.relative_to(input_dir)
    dst = output_dir / rel_path

    if dst.exists() and not OVERWRITE:
        return dst

    wav, sample_rate = torchaudio.load(str(src))
    wav_denoised = reduce_noise_multichannel(wav, sample_rate)

    if CLAMP_OUTPUT:
        wav_denoised = torch.clamp(wav_denoised, -1.0, 1.0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), wav_denoised, sample_rate=sample_rate)
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
    print(
        "noisereduce config: "
        f"stationary={STATIONARY}, prop_decrease={PROP_DECREASE}, "
        f"use_noise_clip={USE_NOISE_CLIP}"
    )
    if USE_NOISE_CLIP:
        print(f"noise clip: {NOISE_START_SEC}-{NOISE_END_SEC}s")
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
        report = output_dir / "noisereduce_failures.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Failure report saved to: {report}")


if __name__ == "__main__":
    main()

# python utils/noisereduce_audio_dir.py
