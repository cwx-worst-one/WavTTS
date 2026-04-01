#!/usr/bin/env python3
"""Compute torchaudio waveform mean/std before and after resampling to 16k.

Features:
- User-selectable root directories.
- Recursive scan for common audio extensions.
- tqdm progress bar.
- Save summary report in doc/ with:
  1) overall statistics
  2) per-directory statistics
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torchaudio
from tqdm import tqdm

AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".alac",
}


@dataclass
class RunningStats:
    """Running stats using sum/sum_sq/count to get global mean/std."""

    sum_x: float = 0.0
    sum_x2: float = 0.0
    count: int = 0

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(torch.float64)
        self.sum_x += float(x.sum().item())
        self.sum_x2 += float((x * x).sum().item())
        self.count += int(x.numel())

    @property
    def mean(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.sum_x / self.count

    @property
    def std(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.mean
        var = self.sum_x2 / self.count - mean * mean
        var = max(var, 0.0)
        return math.sqrt(var)


@dataclass
class GroupStats:
    loaded: RunningStats
    resampled_16k: RunningStats
    files_ok: int = 0
    files_failed: int = 0

    @classmethod
    def new(cls) -> "GroupStats":
        return cls(loaded=RunningStats(), resampled_16k=RunningStats())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute audio mean/std after torchaudio.load and after resampling to 16k Hz."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more root directories to scan recursively.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate for resampling (default: 16000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doc/audio_stats_torchaudio_report.txt"),
        help="Output report file path.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help=(
            "If provided, per-directory keys are relative to this path. "
            "Otherwise uses each file's parent absolute path."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    return parser.parse_args()


def iter_audio_files(roots: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[WARN] root not found: {root}")
            continue
        if root.is_file() and root.suffix.lower() in AUDIO_EXTS:
            files.append(root)
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
    return sorted(files)


def parent_key(path: Path, relative_to: Path | None) -> str:
    parent = path.parent
    if relative_to is None:
        return str(parent)
    try:
        return str(parent.resolve().relative_to(relative_to.resolve()))
    except Exception:
        return str(parent)


def format_stats(name: str, s: GroupStats) -> str:
    return (
        f"[{name}]\n"
        f"  files_ok      : {s.files_ok}\n"
        f"  files_failed  : {s.files_failed}\n"
        f"  loaded_mean   : {s.loaded.mean:.10f}\n"
        f"  loaded_std    : {s.loaded.std:.10f}\n"
        f"  res16k_mean   : {s.resampled_16k.mean:.10f}\n"
        f"  res16k_std    : {s.resampled_16k.std:.10f}\n"
        f"  loaded_samples: {s.loaded.count}\n"
        f"  res16k_samples: {s.resampled_16k.count}\n"
    )


def main() -> None:
    args = parse_args()

    files = iter_audio_files(args.roots)
    if args.max_files is not None:
        files = files[: args.max_files]

    print(f"Found {len(files)} audio files.")

    overall = GroupStats.new()
    by_dir: Dict[str, GroupStats] = defaultdict(GroupStats.new)

    resampler_cache: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}

    for f in tqdm(files, desc="Processing audio", unit="file"):
        key = parent_key(f, args.relative_to)
        try:
            wav, sr = torchaudio.load(str(f))

            overall.loaded.update(wav)
            by_dir[key].loaded.update(wav)

            if sr == args.target_sr:
                wav_16k = wav
            else:
                rs_key = (sr, args.target_sr)
                if rs_key not in resampler_cache:
                    resampler_cache[rs_key] = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=args.target_sr
                    )
                wav_16k = resampler_cache[rs_key](wav)

            overall.resampled_16k.update(wav_16k)
            by_dir[key].resampled_16k.update(wav_16k)

            overall.files_ok += 1
            by_dir[key].files_ok += 1
        except Exception as e:
            overall.files_failed += 1
            by_dir[key].files_failed += 1
            tqdm.write(f"[ERROR] {f}: {e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# torchaudio.load & resample(16k) statistics report")
    lines.append(f"Generated at: {timestamp}")
    lines.append(f"Target sample rate: {args.target_sr}")
    lines.append(f"Total discovered files: {len(files)}")
    lines.append("")

    lines.append("## 1) Overall")
    lines.append(format_stats("OVERALL", overall))

    lines.append("## 2) Per-directory")
    for k in sorted(by_dir.keys()):
        lines.append(format_stats(k, by_dir[k]))

    args.output.write_text("\n".join(lines), encoding="utf-8")

    print(f"Done. Report saved to: {args.output}")
    print(format_stats("OVERALL", overall))


if __name__ == "__main__":
    main()
