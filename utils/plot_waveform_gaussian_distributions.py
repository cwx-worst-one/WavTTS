#!/usr/bin/env python3
"""Plot speech waveform, Gaussian noise, and scaled-waveform distributions.

Usage:
    1. Edit the USER CONFIG section below:
       - AUDIO_DIRS: directories/files to scan recursively
       - SCALE_VALUES: scale factors for waveform amplitudes
       - OUTPUT_DIR: where figures/reports are saved
    2. Run:
       python utils/plot_waveform_gaussian_distributions.py

The script draws publication-style distribution curves of audio sample amplitudes:
    - speech waveform amplitudes from all audio files under AUDIO_DIRS
    - Gaussian noise with configurable mean/std
    - scaled waveform amplitudes, i.e. waveform * scale for each scale value
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np
import soundfile as sf
from tqdm import tqdm

# =============================================================================
# USER CONFIG: edit these values directly.
# =============================================================================

# One or more directories/files. Directories are scanned recursively.
# Example:
# AUDIO_DIRS = [Path("/mnt/bn/jdy-lq-5/chenwenxi/datasets/my_wavs")]
AUDIO_DIRS = [
    Path("/mnt/bn/jdy-lq-5/chenwenxi/data/librispeech/LibriSpeech/test-clean"),
]

# Scale factors used for "Scaled waveform" curves: scaled_samples = samples * scale.
SCALE_VALUES = [5.0, 10.0]

# Output directory for figures and a small statistics report.
OUTPUT_DIR = Path("/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav_mel_dev/doc/test")
OUTPUT_BASENAME = "speech_waveform_gaussian_distribution"

# Audio extensions scanned recursively.
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

# To keep memory and plotting fast, at most this many waveform samples are retained.
# Reservoir sampling keeps the retained samples approximately uniform over the dataset.
MAX_WAVEFORM_SAMPLES = 2_000_000

# Gaussian noise settings. The default is standard Gaussian noise N(0, 1).
# If you really want to match the speech waveform std, set GAUSSIAN_STD = "match_waveform".
GAUSSIAN_MEAN = 0.0
GAUSSIAN_STD: float | str = 1.0
GAUSSIAN_NUM_SAMPLES = 2_000_000
RANDOM_SEED = 1234

# Plot style.
FIG_DPI = 300
FIGSIZE_COMBINED = (7.2, 4.6)
FIGSIZE_THREE_PANEL = (10.5, 3.4)
NUM_BINS = 600
CURVE_SMOOTH_SIGMA_BINS = 2.0
X_PERCENTILE_LIMIT = 99.95  # robustly removes extreme outliers from x-axis limits
X_LIMIT_PAD_RATIO = 0.08
USE_FIXED_XLIM = True
FIXED_XLIM = (-1.5, 1.5)
USE_LOG_YSCALE = False       # True, False
PLOT_FILLED_HISTOGRAM = True
SAVE_PDF = True
SAVE_PNG = True

# =============================================================================
# End of user config.
# =============================================================================


@dataclass
class LoadSummary:
    files_found: int = 0
    files_loaded: int = 0
    files_failed: int = 0
    total_samples_seen: int = 0


def iter_audio_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            files.append(path)
        elif path.is_dir():
            files.extend(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            )
        else:
            print(f"[WARN] Path does not exist or is unsupported: {path}")
    return sorted(set(files))


def load_audio_mono(path: Path) -> np.ndarray:
    """Load an audio file as a 1-D float32 mono waveform.

    soundfile is used first. If it cannot decode a format such as some MP3 files,
    librosa is tried as a fallback when installed.
    """
    try:
        wav, _sr = sf.read(str(path), always_2d=False, dtype="float32")
    except Exception as sf_error:
        try:
            import librosa  # local import keeps soundfile-only runs lightweight

            wav, _sr = librosa.load(str(path), sr=None, mono=False)
            wav = np.asarray(wav, dtype=np.float32)
        except Exception as librosa_error:
            raise RuntimeError(
                f"soundfile failed ({sf_error}); librosa fallback failed ({librosa_error})"
            ) from librosa_error

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        # soundfile returns shape [time, channels]; librosa may return [channels, time].
        if wav.shape[0] <= 16 and wav.shape[0] < wav.shape[1]:
            wav = wav.mean(axis=0)
        else:
            wav = wav.mean(axis=1)
    elif wav.ndim > 2:
        wav = wav.reshape(-1)
    return wav.reshape(-1)


def reservoir_add(
    reservoir: list[float],
    values: np.ndarray,
    max_size: int,
    total_seen_before: int,
    rng: random.Random,
) -> None:
    """Uniform reservoir sampling over a stream of sample amplitudes."""
    if values.size == 0:
        return

    consumed = 0
    available = max_size - len(reservoir)
    if available > 0:
        take = min(available, values.size)
        reservoir.extend(float(x) for x in values[:take])
        consumed = take
        values = values[take:]

    for offset, value in enumerate(values, start=1):
        # Number of stream samples observed after this value is included.
        seen_index = total_seen_before + consumed + offset
        replace_index = rng.randint(0, seen_index - 1)
        if replace_index < max_size:
            reservoir[replace_index] = float(value)


def collect_waveform_samples(files: Sequence[Path]) -> tuple[np.ndarray, LoadSummary]:
    summary = LoadSummary(files_found=len(files))
    reservoir: list[float] = []
    rng = random.Random(RANDOM_SEED)

    for path in tqdm(files, desc="Loading audio", unit="file"):
        try:
            wav = load_audio_mono(path)
            wav = wav[np.isfinite(wav)]
            before = summary.total_samples_seen
            summary.total_samples_seen += int(wav.size)
            summary.files_loaded += 1
            reservoir_add(reservoir, wav, MAX_WAVEFORM_SAMPLES, before, rng)
        except Exception as exc:
            summary.files_failed += 1
            tqdm.write(f"[ERROR] {path}: {exc}")

    return np.asarray(reservoir, dtype=np.float32), summary


def gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return kernel


def density_curve(samples: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(samples, bins=bins, density=True)
    if CURVE_SMOOTH_SIGMA_BINS > 0:
        hist = np.convolve(hist, gaussian_kernel(CURVE_SMOOTH_SIGMA_BINS), mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def robust_xlim(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    pooled = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size > 0])
    if pooled.size == 0:
        return -1.0, 1.0
    hi = float(np.percentile(np.abs(pooled), X_PERCENTILE_LIMIT))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.max(np.abs(pooled))) if pooled.size else 1.0
    hi = max(hi, 1e-4)
    hi *= 1.0 + X_LIMIT_PAD_RATIO
    return -hi, hi


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-12)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2.0 * math.pi))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def beautify_axis(ax: plt.Axes) -> None:
    if USE_LOG_YSCALE:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.75, alpha=0.8)
    ax.grid(True, which="minor", color="#EFEFEF", linewidth=0.5, alpha=0.8)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Sample Amplitude")
    ax.set_ylabel("Probability Density")


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=FIG_DPI)
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{stem}.pdf")


def plot_combined(wave: np.ndarray, noise_mean: float, noise_std: float, xlim: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_COMBINED)
    colors = {
        "wave": "#1f77b4",
        "noise": "#d62728",
        "scaled": ["#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"],
    }
    bins = np.linspace(xlim[0], xlim[1], NUM_BINS + 1)
    x = np.linspace(xlim[0], xlim[1], 2000)

    # Plot Gaussian first so the legend order is:
    # Gaussian noise -> Speech waveform -> Scaled waveform(s).
    ax.plot(
        x,
        normal_pdf(x, noise_mean, noise_std),
        color=colors["noise"],
        linestyle="--",
        label="Gaussian Noise",
        zorder=4,
    )

    cx, cy = density_curve(wave, bins)
    if PLOT_FILLED_HISTOGRAM:
        ax.fill_between(cx, cy, color=colors["wave"], alpha=0.14, linewidth=0)
    ax.plot(cx, cy, color=colors["wave"], label="Speech Waveform", zorder=3)

    for idx, scale in enumerate(SCALE_VALUES):
        scaled = wave * float(scale)
        sx, sy = density_curve(scaled, bins)
        ax.plot(
            sx,
            sy,
            color=colors["scaled"][idx % len(colors["scaled"])],
            linestyle="-.",
            alpha=0.95,
            label=fr"Scaled Waveform ($\times {scale:g}$)",
        )

    ax.set_xlim(*xlim)
    beautify_axis(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor="#B0B0B0", framealpha=0.95)
    fig.tight_layout()
    save_figure(fig, OUTPUT_BASENAME + "_combined")
    plt.close(fig)


def plot_three_panel(wave: np.ndarray, noise_mean: float, noise_std: float, xlim: tuple[float, float]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE_PANEL, sharey=False)
    bins = np.linspace(xlim[0], xlim[1], NUM_BINS + 1)
    x = np.linspace(xlim[0], xlim[1], 2000)

    cx, cy = density_curve(wave, bins)
    axes[0].fill_between(cx, cy, color="#1f77b4", alpha=0.18, linewidth=0)
    axes[0].plot(cx, cy, color="#1f77b4")
    axes[0].set_title("Speech Waveform")

    axes[1].plot(x, normal_pdf(x, noise_mean, noise_std), color="#d62728", linestyle="--")
    axes[1].fill_between(x, normal_pdf(x, noise_mean, noise_std), color="#d62728", alpha=0.14, linewidth=0)
    axes[1].set_title("Gaussian Noise")

    scaled_colors = ["#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
    for idx, scale in enumerate(SCALE_VALUES):
        sx, sy = density_curve(wave * float(scale), bins)
        axes[2].plot(
            sx,
            sy,
            color=scaled_colors[idx % len(scaled_colors)],
            label=fr"$\times {scale:g}$",
        )
    axes[2].set_title("Scaled Waveform")
    axes[2].legend(frameon=True, fancybox=False, edgecolor="#B0B0B0", framealpha=0.95)

    for ax in axes:
        ax.set_xlim(*xlim)
        beautify_axis(ax)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.tight_layout(w_pad=1.2)
    save_figure(fig, OUTPUT_BASENAME + "_three_panel")
    plt.close(fig)


def write_report(summary: LoadSummary, wave: np.ndarray, noise_mean: float, noise_std: float, xlim: tuple[float, float]) -> None:
    lines = [
        "# Speech waveform / Gaussian noise distribution report",
        "",
        f"Audio inputs: {', '.join(str(p) for p in AUDIO_DIRS)}",
        f"Files found: {summary.files_found}",
        f"Files loaded: {summary.files_loaded}",
        f"Files failed: {summary.files_failed}",
        f"Total samples seen: {summary.total_samples_seen}",
        f"Samples used for plotting: {wave.size}",
        "",
        "## Waveform statistics (samples used for plotting)",
        f"mean: {float(np.mean(wave)):.10g}",
        f"std : {float(np.std(wave)):.10g}",
        f"min : {float(np.min(wave)):.10g}",
        f"max : {float(np.max(wave)):.10g}",
        "",
        "## Gaussian noise",
        f"mean: {noise_mean:.10g}",
        f"std : {noise_std:.10g}",
        "",
        "## Scales",
        ", ".join(str(s) for s in SCALE_VALUES),
        "",
        "## Plot range",
        f"xlim: [{xlim[0]:.10g}, {xlim[1]:.10g}]",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{OUTPUT_BASENAME}_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    files = iter_audio_files(AUDIO_DIRS)
    print(f"Found {len(files)} audio files.")
    if not files:
        raise FileNotFoundError("No audio files found. Please edit AUDIO_DIRS in the USER CONFIG section.")

    wave, summary = collect_waveform_samples(files)
    if wave.size == 0:
        raise RuntimeError("No valid waveform samples were loaded.")

    noise_std = float(np.std(wave)) if GAUSSIAN_STD == "match_waveform" else float(GAUSSIAN_STD)
    noise_mean = float(GAUSSIAN_MEAN)

    rng = np.random.default_rng(RANDOM_SEED)
    noise_for_xlim = rng.normal(noise_mean, noise_std, size=min(GAUSSIAN_NUM_SAMPLES, wave.size)).astype(np.float32)
    scaled_for_xlim = [wave * float(scale) for scale in SCALE_VALUES]
    xlim = FIXED_XLIM if USE_FIXED_XLIM else robust_xlim([wave, noise_for_xlim, *scaled_for_xlim])

    plot_combined(wave, noise_mean, noise_std, xlim)
    plot_three_panel(wave, noise_mean, noise_std, xlim)
    write_report(summary, wave, noise_mean, noise_std, xlim)

    print(f"Done. Figures and report saved to: {OUTPUT_DIR}")
    print(f"  - {OUTPUT_BASENAME}_combined.png/pdf")
    print(f"  - {OUTPUT_BASENAME}_three_panel.png/pdf")
    print(f"  - {OUTPUT_BASENAME}_report.txt")


if __name__ == "__main__":
    main()

# python utils/plot_waveform_gaussian_distributions.py
