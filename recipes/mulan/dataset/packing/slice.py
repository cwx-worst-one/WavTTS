import io
import os
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

STR_CH_FIRST = "channels_first"
STR_CH_LAST = "channels_last"


def _resample_load_ffmpeg(
    path: str, sample_rate: int, downmix_to_mono: bool
) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.

    Returns a channel-first audio signal.

    Args:
        path:
        sample_rate:
        downmix_to_mono:

    Returns:
        (audio signal, sample rate)
    """

    def _decode_resample_by_ffmpeg(filename, sr):
        """decode, downmix, and resample audio file"""
        channel_cmd = "-ac 1 " if downmix_to_mono else ""  # downmixing option
        resampling_cmd = f"-ar {str(sr)}" if sr else ""  # downsampling option
        cmd = f'ffmpeg -i "{filename}" {channel_cmd} {resampling_cmd} -f wav -'
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = p.communicate()
        return out

    src, sr = sf.read(io.BytesIO(_decode_resample_by_ffmpeg(path, sr=sample_rate)))
    return src.T, sr


def load_audio(
    path: str or Path, ch_format: str, sample_rate: int, downmix_to_mono: bool = False
) -> Tuple[np.ndarray, int]:
    """A wrapper of librosa.load that:
        - forces the returned audio to be 2-dim,
        - defaults to sr=None, and
        - defaults to downmix_to_mono=False.

    The audio decoding is done by `audioread` or `soundfile` package
    and ultimately, often by ffmpeg. The resampling is done by
    `librosa`'s child package `resampy`.


    Args:
        path: audio file path
        ch_format: one of 'channels_first' or 'channels_last'
        sample_rate: target sampling rate. if None, use the rate of the audio file
        downmix_to_mono:
        resample_by (str): 'librosa' or 'ffmpeg'. it decides backend for
            audio decoding and resampling.
        **kwargs: keyword args for librosa.load - offset, duration, dtype, res_type.

    Returns:
        (audio, sr) tuple
    """
    if ch_format not in (STR_CH_FIRST, STR_CH_LAST):
        raise ValueError(f"ch_format is wrong here -> {ch_format}")

    if os.stat(path).st_size > 40960:
        src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
        if src.ndim == 1:
            src = np.expand_dims(src, axis=0)
        # now always 2d and channels_first

        if ch_format == STR_CH_FIRST:
            return src, sr
        else:
            return src.T, sr

    return None, None


def slice_audio(wav: np.ndarray, sample_rate: int, chunk_in_sec: int = 30):
    """slice audio to chunks."""
    chunk_size = sample_rate * chunk_in_sec
    for i in range(0, len(wav[0]), chunk_size):
        chunk = wav[:, i : i + chunk_size]
        if chunk.shape[1] == chunk_size:
            yield chunk
        if chunk_size / 2 < chunk.shape[1] < chunk_size:
            # pad only if chunk larger than chunk_size/2, eg. >15s
            yield np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[1])), "constant")


def slice_mulan_audio(audio_path: str, audio_id: str = None):
    """slice audio for mulan training."""
    try:
        x, sr = load_audio(
            audio_path, ch_format=STR_CH_FIRST, sample_rate=24000, downmix_to_mono=True
        )
        if audio_id is None:
            filename = Path(audio_path).stem
        else:
            filename = audio_id
        if x is not None:
            chunks = slice_audio(x, sr)
            return [
                {
                    "__key__": f"{filename}_{idx}",
                    "chunk.npy": (c * 32768).astype(np.int16),
                }
                for idx, c in enumerate(chunks)
            ]
        return []
    except Exception:
        return []


def slice_mulan_audio_group(audio_paths: str):
    """slice audio for mulan training.
    The input audio_paths is concatenation of >=1 file paths, looks like
    '/mnt/bd/litang-lq-music-npy12/15m_nonvocal_filter_data/mcc-sound-data-lq-bytedrive-a/q10_2010/6817347654087870465_6817347646173235202.m4a.npy,/mnt/bd/litang-lq-music-npy12/15m_nonvocal_filter_data/mcc-sound-data-lq-bytedrive-a/q10_2010/6817347654079481858_6817347646173235202.npy'
    """
    try:
        audio_paths = audio_paths.split(",")
        data_type = audio_paths[0][-3:]
        audio_id = audio_paths[0].split("/")[-1].split(".")[0].split("_")[0]
        # TODO: assume the sr is 24000, which is true for mcc-15m
        # not guaranteed for other dataset
        sr = 24000

        xs = []
        if data_type == "npy":
            for audio_path in audio_paths:
                x = np.load(audio_path)
                if x is not None:
                    xs.append(x.reshape([1, -1]))
        else:
            for audio_path in audio_paths:
                x, sr = load_audio(
                    audio_path,
                    ch_format=STR_CH_FIRST,
                    sample_rate=24000,
                    downmix_to_mono=True,
                )
                if x is not None:
                    xs.append(x)

        if audio_id is None:
            filename = Path(audio_paths[0]).stem
        else:
            filename = audio_id

        all_chunks = []
        base_idx = 0
        if len(xs) > 0:
            for x in xs:
                chunks = slice_audio(x, sr)
                all_chunks += [
                    {
                        "__key__": f"{filename}_{base_idx + idx}",
                        "chunk.npy": (c * 32768).astype(np.int16)
                        if c.dtype != "int16"
                        else c,
                    }
                    for idx, c in enumerate(chunks)
                ]
                base_idx = len(all_chunks)
        return all_chunks
    except Exception:
        return []
