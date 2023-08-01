# def collect_audio_segments():
import numpy as np
import subprocess
import io
import os
import random
import soundfile as sf
from typing import Tuple
from pathlib import Path

STR_CH_FIRST = "channels_first"
STR_CH_LAST = "channels_last"

class InvalidAudioError(Exception):
    pass

def _resample_load_ffmpeg(
    path: str, sample_rate: int, downmix_to_mono: bool
) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by ffmpeg.

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
    path: str or Path,
    ch_format: str,
    sample_rate: int = None,
    downmix_to_mono: bool = False,
) -> Tuple[np.ndarray, int]:
    """A wrapper of ffmpeg load that:
        - forces the returned audio to be 2-dim,
        - defaults to sr=None, and
        - defaults to downmix_to_mono=False.

    The audio decoding is done by `audioread` or `soundfile` package and ultimately, often by ffmpeg.

    Args:
        path: audio file path
        ch_format: one of 'channels_first' or 'channels_last'
        sample_rate: target sampling rate. if None, use the rate of the audio file
        downmix_to_mono:

    Returns:
        (audio, sr) tuple
    """
    if ch_format not in (STR_CH_FIRST, STR_CH_LAST):
        raise ValueError(f"ch_format is wrong here -> {ch_format}")

    if os.stat(path).st_size > 22050:
        src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
    else:
        raise InvalidAudioError("Given audio is too short!")

    if src.ndim == 1:
        src = np.expand_dims(src, axis=0)
    # now always 2d and channels_first

    if ch_format == STR_CH_FIRST:
        return src, sr
    else:
        return src.T, sr

def adjust_audio_length(
    waveform: np.ndarray, desired_samples: int, num_chunks: int = 1
):
    """Adjust the length of waveform (numpy array).
    Trim off the end if it's too long, add zeros if it's too short.

    Args:
        waveform: 2D audio data.
        desired_samples (int): target length of the audio [sample]
        num_chunks (int): number of chunks per song

    Returns:
        waveform: length-adjusted audio.
    """

    if waveform.shape[1] < desired_samples:
        waveform = np.pad(
            waveform, pad_width=((0, 0), (0, desired_samples - waveform.shape[1]))
        )
    audio_length = waveform.shape[1]

    if num_chunks == 1:  # random crop
        ix = random.randint(0, audio_length - desired_samples)
        waveform = waveform[:, ix : ix + desired_samples]

        return waveform, ix, audio_length
    else:  # multiple chunks for evaluation
        hop = (audio_length - desired_samples) // num_chunks
        waveform = np.array(
            [
                waveform[:, i * hop : i * hop + desired_samples][0]
                for i in range(num_chunks)
            ]
        )

        return waveform

import glob
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

audio_folder = "/mnt/bn/mm-data/projects/mulan/testing_embed/non_vocal_30k/"
target_folder = "/opt/tiger/mulan/non_vocal_30k_clips_20s"
os.makedirs(target_folder, exist_ok=True)
audio_files = glob.glob(f"{audio_folder}/*mp3")

def process_one_audio(audio_path):
    waveform, sr = load_audio(
        path=audio_path,
        ch_format=STR_CH_FIRST,
        sample_rate=24000,
        downmix_to_mono=True,
    )
    if waveform.shape[1] / sr < 30:
        print(audio_path)
        return
    stride = (waveform.shape[1] - 2 * 240_000) // 5
    clips = np.zeros(shape=[3, 2 * 240_000], dtype=np.float64)
    for i in range(1, 4):
        clips[i - 1, :] = waveform[0, i * stride : i * stride + 2 * 240_000]
    clips = (clips * 32768).astype(np.int16)
    audio_id = audio_path.split("/")[-1].replace(".mp3", "_clips.npy")
    np.save(f"{target_folder}/{audio_id}", clips)

p = Pool(8)
with p:
    p.map(process_one_audio, audio_files)
del p

