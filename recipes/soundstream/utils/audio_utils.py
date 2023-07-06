import io
import os
import subprocess
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

STR_CH_FIRST = "channels_first"
STR_CH_LAST = "channels_last"


def is_integer(x: float) -> bool:
    if x - int(x) < 1e-10:
        return True
    else:
        return False


def pad_audio(audio: torch.tensor, segment_samples: int, hop_samples: int) -> torch.tensor:
    r"""Pad the audio with zero in the end so that the length of audio can
    be evenly divided by segment_samples.

    Args:
        audio: (channels_num, audio_samples)
        segment_samples: int
        hop_samples: int

    Returns:
        padded_audio: (channels_num, audio_samples)
    """
    channels_num, audio_samples = audio.shape

    # Number of segments
    segments_num = int(np.ceil(audio_samples / hop_samples))

    pad_samples = segments_num * hop_samples - audio_samples
    padded_audio = F.pad(audio, ((0, pad_samples)), "constant")
    # (channels_num, padded_audio_samples)

    # Pad additional zeros as buffer at both side
    pad_len = (segment_samples - hop_samples) // 2
    padded_audio = F.pad(padded_audio, ((pad_len, pad_len)), "constant")

    return padded_audio



def enframe(audio: torch.tensor, segment_samples: int, hop_samples: int) -> np.array:
    r"""Enframe long audio into segments.

    Args:
        audio: (channels_num, audio_samples)
        segment_samples: int

    Returns:
        segments: (segments_num, channels_num, segment_samples)
    """
    audio_samples = audio.shape[1]
    segments = []

    pad_len = (segment_samples - hop_samples) // 2
    pointer = pad_len

    while pointer - pad_len + segment_samples <= audio_samples:
        start =  pointer - pad_len
        end = start + segment_samples
        segments.append(audio[:, start : end])
        pointer += hop_samples

    segments = torch.stack(segments)
    return segments



def deframe(segments: torch.tensor, hop_samples: int) -> torch.tensor:
    r"""Deframe segments into long audio.

    Args:
        segments: (segments_num, channels_num, segment_samples)

    Returns:
        output: (channels_num, audio_samples)
    """
    (segments_num, n_channel, segment_samples) = segments.shape

    if segments_num == 1:
        return segments[0]

    # (N, n_channel, segment_samples, dim) = x.shape
    output = torch.zeros(
        (
            n_channel,
            (segments_num - 1) * hop_samples + segment_samples,
        )
    ).to(segments.device)
    cnt = torch.zeros_like(output)

    for i in range(segments_num):
        start = i * hop_samples
        end = start + segment_samples

        output[:, start : end] += segments[i]
        cnt[:, start : end] += 1

    # # Take the average
    output /= cnt

    # Remove front and end padding
    assert (
        segment_samples - hop_samples
    ) % 2 == 0, "hop_size must be even percentage of segment_samples"
    pad_len = int(segment_samples - hop_samples) // 2
    if pad_len > 0:
        output = output[:, pad_len:-pad_len]

    return output

class InvalidAudioError(Exception):
    pass

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
        channel_cmd = "-ac 1 " if downmix_to_mono else "-ac 2"  # downmixing option
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


def _resample_load_librosa(
    path: str, sample_rate: int, downmix_to_mono: bool, **kwargs
) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.

    Returns a channel-first audio signal.
    """
    src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono, **kwargs)
    return src, sr


def load_audio(
    path: str or Path,
    ch_format: str,
    sample_rate: int = None,
    downmix_to_mono: bool = False,
    resample_by: str = "ffmpeg",
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """A wrapper of librosa.load that:
        - forces the returned audio to be 2-dim,
        - defaults to sr=None, and
        - defaults to downmix_to_mono=False.

    The audio decoding is done by `audioread` or `soundfile` package and ultimately, often by ffmpeg.
    The resampling is done by `librosa`'s child package `resampy`.


    Args:
        path: audio file path
        ch_format: one of 'channels_first' or 'channels_last'
        sample_rate: target sampling rate. if None, use the rate of the audio file
        downmix_to_mono:
        resample_by (str): 'librosa' or 'ffmpeg'. it decides backend for audio decoding and resampling.
        **kwargs: keyword args for librosa.load - offset, duration, dtype, res_type.

    Returns:
        (audio, sr) tuple
    """
    if ch_format not in (STR_CH_FIRST, STR_CH_LAST):
        raise ValueError(f"ch_format is wrong here -> {ch_format}")

    if os.stat(path).st_size > 22050:
        if resample_by == "librosa":
            src, sr = _resample_load_librosa(
                path, sample_rate, downmix_to_mono, **kwargs
            )
        elif resample_by == "ffmpeg":
            src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
        else:
            raise NotImplementedError(
                f'resample_by: "{resample_by}" is not supposred yet'
            )
    else:
        raise InvalidAudioError("Given audio is too short!")

    if src.ndim == 1:
        src = np.expand_dims(src, axis=0)
    # now always 2d and channels_first

    if ch_format == STR_CH_FIRST:
        return src, sr
    else:
        return src.T, sr


if __name__ == "__main__":
    dummy_input = torch.ones((2, 44100 * 30))
    dummy_input = pad_audio(dummy_input, 44100 * 3)
    print(dummy_input.shape)
    dummy_input = enframe(dummy_input, 44100 * 3)
    print(dummy_input.shape)
    dummy_input = deframe(dummy_input)
    print(dummy_input.shape)
