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

def regular_pad_audio(audio: torch.tensor, segment_samples: int, hop_samples: int) -> torch.tensor:
    # copied from validation pipeline in utils/audio_utils.py

    _, audio_samples = audio.shape

    # Number of segments
    segments_num = int(np.ceil(audio_samples / segment_samples))

    pad_samples = segments_num * segment_samples - audio_samples

    pad = (0, pad_samples)
    padded_audio = F.pad(audio, pad, "constant", 0)
    return padded_audio

def buffer_pad_audio(audio: torch.tensor, segment_samples: int, hop_samples: int) -> torch.tensor:
    r"""Pad the audio with zero in the end so that the length of audio can
    be evenly divided by segment_samples.

    Args:
        audio: (channels_num, audio_samples)

    Returns:
        padded_audio: (channels_num, audio_samples)
    """
    channels_num, audio_samples = audio.shape

    # Number of segments
    segments_num = int(np.ceil(audio_samples / hop_samples))

    pad_samples = segments_num * hop_samples - audio_samples
    pad = (0, pad_samples)
    padded_audio = F.pad(audio, pad, "constant", 0)

    # Pad additional zeros as buffer at both side
    pad_len = (segment_samples - hop_samples) // 2
    if pad_len > 0:
        pad = (pad_len, pad_len)
        padded_audio = F.pad(padded_audio, pad, "constant", 0)

    return padded_audio

def buffer_enframe(audio: torch.tensor, segment_samples: int, hop_samples: int) -> np.array:
    r"""Enframe long audio into segments.

    Args:
        audio: (channels_num, audio_samples)
        segment_samples: int
        hop_samples: int

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

    return torch.stack(segments, dim=0)

def regular_enframe(audio: torch.tensor, segment_samples: int, hop_samples: int) -> np.array:

    audio_samples = audio.shape[1]
    hop_samples = segment_samples // 2 # must be 0.5 hop
    segments = []

    pointer = 0
    while pointer + segment_samples <= audio_samples:
        segments.append(audio[:, pointer : pointer + segment_samples])
        pointer += hop_samples
    return torch.stack(segments, dim=0)


def buffer_overlap_deframe(segments: torch.tensor, segment_samples: int, hop_samples: int, method: str='crossfade') -> torch.tensor:
    r"""Deframe segments into long audio.

    Args:
        segments: (segments_num, channels_num, segment_samples)

    Returns:
        output: (channels_num, audio_samples)
    """
    (segments_num, n_channel, samples) = segments.shape

    if segments_num == 1:
        return segments[0]

    # (N, n_channel, segment_samples, dim) = x.shape
    # torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    output = torch.zeros(
            n_channel,
            (segments_num - 1) * hop_samples + segment_samples,
    ).to(segments.device)

    if method == 'average':
        cnt = np.zeros_like(output)
        for i in range(segments_num):
            start = i * hop_samples
            end = start + segment_samples

            output[:, start : end] += segments[i] 
            cnt[:, start : end] += 1
        output /= cnt

    elif method == 'crossfade':
        overlap_size = segment_samples - hop_samples
        fadein_amp = torch.linspace(0, 1, overlap_size).to(segments.device)
        fadeout_amp = 1 - fadein_amp
        for i in range(segments_num):
            if i == 0:
                output[:, : segment_samples] += segments[i]
            else:
                start = i * hop_samples
                previous_end = (i - 1) * hop_samples + segment_samples
                output[:, (previous_end - overlap_size) : previous_end] *= fadeout_amp
                to_add = segments[i].clone()
                to_add[:, : overlap_size] *= fadein_amp 
                output[:, start : start+segment_samples] += to_add

    # Remove front and end padding
    assert (
        segment_samples - hop_samples
    ) % 2 == 0, "hop_size must be even percentage of segment_samples"
    pad_len = int(segment_samples - hop_samples) // 2
    if pad_len > 0:
        output = output[:, pad_len:-pad_len]

    return output

def concat_deframe(segments: torch.tensor, segment_samples: int, hop_samples: int) -> torch.tensor:
    # copied from validation pipeline in utils/audio_utils.py

    (segments_num, _, segment_samples) = segments.shape

    if segments_num == 1:
        return segments[0]

    output = []
    output.append(segments[0, :, 0 : int(segment_samples * 0.75)])
    for i in range(1, segments_num - 1):
        output.append(
            segments[i, :, int(segment_samples * 0.25) : int(segment_samples * 0.75)]
        )

    output.append(segments[-1, :, int(segment_samples * 0.25) :])
    output = torch.cat(output, dim=-1)
    
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
    path: str,
    ch_format: str = "channels_first",
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


