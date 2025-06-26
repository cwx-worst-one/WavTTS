import os
import io
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
from pydub import AudioSegment


def generate_click_wav_from_beat_result(beats, sr, length=None):
    "Format: [(time, beat_pos), ...]"
    downbeat_times = [t for t, b in beats if b == 1.0]
    other_times = [t for t, b in beats if b != 1.0]
    audio_other = librosa.clicks(times=other_times, sr=sr, click_freq=1000, length=length)
    audio_downbeat = librosa.clicks(times=downbeat_times, sr=sr, click_freq=1500, length=length)
    return pad_add(audio_other, audio_downbeat)


def pad_add(a, b):
    """Support len(a.shape) == 1 or 2
    """
    assert len(a.shape) == len(b.shape)
    if len(a) == len(b): return a + b
    if len(a) < len(b): a, b = b, a
    b = np.pad(b, [(0, a.shape[i] - b.shape[i]) for i in range(len(a.shape))])
    return a + b


def audio_array_to_bytes(audio_arr, sr, format='wav'):
    """Assuming the last dim is channel dim, unless there is only one dim.
    """
    max_abs_value = abs(audio_arr).max()
    if max_abs_value > 1:
        audio_arr = audio_arr / max_abs_value
    audio_arr = (audio_arr * 2**15).astype(np.int16)
    channels = 1 if len(audio_arr.shape) == 1 else audio_arr.shape[-1]
    out_io = io.BytesIO()
    audio_seg = AudioSegment(audio_arr.tobytes(), frame_rate=sr, sample_width=audio_arr.dtype.itemsize, channels=channels)
    audio_seg.export(out_io, format=format)
    out_io.seek(0)
    return out_io.read()


def audio_bytes_to_array(
    raw_audio: bytes,
    sr: float=None
) -> Tuple[np.array, float]:
    audio: AudioSegment = AudioSegment.from_file(io.BytesIO(raw_audio))
    if sr is not None and sr != audio.frame_rate:
        audio = audio.set_frame_rate(sr)
    wav = (np.array(audio.get_array_of_samples()) / 2**15).reshape((-1, audio.channels))
    return wav, audio.frame_rate


def load_wav(path, sr: Optional[int] = None, mono: bool = False) -> np.ndarray:
    wav, sr = librosa.load(path, sr=sr, mono=mono)
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2_147_483_648.0
    return wav


def concat_and_crossfade_audio_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    sample_rate: int,
    crossfade_duration: float,
) -> torch.Tensor:
    """
    Perform crossfade on two audio tensors with sinusoidal ramps for consistent loudness,
    handling different lengths, channels (mono or stereo), and edge cases where the 
    audio length is shorter than the crossfade duration.

    Args:
        tensor1 (torch.Tensor): First audio tensor with shape (channels, samples).
        tensor2 (torch.Tensor): Second audio tensor with shape (channels, samples).
        sample_rate (int): Sample rate of the audio.
        crossfade_duration (float): Duration of crossfade in seconds.

    Returns:
        torch.Tensor: Resulting audio tensor after crossfade.
    """
    if tensor1.shape[0] != tensor2.shape[0]:
        raise ValueError("Input tensors must have the same number of channels.")
    
    crossfade_samples = int(crossfade_duration * sample_rate)

    max_possible_crossfade = min(tensor1.shape[1], tensor2.shape[1])
    actual_crossfade_samples = min(crossfade_samples, max_possible_crossfade)

    if actual_crossfade_samples == 0:
        return torch.cat((tensor1, tensor2), dim=1)

    fade_out_ramp = torch.cos(torch.linspace(0, torch.pi / 2, steps=actual_crossfade_samples)).unsqueeze(0)
    fade_in_ramp = torch.sin(torch.linspace(0, torch.pi / 2, steps=actual_crossfade_samples)).unsqueeze(0)

    fade_out_region = tensor1[:, -actual_crossfade_samples:]
    fade_in_region = tensor2[:, :actual_crossfade_samples]

    faded_out = fade_out_region * fade_out_ramp
    faded_in = fade_in_region * fade_in_ramp

    crossfaded_region = faded_out + faded_in

    result = torch.cat((tensor1[:, :-actual_crossfade_samples], crossfaded_region, tensor2[:, actual_crossfade_samples:]), dim=1)

    return result
