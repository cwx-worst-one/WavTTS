import os
import io
from typing import Tuple

import librosa
import numpy as np
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
