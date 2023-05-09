import os
from typing import Callable, Union

import numpy as np
from pydub import AudioSegment


def _read_from_data(
    data: Union[bytes, np.ndarray],
    sample_rate: int = 16000,
    num_channel: int = 1,
    channel_first: bool = False,
):

    if isinstance(data, np.ndarray):
        if channel_first:
            data = data.T.reshape((-1))
        if data.dtype != np.int16:
            data = _int16(data).tobytes()
    elif not isinstance(data, bytes):
        raise TypeError(
            f"Expecting data is np.ndarray convertible, but got {type(data)}"
        )

    return AudioSegment(
        data=data, sample_width=2, frame_rate=sample_rate, channels=num_channel
    )


def _read_from_file(file, **kwargs):
    return AudioSegment.from_file(file=file, **kwargs)


def _read(input: Union[str, bytes], **kwargs):
    if isinstance(input, str) or isinstance(input, os.PathLike):
        return _read_from_file(input, **kwargs)

    return _read_from_data(input, **kwargs)


def _resample(audio: AudioSegment, tgt_sr: int):
    return audio.set_frame_rate(tgt_sr)


def _mono(audio: AudioSegment):
    return audio.set_channels(1)


def _slice_duration(audio: AudioSegment, start: int = 0, end: int = None):
    if end is None:
        end = len(audio)
    return audio[start:end]


def _slice_sample(
    audio: Union[AudioSegment, np.ndarray], start: int = 0, end: int = None
):
    if isinstance(audio, AudioSegment):
        return audio.get_sample_slice(start_sample=start, end_sample=end)
    elif isinstance(audio, np.ndarray):
        if end is None:
            end = len(audio)
        return audio[start:end]
    else:
        raise TypeError(
            f"Expecting type of audio either AudioSegment or np.ndarray,"
            f" but got {type(audio)}"
        )


def _numpy(audio: AudioSegment, channel_first=False, dtype=None):
    data = np.array(audio.get_array_of_samples()).reshape((-1, audio.channels))
    if channel_first:
        data = data.T
    return data.astype(dtype=dtype)


def _norm(audio: np.ndarray, scale: float = 1 / 32768):
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"Expecting type of audio np.ndarray, but got {type(audio)},"
            f" please call ``.numpy()`` first."
        )
    return audio.astype(np.float32) * scale


def _int16(audio: np.ndarray, scale: float = 32768):
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"Expecting type of audio np.ndarray, but got {type(audio)},"
            f" please call ``.numpy()`` first."
        )
    return (audio * scale).astype(np.int16)


ACTION_MAPPING = {
    "read_file": _read_from_file,
    "read_data": _read_from_data,
    "read": _read,
    "slice_duration": _slice_duration,
    "slice_sample": _slice_sample,
    "mono": _mono,
    "resample": _resample,
    "numpy": _numpy,
    "norm": _norm,
    "int16": _int16,
}


class Action:
    r"""Action is a wrapper of real audio processing operation.

    Here is the list of builtin action names:

    .. code-block:: python

       "read_file": _read_from_file,
        "read_data": _read_from_data,
        "read": _read,
        "slice_duration": _slice_duration,
        "slice_sample": _slice_sample,
        "mono": _mono,
        "resample": _resample,
        "numpy": _numpy,
        "norm": _norm,
        "int16": _int16,

    Args:
        name (str): action name
        func (Callable): for custom actions, user could omit it if using
            builtin actions
    """

    def __init__(self, name: str, *args, func: Callable = None, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        if func is not None:
            self.func = func
        else:
            try:
                self.func = ACTION_MAPPING[self.name]
            except KeyError:
                raise ValueError(f"Unsupported action name {name}")  # noqa

    def __repr__(self):  # pragma: no cover
        return f"Action(name={self.name}, args={self.args}, kwargs={self.kwargs})"

    def __call__(self, input=None):

        if input is None:
            return self.func(*self.args, **self.kwargs)

        return self.func(input, *self.args, **self.kwargs)
