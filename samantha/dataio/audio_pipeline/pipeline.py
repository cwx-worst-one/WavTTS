import copy
from typing import Callable, Dict, Iterable, List, Union

import numpy as np

from .action import Action


class AudioPipeline:
    r"""AudioPipeline is a standardised audio processing tool, which support reading,
    slicing, downmixing, etc. operations. Besides AudioPipeline also support custom
    operations via :func:`apply`.

    .. note::

        This tool use `pydub <https://github.com/jiaaro/pydub>`_ as the backend audio
        processor and internally using ffmpeg to do all audio format conversion.


    AudioPipeline support two ways to create a audio processing pipeline. User can use
    one of them or even mixing.

    1. create pipeline from init args

    .. code-block:: python

        pipeline = AudioPipeline(
            data_iter=inputs,
            pipeline=[
                ["read"],
                ["resample", 16000],
                ["mono"],
                ["slice_sample", 0, 2000],
                {"name": "numpy", "channel_first": True},
                ["norm"],
            ]
        )

    2. create pipeline from dot actions

    .. code-block:: python

        def multiply(audio, factor):
            return audio * factor

        pipeline = (
            AudioPipeline(data_iter=inputs)
            .read()
            .resample(target_sample_rate=16000)
            .mono()
            .slice_sample(0, 2000)
            .numpy()
            .norm()
            .apply("multiply", func=multiply, factor=10)
        )

    3. create pipeline with mixing methods

    .. code-block:: python

        pipeline = (
            AudioPipeline(
                data_iter=inputs,
                pipeline=[
                    ["read"],
                    ["resample", 16000],
                ]
            )
            .mono()
            .slice_sample(10, 2000)
        )

    Supported action specs of pipeline in :func:`__init__` are

    - list of element: [action_name, args(optional), kwargs(optional)]

    .. code-block:: yaml

        - ["numpy"]
        - ["numpy", False, np.float32]
        - ["numpy", False, dict(dtype=np.float32)]
        - ["numpy", dict(channel_first=True, dtype=np.float32)]

    - a dict: **name** must exist in the dict

    .. code-block:: yaml

        - {"name": "numpy"}
        - {"name": "numpy", "channel_first": False}

    Args:
        data_iter (Iterable): iterator to data or file list that need to be processed
        pipeline (Optional[List[Union[Dict, List]]]): a list of operations will be
            applied to data

    """

    def __init__(self, data_iter: Iterable, pipeline: List[Union[Dict, List]] = None):
        self.data_iter = data_iter
        self.pipeline = []
        if pipeline is not None:
            self.build_pipeline(pipeline)

    def read_file(self, **kwargs):
        r"""Append action read audio from file to pipeline.

        Args:
            **kwargs (dict): any parameter that need to passa
                to :func:`pydub.AudioSegment.from_file`

        Returns:
            AudioPipeline: an audio pipeline with action read_file added.

        """
        return self.compose(Action("read_file", **kwargs))

    def read_data(self, channel_first, sample_rate, num_channel):
        r"""Append action read audio from data to pipeline.

        Args:
            channel_first (bool): channel first layout or not
            sample_rate (int): sample rate of the audio
            num_channel (int): num channel of the audio

        Returns:
            AudioPipeline: an audio pipeline with action read_data added.

        """
        return self.compose(
            Action(
                "read_data",
                channel_first=channel_first,
                sample_rate=sample_rate,
                num_channel=num_channel,
            )
        )

    def read(self, **kwargs):
        r"""Append action read to pipeline.

        .. note::

            :func:`read` will automatically switch to :func:`read_file` or
            :func:`read_data` based on what user passed to :py:attr:`data_iter`.

        Args:
            **kwargs: any parameters that pass to :func:`read_file` or
                :func:`read_data`

        Returns:
            AudioPipeline: an audio pipeline with action read added.

        """
        return self.compose(Action("read", **kwargs))

    def resample(self, target_sample_rate):
        r"""Append action resample to pipeline.

        Args:
            target_sample_rate (int): target sample rate.

        Returns:
            AudioPipeline: an audio pipeline with action resample added.

        """
        return self.compose(Action("resample", tgt_sr=target_sample_rate))

    def mono(self):
        r"""Append action mono to pipeline.

        Returns:
            AudioPipeline: an audio pipeline with action mono added.

        """
        return self.compose(Action("mono"))

    def slice_duration(self, start: int = 0, end: int = None):
        r"""Append action slice_duration to pipeline.

        This action slice audio by given time points in milliseconds.

        Args:
            start (int): start time points.
            end (int): end time points.

        Returns:
            AudioPipeline: an audio pipeline with action slice_duration added.

        """
        return self.compose(Action("slice_duration", start=start, end=end))

    def slice_sample(self, start: int = 0, end: int = None):
        r"""Append action slice_sample to pipeline.

        This action slice audio by given index in samples.

        .. warning::

            If ``slice_sample`` called after :func:`numpy`, please make sure
            ``channel_first=False``.

        Args:
            start (int): start index
            end (int):  end index

        Returns:
            AudioPipeline: an audio pipeline with action slice_sample added.

        """
        return self.compose(Action("slice_sample", start=start, end=end))

    def numpy(self, channel_first: bool = False, dtype: np.dtype = None):
        r"""Append action numpy to pipeline.

        This action convert audio from AudioSegment to numpy ndarray.

        Args:
            channel_first (bool): channel first or not
            dtype (np.dtype): target data type

        Returns:
            AudioPipeline: an audio pipeline with action numpy added.

        """

        return self.compose(Action("numpy", channel_first=channel_first, dtype=dtype))

    def norm(self, scale: float = 1 / 32768):
        r"""Append action norm to pipeline.

        Norm value of audio by multiply scale (1/32768).

        Args:
            scale (float): scale that norm using.

        Returns:
            AudioPipeline: an audio pipeline with action norm added.

        """

        return self.compose(Action("norm", scale=scale))

    def int16(self, scale: float = 32768):
        r"""Append action int16 to pipeline.

        Convert data type of audio to int16 for saving disk usage.

        Args:
            scale (float): scale that int16 using.

        Returns:
            AudioPipeline: an audio pipeline with action int16 added.

        """

        return self.compose(Action("int16", scale=scale))

    def apply(self, name: str, func: Callable, *args, **kwargs):
        r"""Append a custom action to pipeline.

        Examples::

            >>> def multiply(audio, factor):
            ...     return audio * factor
            >>> pipeline = (
            ...     AudioPipeline(data_iter=[])
            ...     .apply(name="multipy", func=multiply, factor=10)
            ... )
            >>> print(pipeline)
            AudioPipeline(
                Action(name=multipy, args=(), kwargs={'factor': 10})
            )


        Args:
            name (str): action name
            func (Callable): custom action which could be invoked

        Returns:
            AudioPipeline: an audio pipeline with custom action added.

        """
        return self.compose(Action(name, func=func, *args, **kwargs))

    def append(self, f):
        self.pipeline.append(f)

    def compose(self, *args):
        result = copy.copy(self)
        for arg in args:
            result.append(arg)
        return result

    def build_pipeline(self, pipeline):

        if not isinstance(pipeline, list):
            raise TypeError(f"Expecting pipeline is a list, but got {type(pipeline)}")

        for action in pipeline:
            if isinstance(action, dict):
                name = action.pop("name", None)
                self._compose_action(name, **action)
            elif isinstance(action, list):

                act_len = len(action)
                if act_len > 1:
                    if isinstance(action[-1], dict):
                        self._compose_action(action[0], *action[1:-1], **action[-1])
                    else:
                        self._compose_action(action[0], *action[1:])
                else:
                    self._compose_action(action[0])
            else:
                raise TypeError(
                    f"Unsupported action type of pipeline {type(action)}"
                )  # noqa

    def _compose_action(self, name, *args, **kwargs):
        self.compose(Action(name, *args, **kwargs))

    def __repr__(self):  # pragma: no cover
        repr = "\nAudioPipeline(\n"
        for action in self.pipeline:
            repr += f"  {action}\n"
        repr += ")\n"
        return repr

    def __call__(self):
        for x in self.data_iter:
            for action in self.pipeline:
                x = action(x)
                if x is None:
                    break
            else:
                yield x
