r"""data preprocess functions"""
import logging
from random import randrange

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class ImageReader:
    r"""Read image using PIL and return a tensor.

    Args:
        transform (transforms.Compose): transform functions
    """

    def __init__(self, transform: transforms.Compose):
        self.transform = transform

    def __call__(self, img_path: str) -> torch.Tensor:
        r"""Perform reading.

        Args:
            img_path (str): the image file path
        """
        img = Image.open(img_path)
        return self.transform(img)


class AudioEditor:
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        if sampling_rate <= 0:
            self._raise_config_error("Sampling rate must be positive.")

    def _raise_config_error(self, msg: str):
        raise ValueError(f"Invalid config: {self.__repr__()}. {msg}")

    def _check_audio(self, audio: torch.Tensor):
        if len(audio.shape) != 2:
            raise ValueError(
                f"Audio has incorrect shape: {audio.shape}. "
                "It should be a 2D tensor (num_channels x num_frames)."
            )
        elif audio.shape[0] > audio.shape[1]:
            logging.warn(
                "Number of audio channels is greater than the number of frames. "
                "You may need to swap the dimensions of the input array."
            )


class AudioChunker(AudioEditor):
    r"""Returns multiple chunks from equally distributed positions. If the model's
    receptive field is 2 channels with 30 seconds and number of chunks are 8,
    the output shape is (2, 8, 30 * sampling_rate).
    Note: No padding will be added to the original audio

    Args:
        sampling_rate (int): sampling rate of original audio
        chunk_duration_in_s (float): length in seconds of each chunk
        num_chunks (int): number of chunks
        hop_size (int): hop size of audio chunking. default=None
        is_random (bool): only take a single random chunk if True.
            If false, take all the chunks. default=False

    Returns:
        (torch.Tensor, torch.Tensor): Returns chunked array
        if is_random is disabled:
            output shape:
            torch.Size([channel_size, num_chunks,
                chunk_duration_in_s * sampling_rate])
        if is_random is enabled:
            output shape:
            torch.Size([channel_size, chunk_duration_in_s * sampling_rate])
        and empty padding_mask
    """

    def __init__(
        self,
        sampling_rate: int,
        chunk_duration_in_s: float,
        num_chunks: int = None,
        hop_size: int = None,
        is_random: bool = False,
    ):
        self.sampling_rate = sampling_rate
        self.chunk_duration_in_s = chunk_duration_in_s
        self.num_chunks = num_chunks
        self.hop_size = hop_size
        self.is_random = is_random
        super().__init__(sampling_rate)
        if chunk_duration_in_s <= 0:
            self._raise_config_error("chunk_duration_in_s must be postivie.")

        valid_num_chunks = num_chunks and num_chunks > 1
        valid_hop_size = hop_size and hop_size > 0

        if not valid_num_chunks and not valid_hop_size:
            self._raise_config_error("num_chunks and hop_size both invalid")

        elif valid_num_chunks and valid_hop_size:
            self._raise_config_error(
                "Need one and only one " "valid config from num_chunks and hop_size"
            )

    def __call__(self, audio: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        r"""Perform chunking.
        Args:
            audio (torch.Tensor): the shape should be
            torch.Size([channel_size, audio_length])
            e.g. torch.Size([3, 358870])
        """
        self._check_audio(audio)
        len_chunk = self.chunk_duration_in_s * self.sampling_rate
        len_audio = audio.shape[-1]
        if len_chunk >= len_audio:
            raise ValueError(
                f"len_chunk({len_chunk}) must be less than the"
                f" length of audio({len_audio})"
            )

        chunked_tensor = []
        # Use hop size
        if self.hop_size is not None:
            hop_size = self.hop_size
            num_chunks = 1 + (len_audio - len_chunk) // hop_size
        # Use num_chunks
        else:
            num_chunks = self.num_chunks
            hop_size = (len_audio - len_chunk) // (num_chunks - 1)

        channel_size = len(audio)
        for chan in range(channel_size):
            if self.is_random:
                random_chunk = randrange(num_chunks)
                chunked_tensor.append(
                    audio[
                        chan,
                        int(random_chunk * hop_size) : int(
                            random_chunk * hop_size + len_chunk
                        ),
                    ]
                )
            else:
                chunked_tensor.append(
                    torch.stack(
                        [
                            audio[
                                chan, int(i * hop_size) : int(i * hop_size + len_chunk),
                            ]
                            for i in range(num_chunks)
                        ]
                    )
                )
        chunked_tensor = torch.stack(chunked_tensor)

        padding_mask = torch.zeros_like(chunked_tensor)
        return chunked_tensor, padding_mask

    def __repr__(self):
        return (
            f"AudioChunker("
            f"sampling_rate={self.sampling_rate},"
            f"chunk_duration_in_s={self.chunk_duration_in_s},"
            f"num_chunks={self.num_chunks},"
            f"hop_size={self.hop_size},"
            f"is_randome={self.is_random})"
        )


class AudioLengthModifier(AudioEditor):
    r"""Adjust the length of waveform (numpy array).
    Trim off the end if it's too long, add zeros if it's too short.

    Args:
        target_duration_sec (float): target length of the audio in seconds
        sampling_rate (float): sample rate of the input audio
        random_crop (bool): Randomly crop longer sequences when the
            target sequence is shorter than the input. If False, sample
            from the start of the sequence and truncate the end.

    Returns:
        (torch.Tensor, torch.Tensor): length-adjusted audio.
        (torch.Tensor, torch.Tensor): padding_mask, same shape as waveform
        int: start index of the source audio if target audio length is
            shorter and use randome crop, otherwise 0.
    """

    def __init__(
        self,
        sampling_rate: int,
        target_duration_sec: float,
        is_random_crop: bool = True,
    ):
        self.target_duration_sec = target_duration_sec
        self.is_random_crop = is_random_crop
        super().__init__(sampling_rate)

        if target_duration_sec < 0:
            self._raise_config_error("target_duration_sec must be positive.")

    def __call__(self, audio: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        r"""Perform length modifiying.
        Args:
            audio (torch.Tensor): the shape should be
            torch.Size([channel_size, audio_length])
            e.g. torch.Size([3, 358870])
        """
        self._check_audio(audio)
        num_channels, source_frames = audio.shape
        target_frames = int(self.target_duration_sec * self.sampling_rate)

        start_idx = 0
        if source_frames < target_frames:
            output_audio = F.pad(
                input=audio,
                pad=(0, target_frames - source_frames),
                mode="constant",
                value=0,
            )
            padding_mask = F.pad(
                input=torch.zeros_like(audio),
                pad=(0, target_frames - source_frames),
                mode="constant",
                value=1,
            )
        else:
            if self.is_random_crop:
                start_idx = randrange(source_frames - target_frames)
            output_audio = audio.clone()[:, start_idx : start_idx + target_frames]
            padding_mask = torch.zeros((num_channels, target_frames))

        return output_audio, padding_mask, start_idx

    def __repr__(self):
        return (
            f"AudioLengthModifier("
            f"sampling_rate={self.sampling_rate},"
            f"target_duration_sec={self.target_duration_sec},"
            f"is_random_crop={self.is_random_crop})"
        )


class AudioSlicer:
    r"""Slice an audio by start time and end time.

    Args:
        sample_rate (int): The sample rate of the audio.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def __call__(
        self, audio: np.ndarray, start_time: float = 0, end_time: float = -1
    ) -> torch.Tensor:
        r"""Perform slicing.

        Args:
            audio (np.ndarray): Audio to be sliced, stored in np.ndarray.
            start_time (float): Start time of slice, in second (default: 0).
            end_time (float): End time of slice, in second (default: -1).

        Returns:
            torch.Tensor: Returns tensor of sliced audio
        """
        start = int(start_time * self.sample_rate)
        end = int(end_time * self.sample_rate) if end_time != -1 else audio.shape[0]
        sliced_audio = audio.copy()[start:end]
        return torch.from_numpy(sliced_audio)
