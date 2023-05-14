from typing import Dict, Generator, Optional
import io
import torch
from torchaudio_augmentations import Compose

from recipes.musiclm.transforms.audio import (
    NormalizeAudio,
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
    LoudnessCheck,
    ReadMP3,
)
from recipes.musiclm.transforms.base import TransformBase


class MusicLMTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        audio_key: str,
        sample_range_key: Optional[str] = None,
        min_volume_threshold: Optional[float] = 0.0,
        num_crops: int = 1,
        num_tries: int = 5,
    ) -> None:
        super().__init__()
        self.audio_key = audio_key
        self.sample_range_key = sample_range_key
        self.min_volume_threshold = min_volume_threshold
        self.num_crops = num_crops
        self.num_tries = num_tries

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.normalize_audio_fp32]
        )

        self.random_pad = RandomPad(n_samples=n_samples)
        self.random_crop = RandomResizedCrop(n_samples=n_samples)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        audio = self.base_transform(x[self.audio_key])

        if self.sample_range_key is not None:
            sample_range = self.base_transform(x[self.sample_range_key])
            audio = self.normalize_audio(audio, norm_tensor=sample_range)

        audio = self.random_pad(audio)
        # Return up to self.num_crops crops
        num_crops = 0
        num_tries = 0
        while num_crops < self.num_crops and num_tries < self.num_tries:
            cropped_audio = self.random_crop(audio)
            if (
                self.min_volume_threshold is None
                or torch.mean(torch.abs(cropped_audio)) >= self.min_volume_threshold
            ):
                yield {"audio.npy": cropped_audio}
                num_crops += 1
            num_tries += 1
        self._update_stats(num_crops == 0)


class MCCTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        sample_rate: int,
        metalist_path: str,
        audio_key: str = "mp3",
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold

        with open(metalist_path, "r") as fp:
            self.metalist = [self.metalist2id(line.strip()) for line in fp.readlines()]

        self.is_loud = LoudnessCheck(
            self.sample_rate,
            self.min_volume_threshold,
            self.loudness_ratio_threshold
        )
        self.read_mp3 = ReadMP3(self.sample_rate)
        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [
                self.read_mp3,
                self.to_tensor,
                self.audio_dim,
                self.normalize_audio_fp32
            ]
        )

        self.random_pad = RandomPad(n_samples=n_samples)
        self.random_crop = RandomResizedCrop(n_samples=n_samples)

    def key2id(self, url, key):
        """
        url: 'pipe:hdfs dfs -cat hdfs://harunava/home/byte_speech_sv/data/genre_balanced_mcc_clean/classical.2.57/0.tar'
        key: '6980478627137718274'
        return: 'classical.2.57/0/6980478627137718274'
        """
        sub_folder = url.split("/")[-2]
        tar_name = url.split("/")[-1].split(".")[0]
        return f"{sub_folder}/{tar_name}/{key}"

    def metalist2id(self, path):
        """
        path: '../genre_balanced_mcc_clean_tags/8-bit.0.54/0/0-7196752726452242434.json'
        return: '8-bit.0.54/0/7196752726452242434'
        """
        sub_folder = path.split("/")[-3]
        tar_name = path.split("/")[-2]
        id = path.split("/")[-1].split(".")[0].split("-")[-1]
        return f"{sub_folder}/{tar_name}/{id}"

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        audio = self.base_transform(io.BytesIO(x[self.audio_key]))

        audio = self.random_pad(audio)
        # Return up to self.num_crops crops
        max_num_crops = audio.size(1) // self.n_samples
        num_crops = 0

        if self.key2id(x["__url__"], x["__key__"]) in self.metalist:
            while num_crops < max_num_crops:
                cropped_audio = self.random_crop(audio)
                if self.is_loud(cropped_audio):
                    yield {"audio.npy": cropped_audio}
                num_crops += 1
        else:
            self._update_stats(skipped=True)
