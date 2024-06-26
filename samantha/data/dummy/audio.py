import random

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from samantha.data.audio.types import AudioDataResult, AudioMeta, SegmentInfo
from samantha.data.base import DataResult, LightningDataModuleBase
from samantha.transforms.audio import Pad


class DummyAudioDataset(IterableDataset):
    def __init__(
        self,
        data_type: str,
        sample_rate: int,
        n_channels: int,
        min_duration: int,
        max_duration: int,
    ):
        self.urls = []
        self.data_type = data_type
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.min_audio_samples = min_duration * sample_rate
        self.max_audio_samples = max_duration * sample_rate

    def generate(self):
        while True:
            idx = random.randint(0, 9999999)
            n_frames = random.randint(self.min_audio_samples, self.max_audio_samples)
            audio = torch.randn((self.n_channels, n_frames))
            # input_length = audio.shape[-1]

            segment_info = SegmentInfo(
                meta=AudioMeta(
                    path=idx,
                    duration=n_frames / self.sample_rate,
                    sample_rate=self.sample_rate,
                ),
                seek_time=0,
                n_frames=n_frames,
                total_frames=n_frames,
                sample_rate=self.sample_rate,
                channels=self.n_channels,
                data_type=self.data_type,
            )
            result = AudioDataResult(
                audio=audio, shard=None, key=None, segment_info=segment_info, index=None
            )
            yield result

    def __iter__(self):
        return iter(self.generate())


class DummyAudioDataModule(LightningDataModuleBase):
    def __init__(
        self,
        data_type: str,
        sample_rate: int,
        n_channels: int,
        min_duration: float,
        max_duration: float,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):

        dataset = DummyAudioDataset(
            data_type, sample_rate, n_channels, min_duration, max_duration
        )

        super().__init__(
            train_dataset=dataset,
            validation_dataset=dataset,
            test_dataset=dataset,
            predict_dataset=dataset,
            batch_size=batch_size,
            shuffle=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def collate_fn(self, batch) -> DataResult:

        collated = AudioDataResult(
            audio=[],
            key=[],
            shard=[],
            segment_info=[],
            index=[],
            # input_length=[],
        )
        keys = collated.keys()

        for b in batch:
            for k in keys:
                collated[k].append(b[k])

        # max_length = max(collated.input_length)
        # pad = Pad(max_length, value=0.0)
        # for i in range(batch_size):
        #     audio = collated.audio[i]
        #     collated.audio[i] = pad(audio)

        collated.audio = torch.stack(collated.audio, dim=0)
        return collated


if __name__ == "__main__":
    datamodule = DummyAudioDataModule(
        sample_rate=44100,
        n_channels=1,
        min_duration=30,
        max_duration=30,
        batch_size=8,
        num_workers=8,
    )

    dataloader = datamodule.train_dataloader()

    for batch in tqdm(dataloader):
        pass
