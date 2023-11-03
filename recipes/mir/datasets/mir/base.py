# base

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torchaudio
import webdataset as wds
from samantha.dataio.batching import BucketBatcher
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose
from tqdm import tqdm

BatchedStr = Union[str, List[str]]


@dataclass
class DataResult:
    shard: BatchedStr
    key: BatchedStr


def collate_batch(batch) -> DataResult:
    class_keys = vars(batch[0]).keys()
    init_args = {k: [] for k in class_keys}
    new_class = batch[0].__class__(**init_args)

    for b in batch:
        for k in class_keys:
            new_class.__dict__[k].append(b.__dict__[k])

    for k in class_keys:
        if type(new_class.__dict__[k][0]) == torch.Tensor:
            new_class.__dict__[k] = torch.stack(new_class.__dict__[k], dim=0)
    return new_class


def _load_waveform(path: str, exp_sample_rate: int):
    waveform, sample_rate = torchaudio.load(path)
    if exp_sample_rate != sample_rate:
        raise ValueError(
            f"sample rate should be {exp_sample_rate}, but got {sample_rate}"
        )
    return waveform, sample_rate














# mir.base

import logging
from abc import abstractmethod, abstractproperty
from typing import Any, Iterable, List, Optional

import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose

from samantha.dataio.batching import BucketBatcher
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)
from samantha.utils.hdfs_helper import ARNOLD_REGION

# from samantha.utils.logger import RankedLogger

# logger = RankedLogger()

logger = logging.getLogger(__name__)

_hdfs_root = {
    "US": "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/mir_benchmark",
    "CN": "hdfs://haruna/home/byte_speech_sv/data/music/mir_benchmark",
}

class MIRDataModuleBase:
    _splits = []
    _sample_rate = None

    def __init__(self, split: str, resample: bool):
        super().__init__()

        if split not in self._splits:
            raise NotImplementedError(f"{split} does not exist for this dataset")

        self.split = split
        self.resample = resample
        self.logger = logger

        self.init_provider()

    @abstractmethod
    def transform(self, items: Iterable[Any]) -> Iterable[Any]:
        pass

    @property
    def _root(self):
        if ARNOLD_REGION is None:
            raise Exception(
                f"ARNOLD_REGION must be defined as either: {_hdfs_root.keys()}"
            )
        return _hdfs_root[ARNOLD_REGION]

    @property
    def is_train(self):
        return self.split == "train"

    def init_provider(self):
        urls = self.pipe_shard_urls(self.shard_urls)

        provider = wds.DataPipeline(
            self.shard_sampler(urls),
            wds.tarfile_to_samples(),
            wds.shuffle(100),  #  TODO!!
            wds.decode(),
            self.transform,
        )
        self.provider = provider
        self.logger.info(
            f"Collected {len(urls)} shard urls for data split: {self.split}"
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def shard_sampler(self):
        if self.resample:
            return wds.ResampledShards
        return wds.SimpleShardList

    @abstractproperty
    def shard_urls(self) -> List[str]:
        return []

    @staticmethod
    def pipe_shard_urls(shard_urls):
        return [f"pipe: hdfs dfs -cat {url}" for url in shard_urls]



class WebDataModuleBase(LightningDataModuleBase):
    def __init__(
        self,
        train_dataset,
        batch_size: int,
        shuffle_buffer_size: int,
        validation_dataset=None,
        test_dataset=None,
        predict_dataset=None,
        num_workers: int = 8,
        pin_memory: bool = True,
        batcher: Optional[BucketBatcher] = None,
        batch_size_valid: Optional[int] = None,
        batch_size_test: Optional[int] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.train_dataset = train_dataset
        self.shuffle_buffer_size = shuffle_buffer_size
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.batcher = batcher
        self.batch_size_valid = batch_size_valid
        self.batch_size_test = batch_size_test

        if self.batch_size_valid is None:
            self.batch_size_valid = self.batch_size
        if self.batch_size_test is None:
            self.batch_size_test = self.batch_size

    def collate_fn(self, batch):
        return collate_batch(batch)

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                # TODO: this can be more elegant
                if hasattr(batch[0], "audio"):
                    max_length = max([self.batcher.length_fn(item) for item in batch])
                    random_pad = RandomPad(max_length)
                    for idx in range(len(batch)):
                        batch[idx].audio = random_pad(batch[idx].audio)
                yield batch

    def DataLoader(self, dataset: wds.DataPipeline, batch_size: int, num_workers: int):
        dataset.pipeline.append(wds.shuffle(self.shuffle_buffer_size))
        if self.batcher is None:
            dataset.pipeline.append(
                wds.batched(batch_size, collation_fn=self.collate_fn, partial=False)
            )
        if self.batcher is None:
            collate_fn = None
        else:
            collate_fn = self.collate_fn
        return DataLoader(
            dataset, batch_size=None, num_workers=num_workers, collate_fn=collate_fn
        )

    def train_dataloader(self):
        # provider = self.train_dataset.provider
        return self.DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # provider = self.validation_dataset.provider
        return self.DataLoader(
            self.validation_dataset,
            self.batch_size_valid,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        # provider = self.test_dataset.provider
        return self.DataLoader(
            self.test_dataset,
            self.batch_size_test,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        # provider = self.predict_dataset.provider
        return self.DataLoader(
            self.predict_dataset,
            self.batch_size,
            num_workers=self.num_workers,
        )


class BaseAudioTransform:
    def __init__(self):
        self.transform = Compose(
            [
                ToTensor(),
                NormalizeAudioToFloat32(),
                SetAudioDimensions(),
            ]
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
