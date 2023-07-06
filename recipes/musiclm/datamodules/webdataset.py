import logging
import random
from typing import Any, List

import pytorch_lightning as pl
from braceexpand import braceexpand
import webdataset as wds
from webdataset import WebDataset, WebLoader
from webdataset.compat import FluidInterface
from webdataset.pipeline import DataPipeline
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def return_self(x: Any) -> Any:
    return x


def read_urls_from_file(fp: str) -> List[str]:
    with open(fp) as f:
        return f.read().splitlines()


def expand_urls(urls: List[str]) -> List[str]:
    expanded = []
    for u in urls:
        if type(u) == str:
            expanded.extend(list(braceexpand(u)))
        elif type(u) == list:
            expanded.extend(u)
    logger.info(f"Number of URLs: {len(expanded)}")
    return expanded


class SampleEqually(DataPipeline, FluidInterface):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        while True:
            for source in sources:
                try:
                    yield next(source)
                except StopIteration:
                    return


class RandomSample(DataPipeline, FluidInterface):
    def __init__(self, datasets: List[WebDataset], probabilities: List[float]):
        super().__init__()
        self.datasets = datasets
        self.probabilities = probabilities

    def sample_single_idx(self, samples):
        return random.choices(range(len(samples)), weights=self.probabilities, k=1)[0]

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        while True:
            source_idx = self.sample_single_idx(sources)
            try:
                yield next(sources[source_idx])
            except StopIteration:
                return


class WebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 100,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return WebLoader(
            dataset=self.train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return WebLoader(
            dataset=self.validation_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return WebLoader(
            dataset=self.predict_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 100,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset,
            wds.shuffle(self.shuffle_buffer_size),
            wds.to_tuple("audio"),
            wds.batched(self.batch_size),
        )
        return DataLoader(train_dataset_batched, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        validation_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.to_tuple("audio"),
            wds.batched(self.batch_size),
        )
        return DataLoader(validation_dataset_batched, batch_size=None, num_workers=self.num_workers)

    def predict_dataloader(self):
        predict_dataset_batched = DataPipeline(
            self.predict_dataset,
            # wds.shuffle(self.shuffle_buffer_size),
            wds.to_tuple("audio"),
            wds.batched(self.batch_size),
        )
        return DataLoader(predict_dataset_batched, batch_size=None, num_workers=self.num_workers)
