import logging
import random
from typing import Any, List

import torch
import pytorch_lightning as pl
from braceexpand import braceexpand
from webdataset import WebDataset, WebLoader
from webdataset.compat import FluidInterface
from webdataset.pipeline import DataPipeline

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

class ConcatDatasetsWithinBatch(DataPipeline, FluidInterface):
    def __init__(self, datasets: List[WebDataset], resampled: bool = False):
        super().__init__()
        self.datasets = datasets
        self.resampled = resampled

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        while True:
            data = []
            for source in sources:

                # Get next batch from source if available
                try:
                    source_data = next(source)
                except StopIteration:
                    source_data = None
                data.append(source_data)

            # If at least one batch available, return it
            if all( source_data is None for source_data in data):
                return
            else:
                yield tuple(data)


class ConcatDatasetsWithinBatch(DataPipeline, FluidInterface):
    def __init__(self, datasets: List[WebDataset], resampled: bool = False):
        super().__init__()
        self.datasets = datasets
        self.resampled = resampled

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        while True:
            data = []
            for source in sources:

                # Get next batch from source if available
                try:
                    source_data = next(source)
                except StopIteration:
                    source_data = None
                data.append(source_data)

            # If at least one batch available, return it
            if all( source_data is None for source_data in data):
                return
            else:
                yield tuple(data)



class WebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        validation_dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
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
            dataset=self.train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
