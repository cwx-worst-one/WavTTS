import logging
from abc import abstractmethod
from typing import Generator

import torch.nn as nn

from recipes.mulan.transforms.base import TransformBase

logger = logging.getLogger(__name__)


class BufferPreprocessorBase:
    def __init__(self, sample_rate: int, transforms: TransformBase) -> nn.Module:
        self.sample_rate = sample_rate
        self.transforms = transforms

    @abstractmethod
    def train_buffer_preprocessor(self, batch: Generator) -> Generator:
        pass


class WebDatasetBufferPreprocessor(BufferPreprocessorBase):
    def __init__(self, sample_rate: int, transforms: TransformBase):
        super().__init__(sample_rate=sample_rate, transforms=transforms)

    def train_buffer_preprocessor(self, buffer: Generator) -> Generator:
        for item in buffer:
            yield from self.transforms(item)
