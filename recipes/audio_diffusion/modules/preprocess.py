import logging
from abc import abstractmethod
from typing import Dict, Generator

import torch.nn as nn

logger = logging.getLogger(__name__)


class BufferPreprocessorBase:
    def __init__(self, sample_rate: int, transforms: nn.Module) -> nn.Module:
        self.sample_rate = sample_rate
        self.transforms = transforms
        logger.info(f"Preprocessor transforms:\n{self.transforms}")

    @abstractmethod
    def train_buffer_preprocessor(self, batch: Generator) -> Generator:
        pass


class WebDatasetBufferPreprocessor(BufferPreprocessorBase):
    def __init__(self, sample_rate: int, transforms: Dict[str, nn.Module]):
        super().__init__(sample_rate=sample_rate, transforms=transforms)
        self.count = 0
        self.skipped = 0

    def train_buffer_preprocessor(self, buffer: Generator) -> Generator:
        for item in buffer:
            self.count += 1
            item = self.transforms(item)
            if item.get("__skip__", False):
                self.skipped += 1
                if self.skipped % 100 == 0:
                    print(f"Skipped {self.skipped}/{self.count} items")
                continue
            yield item
