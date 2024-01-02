import logging
from typing import Any

logger = logging.getLogger(__name__)


class LiteDataModule:
    def __init__(
        self,
        train_dataloader: Any = None,
        val_dataloader: Any = None,
        test_dataloader: Any = None,
    ):
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader
