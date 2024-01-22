import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from pytorch_lightning import LightningDataModule

    _DM_BASE_CLS = LightningDataModule
except ImportError:
    _DM_BASE_CLS = object


class LiteDataModule(_DM_BASE_CLS):
    def __init__(
        self,
        train_dataloader: Any = None,
        val_dataloader: Any = None,
        test_dataloader: Any = None,
        predict_dataloader: Any = None,
    ):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self._predict_dataloader = predict_dataloader

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def predict_dataloader(self):
        return self._predict_dataloader
