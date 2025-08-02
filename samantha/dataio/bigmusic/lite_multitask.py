from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union

from cruise import CruiseDataModule
from lite.module.datamodule import (
    DataModuleMultiTaskConfig,
    LiteDataModule,
    _MultiIterableDatasetArgs,
)


class MusicLiteDataModule(LiteDataModule, CruiseDataModule):

    def __init__(self, config, **kwargs):
        LiteDataModule.__init__(self, config=config, **kwargs)
        CruiseDataModule.__init__(self)
        self.train_dataset_args: Optional[_MultiIterableDatasetArgs] = None
        self.val_dataset_args: Optional[_MultiIterableDatasetArgs] = None
        self.predict_dataset_args: Optional[_MultiIterableDatasetArgs] = None
        self.save_hparams()

    def setup(self):
        LiteDataModule.setup(self)

    def load_state_dict(self, state_dict):
        LiteDataModule.load_state_dict(self, state_dict)

    def state_dict(self):
        return LiteDataModule.state_dict(self)

    def train_dataloader(self):
        return LiteDataModule.train_dataloader(self)

    def val_dataloader(self):
        return LiteDataModule.val_dataloader(self)

    def predict_dataloader(self):
        return LiteDataModule.predict_dataloader(self)

    def teardown(self):
        LiteDataModule.teardown(self)
