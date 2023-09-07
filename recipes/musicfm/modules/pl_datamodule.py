import pytorch_lightning as pl
from webdataset import WebLoader


class WebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        validation_dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        test_dataset=None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
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
            # num_workers=self.num_workers,
            num_workers=1,  # ensure no duplicate reads due to resampled=True
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset:
            return WebLoader(
                dataset=self.test_dataset,
                batch_size=None,
                shuffle=False,
                # num_workers=self.num_workers,
                num_workers=1,  # ensure no duplicate reads due to resampled=True
                pin_memory=self.pin_memory,
            )
        return None
