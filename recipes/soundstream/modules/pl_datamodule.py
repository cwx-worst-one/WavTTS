import pytorch_lightning as pl


class SoundstreamDataModule(pl.LightningDataModule):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
