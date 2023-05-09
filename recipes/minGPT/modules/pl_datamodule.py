import pytorch_lightning as pl


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, train_dataloader):
        super().__init__()
        self._train_dataloader = train_dataloader

    def train_dataloader(self):
        return self._train_dataloader
