from pytorch_lightning import LightningDataModule


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        predict_dataloader=None,
    ):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self._predict_dataloader = predict_dataloader

    def train_dataloader(self):
        return self._train_dataloader

    def predict_dataloader(self):
        return self._predict_dataloader
