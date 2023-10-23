import pytorch_lightning as pl
from recipes.beat.utils.inference_dataset import AudioInferenceDataModule


class ChordDataModule(AudioInferenceDataModule):
    def __init__(self, train_dataloader, val_dataloader, sampling_rate=16000, mono=True, **kwargs):
        super().__init__(sampling_rate, mono, **kwargs)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
