import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from recipes.waveformvae.dataset.train import collate_fn


class WaveformVAEDataModule(pl.LightningDataModule):
    def __init__(self, train_dataloader=None, val_dataloader=None):
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
