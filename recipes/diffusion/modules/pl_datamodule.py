import pytorch_lightning as pl
import webdataset as wds
from torch.utils.data import DataLoader
from recipes.soundstream.dataset.utils import collate_fn

class DiffusionDataModule(pl.LightningDataModule):
    def __init__(self, 
        train_dataset, 
        val_dataset,
        train_batch_size=72,
        val_batch_size=72,
        train_num_workers=4,
        val_num_workers=4,
        sample_buffer_size=600,
        num_steps_per_val=10,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.sample_buffer_size = sample_buffer_size
        self.num_steps_per_val = num_steps_per_val


    def train_dataloader(self):
        train_dataset_batched = wds.DataPipeline(
            self.train_dataset,
            wds.shuffle(self.sample_buffer_size),
            wds.batched(self.train_batch_size, collation_fn=collate_fn),
        )
        return DataLoader(train_dataset_batched, batch_size=None, num_workers=self.train_num_workers)


    def val_dataloader(self):
        val_dataset_batched = wds.DataPipeline(
            self.val_dataset,
            wds.batched(self.val_batch_size, collation_fn=collate_fn),
        ).with_epoch(self.num_steps_per_val)
        
        return DataLoader(val_dataset_batched, batch_size=None, num_workers=self.val_num_workers)
