import pytorch_lightning as pl
import webdataset as wds
import torch
from torch.utils.data import DataLoader

from recipes.t5_mulan.dataset.val import KaggleValDataset


class MuLanDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        batch_size=72,
        val_batch_size=128,
        sample_buffer_size=600,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.sample_buffer_size = sample_buffer_size

    def train_dataloader(self):
        train_dataset_batched = wds.DataPipeline(
            self.train_dataset,
            wds.shuffle(self.sample_buffer_size),
            wds.batched(self.batch_size, collation_fn=self._collate_training),
        )
        return DataLoader(train_dataset_batched, batch_size=None, num_workers=4)

    def val_dataloader(self):
        dataset = KaggleValDataset(nodesplitter=lambda x: x)
        val_dataset_batched = wds.DataPipeline(
            dataset,
            wds.batched(self.val_batch_size, collation_fn=self._collate_kaggle_val),
        )
        return DataLoader(val_dataset_batched, batch_size=None, num_workers=4)

    def _collate_kaggle_val(self, samples):
        """Collate function for kaggle val dataset."""
        batch = {}
        for k in samples[0].keys():
            batch[k] = [sample[k] for sample in samples]
        batch["audio"] = torch.cat(batch["audio"])
        batch["music_id"] = torch.tensor(batch["music_id"])
        batch["text_embeds"] = torch.stack(batch["text_embeds"])
        batch["text_embeds_mask"] = torch.stack(batch["text_embeds_mask"])
        batch["aspect_list_embeds"] = torch.stack(batch["aspect_list_embeds"])
        batch["aspect_list_embeds_mask"] = torch.stack(batch["aspect_list_embeds_mask"])
        return batch

    def _collate_training(self, samples):
        """Collate function for kaggle val dataset."""
        batch = {}
        for k in samples[0].keys():
            batch[k] = [sample[k] for sample in samples]
        batch["audio"] = torch.cat(batch["audio"])
        batch["music_id"] = torch.tensor(batch["music_id"])
        batch["text_embeds"] = torch.stack(batch["text_embeds"])
        batch["text_embeds_mask"] = torch.stack(batch["text_embeds_mask"])
        return batch


if __name__ == "__main__":
    from recipes.t5_mulan.dataset.mcc import MCCN2MDatasetApril
    train_dataset = MCCN2MDatasetApril()
    datamodule = MuLanDataModule(train_dataset)
    for batch in datamodule.train_dataloader():
        print(batch.keys())
        for k in ['audio', 'text_embeds', 'text_embeds_mask']:
            print(k, batch[k].shape)
        print("music_id", type(batch["music_id"]), batch["music_id"])
        break
    for batch in datamodule.val_dataloader():
        print(batch.keys())
        for k in ['audio', 'text_embeds', 'text_embeds_mask', 'aspect_list_embeds', 'aspect_list_embeds_mask']:
            print(k, batch[k].shape)
        print("music_id", type(batch["music_id"]), batch["music_id"])
        break
