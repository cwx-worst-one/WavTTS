import cruise
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from recipes.unit2speech.dataset.audio import AudioDataset


class U2SDataModule(cruise.CruiseDataModule):
    def __init__(self, wav_list, batch_size, num_workers=4, limit=None, use_formant_shift=False, return_path=False):
        super().__init__()
        self.save_hparams()
    
    def setup(self):
        self.train_dataset = AudioDataset(
            self.hparams.wav_list,
            self.hparams.limit,
            use_formant_shift=self.hparams.use_formant_shift,
        )

    def train_dataloader(self):
        dist_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        batch_sampler = BatchSampler(
            dist_sampler,
            batch_size=self.hparams.batch_size,
            drop_last=False,
        )
        dataloader = DataLoader(
            self.train_dataset,
            sampler=batch_sampler,
            collate_fn=lambda batch: batch[0],
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def predict_dataloader(self):
        self.predict_dataset = AudioDataset(
            self.hparams.wav_list,
            self.hparams.limit,
            use_formant_shift=self.hparams.use_formant_shift,
            return_path=self.hparams.return_path,
        )
        dist_sampler = DistributedSampler(self.predict_dataset, shuffle=True)
        batch_sampler = BatchSampler(
            dist_sampler,
            batch_size=self.hparams.batch_size,
            drop_last=False,
        )
        dataloader = DataLoader(
            self.predict_dataset,
            sampler=batch_sampler,
            collate_fn=lambda batch: batch[0],
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        return None

if __name__ == "__main__":
    wav_list = "en_1400h.list"
    bs = 10
    dm = U2SDataModule(wav_list, batch_size=bs)
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch[0][0].shape, batch[0][1].shape)
