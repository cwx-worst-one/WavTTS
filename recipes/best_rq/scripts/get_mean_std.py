from recipes.best_rq.modules.lit_datamodule import DataModule
from recipes.musiclm.datasets.mcc import PGCDataset
import torch
from tqdm import tqdm


class Preprocessor:
    def __init__(self, dataloader):
        super(Preprocessor, self).__init__()
        # initialize
        self.total_samples = 0.0
        self.mean = torch.zeros((128))
        self.std = torch.zeros((128))

        # prepare data loader
        self.dataloader = dataloader

    def update(self, x):
        print("[Batch] mean:", x.mean(dim=0))
        print("[Batch] std:", x.std(dim=0))
        # count total samples
        num_samples = x.size(0)
        self.total_samples += num_samples

        # update mean
        new_mean = x.mean(dim=0)
        delta = new_mean - self.mean
        self.mean += delta * num_samples / self.total_samples
        new_std = ((x - new_mean) * (x - self.mean)).sum(dim=0)
        self.std += new_std

    def iterate(self, num_iter):
        i = 0
        for x in self.dataloader:
            feature = x["feature"][:, :, :-1].transpose(1, 2).reshape((-1, 128))
            self.update(feature)
            i += 1
            print("iter: %d" % i)
            print("[EST] mean:", self.mean)
            print("[EST] std:", torch.sqrt(self.std / (self.total_samples - 1)))
            if i % 100 == 0:
                torch.save(self.mean, f"/mnt/bn/zongyu-lq/features/best_rq/mv/pgc600k_mean_{i}.pt")
                torch.save(torch.sqrt(self.std / (self.total_samples - 1)), f"/mnt/bn/zongyu-lq/features/best_rq/mv/pgc600k_std_{i}.pt")
            if i >= num_iter:
                return



if __name__ == "__main__":
    mcc_dataset = PGCDataset(
        sample_rate=24000,
        duration=30,
        normalize_audio=True,
        resampled=True,
        shardshuffle=True,
    )
    datamodule = DataModule(
        batch_size=120,
        mask_hop=0.4,
        mask_prob=0.5,
        feature_mean=0,
        feature_std=1,
        num_workers=2,
        shuffle_buffer_size=360,
        train_dataset=mcc_dataset,
        sample_rate=24000,
        n_fft=2048,
        hop_length=240,
        n_mels=128,
    )
    mcc_loader = datamodule.train_dataloader()
    p = Preprocessor(mcc_loader)
    p.iterate(100_000_000)
