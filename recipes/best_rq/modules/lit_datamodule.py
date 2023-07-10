from functools import partial

import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchaudio_augmentations import Compose
from webdataset.pipeline import DataPipeline


class NormalizeFeature(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeFeature, self).__init__()
        self.mean = mean.unsqueeze(0).unsqueeze(-1)
        self.std = std.unsqueeze(0).unsqueeze(-1)

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x


def masking(x, sample_rate, mask_hop, hop_length, mask_prob):
    mx = x.clone()
    b, t = mx.shape
    len_masking_raw = int(sample_rate * mask_hop)
    len_masking_token = int(sample_rate / hop_length / 2 / 2 * mask_hop)

    # get random mask indices
    start_indices = torch.rand(b, t // len_masking_raw) < mask_prob
    time_domain_masked_indices = torch.nonzero(
        start_indices.repeat_interleave(len_masking_raw, dim=1)
    )
    token_domain_masked_indices = torch.nonzero(
        start_indices.repeat_interleave(len_masking_token, dim=1)
    )

    # mask with random values
    masking_noise = torch.randn(len(time_domain_masked_indices)) * 0.1  # 0 mean 0.1 std
    mx[tuple(time_domain_masked_indices.t())] = masking_noise
    return mx, token_domain_masked_indices


def collation_fn(batch, feature_fn, sample_rate, mask_hop, hop_length, mask_prob):
    res = {"audio": []}
    for item in batch:
        res["audio"].append(item["audio"].squeeze(0))
    res["audio"] = torch.stack(res["audio"], dim=0)
    res["masked_audio"], res["token_domain_masked_indices"] = masking(
        res["audio"], sample_rate, mask_hop, hop_length, mask_prob
    )
    res["feature"] = feature_fn(res["audio"])
    res["masked_feature"] = feature_fn(res["masked_audio"])
    return res


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        mask_hop,
        mask_prob,
        feature_mean,
        feature_std,
        num_workers: int = 8,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 100,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        sample_rate: int = 24000,
        n_fft: int = 2048,
        hop_length: int = 240,
        n_mels: int = 128,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        if isinstance(feature_mean, str):
            self.feature_mean = torch.load(feature_mean)
        else:
            self.feature_mean = torch.tensor(feature_mean)
        if isinstance(feature_std, str):
            self.feature_std = torch.load(feature_std)
        else:
            self.feature_std = torch.tensor(feature_std)

        self.feature_fn = Compose(
            [
                MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                ),
                AmplitudeToDB(),
                NormalizeFeature(self.feature_mean, self.feature_std),
            ]
        )

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset,
            wds.shuffle(self.shuffle_buffer_size),
            wds.batched(
                self.batch_size,
                collation_fn=partial(
                    collation_fn,
                    feature_fn=self.feature_fn,
                    sample_rate=self.sample_rate,
                    mask_hop=self.mask_hop,
                    hop_length=self.hop_length,
                    mask_prob=self.mask_prob,
                ),
            ),
        )
        return DataLoader(
            train_dataset_batched, batch_size=None, num_workers=self.num_workers
        )

    def val_dataloader(self):
        validation_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.batched(
                self.batch_size,
                collation_fn=partial(
                    collation_fn,
                    feature_fn=self.feature_fn,
                    sample_rate=self.sample_rate,
                    mask_hop=self.mask_hop,
                    hop_length=self.hop_length,
                    mask_prob=self.mask_prob,
                ),
            ),
        )
        return DataLoader(
            validation_dataset_batched, batch_size=None, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        predict_dataset_batched = DataPipeline(
            self.predict_dataset,
            wds.shuffle(self.shuffle_buffer_size),
            wds.batched(
                self.batch_size,
                collation_fn=partial(
                    collation_fn,
                    feature_fn=self.feature_fn,
                    sample_rate=self.sample_rate,
                    mask_hop=self.mask_hop,
                    hop_length=self.hop_length,
                    mask_prob=self.mask_prob,
                ),
            ),
        )
        return DataLoader(
            predict_dataset_batched, batch_size=None, num_workers=self.num_workers
        )
