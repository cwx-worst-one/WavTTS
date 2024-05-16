import os

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader

# from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import AmplitudeToDB
from webdataset.pipeline import DataPipeline

from recipes.musiclm.datasets.mcc import MCC40MDataset
from samantha.dataio.dataset import MultiIterableDataset
from samantha.transforms.audio import MelSpectrogram


class Preprocessor:
    def __init__(self, dataloader, melspec, output_dir):
        super(Preprocessor, self).__init__()
        # initialize
        self.total_samples = 0.0
        self.mean = torch.zeros((128))
        self.std = torch.zeros((128))

        # prepare data loader
        self.dataloader = dataloader
        self.melspec = melspec
        self.amp2db = AmplitudeToDB()
        self.output_dir = output_dir

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
            if isinstance(x, list):
                x = x[0]
            x = x.squeeze(1)
            feature = self.amp2db(
                self.melspec(x)[:, :, :-1].transpose(1, 2).reshape((-1, 128))
            )
            self.update(feature)
            i += 1
            print("iter: %d" % i)
            print("[EST] mean:", self.mean)
            print("[EST] std:", torch.sqrt(self.std / (self.total_samples - 1)))
            if i % 100 == 0:
                cmvn = torch.stack(
                    [self.mean, torch.sqrt(self.std / (self.total_samples - 1))], dim=0
                ).numpy()
                np.save(f"{self.output_dir}/mcc_{i:>04}.npy", cmvn)
                print(cmvn)
            if i >= num_iter:
                return


if __name__ == "__main__":
    sample_rate = 24000
    duration = 30
    num_workers = 0
    shuffle_buffer_size = 100
    batch_size = 40
    mcc40m = MCC40MDataset(
        url2index="/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores/npy_url2idx.txt",
        audio_key="audio.npy",
        sample_rate=sample_rate,
        duration=duration,
        normalize_audio=True,
        max_num_crops=3,
        min_volume_threshold=0.05,
        loudness_ratio_threshold=0.2,
        resampled=True,
        shardshuffle=True,
    )
    mcc60m = MCC40MDataset(
        url2index="/mnt/bn/audio-diffusion/data/vocal_mcc/mcc_60m_url2index_final.tsv",
        audio_key="mp3",
        sample_rate=sample_rate,
        duration=duration,
        normalize_audio=True,
        max_num_crops=3,
        min_volume_threshold=0.05,
        loudness_ratio_threshold=0.2,
        resampled=True,
        shardshuffle=True,
        aed_filtered=False,
        avoid_vocal=False,
    )
    dataset = MultiIterableDataset(datasets=[mcc40m, mcc60m], weights=[1, 1])
    melspec = MelSpectrogram(
        sample_rate=sample_rate, n_mels=128, n_fft=2048, hop_length=240
    )
    dataset_batched = DataPipeline(
        dataset,
        wds.shuffle(shuffle_buffer_size),
        wds.to_tuple("audio"),
        wds.batched(batch_size),
    )
    dataloader = DataLoader(dataset_batched, batch_size=None, num_workers=num_workers)
    output_dir = "/mnt/bn/zongyu-lq/features/bestrq/cmvn"
    os.makedirs(output_dir, exist_ok=True)
    p = Preprocessor(dataloader, melspec, output_dir)
    p.iterate(100_000_000)
