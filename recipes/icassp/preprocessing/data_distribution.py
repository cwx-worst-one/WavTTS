import sys
import json
import numpy as np
from einops import rearrange
from webdataset import WebLoader, WebDataset

from recipes.audio_diffusion.modules.datasets.webdataset_ext import MultiWebDataset
from recipes.audio_diffusion.conf.parse_shard_list import get_train_shards
from recipes.audio_diffusion.modules.preprocess import WebDatasetBufferPreprocessor
from recipes.icassp.modules.data_transform import FMAAudioTransformsForStat


# Welford's method for iterative updating
class Preprocessor:
    def __init__(
        self, 
        batch_size=16, 
        num_workers=4, 
        hop_length=240, 
        n_fft=2048, 
        total_samples=0,
        ):
        super(Preprocessor, self).__init__()

        # initialize
        self.stat = {
            "spec_cnt": 0,
            "spec_mean": 0.0,
            "spec_std": 0.0,
            "melspec_cnt": 0,
            "melspec_mean": 0.0,
            "melspec_std": 0.0,
            "cqt_cnt": 0,
            "cqt_mean": 0.0,
            "cqt_std": 0.0,
            "mfcc_cnt": 0,
            "mfcc_mean": 0.0,
            "mfcc_std": 0.0,
            "chromagram_cnt": 0,
            "chromagram_mean": 0.0,
            "chromagram_std": 0.0,
        }

        web_dataset = WebDataset("/mnt/bn/transcription/fma/incoming/sami_ai/datasets/fma_24kHz_{0..9}/train/shards-{0000..0070}.tar")
        data_transform = FMAAudioTransformsForStat(30, 24000, hop_length, n_fft)
        buffer_processor = WebDatasetBufferPreprocessor(sample_rate=24000, transforms=data_transform)
        # train_dataset = multi_dataset.shuffle(100).compose(buffer_processor.train_buffer_preprocessor).to_tuple("mel_spec chromatic_spec mfcc chromagram").batched(batch_size)
        train_dataset = web_dataset.shuffle(100).decode().compose(buffer_processor.train_buffer_preprocessor).to_tuple("spec mel_spec cqt mfcc chromagram").batched(batch_size)
        self.loader = WebLoader(train_dataset, num_workers=num_workers)

    def update(self, x, feature_name):
        # count total samples
        num_samples = len(x.flatten())
        self.stat["%s_cnt" % feature_name] += num_samples
        
        # update mean
        new_mean = x.mean().numpy()
        delta = new_mean - self.stat["%s_mean" % feature_name]
        self.stat["%s_mean" % feature_name] += delta * num_samples / self.stat["%s_cnt" % feature_name]

        # update std
        new_std = ((x - new_mean) * (x - self.stat["%s_mean" % feature_name])).sum().numpy()
        self.stat["%s_std" % feature_name] += new_std

    def iterate(self, num_iter):
        i = 0
        while i < num_iter:
            for features in self.loader:
                for feature, feature_name in zip(features, ["spec", "melspec", "cqt", "mfcc", "chromagram"]):
                    self.update(feature, feature_name)
                i += 1
                print("iter: %d" % i)
                if i % 100 == 0:
                    self.save_features()

    def save_features(self):
        stat = {k: v for k, v in self.stat.items()}
        for k in stat.keys():
            if k[-3:] == "std":
                stat[k] = np.sqrt(stat[k] / (stat[k[:-3] + "cnt"] - 1))
        print(stat)
        with open("/mnt/bn/audio-diffusion/pretrained_models/musicfm/fma_classic_stats.json", "w") as file:
            json.dump(stat, file)


if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iter = int(sys.argv[3])
    hop_length = 240
    n_fft = 2048
    p = Preprocessor(batch_size, num_workers, hop_length, n_fft)
    p.iterate(num_iter)
