import sys
from collections import Counter

from webdataset import WebLoader

from recipes.audio_diffusion.conf.parse_shard_list import get_train_shards
from recipes.audio_diffusion.modules.datasets.webdataset_ext import MultiWebDataset
from recipes.audio_diffusion.modules.preprocess import WebDatasetBufferPreprocessor
from recipes.musicfm.models.classic import ClassicMusicFM
from recipes.musicfm.modules.data_transform import AudioTransforms


class Monitor:
    def __init__(self, batch_size=16, num_workers=4):
        super(Monitor, self).__init__()

        # prepare data loader
        urls = get_train_shards(
            ["/mnt/bn/audio-diffusion/data/resso_original_shard_list.txt"]
        )
        multi_dataset = MultiWebDataset(urls, [1.0])
        data_transform = AudioTransforms(30, 24000)
        buffer_processor = WebDatasetBufferPreprocessor(
            sample_rate=24000, transforms=data_transform
        )
        train_dataset = (
            multi_dataset.shuffle(100)
            .compose(buffer_processor.train_buffer_preprocessor)
            .to_tuple("audio")
            .batched(batch_size)
        )
        self.loader = WebLoader(train_dataset, num_workers=num_workers)
        self.model = ClassicMusicFM(
            stat_path="/mnt/bn/audio-diffusion/pretrained_models/musicfm/global_stats.json"
        )

    def iterate(self, num_iter):
        i = 0
        # all_tokens = {k: [] for k in ["mel", "chromatic", "mfcc", "chromagram"]}
        unique_tokens = {k: [] for k in ["mel", "chromatic", "mfcc", "chromagram"]}
        while i < num_iter:
            for wav in self.loader:
                wav = wav[0][0].squeeze(1)
                tokens = self.model.get_targets(wav)
                for key in tokens.keys():
                    ts = tokens[key].flatten().tolist()
                    unique_tokens[key] = list(set(unique_tokens[key] + ts))
                    # all_tokens[key] += ts
                    if i % 10 == 0:
                        print(
                            "[%d] %.4f utilized for [%s]"
                            % (i, len(unique_tokens[key]) / 8192, key)
                        )
                        # print(Counter(all_tokens[key]))
                i += 1


if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iter = int(sys.argv[3])
    p = Monitor(batch_size, num_workers)
    p.iterate(num_iter)
