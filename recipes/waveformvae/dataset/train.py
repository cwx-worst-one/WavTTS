from torch.utils.data import Dataset
import torch
import numpy as np
import random
from tqdm import tqdm
from recipes.waveformvae.utils.audio_utils import wav_load
from samantha.dataio.parquet import ParquetDataset

import webdataset as wds
from torch.utils.data import IterableDataset


class TrainDataset(Dataset):
    def __init__(self, paths: list, segment_length: int, sampling_rate=24000):
        super(TrainDataset, self).__init__()
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.metas = self.get_metadata(paths)
        self.dataset_length = len(self.metas)

    def get_metadata(self, paths):
        metas = []
        cnt = 0
        for path in paths:
            with open(path, "r") as f:
                lines = f.readlines()
                for i in tqdm(range(len(lines))):
                    l = lines[i]
                    cnt += 1
                    wav_fp, lab_fp, wav_len, sr, phone_len = l.strip().split('|')
                    if (int(wav_len) / int(sr)) > (self.segment_length / 24000):
                        metas.append([wav_fp, wav_len, sr])
        return metas

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        wav_path, wav_len, sr = self.metas[index]
        wav = wav_load(path=wav_path, sr=self.sampling_rate, mono=True)
        wav = wav / max(0.01, np.max(np.abs(wav)))
        if wav.shape[0] - self.segment_length <= 0:
            return torch.zeros(self.segment_length)
        else:
            s = random.randint(0, wav.shape[0] - self.segment_length)
            # return torch.from_numpy(wav[s:s + self.segment_length])
            return torch.FloatTensor(wav[s:s + self.segment_length])


def collate_fn(batch):
    return torch.stack(batch).unsqueeze(1)


class WDSDataset(IterableDataset):
    """waveform vae dataset

    Args:
        hp (dict): hparams for transform
        data_id (int): dataset id, default is [this](https://bigspeech.bytedance.net/data/collection/109/basic?version=2)
        meta_paths (list): set meta_path
    """
    def __init__(self, hp, data_id=609, meta_paths=None, audio_key='wav'): 
        self.mel_win = hp.get('mel_win', 20)
        self.hop_size = hp.get('hop_size', 600)
        self.segment_size = self.mel_win * self.hop_size
        self.sample_rate = hp.get('sample_rate', 24000)
        self.audio_key = audio_key
        self.hp = hp

        if meta_paths:
            urls = []
            print(meta_paths)
            for meta_path in meta_paths:
                if 'tar_list' in meta_path:
                    with open(meta_path, 'r') as f:
                        lines = [l.strip() for l in f]
                    for l in lines:
                        urls.append("pipe: hdfs dfs -cat {}".format(l))
                else:
                    urls.append("pipe: hdfs dfs -cat {}".format(meta_path))
                for url in urls:
                    print(url)
        else:
            urls = None
        
        self.dataset = (
            ParquetDataset(data_id=data_id, data_urls=urls, resampled=True)
            .map(self._normalize)  # 应用自定义处理函数
            .shuffle(2000)  # buffer size
            .with_length(5_000_000)
        )

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    # def _normalize(self, item):
    #     wav = item['npy']
    #     if wav.dtype == np.int16:
    #         wav = wav / 32768.0
    #     elif wav.dtype == np.int32:
    #         wav = wav / 2_147_483_648.0
    #     elif wav.dtype in [np.float32, np.float64]:
    #         wav = wav
    #     else:
    #         raise Exception("Not support data type: {}".format(wav.dtype))
    #     # 单通道
    #     if len(wav.shape) >= 2:
    #         wav = wav[0]
    #     scale = max(0.001, np.max(np.abs(wav)))
    #     wav = wav.astype(np.float32)
    #     # random slice
    #     wav_len = wav.shape[0]
    #     if wav_len < self.hop_size * 32:
    #         rand_slice = np.zeros(shape=[self.segment_size,], dtype='float32')
    #     elif wav_len < self.segment_size:
    #         wav = np.pad(wav, (0, self.segment_size - wav_len))
    #         rand_slice = wav
    #     else:
    #         beg = np.random.randint(low=0, high=wav_len - self.segment_size + 1)
    #         rand_slice = wav[beg : beg + self.segment_size]
    #     # 音量归一化
    #     rand_slice = rand_slice / scale * 0.95

    #     rand_slice = torch.from_numpy(rand_slice).float().unsqueeze(0)
    #     # 历史遗留问题，不解决
    #     # return 1, rand_slice, rand_slice, 1

    #     # align to TrainDataset
    #     return rand_slice

    def _normalize(self, item):
        if "npy" in item:
            wav = item['npy']
        else:
            wav = np.frombuffer(item[self.audio_key ][44:], dtype=np.int16)
        if wav.dtype == np.int16:
            wav = wav / 32768.0
        elif wav.dtype == np.int32:
            wav = wav / 2_147_483_648.0
        elif wav.dtype in [np.float32, np.float64]:
            wav = wav
        else:
            raise Exception("Not support data type: {}".format(wav.dtype))
        # 单通道
        if len(wav.shape) >= 2:
            wav = wav[0]
        scale = max(0.001, np.max(np.abs(wav)))
        wav = wav.astype(np.float32)
        wav = wav / scale * 0.95

        # random slice
        if wav.shape[0] - self.segment_size <= 0:
            return torch.zeros(self.segment_size)
        else:
            s = random.randint(0, wav.shape[0] - self.segment_size)
            return torch.from_numpy(wav[s:s + self.segment_size])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    hp = dict(
        mel_win=40,
        hop_size=600,
        sample_rate=24000,
    )
    dataset = WDSDataset(hp, data_id=609) 
    data = next(iter(dataset))
    print(type(data))

