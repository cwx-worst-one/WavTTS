import hashlib
import random

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

import recipes.mulan.dataset.utils as utils
from recipes.soundstream.dataset.utils import collate_fn, fix_hash

DATASET_URLS = [
    f"pipe:hdfs dfs -cat /home/byte_speech_sv/jingsong.gao/mcc_n2m/mcc_n2m_{i:04d}.tar"
    for i in range(6379)
]

class MCCDataset(IterableDataset):
    def __init__(
        self,
        music_len=30720,
        mode="train",
        **kwargs,
    ):
        assert mode in ["train", "val"]

        self.music_len = music_len

        # create subsets
        if mode == "train":
            DATASET_URLS_sub = DATASET_URLS[:-1]
        elif mode == "val":
            DATASET_URLS_sub = DATASET_URLS[-1:]
        self.mode = mode

        self.dataset = (
            wds.WebDataset(DATASET_URLS_sub, **kwargs)
            .decode()
            .map(self._process_audio)
        )

    def __len__(self):
        if self.mode == "val":
            return self.num_steps_per_val

    def _process_audio(self, data):
        audio = data["audio.npy"]
        audio = (audio / 32768.0).astype("float32")

        if self.mode == "train":
            if audio.shape[-1] < self.music_len:
                audio = np.pad(
                    audio, ((0, 0), (0, self.music_len - audio.shape[-1])), "constant"
                )
                start_idx = 0
            else:
                start_idx = random.randint(0, audio.shape[-1] - self.music_len)
        else:
            mid_point = int(audio.shape[-1] // 2)
            if mid_point + self.music_len > audio.shape[-1]:
                audio = np.pad(
                    audio,
                    ((0, 0), (0, mid_point + self.music_len - audio.shape[-1])),
                    "constant",
                )
                start_idx = 0
            else:
                start_idx = mid_point

        audio = torch.from_numpy(audio[..., start_idx : start_idx + self.music_len]).float()

        data["audio"] = audio[None,]
        data["music_id"] = fix_hash(data["__key__"])
        return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    dataset = MCCDataset( 
        batch_size=2,
        sample_buffer_size=2,
        music_len=15000,
        mode="train",
        num_steps_per_val=10
    )

    for i, batch in enumerate(dataset):
        print(i)
        print(batch['music_id'])
        print(batch['audio'].shape)
        if i == 10:
            break