import hashlib
import random

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

import recipes.mulan.dataset.utils as utils
from recipes.soundstream.dataset.utils import collate_fn

ECALS_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/ecals/ecals_{i:04d}.tar" for i in range(445)
]


class ECALSDataset(IterableDataset):
    def __init__(
        self,
        batch_size,
        sample_buffer_size,
        mode="train",
        num_steps_per_val=10,
        **kwargs,
    ):
        assert mode in ["train", "val"]
        # create subsets
        if mode == "train":
            ECALS_URLS_sub = ECALS_URLS[:-1]
        elif mode == "val":
            ECALS_URLS_sub = ECALS_URLS[-1:]
        self.mode = mode

        self.dataset = (
            wds.WebDataset(ECALS_URLS_sub, **kwargs)
            .decode()
            .map(self._process_audio)
            .shuffle(sample_buffer_size)
            .batched(batch_size, collation_fn=collate_fn)
        )

        if mode == "val":
            self.dataset = self.dataset.with_epoch(num_steps_per_val)
        self.num_steps_per_val = num_steps_per_val

    def __len__(self):
        if self.mode == "val":
            return self.num_steps_per_val

    def _process_audio(self, data):
        audio = data["audio.npy"]
        audio = (audio / 32768.0).astype("float32")

        if self.mode == "train":
            music_len = 30720
            if audio.shape[-1] < music_len:
                audio = np.pad(
                    audio, ((0, 0), (0, music_len - audio.shape[-1])), "constant"
                )
                start_idx = 0
            else:
                start_idx = random.randint(0, audio.shape[-1] - music_len)
        else:
            music_len = 30720 * 8
            mid_point = int(audio.shape[-1] // 2)
            if mid_point + music_len > audio.shape[-1]:
                audio = np.pad(
                    audio,
                    ((0, 0), (0, mid_point + music_len - audio.shape[-1])),
                    "constant",
                )
                start_idx = 0
            else:
                start_idx = mid_point

        audio = torch.from_numpy(audio[..., start_idx : start_idx + music_len]).float()

        data["audio"] = audio[None,]
        music_id = data["meta.json"]["music_id"]
        data["music_id"] = data["__key__"]
        return data

    def __iter__(self):
        return iter(self.dataset)
