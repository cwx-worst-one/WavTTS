import hashlib
import random
import subprocess

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from recipes.soundstream.dataset.utils import collate_fn

HDFS_PATH = (
    "hdfs://harunava/home/byte_speech_sv/data/resso/resso_1.3m_48khz_stereo_15_03_23/"
)
# get the return from a subprocess call in a list
def get_hdfs_paths(command):
    out = subprocess.check_output(command, shell=True).decode("utf-8").split("\n")[1:-1]
    out = [f"pipe:hdfs dfs -cat {x.split(' ')[-1]}" for x in out]
    return out


class RessoDataset(IterableDataset):
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
            ECALS_URLS_sub = get_hdfs_paths(f"hdfs dfs -ls {HDFS_PATH}")[3:-1]
        elif mode == "val":
            ECALS_URLS_sub = get_hdfs_paths(f"hdfs dfs -ls {HDFS_PATH}")[-1:]
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
        print(data.keys())
        audio = data["audio.npy"]
        audio = (audio / 32768.0).astype("float32")
        import soundfile as sf

        sf.write("test.wav", audio, 48000)
        assert 1 == 2

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


if __name__ == "__main__":

    dataset = RessoDataset(
        batch_size=5, sample_buffer_size=1, mode="train", resampled=True
    )

    for i, batch in enumerate(dataset):
        print(i)
        print(batch["music_id"])
        print(batch["audio"].shape)
        if i == 10:
            break
