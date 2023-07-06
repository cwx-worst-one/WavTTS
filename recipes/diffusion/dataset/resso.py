import io
import hashlib
import random

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

import recipes.mulan.dataset.utils as utils
from recipes.soundstream.dataset.utils import collate_fn, fix_hash

from pydub import AudioSegment

# read line from txt with delimiter '\n'
with open('recipes/diffusion/assets/resso_1.3m_mss_shard_list.txt', 'r') as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].replace('\n', '')

DATASET_URLS = [
    f"pipe:hdfs dfs -cat {l}"
    for l in lines
]
SAMPLE_RATE = 24000

def normalize_audio_to_float32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor

    info = torch.iinfo(tensor.dtype)
    abs_max = 2 ** (info.bits - 1)
    return tensor.float() / abs_max

def read_mp3(mp3_byte):
    audio = AudioSegment.from_file(mp3_byte, format="mp3")
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    wav = np.asarray(audio.get_array_of_samples())
    return wav


class RessoDataset(IterableDataset):
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
            DATASET_URLS_sub = DATASET_URLS[2:-1]
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
        audio = read_mp3(io.BytesIO(data["mss_acc"]))
        audio = torch.from_numpy(audio)
        audio = normalize_audio_to_float32(audio).float()

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

        audio = audio[..., start_idx : start_idx + self.music_len]

        data["audio"] = audio[None,]
        data["music_id"] = fix_hash(data["__key__"])
        return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    dataset = RessoDataset( 
        music_len=240000,
        mode="train",
    )

    for i, batch in enumerate(dataset):
        print(i)
        print(batch['music_id'])
        print(batch['audio'].shape)
        import soundfile as sf
        sf.write(f'test/{i}.wav', batch['audio'].numpy().T, 24000)
        if i == 10:
            break