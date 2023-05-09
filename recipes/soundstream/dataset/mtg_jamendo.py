import os
import random

import numpy as np
import torch
import torchaudio

BASE_DIR = "/mnt/bd/sami-weitsung-2/mss_datasets/mtg_jamendo_raw/"


def get_file_list(folder, ext=".mp3"):
    fs = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(ext):
                fs.append(os.path.join(root, f))
    return fs


# torch dataset
class MTGJamendoDataset(torch.utils.data.Dataset):
    def __init__(self, num_steps=None, mode="train"):

        assert mode in ["train", "val"]

        fs = get_file_list(BASE_DIR)
        if mode == "train":
            fs = fs[:-50]
        else:
            fs = fs[-50:]
        self.fs = fs
        self.mode = mode
        if num_steps is None:
            self.len = len(fs)
        else:
            self.len = num_steps

    def __getitem__(self, index):
        # random sample an index from fs if training
        if self.mode == "train":
            index = random.randint(0, len(self.fs) - 1)
        f = self.fs[index]
        audio, _ = torchaudio.load(f)
        audio = audio.numpy()
        # to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0, keepdims=True)

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
            music_len = 30720 * 8 * 2
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

        return {"audio": audio, "music_id": f.split("/")[-1].split(".")[0]}

    def __len__(self):
        return self.len


def worker_init_fn(worker_id):
    np.random.seed()
    random.seed()


if __name__ == "__main__":
    dataset = MTGJamendoDataset()

    loader = torch.utils.data.DataLoader(dataset, batch_size=5, num_workers=4)
    for d in loader:
        print(d["audio"].shape)
        print(d["music_id"])
        break
