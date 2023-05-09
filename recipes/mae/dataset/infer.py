import numpy as np
import torch
from torch.utils.data import Dataset


class MuLanInferDataset(Dataset):
    def __init__(self, meta_path, window_size=240_000, hop_size=120_000):
        self.meta_path = meta_path
        self.window_size = window_size
        self.hop_size = hop_size
        self.meta = self.load_meta()
        self.num_samples = len(self.meta)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Read the music
        music_id, filepath = self.meta[index]
        music = (np.load(filepath) / 32768.0).astype(np.float32)
        music_len = music.shape[0]
        # Pad to factor of window size
        music = np.pad(music, (0, self.window_size - (music_len % self.window_size)))
        music_len = music.shape[0]
        # Slice the music into windows
        music_ids = []
        music_windows = []
        for i in range(0, music_len, self.hop_size):
            # Pad to window size
            if i + self.window_size > music_len:
                music_window = np.pad(
                    music[i:], (0, self.window_size - (music_len - i))
                )
            else:
                music_window = music[i : i + self.window_size]
            # convert to torch tensor
            music_window = torch.from_numpy(music_window)
            # Append to list
            music_ids.append(music_id)
            music_windows.append(music_window)
        return [music_ids, music_windows]

    def load_meta(self):
        meta = []
        with open(self.meta_path, "r") as f:
            for line in f:
                filepath, _, sample_rate = line.strip().split("|")
                assert sample_rate == "24000"
                music_id = filepath.split("/")[-1].split(".")[0]
                meta.append((music_id, filepath))
        return meta
