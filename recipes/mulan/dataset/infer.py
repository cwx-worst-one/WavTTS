import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from samantha.utils.hdfs_helper import hopen


class MuLanInferDataset(IterableDataset):
    def __init__(
        self, meta_path, batch_size=128, window_size=240_000, hop_size=120_000
    ):
        self.meta_path = meta_path
        self.window_size = window_size
        self.hop_size = hop_size
        self.meta = self.load_meta()
        self.pipeline = wds.DataPipeline(
            [self.meta],
            wds.split_by_node,
            wds.split_by_worker,
            wds.map(self._process_audio),
            wds.batched(batch_size, collation_fn=self._collate_fn),
        )
        self.num_samples = len(self.meta)

    def __iter__(self):
        return iter(self.pipeline)

    def _process_audio(self, data):
        # Read the music
        filepath = data[1]
        music = (np.load(filepath) / 32768.0).astype(np.float32)
        # If shape is (1,T), reshape to (T,)
        if len(music.shape) == 2:
            music = music[0]
        # Case 1: music is shorter than 30s (sr = 24000), drop it
        if music.shape[0] < 30 * 24_000:
            return None
        # Case 2: music is exactly 30s, split to 3*10s parts
        if music.shape[0] == 30 * 24_000:
            music_windows = []
            for i in range(3):
                start = int(i * 10 * 24_000)
                end = int((i + 1) * 10 * 24_000)
                music_window = music[start:end]
                music_window = torch.from_numpy(music_window)
                music_windows.append(music_window)
            return [music_windows]
        # Case 3: music is longer than 30s, first split to 3 parts
        # then split first 10s from each part
        music_len = music.shape[0]
        music_windows = []
        for i in range(3):
            start = int(i * music_len / 3)
            end = int((i + 1) * music_len / 3)
            music_part = music[start:end]
            music_window = music_part[: self.window_size]
            if music_window.shape[0] < self.window_size:
                music_window = np.pad(
                    music_window, (0, self.window_size - music_window.shape[0])
                )
            music_window = torch.from_numpy(music_window)
            music_windows.append(music_window)
        return [music_windows]

    def load_meta(self):
        meta = []
        with hopen(self.meta_path, "r") as f:
            for i, line in enumerate(f):
                filepath = line.strip().split("|")[0]
                meta.append((i, filepath))
        return meta

    def _collate_fn(self, batch):
        music_windows = []
        for sample in batch:
            music_windows.append(sample[0])
        music_windows = torch.stack(music_windows)
        return {"audio": music_windows}


if __name__ == "__main__":
    x = MuLanInferDataset(
        "/mlx_devbox/users/jingsonggao/playground/samantha/audio.txt"
    )
    loader = wds.WebLoader(x, batch_size=None, num_workers=4)
    for a in loader:
        pass
