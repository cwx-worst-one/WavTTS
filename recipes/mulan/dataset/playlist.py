import random

import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.mulan.dataset.utils as utils

PLAYLIST_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/playlist_v5/playlist_{i:04d}.tar"
    for i in range(2598)
]


class PlaylistDataset(IterableDataset):
    def __init__(self, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.dataset = (
            wds.WebDataset(PLAYLIST_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(self._process_aed)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        descs = meta["descriptions"]
        # random choose one description
        data["text"] = random.choice(descs)
        data["data_source"] = meta["data_source"]
        try:
            data["music_id"] = int(meta["music_id"])
        except Exception:
            return None
        return data

    def _process_aed(self, data):
        # TODO add aed logic
        # aed_result = data["aed.npy"]
        return data

    def __iter__(self):
        return iter(self.dataset)
