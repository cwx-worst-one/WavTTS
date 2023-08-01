import random

import numpy as np
import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.mulan.dataset.utils as utils

PLAYLIST_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/playlist_v5/playlist_{i:04d}.tar"
    for i in range(2598)
]


class PlaylistDataset(IterableDataset):
    def __init__(self, mode="train", seq_len=250, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(PLAYLIST_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(self._aed_filter)
            .map(utils.tokenize_text(self.tokenizer, mode, seq_len))
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

    def _aed_filter(self, data):
        aed = data["aed.npy"]
        # class label: https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/metadata/class_labels_indices.csv
        aed_music_related = np.concatenate([aed[27:30], aed[32:38], aed[137:282]])
        if np.max(aed_music_related) > 0.4:
            return data
        else:
            return None

    def __iter__(self):
        return iter(self.dataset)
