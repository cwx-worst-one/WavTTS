import random

import numpy as np
import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.audio_lm.datasets.utils as utils

PLAYLIST_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/playlist_v5/playlist_{i:04d}.tar"
    for i in range(5)  # 2598
]


class PlaylistDataset(IterableDataset):
    def __init__(self, duration=10, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(PLAYLIST_URLS, **kwargs)
            .decode()
            .map(utils.process_audio(duration))
            .map(self._process_text)
            .map(self._aed_filter)
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

    def _aed_filter(self, data):
        aed = data["aed.npy"]
        # class label:
        #  https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/metadata/class_labels_indices.csv  # noqa
        aed_music_related = np.concatenate([aed[27:30], aed[32:38], aed[137:282]])
        if np.max(aed_music_related) > 0.4:
            return data
        else:
            return None

    def __iter__(self):
        return iter(self.dataset)
