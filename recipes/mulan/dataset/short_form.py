import random

import numpy as np
import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.mulan.dataset.utils as utils

SHORTFORM_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/shortform/shortform_{i:04d}.tar"
    for i in range(10000)
] + [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/shortform/shortform_{i:05d}.tar"
    for i in range(10000, 21367)
]


class ShortFormDataset(IterableDataset):
    def __init__(self, mode="train", seq_len=250, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(SHORTFORM_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(self._aed_filter)
            .map(utils.tokenize_text(self.tokenizer, mode, seq_len))
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        source = meta["data_source"]

        text = ""
        if source == "short_form":
            target_ks = ["genre", "mood", "theme", "language"]
            random.shuffle(target_ks)  # randomize the order of the tags
            for field in meta:
                if field == "meta_song_title":
                    text += f"'{meta[field]}' "
                elif field == "meta_song_author":
                    text += f"by {meta[field]} "
                elif field == "meta_song_album_name":
                    text += f"from {meta[field]}: "
                elif field in target_ks:
                    if meta[field] != "":
                        text += meta[field] + " "
            text = text[:-1]
        else:
            # drop data from other sources
            return None

        # drop data without text
        if text == "":
            return None
        data["text"] = text
        data["data_source"] = source
        data["music_id"] = meta["music_id"]
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
