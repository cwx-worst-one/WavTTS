import random

import numpy as np
import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.mulan.dataset.utils as utils

LONGFORM_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/longform/longform_{i:04d}.tar"
    for i in range(1808)
]


class LongFormDataset(IterableDataset):
    def __init__(self, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(LONGFORM_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(self._aed_filter)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        source = meta["data_source"]
        text = ""
        if source == "long_form":
            # preserve only from 'Canada', 'United States'
            filtered_titles = []
            for title, country in zip(meta["item_titles"], meta["item_country_names"]):
                if country in ["Canada", "United States"]:
                    filtered_titles.append(title)
            # drop the sample if none of the texts from us, ca
            if not filtered_titles:
                return None
            text = random.choice(filtered_titles)
            text = utils.remove_emoji(text)
            # remove hash tag to ,
            text = text.replace("#", ",")
            country_names = meta["item_country_names"]
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
