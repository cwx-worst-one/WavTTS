import random

import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.mulan.dataset.utils as utils

TTQUERY_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/tt_query_700k/tt_query_{i:04d}.tar"
    for i in range(663)
]


class TTQueryDataset(IterableDataset):
    def __init__(self, mode="train", seq_len=250, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.dataset = (
            wds.WebDataset(TTQUERY_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode, seq_len))
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        querys = meta["querys"]
        # shuffle querys to make it more robust
        random.shuffle(querys)
        data["text"] = " ".join(querys)
        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]
        return data

    def __iter__(self):
        return iter(self.dataset)
