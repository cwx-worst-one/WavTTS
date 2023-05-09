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
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.dataset = (
            wds.WebDataset(LONGFORM_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(self._process_aed)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        # TODO implement this logic
        # data["text"] = " ".join(tags)
        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]
        return data

    def _process_aed(self, data):
        # TODO implement this logic
        return data

    def __iter__(self):
        return iter(self.dataset)
