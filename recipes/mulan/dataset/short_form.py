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
    def __init__(self, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.dataset = (
            wds.WebDataset(SHORTFORM_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(self._process_aed)
            # .map(utils.tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        # TODO text logic
        # # shuffle tags to make it more robust
        # random.shuffle(tags)
        # data["text"] = " ".join(tags)
        data["data_source"] = meta["data_source"]
        data["music_id"] = meta["music_id"]
        return data

    def _process_aed(self, data):
        # TODO aed logic
        return data

    def __iter__(self):
        return iter(self.dataset)
