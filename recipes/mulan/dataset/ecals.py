import ast
import pickle
import random

import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

import recipes.mulan.dataset.utils as utils

ECALS_URLS = [
    f"pipe:hdfs dfs -cat {utils.HDFS_BASE}/ecals/ecals_{i:04d}.tar" for i in range(445)
]


class ECALSDataset(IterableDataset):
    def __init__(self, mode="train", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        with open("assets/ecals_gpt3_expansion.pkl", "rb") as f:
            self.ecals_gpt3 = pickle.load(f)
        self.dataset = (
            wds.WebDataset(ECALS_URLS, **kwargs)
            .decode()
            .map(utils.process_audio)
            .map(self._process_text)
            .map(utils.tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        meta = data["meta.json"]
        source = meta["data_source"]
        text = ""
        title = meta["title"]
        artist_name = meta["artist_name"]
        year = meta["year"]
        tags = ast.literal_eval(meta["tag"])
        # shuffle tags to make it more robust
        random.shuffle(tags)
        # add with 80% probability
        tag_subset = []
        for tag in tags:
            if random.random() < 0.8:
                tag_subset.append(tag)
        if int(year) != 0:
            text = f"'{title}' by {artist_name}: {year} {' '.join(tag_subset)}"
        else:
            text = f"'{title}' by {artist_name}: {' '.join(tag_subset)}"

        # add gpt3 expansion
        music_id = meta["music_id"]
        if music_id in self.ecals_gpt3:
            # use gpt3 expansion
            text = self.ecals_gpt3[music_id].replace("\n", "")
        data["text"] = text
        data["data_source"] = source
        data["music_id"] = utils.fix_hash(music_id)
        return data

    def __iter__(self):
        return iter(self.dataset)
