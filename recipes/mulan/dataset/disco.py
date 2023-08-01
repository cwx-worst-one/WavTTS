import os
import random

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import webdataset as wds
from recipes.mulan.dataset.utils import *
from samantha.dataio.webdataset.extension import IndexedWebDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DiscoDataset(IterableDataset):
    def __init__(self, mode="train", seq_len=250, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        url2idx = "/mnt/bn/audio-diffusion/data/disco/10M_url2idx.txt"
        self.dataset = (
            IndexedWebDataset(url2idx, handler=wds.warn_and_continue, **kwargs)
            .decode()
            .map(process_audio)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode, seq_len))
        )

    def _process_text(self, data):
        meta = data["__index_data__"]
        data["music_id"] = fix_hash(data["__key__"])
        if meta["video_description_youtube_language"] != "en":
            text = ""
        else:
            attr_list = ["video_title_youtube", "track_name_spotify", "search_query_youtube",
                        "video_description_youtube", "primary_artist_name_spotify"]
            # random choose 1-len(attr_list) attributes
            attr_keys = random.sample(attr_list, random.randint(1, len(attr_list)))
            attr_values = [meta.get(a) for a in attr_keys]
            attr_values = [a for a in attr_values if a]
            text = " ".join(attr_values)
        data["text"] = text.strip()
        return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":

    d = DiscoDataset()
    for i, data in enumerate(d):
        print(data.keys())
        if i > 10:
            break
