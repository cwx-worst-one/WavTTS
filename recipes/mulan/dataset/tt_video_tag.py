import json
import os
import random
import pandas as pd

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import time
import webdataset as wds
from recipes.mulan.dataset.utils import *
from samantha.dataio.webdataset.extension import IndexedWebDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# /mnt/bn/audio-diffusion/data/qmusic/
URL2IDX_Map = {
    "ttv": {
        "train": "/mnt/bn/audio-diffusion/data/tt_video_tag/tt_video_tag_url2index.txt",
    },

}

map_df = pd.read_csv("/mnt/bn/audio-diffusion/data/tt_video_tag/tt_video_tag_map.csv")
idlst = map_df['label_id'].unique().tolist()
TAG_MAP= dict()
for idx in idlst:
    namelst = map_df[map_df['label_id']==idx]['label_name'].unique().tolist()
    assert len(namelst) == 1
    TAG_MAP[str(idx)] = namelst[0]


def return_self(x):
    return x

class TTVDataset(IterableDataset):
    def __init__(self, name="ttv", mode="train", tokenizer_name="bert", seq_len=500, tok_path=None, **kwargs):
        if tokenizer_name =="bert":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        elif tokenizer_name == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.name = name
        url2idx = URL2IDX_Map[name][mode]
        self.dataset = (
            IndexedWebDataset(url2idx, handler=wds.warn_and_continue, **kwargs)
            .decode()
            .map(process_audio)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode, seq_len))
        )

    def _process_text(self, data):
        if self.name == "ttv":
            meta = data['__index_data__']
            tag_keys = ['lab_diversity4_tier1_tag', 'lab_diversity4_tier2_tag', 'lab_diversity4_tier3_tag',
                        'item_mt_diversity_tier3_tags', 'lab_diversity_tier4']
            
            idxs = list()
            for tagk in tag_keys:
                idxs += meta[tagk]
            text = ",".join([TAG_MAP.get(tagid, " ") for tagid in idxs])
            # TODO use llama v2 to expand query?
            data["text"] = text.strip()
            data["music_id"] = int(data["__key__"])
            return data           
        else:
            return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    # /mnt/bn/audio-diffusion/data/qmusic/
    d = IndexedWebDataset("/mnt/bn/audio-diffusion/data/tt_video_tag/tt_video_tag_url2index.txt").decode()
    import time   

    name = "ttv"
    mode = "train"
    d = TTVDataset(name=name, mode="train")
    for item in d:
        print(item['text'])
        break
    # start = 0
    # for item in d:
    #     print(start)
    #     start += 1
    #     print(item['text'])
    #     # print(item['__index_data__'])
    #     break
    # end = time.time()
