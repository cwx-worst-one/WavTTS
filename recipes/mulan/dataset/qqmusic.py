import json
import os
import random

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import time
import webdataset as wds
from recipes.mulan.dataset.utils import *
from samantha.dataio.webdataset.extension import IndexedWebDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# /mnt/bn/audio-diffusion/data/qmusic/
URL2IDX_Map = {
    "qq": {
        "train": "/mnt/bn/audio-diffusion/data/qmusic/url2idx.txt",
        "val": "/mnt/bn/audio-diffusion/data/qmusic/url2idx_test.txt",
        "demo": "/mnt/bn/audio-diffusion/data/qmusic/url2idx_retrival.txt",
    },
    "zh_600k": {
        "train": "/mnt/bn/audio-diffusion/data/mcc_pgc_600k/gpt_zh_url2idx.txt",
    }
}

def return_self(x):
    return x

class CMDataset(IterableDataset):
    def __init__(self, name="qq", mode="train", tokenizer_name="bert-base-chinese",**kwargs):
        if tokenizer_name =="chinese":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer_name == "multiligual":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.name = name
        url2idx = URL2IDX_Map[name][mode]
        self.dataset = (
            IndexedWebDataset(url2idx, handler=wds.warn_and_continue, **kwargs)
            .decode()
            .map(process_audio)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        if self.name == "qq":
            meta = data['__index_data__']
            text = meta["desc"]
            data["text"] = text
            data["music_id"] = int(data["__key__"])
            return data
        elif self.name == 'zh_600k':
            meta = data['__index_data__']
            text = meta["gpt_text_zh"]
            data["text"] = text
            data["music_id"] = int(data["__key__"])
            return data            
        else:
            return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    # /mnt/bn/audio-diffusion/data/qmusic/
    d = IndexedWebDataset("/mnt/bn/audio-diffusion/data/qmusic/url2idx_test.txt").decode()
    import pickle
    import numpy as np
    import soundfile as sf
    def towav(arry, mid):
        arry = (arry/32768.).astype(np.float32)
        sf.write(f"./qqmusic/{mid}.wav", arry, 24000)
    
    info = {"name":[],
            "song_id":[],
            "desc": []
            }

    items = list()
    for item in d:
        items.append(item)
        if len(items) > 200:
            break
    import random
    random.shuffle(items)
    for item in items:
        # print(item.keys())
        info['name'].append(item['__index_data__']['name'])
        info['song_id'].append(item['__index_data__']['song_id'])
        info['desc'].append(item['__index_data__']['desc'])
        towav(item['audio.npy'], item['__index_data__']['song_id'])
        if len(info['name']) > 100:
            break
    import pandas as pd
    df = pd.DataFrame.from_dict(info)
    df.to_csv("./qqmusic/qqsample.csv", index=False)
    
    # for audio_type in ["vocal", "nonvocal"]:
    #     for name in ["mcc3.5m", "mcc600k", "metacritic", "everynoise"]:
    #         if name.startswith("mcc") and audio_type == "nonvocal":
    #             continue
    #         if name == "mcc3.5m" and audio_type == "vocal":
    #             print(f"Testing {audio_type} {name}")
    #             start = time.time()
    #             d = MMEDataset(name=name, audio_type=audio_type, mode="train")
    #             start = 0
    #             for item in d:
    #                 print(start)
    #                 start += 1
    #                 print(item['text'])
    #                 # print(item['__index_data__'])
    #                 break
    #                 print("\n")
    #                 time.sleep(2)
    #             end = time.time()
