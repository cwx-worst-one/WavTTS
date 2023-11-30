import json
import os
import random

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import time
import webdataset as wds
from recipes.mulan.dataset.utils import *
from samantha.dataio.webdataset.extension import IndexedWebDataset
import json
import librosa
import io
from samantha.dataio.parquet.parquet_dataset import ParquetDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# /mnt/bn/audio-diffusion/data/qmusic/
def pattern_apply(text_content):
    try:
        pattern = r'"text":\s*"([^"]+)"'
        match_res = re.search(pattern, text_content)
        if match_res:
            extracted_text = match_res.group(1)
        else:
            return ""
    except:
        print("wrong text!")
        return ""
    return extracted_text

# for item in d:
#     metadata = json.loads(item["meta"])
#     print(metadata)
#     print("gpt_text:", gpt_text)
#     print("artist_name:", artist_name_res)
#     print("track_name:", track_name_res)
#     print("album_name:", album_name_res)
# print("Total items:", cnt)


def return_self(x):
    return x

class CMDataset(IterableDataset):
    def __init__(self, name="en_600k", mode="train",**kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.name = name
        dataset_id = 93
        self.dataset = (
            ParquetDataset(dataset_id, handler=wds.warn_and_continue, **kwargs)
            .map(process_audio_pq)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        metadata = json.loads(data["meta"])
        if self.name == "en_600k":
            try:
                gpt_text = metadata.get('lyrics', "")
                gpt_text = gpt_text ["result"][0]["text"]
            except:
                gpt_text = ""
            # artist_name = metadata.get('artist_name', "")
            # track_name = metadata.get('song_title', "")
            # artist_name_res = pattern_apply(artist_name)
            # track_name_res = pattern_apply(track_name)
            data["text"] = ". ".join([gpt_text])
            data["music_id"] = fix_hash(data["__key__"])
            return data  
        else:
            return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    dataset = CMDataset(name="en_600k", mode="train")
    for item in dataset:
        print(item.keys())
        print(item["audio"])
        print(item["text"])
        break
        #breakpoint()
