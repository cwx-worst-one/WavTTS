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

def process_audio_pq(data, segment = None):
    audio, _ = librosa.load(io.BytesIO(data["wav"]), sr = None)
    data["audio.npy"] = audio
    del data["wav"]
    data = process_audio(data, segment)
    return data

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

class SSTKDataset(IterableDataset):
    def __init__(self, name="sstk", mode="train",**kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.name = name
        dataset_id = 105
        self.dataset = (
            ParquetDataset(dataset_id, handler=wds.warn_and_continue, **kwargs)
            .map(self._process_audio)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )
    def _process_audio(self, data):
        metadata = json.loads(data["meta"])

        #"vad": {"segment": [{"start": 1840, "end": 4210}, {"start": 48720, "end": 49700}, {"start": 53050, "end": 54850}, {"start": 86020, "end": 88240}, {"start": 88530, "end": 89540}], 
        # "extra": {"voice_duration_in_seconds": 8.38, "audio_duration_in_seconds": 154.10526, "voice_proportion": 0.054378}} 
        segment = metadata.get("vad", {}).get("segment", None)
        #print("segment in _process_audio", segment)
        return process_audio_pq(data, segment)


    def _process_text(self, data):
        metadata = json.loads(data["meta"])
        if self.name == "sstk":
            title = metadata.get('title', "")
            description = metadata.get('description', "")
            keywords = metadata.get('keywords', "")
            genres = metadata.get('genres', "")
            instruments = metadata.get('instruments', "")
            text_fields = [title, description, keywords, genres, instruments]
            data["text"] = ". ".join([t for t in text_fields if t])
            data["music_id"] = fix_hash(data["__key__"])
            return data  
        else:
            return data

    def __iter__(self):
        return iter(self.dataset)


if __name__ == "__main__":
    dataset = SSTKDataset(name="sstk", mode="train")
    cnt = 0
    for item in dataset:
        cnt += 1
        print(item.keys())
        print(item["audio"])
        print(item["text"])
        if cnt == 5:
            break
        #breakpoint()
        
