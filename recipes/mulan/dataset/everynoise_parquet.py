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

def process_audio_pq(data, segment = None):
    audio, _ = librosa.load(io.BytesIO(data["wav"]), sr = None)
    data["audio.npy"] = audio
    del data["wav"]
    data = process_audio(data, segment)
    return data

class EverynoiseDataset(IterableDataset):
    def __init__(self, name="everynoise", mode="train", text_pick = "random", dataset_id=312, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
        self.name = name
        self.text_pick = text_pick
        self.dataset = (
            ParquetDataset(dataset_id, handler=wds.warn_and_continue, **kwargs)
            .map(self._process_audio)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )
    def _process_audio(self, data):
        return process_audio_pq(data, segment=None)


    def _process_text(self, data):
        metadata = json.loads(data["meta"])
        nonvocal = metadata['music_tagging']['Language']['Non-vocal']
        if nonvocal < 0.55: return None
        data["text"] = _process_everynoise(metadata)
        data["music_id"] = fix_hash(data["__key__"])
        return data

    def __iter__(self):
        return iter(self.dataset)

def sample_pct(arr, dropout=0.5, min_examples=1):
    random.shuffle(arr)
    if len(arr) * (1 - dropout) <= min_examples:
        return arr
    return [a for idx, a in enumerate(arr) if random.random() >= dropout]

def _process_everynoise(meta, dropout=0.2):
    if random.random() < 0.05:
        try:
            return f"title: {meta['raw']['name']} album_name: {meta['raw']['name']}"
        except: pass
    if 'genres' in meta['raw'] and meta['raw']['genres']:
        spotify_genres = meta['raw']['genres']
    else:
        spotify_genres = []
    
    everynoise_genres = []
    if 'everynoise_genre' in meta:
        everynoise_genres.append(meta['everynoise_genre'])
    if 'everynoise_trending' in meta:
        everynoise_genres.append(meta['everynoise_trending']['genre'])
    genres = [g for g in set(spotify_genres + everynoise_genres) if g]
    return ', '.join(sample_pct(genres, dropout=dropout, min_examples=1))

if __name__ == "__main__":
    dataset = EverynoiseDataset(name="sstk", mode="train")
    cnt = 0
    for item in dataset:
        cnt += 1
        print(item.keys())
        print(item["audio"])
        print(item["text"])
        if cnt == 5:
            break
        #breakpoint()
        
