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
       # print("wrong text!")
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

def isEmpty(v):

    if v is None or v == "\\N" or len(v.strip()) == 0:
        return True
    return False

class SSTKDataset(IterableDataset):
    def __init__(self, name="sstk", mode="train", text_pick = "random", dataset_id=105, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.name = name
        self.text_pick = text_pick
        self.dataset = (
            ParquetDataset(dataset_id, handler=wds.warn_and_continue, **kwargs)
            .map(self._process_audio)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )
    def _process_audio(self, data):
        metadata = json.loads(data["meta"])
        vad = metadata.get("vad", {})

        # Only ignore vocal segments if vocals detected in metadata. Otherwise VAD will cut out saxophone.
        text = ""
        for key in ["description", "title", "keywords", "genres", "instruments"]:
            if key in metadata:
                text += str(key)
        has_vocal_metadata = 'vocal' in text.lower()
        voice_proportion = vad.get("extra", {}).get("voice_proportion", 0.0)
        if has_vocal_metadata or voice_proportion > 0.5:
            segment = metadata.get("vad", {}).get("segment", None)
        else:
            segment = None


        #"vad": {"segment": [{"start": 1840, "end": 4210}, {"start": 48720, "end": 49700}, {"start": 53050, "end": 54850}, {"start": 86020, "end": 88240}, {"start": 88530, "end": 89540}], 
        # "extra": {"voice_duration_in_seconds": 8.38, "audio_duration_in_seconds": 154.10526, "voice_proportion": 0.054378}} 
        # segment = metadata.get("vad", {}).get("segment", None)
        #print("segment in _process_audio", segment)
        return process_audio_pq(data, segment=segment)


    def _process_text(self, data):
        metadata = json.loads(data["meta"])
        text_fields = {}
        for key in ["description", "title", "keywords", "genres", "instruments"]:
            v = metadata.get(key)
            if isEmpty(v): continue
            text_fields[key] = v.strip()
        if len(text_fields) == 0: pass
        if self.text_pick == "sstk_dropout":
            if random.random() <= 0.3:
                if random.random() < 0.8 and "description" in text_fields:
                    data["text"] = text_fields["description"]
                elif "title" in text_fields:
                    data["text"] = text_fields["title"]
                
            def sample_pct(arr, dropout=0.5, min_examples=1):
                random.shuffle(arr)
                if len(arr) * dropout <= min_examples:
                    return arr[:min_examples]
                return [a for idx, a in enumerate(arr) if random.random() >= dropout]

            if not data.get("text"):
                keywords = []
                if "keywords" in text_fields:
                    kw = [t.strip() for t in text_fields["keywords"].split(",")]
                    kw = sample_pct(kw, 0.5, 5)
                    keywords.extend(kw)
                if "genres" in text_fields:
                    g = [t.strip() for t in text_fields["genres"].split(",")]
                    g = sample_pct(g, 0.2, 0)
                    keywords.extend(g)
                if "instruments" in text_fields:
                    i = [t.strip() for t in text_fields["instruments"].split(",")]
                    i = sample_pct(i, 0.3, 2)
                    keywords.extend(i)
                keywords = list(set(keywords))
                random.shuffle(keywords)
                if random.random() < 0.5:
                    keywords = [k.lower() for k in keywords]
                else:
                    keywords = [k.capitalize() for k in keywords]
                if random.random() < 0.5:
                    data["text"] = ", ".join(keywords)
                else:
                    data["text"] = " ".join(keywords)
        elif self.text_pick == "random":
                if random.random() < 0.2: # instruments: a, b, c
                    text_fields = [f"{key}: {value}" for key, value in text_fields.items()]
                else:
                    text_fields = list(text_fields.values())
                selected_fields = random.sample(text_fields, k=random.randint(1, len(text_fields)))
                if random.random() < 0.4:
                    selected_text = " ".join([" ".join(t.split(',')) for t in selected_fields])
                elif random.random() < 0.4:
                    selected_text = ". ".join(selected_fields).lower()
                else:
                    selected_text = ". ".join(selected_fields)
                data["text"] = selected_text
        else:
            data["text"] = " ".join([" ".join(t.split(',')) for t in text_fields.values()])
        data["music_id"] = fix_hash(data["__key__"])
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
        
