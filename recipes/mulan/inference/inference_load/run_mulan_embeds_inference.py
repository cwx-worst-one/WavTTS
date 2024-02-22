import io
import os
import glob
import numpy as np
import pickle
import torch
import torch
import subprocess
import soundfile as sf
from typing import Tuple
from pathlib import Path
from tqdm import tqdm
from typing import List
import subprocess
from scipy.io.wavfile import write
import json
import os
import random

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import time
import webdataset as wds
from recipes.mulan.dataset.utils import *
import json
import librosa
import io
from samantha.dataio.parquet.parquet_dataset import ParquetDataset
# Load tokenizer
from transformers import AutoTokenizer

device = "cuda:0"

# Load mulan model
from mulan_modules import create_mulan_model



ckpt_path = "/mnt/bn/audio-diffusion/xuchen/mulan/models/mulan-step=014800-kaggle.ckpt"
mulan_module = create_mulan_model(ckpt_path, device=device).eval()


## TODO: replace with your own audio folder (input) and audio embed folder (output)
## Assume one audio file is either one npy file or one wav file
# audio_folder = "/mnt/bn/mm-data/user/xuchen.song/mulan_tag/audio_sample"
audio_folder = "/opt/tiger/samantha/audio_samples"
os.makedirs(audio_folder, exist_ok=True)
audio_embed_folder = audio_folder.replace("/audio_sample", "/audio_embeds_mulan_30ks")
text_embed_folder = audio_folder.replace("/audio_sample", "/text_embeds_mulan")
os.makedirs(audio_embed_folder, exist_ok=True)
os.makedirs(text_embed_folder, exist_ok=True)

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
def extract_segments(wav, sr=24000, duration: int = 10, stride: int = 5):
    """Extract audio segments of `duration` seconds every `stride` seconds."""
    wavs = torch.tensor(wav).unfold(1, sr * duration, sr * stride)[
        0, :, : sr * duration
    ]
    return wavs
def process_audio_pq(data, segment = None):
    audio, _ = librosa.load(io.BytesIO(data["wav"]), sr = None)
    data["audio.npy"] = audio
    del data["wav"]
    data = process_audio(data, segment)
    return data
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



def infer_audio_embeds():
    """Assume the audios are stored in npy or wav format."""
    cnt = 0
    dataset = SSTKDataset(name="sstk", mode="train")
    for item in dataset:

        audio_wav = item["audio"]
        audio_id = item["uttid"]
        audio_npy =  audio_wav.numpy()[0, :]
        write(audio_folder +"/"+ audio_id+".wav", 24000, audio_npy)
        # assert (
        #     audio_wav.ndim == 1
        # ), """The audios are expected to be mono wave of shape [T]"""

        # 3. split the complete audio into a minibatch of segments
        audio_segments = extract_segments(
            audio_wav.reshape([1, -1]), sr=24000, duration=10, stride=10
        ).to(
            device
        )  # [M, sr*duration]

        # 4. Call mulan model to calculate audio embeds
        with torch.no_grad():
            music_emb = (
                # music_encoder="mut_sstk
                mulan_module.music_encoder(audio_segments.unsqueeze(1))
                .cpu()
                .data.numpy()
            )
        print("music_emb", music_emb.shape)
        np.save(f"{audio_embed_folder}/{audio_id}.mulan_emb.npy", music_emb)
        cnt += 1
        if cnt == 3:
            break


print("\nInferring the audio embeds...\n")
infer_audio_embeds()


""""Hard coded method, the list of text pool"""


def prepare_texts():
    text_pool_1 = [
        "EDM Techno",
        "EDM House",
        "EDM Disco",
        "Dubstep",
        "Electronic Ambient",
        "8 Bit / Chiptune",
    ]
    return [text_pool_1]


def infer_text_embeds(
    mulan_module,
    text_dset: List[str],
    tokenizer: AutoTokenizer = None,
    batch_size=128,
    max_length=200,
    device="cuda:0",
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

    embeds = []
    text_dset = [s.lower() for s in text_dset]

    with torch.no_grad():
        for xl in tqdm(range(0, len(text_dset), batch_size)):
            xr = min(len(text_dset), xl + batch_size)

            data = {}
            encodings = tokenizer(
                text_dset[xl:xr],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            data["input_ids"] = encodings["input_ids"].to(device)
            data["token_type_ids"] = encodings["token_type_ids"].to(device)
            data["attention_mask"] = encodings["attention_mask"].to(device)

            embeds.append(mulan_module.text_encoder(**data).cpu().data.numpy())
    return np.concatenate(embeds)

print("\nInferring the text embeds...\n")
text_pools = prepare_texts()

for i in range(len(text_pools)):
    text_pool = text_pools[i]
    text_embeds = infer_text_embeds(mulan_module, text_pool)
    pickle.dump(
        [text_pool, text_embeds], open(f"{text_embed_folder}/text_pool{i}.pkl", "wb")
    )
