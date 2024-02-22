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
from samantha.dataio.webdataset.extension import IndexedWebDataset
import json
import librosa
import io
from samantha.dataio.parquet.parquet_dataset import ParquetDataset
# Load tokenizer
from transformers import AutoTokenizer

device = "cuda:0"

# Load mulan model
from mulan_modules import create_mulan_model


## TODO: replace with your own mulan model checkpoint
# ckpt_path = "/mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=014000-median_rank_1=160-kaggle.ckpt"
# ckpt_path = "/mnt/bn/mm-data/user/xuchen.song/mulan_tag/mulan-step=024600-kaggle.ckpt"
# prepare the following ckpt_path using these two commands:
# cd /opt/tiger/arnold_starter/
# hdfs dfs get hdfs://haruna/home/byte_speech_sv/mulan/experiments/MuLan_large/qqmusic_1115_callbacks/checkpoints/mulan-step=012800-kaggle.ckpt
ckpt_path = "mulan-step=012800-kaggle-minimal.ckpt"
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

class CMDatasetPar(IterableDataset):
    def __init__(self, name="qq", mode="train", tokenizer_name="chinese",**kwargs):
        if tokenizer_name =="chinese":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer_name == "multiligual":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.name = name
        dataset_id = 401
        self.dataset = (
            ParquetDataset(dataset_id, handler=wds.warn_and_continue, **kwargs)
            .map(process_audio_pq)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )

    def _process_text(self, data):
        metadata = json.loads(data["meta"])
        if self.name == "qq":
            gpt_text = metadata.get('gpt_text', "")
            artist_name = metadata.get('artist_name', "")
            track_name = metadata.get('track_name', "")
            album_name = metadata.get('album_name', "")
            artist_name_res = pattern_apply(artist_name)
            track_name_res = pattern_apply(track_name)
            album_name_res = pattern_apply(album_name)
            data["text"] = ". ".join([artist_name_res, track_name_res, album_name_res, gpt_text])
            data["music_id"] = fix_hash(data["__key__"])
            return data  
        else:
            return data

    def __iter__(self):
        return iter(self.dataset)
def infer_audio_embeds():
    cnt = 0
    dataset = CMDatasetPar(name="qq", mode="test")
    for item in dataset:
        # print(item.keys())
        # print(item["audio"])
        # print(item["text"])
        audio_id = item["uttid"]
        audio_wav = item["audio"]
        #print("audio_wav", audio_wav.shape)
        #audio_load torch.Size([1, 240000])
        audio_segments = extract_segments(
                audio_wav.reshape([1, -1]), sr=24000, duration=10, stride=10
            ).to(
                device
            )  # [M, sr*duration]

            # 4. Call mulan model to calculate audio embeds
        with torch.no_grad():
            music_emb = (
                mulan_module.music_encoder(audio_segments.unsqueeze(1))
                .cpu()
                .data.numpy()
            )
        print("music_emb", music_emb.shape)
        np.save(f"{audio_embed_folder}/{audio_id}.mulan_emb.npy", music_emb)
        cnt += 1
        if cnt == 10000:
                break

    #break
    #breakpoint()

# def infer_audio_embeds():
#     """Assume the audios are stored in npy or wav format."""
#     cnt = 0
#     dataset = SSTKDataset(name="sstk", mode="test")
#     for item in dataset:
#         # print(item.keys())
#         # print(item["audio"])
#         # print(item["text"])
#         audio_wav = item["audio"]
#         audio_id = item["uttid"]
#         audio_npy =  audio_wav.numpy()[0, :]
#         write(audio_folder +"/"+ audio_id+".wav", 24000, audio_npy)
#         # assert (
#         #     audio_wav.ndim == 1
#         # ), """The audios are expected to be mono wave of shape [T]"""

#         # 3. split the complete audio into a minibatch of segments
#         audio_segments = extract_segments(
#             audio_wav.reshape([1, -1]), sr=24000, duration=10, stride=10
#         ).to(
#             device
#         )  # [M, sr*duration]

#         # 4. Call mulan model to calculate audio embeds
#         with torch.no_grad():
#             music_emb = (
#                 mulan_module.music_encoder(audio_segments.unsqueeze(1))
#                 .cpu()
#                 .data.numpy()
#             )
#         # print("music_emb", music_emb.shape)
#         np.save(f"{audio_embed_folder}/{audio_id}.mulan_emb.npy", music_emb)
#         cnt += 1
#         if cnt == 30000:
#             break


print("\nInferring the audio embeds...\n")
infer_audio_embeds()


""""Hard coded method, the list of text pool"""


def prepare_texts():
    text_pool_1 = [
        "流行",
        "摇滚",
        "民谣",
        "说唱",
        "民谣",
        "古风",
        "电子流行",
        "电子",
        "抒情",
        "民歌",
        "儿童歌曲",
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
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

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
