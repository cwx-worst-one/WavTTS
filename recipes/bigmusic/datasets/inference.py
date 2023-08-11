import itertools
import os
import json
import librosa
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from recipes.bigmusic.datasets.lyrics import transform_dataset
from recipes.bigmusic.datasets.transforms.lyrics import LyricsTokenTransform, AddConditionsTransform, AddMulanVocalTagTransform
from recipes.musiclm.inference.utils import load_wav

default_prompt_path = Path(__file__).absolute().parent/'inference_prompts/default.json'

def inference_dataset_from_prompt(prompt_path, conditions, batch_size=8, max_items=16, run_combinations=True, is_lyrics_prompt=True):
    if is_lyrics_prompt:
        return inference_dataset_from_lyrics_prompt(prompt_path, conditions, batch_size, max_items, run_combinations)
    assert conditions == "mulan_text"
    return inference_dataset_from_instrumental_prompt(prompt_path, conditions, batch_size, max_items)
    
def inference_dataset_from_lyrics_prompt(prompt_path, conditions="mulan_text,lyrics_tokens", batch_size=8, max_items=16, run_combinations=True):
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
    if 'mulan_audio' in prompts:
        prompts['mulan_audio'] = [load_wav(wav_path) for wav_path in prompts['mulan_audio']]
    if 'vocal_audio' in prompts:
        prompts['vocal_audio'] = [load_wav(wav_path) for wav_path in prompts['vocal_audio']]
    if run_combinations:
        lyrics_prompt_pairs = itertools.product(*list(prompts.values()))
    else:
        lyrics_prompt_pairs = zip(*list(prompts.values()))

    item_keys = list(prompts.keys())
    items = []
    for idx, pair in enumerate(lyrics_prompt_pairs):
        if max_items and idx == max_items:
            break
        item = { key:value for key,value in zip(item_keys,pair) }
        items.append(item)
    if 'lyrics_tokens' in conditions:
        # for mix mulan, we need to add 'vocal' tag to text prompt to generate vocals
        segment_transforms = [LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len=150), AddMulanVocalTagTransform()]
    else:
        segment_transforms = []
    batch_transforms=[AddConditionsTransform(conditions)]
    dataset = WebPipeline(items, pipeline=[])
    return transform_dataset(dataset, segment_transforms=segment_transforms, batch_transforms=batch_transforms, batch_size=batch_size)

def inference_dataset_from_instrumental_prompt(prompt_path, conditions="mulan_text", batch_size=8, max_items=16):
    items = []
    _, ext = os.path.splitext(prompt_path)
    if ext == ".csv":
        df = pd.read_csv(prompt_path)
        for _, row in df.iterrows():
            items.append(
                {
                    "category": row["category"],
                    "mulan_text": row["text"]
                }
            )
            if max_items and len(items) == max_items:
                break
    else:
        with open(prompt_path, "r") as fp:
            for line in fp.readlines():
                items.append(
                    {
                        "category": "demos",
                        "mulan_text": line.strip()
                    }
                )
                if max_items and len(items) == max_items:
                    break
    batch_transforms=[AddConditionsTransform(conditions)]
    dataset = WebPipeline(items, pipeline=[])
    return transform_dataset(dataset, batch_transforms=batch_transforms, batch_size=batch_size)
