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
from recipes.bigmusic.datasets.transforms.lyrics import LyricsTokenTransform, AddConditionsTransform, AddMulanVocalTagTransform, MetadataT5Transform
from recipes.musiclm.inference.utils import load_wav

default_prompt_path = Path(__file__).absolute().parent/'inference_prompts/default.json'

def inference_dataset_from_prompt(prompt_path, conditions="style_text,lyrics_tokens", batch_size=8, max_items=16, lyrics_max_seq_len=150, run_combinations=False):
    if prompt_path == "validation":
        return inference_validation(batch_size=batch_size)
    prompt_path = Path(prompt_path)
    if prompt_path.suffix == '.json':
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
    elif prompt_path.suffix == '.csv':
        df = pd.read_csv(prompt_path)
        prompts = df.to_dict('list')
    elif prompt_path.suffix == '.txt':
        with open(prompt_path, "r") as fp:
            prompts = [{ "style_text": line.strip() } for line in fp.readlines()]

    if 'style_audio' in prompts:
        prompts['style_audio'] = [load_wav(wav_path) for wav_path in prompts['style_audio']]
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
        segment_transforms = [LyricsTokenTransform.init_espeak_tokenizer(
            lyrics_max_seq_len=lyrics_max_seq_len)]
        if 'style_tokens' in conditions: # t5 case: add t5 tokenizer
            segment_transforms.append(MetadataT5Transform(max_seq_len=lyrics_max_seq_len))
        else: # mulan vocal case. add vocal tag to prompt to generate vocals
            segment_transforms.append(AddMulanVocalTagTransform())
    else:
        segment_transforms = []
    batch_transforms=[AddConditionsTransform(conditions)]
    dataset = WebPipeline(items, pipeline=[])
    return transform_dataset(dataset, segment_transforms=segment_transforms, batch_transforms=batch_transforms, batch_size=batch_size)
