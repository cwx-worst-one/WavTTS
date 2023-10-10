import itertools
import os
import json
import librosa
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from recipes.bigmusic.datasets.lyrics import transform_dataset, default_batch_fn
from recipes.bigmusic.datasets.transforms.lyrics import LyricsTokenTransform, AddConditionsTransform, StyleTextT5Transform, MCCMetadataTextTransform
from recipes.musiclm.inference.utils import load_wav

default_prompt_path = Path(__file__).absolute().parent/'inference_prompts/default.json'

def prompt_path_to_items(prompt_path):
    if isinstance(prompt_path, dict): # prompt path is already an item list
        # Format expects { 'style_audio': [], 'style_text': [], 'lyrics': [] }
        return prompt_path
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
    return prompts

def rewrite_mcc_tags_to_mulan_tags(style_text):
    genre_map = {
        # Mood map
        "Cute": "Cute/Playful",
        "Dynamic": "Dynamic/Energetic",
        "Sorrow": "Sorrow/Sad",
        "Angry": "Angry/Aggressive",
        "Tense": "Thrilling/Suspenseful/Tense",
        "Sweet": "Calm/Relaxing", # "Dreamy/Ethereal"
        # Genre map
        "Trap Rap": "Hip Hop"
    }
    rewrite = style_text
    for k,v in genre_map.items():
        rewrite = rewrite.replace(k.lower(), v.lower())
    print('Rewriting', style_text, rewrite)
    return rewrite

def inference_dataset_from_prompt(prompt_path, conditions="style_text,lyrics_tokens", batch_size=8, max_items=16, lyrics_max_seq_len=400, run_combinations=False, enable_punctuation=False, use_mulan_v2_genres=False):
    prompts = prompt_path_to_items(prompt_path)
    if 'text_category' in prompts: # fix csv formatting
        prompts['category'] = prompts.pop('text_category')
    if 'text_prompt' in prompts: # fix csv formatting
        prompts['style_text'] = prompts.pop('text_prompt')
    if 'style_audio' in prompts:
        prompts['style_audio'] = [load_wav(wav_path) for wav_path in prompts['style_audio']]
    if 'vocal_audio' in prompts:
        prompts['vocal_audio'] = [load_wav(wav_path) for wav_path in prompts['vocal_audio']]
    if use_mulan_v2_genres and 'style_text' in prompts:
        prompts['style_text'] = [rewrite_mcc_tags_to_mulan_tags(style_text) for style_text in prompts['style_text']]
    if run_combinations:
        lyrics_prompt_pairs = itertools.product(*list(prompts.values()))
    else:
        lyrics_prompt_pairs = zip(*list(prompts.values()))

    if 'lyrics_tokens' in conditions:
        segment_transforms = [LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len=lyrics_max_seq_len, truncate_long_lyrics=True, enable_punctuation=enable_punctuation)]
        if 'style_text' not in prompts: # style text not provided. must generate own
            if 'metadata' in prompts:
                print('WARNING: style_text not provided. Using metadata to generate style prompt')
                # mcc metadata provided. use rewrite method
                segment_transforms.append(MCCMetadataTextTransform())
            else:
            #     segment_transforms.append(RandomGenreTextTransform())
                raise Exception('Could not find style text')
        if 'style_tokens' in conditions: # t5 case: add t5 tokenizer
            # TODO: (AS) pass max_seq_len parameter to transform
            segment_transforms.append(StyleTextT5Transform())
    else: # instrumental use case
        segment_transforms = []

    item_keys = list(prompts.keys())
    items = []
    for idx, pair in enumerate(lyrics_prompt_pairs):
        if max_items and idx == max_items:
            break
        item = { key:value for key,value in zip(item_keys,pair) }
        items.append(item)
    batch_transforms=[AddConditionsTransform(conditions)]
    dataset = WebPipeline(items, pipeline=[])
    batch_fn = default_batch_fn(batch_size)
    return transform_dataset(dataset, segment_transforms=segment_transforms, batch_transforms=batch_transforms, batch_fn=batch_fn)
