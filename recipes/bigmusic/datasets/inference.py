import itertools
import os
import json
import librosa
import torch
from pathlib import Path
from torch.utils.data import Dataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from recipes.bigmusic.datasets.lyrics import transform_dataset
from recipes.bigmusic.datasets.transforms.lyrics import LyricsTokenTransform, AddConditionsTransform, AddMulanVocalTagTransform
from recipes.musiclm.inference.utils import load_wav

default_prompt_path = Path(__file__).absolute().parent/'inference_prompts/default.json'

class ItemDataset(Dataset):
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
    
def dataset_from_prompt(prompt_path=None, conditions="mulan_text,lyrics_tokens", batch_size=8, max_items=16, run_combinations=True):
    if prompt_path is None or not os.path.exists(prompt_path):
        print('Prompt path not found. Using default', default_prompt_path)
        prompt_path = default_prompt_path
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
    if 'mulan_audio' in prompts:
        prompts['mulan_audio'] = [load_wav(wav_path) for wav_path in prompts['mulan_audio']]
    if 'vocal_audio' in prompts:
        prompts['vocal_audio'] = [load_wav(wav_path) for wav_path in prompts['vocal_audio']]
    if run_combinations:
        lyrics_prompt_pairs = itertools.product(*list(prompts.values()))
    else:
        lyrics_prompt_pairs = itertools.zip_longest(*list(prompts.values()), fillvalue=None)

    item_keys = list(prompts.keys())
    items = []
    for idx, pair in enumerate(lyrics_prompt_pairs):
        if max_items and idx == max_items:
            break
        item = { key:value for key,value in zip(item_keys,pair) }
        items.append(item)
    if 'lyrics_tokens' in conditions:
        # for mix mulan, we need to add 'vocal' tag to text prompt to generate vocals
        segment_transforms = [LyricsTokenTransform(lyrics_max_seq_len=150, allow_unknown=False), AddMulanVocalTagTransform()]
    else:
        segment_transforms = []
    batch_transforms=[AddConditionsTransform(conditions)]
    dataset = WebPipeline(items, pipeline=[])
    return transform_dataset(dataset, segment_transforms=segment_transforms, batch_transforms=batch_transforms, batch_size=batch_size)

