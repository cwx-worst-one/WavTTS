import itertools
import os
import json
import glob
import torch
from typing import List
import pandas as pd
from pathlib import Path
from functools import partial

from torchaudio_augmentations import Compose

from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    SetAudioDimensions,
    ToTensor,
)

from recipes.bigmusic.datasets.svs import SVSInferTransforms, override_parameter
from recipes.bigmusic.datasets.svs import collate_fn as future_function
from recipes.bigmusic.datasets.lyrics import (
    transform_dataset,
    default_batch_fn,
    dictionary_collate,
)
from recipes.bigmusic.datasets.mir_data_util import MACRO_STYLE_MAP
from recipes.bigmusic.datasets.transforms.lyrics import (
    LyricsTokenTransform,
    AddConditionsTransform,
    StyleTextT5Transform,
    MCCMetadataTextTransform,
    AddDurationTransform,
)
from recipes.bigmusic.datasets.transforms.structure import (
    IntensityTransform,
)
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
        # NOTE: Please make sure the numerical columns are always filled.
        # Remove columns that only contain na. Columns can be partially filled.
        # The caller is responsible for data validation.
        df = df.dropna(axis='columns', how='all')
        # For string columns, replace na with ""
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].fillna("")
        prompts = df.to_dict('list')
    elif prompt_path.suffix == '.txt':
        with open(prompt_path, "r") as fp:
            prompts = [{ "style_text": line.strip() } for line in fp.readlines()]
    return prompts

def inference_dataset_from_prompt(
    prompt_path,
    conditions="style_text,lyrics_tokens",
    batch_size=8,
    max_items=None,
    lyrics_max_seq_len=400,
    run_combinations=False,
    enable_punctuation=True,
    lang='en',
    dataset_mode="truncate_length",
    transform_style_text=True,
    extra_params=None,
):
    prompts = prompt_path_to_items(prompt_path)
    if 'index' in prompts:
        prompts['index'] = [str(x) for x in prompts['index']]
    if 'text_category' in prompts: # fix csv formatting
        prompts['category'] = prompts.pop('text_category')
    if 'text_prompt' in prompts: # fix csv formatting
        prompts['style_text'] = prompts.pop('text_prompt')
    elif 'text' in prompts:
        prompts['style_text'] = prompts.pop('text')
    if 'style_category' in conditions and 'style_category' not in prompts:
        prompts['style_category'] = prompts['style_text']
    if 'style_audio' in prompts:
        prompts['style_audio'] = load_and_normalize_wavs(prompts['style_audio'])
    if 'vocal_audio' in prompts:
        additional_transforms = [voice_clone_transform(extra_params)] if extra_params.get('app_type') == 'vclone' else []
        prompts['vocal_audio'] = load_and_normalize_wavs(prompts['vocal_audio'], additional_transforms)
    if 'intensity_audio' in prompts:
        prompts['intensity_audio'] = load_and_normalize_wavs(prompts['intensity_audio'])
    if 'structure' in prompts:
        prompts['structure'] = [None if x == "random" else json.loads(x) for x in prompts['structure']]
    elif 'structure' in conditions:
        # Use random structure by default
        prompts['structure'] = [None] * len(prompts[next(iter(prompts.keys()))])
    if 'semantic_tokens' in prompts:
        prompts['semantic_tokens'] = [torch.load(fp) for fp in prompts['semantic_tokens']]

    # Process lyrics and style_text
    if 'rewrite_lyrics' in prompts:  # override lyrics with rewrite_lyrics
        rewritten_lyrics = process_lyrics(prompts.pop('rewrite_lyrics'))
        prompts['lyrics'] = [
            rewritten if rewritten else original.strip()
            for original, rewritten in zip(prompts['lyrics'], rewritten_lyrics)
        ]
    if 'style_text' in prompts and transform_style_text:
        prompts['style_text'] = process_style_text(prompts['style_text'])

    if run_combinations:
        lyrics_prompt_pairs = itertools.product(*list(prompts.values()))
    else:
        lyrics_prompt_pairs = zip(*list(prompts.values()))

    segment_transforms = []
    if 'lyrics_tokens' in conditions:
        if lang == 'en':
            segment_transforms.append(
                LyricsTokenTransform.init_espeak_tokenizer(
                    lyrics_max_seq_len=lyrics_max_seq_len,
                    dataset_mode=dataset_mode,
                    enable_punctuation=enable_punctuation,
                    validate_ascii=True,
                )
            )
        elif lang == 'zh_wp':
            segment_transforms.append(
                LyricsTokenTransform.init_zh_tokenizer(
                    lyrics_max_seq_len=lyrics_max_seq_len,
                    dataset_mode=dataset_mode,
                    enable_punctuation=enable_punctuation,
                )
            )
        elif lang == 'zh_phone':
            segment_transforms.append(
                LyricsTokenTransform.init_sami_tokenizer(
                    lyrics_max_seq_len=lyrics_max_seq_len,
                    dataset_mode=dataset_mode,
                    enable_punctuation=enable_punctuation,
                )
            )

        if 'style_text' in conditions and 'style_text' not in prompts:
            # style text not provided. must generate own
            if 'metadata' in prompts:
                print('WARNING: style_text not provided. Using metadata to generate style prompt')
                # mcc metadata provided. use rewrite method
                segment_transforms.append(MCCMetadataTextTransform('Vocal'))
            else:
                raise Exception('Could not find style text')
        if 'style_tokens' in conditions: # t5 case: add t5 tokenizer
            # TODO: (AS) pass max_seq_len parameter to transform
            segment_transforms.append(StyleTextT5Transform())
    if 'intensity' in conditions:
        segment_transforms.append(
            IntensityTransform(
                audio_key="intensity_audio" if "intensity_audio" in prompts else "style_audio",
                sample_rate=extra_params["sample_rate"],
                calculation_mode=extra_params.get("intensity_calculation", "mean"),
                intensity_hz=extra_params.get("intensity_hz", 1),
            )
        )

    batch_transforms=[AddConditionsTransform(conditions)]
    if 'duration' in conditions:
        batch_transforms.append(AddDurationTransform(extra_params["duration"]))
    
    item_keys = list(prompts.keys())
    items = []
    for idx, pair in enumerate(lyrics_prompt_pairs):
        if max_items and idx == max_items:
            break
        item = { key:value for key,value in zip(item_keys,pair) }
        items.append(item)

    dataset = WebPipeline(items, pipeline=[])
    batch_fn = default_batch_fn(
        batch_size,
        collation_fn=partial(dictionary_collate, remove_invalid=False),
    )
    return transform_dataset(
        dataset,
        segment_transforms=segment_transforms,
        batch_transforms=batch_transforms,
        batch_fn=batch_fn,
    )

def load_and_normalize_wavs(wav_paths, additional_transforms=()):
    audio_transforms = Compose([ToTensor(), SetAudioDimensions(), *additional_transforms])
    wavs = [audio_transforms(load_wav(wav_path)) for wav_path in wav_paths]
    return wavs

def voice_clone_transform(extra_params):
    sample_rate = extra_params['sample_rate']
    voice_clone_duration = extra_params.get('voice_clone_duration', -1)
    return lambda vocal_audio: vocal_audio[:, :(voice_clone_duration * sample_rate)]

def process_lyrics(lyrics_list: List[str]) -> List[str]:
    """Move in-line leading section tags out as single lines."""
    def add_section_tags_to_lyrics(text: str) -> str:
        """Add section tags based on the number of lines.
        This function is also a reference of the web demo's text processing.
        """
        intro_tag, outro_tag, verse_tag, chorus_tag = "[intro]", "[outro]", "[verse]", "[chorus]"
        lines = text.split("\n")
        n_lines = len(lines)
        if n_lines in [1, 2]:  # intro + verse + outro
            result = [intro_tag, verse_tag] + lines + [outro_tag]
        elif n_lines in [3, 4]: # verse + outro
            result = [verse_tag] + lines + [outro_tag]
        elif n_lines == 6:  # verse + chorus
            result = [verse_tag] + lines[:2] + [chorus_tag] + lines[2:]
        else:  # verse + chorus
            result = [verse_tag] + lines[:n_lines//2] + [chorus_tag] + lines[n_lines//2:]
        return "\n".join(result)

    return [add_section_tags_to_lyrics(lyrics) if lyrics.strip() else "" for lyrics in lyrics_list]

def process_style_text(style_text_list: List[str]) -> List[str]:
    """Auto-convert macro style text into separate sub-category text seaprated by '|'."""
    def process_one(text: str) -> str:
        separator = "|"
        # Treat the text as formatted if there is any separator in the text
        if separator in text:
            return text
        return MACRO_STYLE_MAP.get(text, MACRO_STYLE_MAP["Pop"])
    return [process_one(text) for text in style_text_list]

def inference_svs_dataset_from_prompt(
    input_txt_pattern,
    conditions="style_text,lyrics_tokens",
    batch_size=8,
    style_prompt_path: str = '',
    vocal_prompt_duration: float = 10.0,
    pitch_shift: int = 0,
    segment_max_phone_len = 0,
    segment_max_leadsheet_len = 0,
    target_spkr_name='Krista',
    vocal_prompt_number: int = 1,
    **kwargs,
):

    collate_fn = override_parameter(future_function, conditions=conditions)

    items = []
    for f in glob.glob(input_txt_pattern+"/*.txt", recursive=True):
        if os.path.exists(f) and len(open(f).readlines()) > 1:
            items.append(f)
    segment_transforms = [SVSInferTransforms(style_prompt_path=style_prompt_path,
                                            vocal_prompt_duration=vocal_prompt_duration,
                                            pitch_shift=pitch_shift,
                                            segment_max_phone_len=segment_max_phone_len,
                                            segment_max_leadsheet_len=segment_max_leadsheet_len,
                                            target_spkr_name=target_spkr_name,
                                            vocal_prompt_number=vocal_prompt_number,
                                            **kwargs,)]

    dataset = WebPipeline(items, pipeline=[])
    batch_fn = default_batch_fn(batch_size, collation_fn=collate_fn)
    return transform_dataset(
        dataset,
        segment_transforms=segment_transforms,
        batch_transforms=[],
        batch_fn=batch_fn,
    )
