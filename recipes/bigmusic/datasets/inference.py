import copy
import itertools
import os
import json
import glob
import torch
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from functools import partial, reduce
import operator

from torchaudio_augmentations import Compose

from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    SetAudioDimensions,
    ToTensor,
)
import samantha.utils.hdfs_helper as hh

from recipes.bigmusic.datasets.svs import SVSInferTransforms, override_parameter
from recipes.bigmusic.datasets.svs import collate_fn as future_function
from recipes.bigmusic.datasets.lyrics import (
    transform_dataset,
    default_batch_fn,
    dictionary_collate,
)
from recipes.bigmusic.datasets.mir_data_util import (
    rewrite_style_input_to_multi_tag,
    rewrite_style_input_to_sa_tag,    
    rewrite_style_input_to_multi_tag_v3,
    rewrite_style_input_to_multi_tag_combo_v3,
    ARTIST_ID_MAP_V2,
    KEY_ID_MAP,
    TEMPO_LABEL_ID_MAP,
    tempo_to_label,
)
from recipes.bigmusic.datasets.transforms.lyrics import (
    LyricsTokenTransform,
    AddConditionsTransform,
    StyleTextT5Transform,
    MCCMetadataTextTransform,
    AddDurationTransform,
)
from recipes.bigmusic.datasets.transforms.mir_transforms import ChordSeqTokenTransform
from recipes.bigmusic.datasets.transforms.structure import (
    IntensityTransform,
)
from recipes.bigmusic.datasets.utils.zh_lyrics_proc import SongLyrics
from recipes.datasets.mcc.sami_tokenizer import section_parens
from recipes.musiclm.inference.utils import load_wav
from recipes.musiclm.utils.dist import local_zero_first

default_prompt_path = Path(__file__).absolute().parent/'inference_prompts/default.json'

SEGMENT_TRANSFORMS = None

def prompt_path_to_items(prompt_path, cache_dir='.prompt_cache'):
    if isinstance(prompt_path, dict): # prompt path is already an item list
        # Format expects { 'style_audio': [], 'style_text': [], 'lyrics': [] }
        return prompt_path
    if prompt_path.startswith("hdfs://"):
        with local_zero_first():
            local_path = f"{cache_dir}/{os.path.basename(prompt_path)}"
            if not os.path.exists(local_path):
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                if not hh.get(prompt_path, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {prompt_path}.")
            prompt_path = local_path

    prompt_path = Path(prompt_path)
    if prompt_path.suffix == '.json':
        with open(prompt_path, 'r', encoding='utf-8') as f:
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
        with open(prompt_path, "r", encoding='utf-8') as fp:
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
    rewrite_target="sa_tag",
    is_inference=False,
    front_results=None,
    rewrite_lyrics=True,
    disable_multitag=False,
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
        prompts['vocal_prompt'] = prompts['style_audio']
        prompts['style_audio'] = load_and_normalize_wavs(prompts['style_audio'])
    if 'vocal_audio' in prompts:
        additional_transforms = [voice_clone_transform(extra_params)] if extra_params.get('app_type') == 'vclone' else []
        prompts['vocal_audio'] = load_and_normalize_wavs(prompts['vocal_audio'], additional_transforms)
    if 'intensity_audio' in prompts:
        prompts['intensity_audio'] = load_and_normalize_wavs(prompts['intensity_audio'])
    if 'beat_audio' in prompts:
        prompts['beat_audio'] = load_and_normalize_wavs(prompts['beat_audio'])
    if 'structure' in prompts:
        prompts['structure'] = [None if x == "random" else json.loads(x) for x in prompts['structure']]
    elif 'structure' in conditions:
        # Use random structure by default
        prompts['structure'] = [None] * len(prompts[next(iter(prompts.keys()))])
    if 'offset' in conditions:
        if extra_params is not None:
            offset = extra_params.get('offset', 0)
        else:
            offset = 0
        prompts['offset'] = [offset] * len(prompts[next(iter(prompts.keys()))])
    if 'semantic_tokens' in prompts:
        prompts['semantic_tokens'] = [torch.load(fp) for fp in prompts['semantic_tokens']]
    if 'chord_seq' in prompts:
        prompts['chord_seq'] = [x.split() for x in prompts['chord_seq']]    # convert "C:maj G:maj" to ["C:maj", "G:maj"]

    # Below are query rewritting logics for Chinese Lyrics2song. 
    if lang.startswith("zh_"):
        prompts = process_zh_prompts(prompts, conditions.split(","), rewrite_target, transform_style_text, rewrite_lyrics, disable_multitag)

    if run_combinations:
        lyrics_prompt_pairs = itertools.product(*list(prompts.values()))
    else:
        lyrics_prompt_pairs = zip(*list(prompts.values()))

    global SEGMENT_TRANSFORMS
    if SEGMENT_TRANSFORMS:
        segment_transforms = SEGMENT_TRANSFORMS
    else:
        segment_transforms = []
        if 'lyrics_tokens' in conditions:
            if lang == 'en':
                segment_transforms.append(
                    LyricsTokenTransform.init_espeak_tokenizer(
                        lyrics_max_seq_len=lyrics_max_seq_len,
                        dataset_mode=dataset_mode,
                        enable_punctuation=enable_punctuation,
                        validate_ascii=False,
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
                if is_inference:
                    segment_transforms.append(
                        LyricsTokenTransform.init_sami_inference_tokenizer(
                            lyrics_max_seq_len=lyrics_max_seq_len,
                            dataset_mode=dataset_mode,
                            vocab_type="phoneme"
                        )
                    )
                else:  
 
                    segment_transforms.append(
                        LyricsTokenTransform.init_sami_tokenizer(
                            lyrics_max_seq_len=lyrics_max_seq_len,
                            dataset_mode=dataset_mode,
                            enable_punctuation=enable_punctuation,
                            normalize_tags=True,  # support all kinds of section tags
                            vocab_type="phoneme"
                        )
                    )
            elif lang == 'zh_phonetone':
                if is_inference:
                    segment_transforms.append(
                        LyricsTokenTransform.init_sami_inference_tokenizer(
                            lyrics_max_seq_len=lyrics_max_seq_len,
                            dataset_mode=dataset_mode,
                            enable_punctuation=enable_punctuation,
                            normalize_tags=True,  # support all kinds of section tags
                            vocab_type="phoneme+tone"
                        )
                    )
                else:
                    segment_transforms.append(
                        LyricsTokenTransform.init_sami_tokenizer(
                            lyrics_max_seq_len=lyrics_max_seq_len,
                            dataset_mode=dataset_mode,
                            enable_punctuation=enable_punctuation,
                            normalize_tags=True,  # support all kinds of section tags
                            vocab_type="phoneme+tone"
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
        SEGMENT_TRANSFORMS = segment_transforms

    if 'chord_seq' in conditions:
        segment_transforms.append(
            ChordSeqTokenTransform()
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
    
    # HACK:如果处于推理状态（非batch），那么将front_results添加到第一个item
    if is_inference and front_results is not None and items:
        items[0]['front_results'] = front_results
        return items
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

def process_zh_lyrics(lyrics_list: List[str], genres: Optional[List[str]], rewrite_lyrics: bool) -> List[str]:
    def split_and_normalize_one(text: str, genre: str) -> str:
        if rewrite_lyrics:
            return str(SongLyrics.parse(text).process(genre))
        return "\n".join(list(filter(lambda l: len(l) > 0, map(lambda l: l.strip(), text.split("\n")))))

    if genres is None:
        genres = ["empty"] * len(lyrics_list)

    return [split_and_normalize_one(text, genre) for text, genre in zip(lyrics_list, genres)]

def process_zh_style_text(style_text_list: List[str], rewrite_target="", disable_multitag=False) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Auto-convert macro style text into separate sub-category text seaprated by '|'."""
    def process_one(text: str) -> Tuple[str, str, str, int]:
        """Expecting input style text in the format of "SA_genre|SA_mood|SA_gender" where each field can be optional."""
        if disable_multitag:
            # To support infer with pretrain ckpts
            # https://bytedance.us.larkoffice.com/docx/PddVdJ0ScoLqjExc87Luh0Mjs0c#part-E3sadoTLdoodGbxXrP8uggCestg
            text=text.split("|")[0]
        if "|" not in text:
            text = text + "||" # For backward compatibility, support top genre only style text
        # TODO: also expand to key and tempo_label.
        if rewrite_target == "multi_tag":
            return rewrite_style_input_to_multi_tag(text)
        if rewrite_target == "sa_tag":
            return rewrite_style_input_to_sa_tag(text)
        if rewrite_target == "multi_tag_v3":
            return rewrite_style_input_to_multi_tag_v3(text)
        if rewrite_target == "multi_tag_combo_v3":
            return rewrite_style_input_to_multi_tag_combo_v3(text)
        else:
            raise NotImplementedError(f"Unsupported rewrite_taget: {rewrite_target}")
    tags, keys, tempo_labels, speaker_ids = tuple(zip(*[process_one(style_text) for style_text in style_text_list]))
    return list(tags), list(keys), list(tempo_labels), list(speaker_ids)

def multitags_to_speaker_ids(original_speaker_ids: Optional[List[int]], override_speaker_ids: Optional[List[int]], n_songs: Optional[int] = None) -> List[int]:
    """Extract voice tags and return speaker_ids. Override the original ID unless the original ID represents an artist."""
    if n_songs is None:
        if original_speaker_ids is not None:
            n_songs = len(original_speaker_ids)
        elif override_speaker_ids is not None:
            n_songs = len(override_speaker_ids)
        else:
            raise ValueError("Must pass song number if both speaker ids are Nones")

    empty_id = ARTIST_ID_MAP_V2["zh_empty"]
    female_id = ARTIST_ID_MAP_V2["Female"]
    male_id = ARTIST_ID_MAP_V2["Male"]
    child_id = ARTIST_ID_MAP_V2["Child"]

    if original_speaker_ids is None:
        original_speaker_ids = [empty_id] * n_songs
    if override_speaker_ids is None:
        override_speaker_ids = [empty_id] * n_songs

    def process_one(original_speaker_id: int, override_speaker_id: int) -> int:
        if original_speaker_id not in [empty_id, female_id, male_id, child_id]:  # any speaker_id other than these is artist id
            return original_speaker_id
        return override_speaker_id
    return [process_one(original, override) for original, override in zip(original_speaker_ids, override_speaker_ids)]

def process_zh_prompts(
    prompts: Dict,
    conditions: List[str],
    rewrite_target: str,
    transform_style_text: bool = True,
    rewrite_lyrics: bool = True,
    disable_multitag: bool = False,
) -> Dict:
    """
    - Reformat lyrics.
    - Reading additional speaker/key/tempo labels.
    - Expand style text input to detailed labels.
    """
    prompts = copy.deepcopy(prompts)
    n_songs = len(prompts['lyrics'])

    prompts["lyrics"] = [lyrics.strip() for lyrics in prompts["lyrics"]]

    # Add section tag to the optional "rewrite_lyrics" column
    # if 'rewrite_lyrics' in prompts:  # override lyrics with rewrite_lyrics
    #     rewritten_lyrics = add_section_tags_to_lyrics(prompts.pop('rewrite_lyrics'))
    #     prompts['lyrics'] = [
    #         rewritten if rewritten else original
    #         for original, rewritten in zip(prompts['lyrics'], rewritten_lyrics)
    #     ]

    # Rewrite style_text if `transform_style_text` is True
    # We expect the style_text to be in the format of "SA_genre|SA_mood|SA_gender"
    # For example "Pop||", or "Jazz|Happy|" or "Pop||Female"
    # The output will be in the same format of Multi-tag
    # Before rewrite style_text, save the original style_text for demo video display.
    prompts['original_style_text'] = prompts['style_text']
    if transform_style_text:
        lyrics_prompts = prompts.get("prompt", [])
        if lyrics_prompts:
            qs = []
            for lyrics_p, style_p in zip(lyrics_prompts, prompts['style_text']):
                qs.append(lyrics_p[:10] + '\n' + style_p.split('|')[0])
            prompts['original_style_text'] = qs
        prompts['style_text'], key_from_tag, tempo_label_from_tag, speaker_id_from_tag = process_zh_style_text(prompts['style_text'], rewrite_target, disable_multitag)
    else:
        key_from_tag, tempo_label_from_tag, speaker_id_from_tag = None, None, None        

    # process lyrics based on style (the first item in style text should always be genre)
    prompts['lyrics'] = process_zh_lyrics(prompts['lyrics'], [s.split("|")[0] for s in prompts['style_text']], rewrite_lyrics)

    # Override speaker_id if using multitag (there is a category dedicated to the voice), defaults to 0.
    if 'speaker_id' in conditions:
        prompts['speaker_id'] = multitags_to_speaker_ids(prompts.get("speaker_id"), speaker_id_from_tag, n_songs)

    if 'key' in conditions:
        if 'key' in prompts:  # P1: explicit inputs from csv.
            key_text = prompts['key']
        elif key_from_tag:  # P2: expansion result
            key_text = key_from_tag
        else:  # P3: set default empty key
            key_text = ['N'] * n_songs
        # _key and _tempo_label are text label, prompts['key'] and prompts['tempo_label'] are indices.
        prompts['key'] = [KEY_ID_MAP['N' if s == '' else s] for s in key_text]  # replace empty string with "N"
    
    if 'tempo_label' in conditions:
        if 'tempo_label' in prompts:  # P1: explicit inputs from csv.
            tempo_label_text = prompts['tempo_label']
        elif tempo_label_from_tag:  # P2: expansion result
            tempo_label_text = tempo_label_from_tag
        else:  # P3: set default empty tempo label
            tempo_label_text = [''] * n_songs
        prompts['tempo_label'] = [TEMPO_LABEL_ID_MAP[tl] for tl in tempo_label_text]

    return prompts

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
