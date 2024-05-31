from importlib.metadata import metadata
from typing import Any, Dict, List, Generator, Optional, Callable
import io
import json
import torch
import webdataset as wds
from dataclasses import dataclass
import math
import librosa
import numpy as np
from transformers import T5Tokenizer
from recipes.bigmusic.datasets.tokenizers.cmu_phonemes import CMUPhonemeTokenizer
from recipes.bigmusic.utils.format_utils import normalize_text, normalize_text_sami_tokenizer, format_section_tags
from transformers import Wav2Vec2PhonemeCTCTokenizer
from recipes.datasets.mcc.sami_tokenizer import SamiOfflineTokenizer
from recipes.musiclm.utils.dist import local_zero_first
import random
from recipes.bigmusic.utils.format_utils import rewrite_metadata, rewrite_playlist_labels
from functools import partial
from collections import Counter
import ast

def pad_crop(sequence, seq_len, dtype, padding_value=0):
    item_pad = torch.full((seq_len,), fill_value=padding_value, dtype=dtype)
    item_pad[:len(sequence)] = torch.as_tensor(sequence[:seq_len])
    return item_pad


class RenameAudioKeyTransform():
    "Converts mcc dataloaders to work with bigmusic models"
    def __call__(self, item):
        if 'text' in item:
            item['lyrics'] = item.pop('text')

        wavs = item.pop('audio')
        if len(wavs.shape) == 2:
            wavs = wavs.squeeze(0)
        return { **item, 'style_audio': wavs, 'target_audio': wavs }

class MCCMetadataTextTransform():
    def __init__(self, metadata_type="Vocal"):
        self.metadata_type = metadata_type

    def _call_once(self, item):
        metadata_category = rewrite_metadata_categories(item.get('metadata', {}))
        metadata_string = rewrite_metadata(item.get('metadata', {}), type=self.metadata_type)
        return {
            **item, 'style_text': metadata_string, 'style_category': metadata_category
        }

    def __call__(self, item):
        if isinstance(item, list): # perform batch transform
            return [self._call_once(i) for i in item]
        return self._call_once(item)

class SSTKMetadataTextTransform():
    def _call_once(self, item):
        metadata = item.get('metadata', {})
        text_fields = []
        for key in ["title", "description", "keywords", "genres", "instruments"]:
            v = metadata.get(key)
            if v is None or v == "\\N":
                continue
            text_fields.append(v)
        random.shuffle(text_fields)
        metadata_string = text_fields[0] if len(text_fields) > 0 else ""
        return {
            **item, 'style_text': metadata_string
        }

    def __call__(self, item):
        if isinstance(item, list): # perform batch transform
            return [self._call_once(i) for i in item]
        return self._call_once(item)

class BillboardV2MetadataTextTransform():
    def _call_once(self, item):
        # metadata = item.get('metadata', {})
        metadata_string = BillboardV2MetadataTextTransform.billboard_to_compat_style_text(item['metadata'])
        return {
            **item, 'style_text': metadata_string
        }

    def __call__(self, item):
        if isinstance(item, list): # perform batch transform
            return [self._call_once(i) for i in item]
        return self._call_once(item)

    @staticmethod
    def billboard_to_compat_style_text(index: dict) -> str:
        meta_dict = {}

        ## METADATA
        meta_dict["final_mood"] = None
        meta_dict["final_genre"] = index.get("genre")  # 'style'

        vocal_tags = [l for l in index["tags"]["vocal"] for l in l]
        vocal_tags = Counter(vocal_tags)

        if vocal_tags["gender_male"] > vocal_tags["gender_female"]:
            meta_dict["merge_aed"] = "Male"
        else:
            meta_dict["merge_aed"] = "Female"

        style_text = rewrite_metadata(meta_dict)
        return style_text

class SpotifyMetadataTextTransform():
    def __init__(self, max_genres=None, min_year=2000):
        self.max_genres = max_genres
        self.min_year = min_year

    def _call_once(self, item):
        metadata = item.get('metadata', {})
        genres = metadata['genres']
        if isinstance(metadata['year'], str):
            metadata['year'] = int(metadata['year']) if metadata['year'].isdigit() else None
        if self.min_year is not None and 'year' in metadata and metadata['year'] and metadata['year'] < 2000:
            return None
        if not isinstance(genres, list):     
            try:
                genres = json.loads(genres)
            except json.JSONDecodeError as e: # to fix single quote arrays
                genres = ast.literal_eval(genres)
        
        if self.max_genres:
            genres = random.sample(genres, min(len(genres), self.max_genres))
        metadata_string = ', '.join(genres)
        return {
            **item, 'style_text': metadata_string, 'style_category': genres
        }

    def __call__(self, item):
        if isinstance(item, list): # perform batch transform
            return [self._call_once(i) for i in item]
        return self._call_once(item)

MCC_MOOD = ['Angry', 'Chill', 'Cute', 'Dynamic', 'Excited', 'Happy', 'Lonely', 'Romantic', 'Sorrow', 'Sweet', 'Tense', 'nan']
MCC_GENRE = ['Blues', 'Country', 'EDM', 'Jazz', 'Metal', 'New Age', 'Pop', 'R&B', 'Reggae', 'Rock', 'Trap Rap', 'nan']
MCC_VOICE = ['Female', 'Male', 'nan']

class RandomGenreTextTransform(MCCMetadataTextTransform):
    "Randomly samples genre, mood, vocals. This is for non-MCC datasets where we don't have metadata"
    def __call__(self, item):
        metadata_item = {
            'metadata': {
                'final_mood': random.choice(MCC_MOOD),
                'final_genre': random.choice(MCC_GENRE),
                'merge_aed': random.choice(MCC_VOICE),
                
            }
        }
        metadata_string = rewrite_metadata(metadata_item, type=self.metadata_type)
        return {
            **item, 'style_text': metadata_string
        }

def rewrite_metadata_categories(metadata):
    genre = metadata.get('final_genre', '')
    mood = metadata.get('final_mood', '')
    gender = metadata.get('merge_aed', '')
    if genre == 'nan' or genre is None: genre = ''
    if mood == 'nan' or mood is None: mood = ''
    if gender == 'nan' or gender is None: gender = ''
    
    if 'Female' in gender: gender = 'Female'
    elif 'Male' in gender: gender = 'Male'
    else: gender = ''

    return ','.join([genre, mood, gender]).split(',')

class StyleTextT5Transform():
    def __init__(self, max_seq_len: int=50):
        super().__init__()
        self.text_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.max_seq_len = max_seq_len

    def __call__(self, item):
        metadata_string = item['style_text']
        style_tokens = torch.LongTensor(self.text_tokenizer.encode(metadata_string, padding='max_length', max_length=self.max_seq_len))
        return {
            **item, 'style_tokens': style_tokens
        }

# Segment Transforms
class SemanticTokenLengthTransform():
    def __init__(self, sample_rate=24000, semantic_frame_rate=25, audio_key="target_audio"):
        self.sample_rate = sample_rate
        self.semantic_frame_rate = semantic_frame_rate
        self.audio_key = audio_key

    def __call__(self, item):
        target_audio = item[self.audio_key]
        audio_length = target_audio.shape[-1]
        seq_length = audio_length * self.semantic_frame_rate // self.sample_rate
        return { **item, 'target_tokens_length': seq_length }

class LyricsTokenTransform():
    def __init__(self, lyrics_tokenizer, pad_id, lyrics_max_seq_len:int=None, item_key="lyrics", normalization_fn=normalize_text, dataset_mode: str = "fixed_length", handler: Callable = wds.ignore_and_continue):
        self.lyrics_tokenizer = lyrics_tokenizer
        self.lyrics_max_seq_len = lyrics_max_seq_len
        self.item_key = item_key
        self.pad_id = pad_id
        if lyrics_max_seq_len is None: # no max sequence - switch to variable length mode
            dataset_mode = "variable_length"
        assert dataset_mode in ["fixed_length", "variable_length", "truncate_length"], "Unsupported overflow mode. Must be drop or truncate"
        self.dataset_mode = dataset_mode
        self.normalization_fn = normalization_fn
        self.handler = handler

    def tokenize(self, item, lyrics_text, **kwargs):
        token_dict = self.lyrics_tokenizer(lyrics_text, return_tensors='pt', padding=False, return_length=True, **kwargs)
        input_ids = token_dict['input_ids'].squeeze(0)
        lyrics_length = token_dict['length'].squeeze(0).item() # return int item instead of tensor
        if self.dataset_mode == "variable_length": # return length
            return { **item, 'lyrics_tokens': input_ids, 'lyrics_normalized_text': lyrics_text, 'lyrics_tokens_length': lyrics_length }
        # drop overflowed tokens
        if self.dataset_mode == "fixed_length" and lyrics_length > self.lyrics_max_seq_len:
            return None        
        # for truncate and drop, pad/crop tensor.
        input_ids = pad_crop(input_ids, self.lyrics_max_seq_len, torch.int, padding_value=self.pad_id)
        lyrics_length = self.lyrics_max_seq_len
        return { **item, 'lyrics_tokens': input_ids, 'lyrics_normalized_text': lyrics_text, 'lyrics_tokens_length': lyrics_length }
    
    def __call__(self, item):
        lyrics_text = item[self.item_key]
        try:
            if self.normalization_fn:
                lyrics_text = self.normalization_fn(lyrics_text)
            return self.tokenize(item, lyrics_text)
        except Exception as e:
            self.handler(e)
            return None

    @classmethod
    def init_cmu_tokenizer(cls, lyrics_max_seq_len, allow_unknown=False, **kwargs):
        cmu_tokenizer = CMUPhonemeTokenizer(allow_unknown=allow_unknown)
        return cls(cmu_tokenizer, cmu_tokenizer.pad_id, lyrics_max_seq_len, **kwargs)
    @classmethod
    def init_zh_tokenizer(cls, lyrics_max_seq_len, enable_punctuation=True, **kwargs):
        from transformers import BertTokenizer
        normalization_fn = partial(normalize_text, enable_punctuation=enable_punctuation)
        zh_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        zh_tokenizer.add_special_tokens({'additional_special_tokens': [" <n> "]})
        return cls(zh_tokenizer, zh_tokenizer.pad_token_id, lyrics_max_seq_len, normalization_fn=normalization_fn, **kwargs)

    @classmethod
    def init_sami_tokenizer(cls, lyrics_max_seq_len, enable_punctuation=True, normalize_tags=True, vocab_type="phoneme", **kwargs):
        from recipes.datasets.mcc.sami_tokenizer import SamiTokenizer
        normalization_fn = partial(
            normalize_text_sami_tokenizer,
            enable_punctuation=enable_punctuation,
            normalize_tags=normalize_tags
        )
        zh_phoneme_tokenizer = SamiTokenizer(vocab_type=vocab_type)
        return cls(zh_phoneme_tokenizer, 0, lyrics_max_seq_len, normalization_fn=normalization_fn, **kwargs)
    
    #推理侧使用
    @classmethod
    def init_sami_inference_tokenizer(cls, lyrics_max_seq_len, vocab_type="phoneme", **kwargs):
        from recipes.datasets.mcc.sami_tokenizer import SamiInferenceTokenizer
        normalization_fn = None
        zh_phoneme_tokenizer = SamiInferenceTokenizer(vocab_type=vocab_type)
        return cls(zh_phoneme_tokenizer, 0, lyrics_max_seq_len, normalization_fn=normalization_fn, item_key="front_results", **kwargs)
    
    @classmethod
    def init_sami_offline_tokenizer(cls, lyrics_max_seq_len, enable_punctuation=True, vocab_type="phoneme", **kwargs):
        from recipes.datasets.mcc.sami_tokenizer import SamiOfflineTokenizer
        normalization_fn = lambda x: x
        zh_phoneme_tokenizer = SamiOfflineTokenizer(vocab_type=vocab_type)
        return cls(zh_phoneme_tokenizer, 0, lyrics_max_seq_len, normalization_fn=normalization_fn, item_key="phoneme", **kwargs)

    @classmethod
    def init_espeak_tokenizer(cls, lyrics_max_seq_len, enable_punctuation=True, validate_ascii=False, **kwargs):
        def _normalize_text(text: str):
            if validate_ascii: 
                assert text.isascii(), f"Error tokenizing non-ascii lyrics: {text}"
            text = format_section_tags(text)
            return normalize_text(text, enable_punctuation=enable_punctuation)
        with local_zero_first():
            espeak_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
            espeak_tokenizer._add_tokens(["<n>", "<verse>", "<chorus>", "<intro>", "<bridge>", "<inst>", "<silence>"])
        import logging, phonemizer
        # To silence espeak logging warnings: "WARNING - words count mismatch on 100.0% of the lines"
        phonemizer.logger.get_logger().setLevel(logging.ERROR)
        return cls(espeak_tokenizer, espeak_tokenizer.pad_token_id, lyrics_max_seq_len, normalization_fn=_normalize_text, **kwargs)


class LyricsTokenSamiTransform(LyricsTokenTransform):
    """A LyricsTokenTransform that is designed for SAMI tokenizers and tag dropout for CFG"""
    def __init__(
        self,
        lyrics_tokenizer,
        pad_id,
        lyrics_max_seq_len:int=None,
        item_key="lyrics",
        normalization_fn=normalize_text,
        dataset_mode: str = "fixed_length",
        handler: Callable = wds.ignore_and_continue,
        use_controller_cfg: bool = False,
    ):
        # Only subclasses of SamiOfflineTokenizer have arugment `dropout_section_tag`
        assert issubclass(type(lyrics_tokenizer), SamiOfflineTokenizer)
        super().__init__(
            lyrics_tokenizer=lyrics_tokenizer,
            pad_id=pad_id,
            lyrics_max_seq_len=lyrics_max_seq_len,
            item_key=item_key,
            normalization_fn=normalization_fn,
            dataset_mode=dataset_mode,
            handler=handler,
        )
        self.use_controller_cfg = use_controller_cfg
    
    def tokenize(self, item, lyrics_text):
        # dropout_section_tags will be passed into the tokenizer's __call__
        out_item = super().tokenize(item, lyrics_text, dropout_section_tags=False)
        if self.use_controller_cfg:
            cfg_item = super().tokenize(item, lyrics_text, dropout_section_tags=True)
            out_item = {
                **out_item,
                "lyrics_tokens_cfg": cfg_item["lyrics_tokens"],
                "lyrics_normalized_text_cfg": cfg_item["lyrics_normalized_text"],
                "lyrics_tokens_length_cfg": cfg_item["lyrics_tokens_length"],
            }
        return out_item


class AddConditionsTransform():
    def __init__(self, conditions=""):
        self.conditions = conditions

    def __call__(self, item):
        return { **item, 'conditions': self.conditions }

class RandomConditionsTransform():
    def __init__(self, conditions):
        self.conditions = conditions

    def __call__(self, item):
        random_condition = random.choice(self.conditions)
        return { **item, 'conditions': random_condition }

class AddMulanVocalTagTransform():
    # Mix mulan requires 'vocal' tag for vocal music generation
    def __call__(self, item):
        style_text = item['style_text'] + ' vocal'
        return { **item, 'style_text': style_text }

class VocalChromaTransform():
    "Transform for conditioning on vocal chromagram"
    def __init__(self, sample_rate=24_000, sample_duration=10):
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration

    # 10sec = 59 timesteps. In general, multiple duration by 6 to get max_len
    @staticmethod
    def get_chromagram(y, sr, hop_length=2**12, n_fft=2**14, max_len=60):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        silence = chroma.min(0) > 0.5
        notes = chroma.argmax(0)
        notes[silence] = 12
        notes_padded = np.full((max_len), 12)
        timesteps = min(max_len, len(notes))
        notes_padded[:timesteps] = notes[:timesteps]
        return notes_padded
        # return np.pad(notes, pad_width=((0, 0), (0, 60-notes.shape[-1])), mode='constant', constant_values=(12, 12))
    
    def __call__(self, item):
        if item is None or 'vocal_audio' not in item: return item
        cropped_vocals = item['vocal_audio']
        vocal_chroma = VocalChromaTransform.get_chromagram(cropped_vocals.numpy(), self.sample_rate, max_len=int(self.sample_duration*6))
        return { **item, 'vocal_chroma': vocal_chroma }


class AddDurationTransform():
    def __init__(self, duration):
        assert isinstance(duration, int)
        self.duration = duration

    def __call__(self, item):
        if 'duration' in item:
            durations = set()
            for d in item['duration']:
                durations.add(d.cpu().item() if isinstance(d, torch.Tensor) else d)
            assert len(durations) == 1, f"Duration in batch must be the same, got: {durations}"
            item['duration'] = list(durations)[0]
            return item
        else:
            return { **item, 'duration': self.duration }
