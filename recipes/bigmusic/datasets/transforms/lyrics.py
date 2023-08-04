from typing import Any, Dict, List, Generator, Optional, Callable
import io
import torch
import webdataset as wds
from torchaudio_augmentations import Compose
from dataclasses import dataclass
import math
import librosa
import numpy as np
from recipes.bigmusic.datasets.tokenizers.phoneme_tokenizer import PhonemeTokenizer
from transformers import T5Tokenizer

class MCCInstrumentalBatchTransform():
    "Converts musiclm dataloader to work with bigmusic models"
    def __call__(self, item):
        wavs = item.pop('audio')
        if len(wavs.shape) == 3:
            wavs = wavs.squeeze(1)
        return { **item, 'mulan_audio': wavs, 'target_audio': wavs }
    
class MetadataT5Transform():
    def __init__(self, max_seq_len: int=50):
        self.text_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.max_seq_len = max_seq_len

    def _mcc_metadata_to_string(self, item):
        metadata = item['metadata']
        fields = {
            'final_genre': 'genre',
            'final_mood': 'mood',
            'final_theme': 'theme',
        }
        metadata_string = ""
        for k in fields.keys():
            if k not in metadata: continue
            val = metadata[k]
            if val is None or val == 'nan': continue
            metadata_string += f'{fields[k]}: {metadata[k]} '
        return metadata_string

    def __call__(self, item):
        metadata_string = self._mcc_metadata_to_string(item['metadata'])
        metadata_tokens = torch.LongTensor(self.text_tokenizer.encode(metadata_string, padding='max_length', max_length=150))
        return {
            **item, 'metadata_tokens': metadata_tokens, 'metadata_text': metadata_string
        }
class MetadataMulanTextTransform():
    def _mcc_metadata_to_string(self, item):
        metadata = item['metadata']
        keys = ['final_genre', 'final_mood', 'final_theme']
        fields = []
        for k in keys:
            val = metadata.get(k)
            if val is None or val == 'nan': continue
            fields.extend(val.split(','))
        return ' '.join(fields)

    def __call__(self, item):
        metadata_string = self._mcc_metadata_to_string(item)
        return {
            **item, 'text': metadata_string
        }
class MetadataMulanTextTransformVocalTag(MetadataMulanTextTransform):
    def __call__(self, item):
        metadata_string = self._mcc_metadata_to_string(item)
        metadata_string = 'vocal ' + metadata_string
        return {
            **item, 'mulan_text': metadata_string
        }
    
# Segment Transforms
class LyricsTokenTransform():
    def __init__(self, lyrics_max_seq_len: int, lyrics_tokenizer=None, allow_unknown=False, handler: Callable = wds.ignore_and_continue):
        if lyrics_tokenizer is None:
            lyrics_tokenizer = PhonemeTokenizer(allow_unknown=allow_unknown)
        self.lyrics_tokenizer = lyrics_tokenizer
        self.lyrics_max_seq_len = lyrics_max_seq_len
        self.handler = handler

    def __call__(self, item):
        try:
            lyrics_text = item['lyrics']
            lyrics_tokens = self.lyrics_tokenizer(lyrics_text)
            if len(lyrics_tokens) > self.lyrics_max_seq_len:
                return None
        except Exception as e:
            self.handler(e)
            return None
        num_pad = self.lyrics_max_seq_len - len(lyrics_tokens)
        if num_pad > 0:
            pad_id = self.lyrics_tokenizer.pad_id
            lyrics_tokens += [pad_id for _ in range(num_pad)]
        else: 
            lyrics_tokens = lyrics_tokens[:self.lyrics_max_seq_len]
        return { **item, 'lyrics_tokens': torch.LongTensor(lyrics_tokens) }    

class AddConditionsTransform():
    def __init__(self, conditions=""):
        self.conditions = conditions

    def __call__(self, item):
        return { **item, 'conditions': self.conditions }

class AddMulanVocalTagTransform():
    # Mix mulan requires 'vocal' tag for vocal music generation
    def __call__(self, item):
        mulan_text = 'vocal ' + item['mulan_text'] 
        return { **item, 'mulan_text': mulan_text }

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
