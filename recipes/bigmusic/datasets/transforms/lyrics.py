from importlib.metadata import metadata
from typing import Any, Dict, List, Generator, Optional, Callable
import io
import torch
import webdataset as wds
from torchaudio_augmentations import Compose
from dataclasses import dataclass
import math
import librosa
import numpy as np
from transformers import T5Tokenizer
from recipes.bigmusic.datasets.tokenizers.cmu_phonemes import CMUPhonemeTokenizer
from transformers import Wav2Vec2PhonemeCTCTokenizer
from recipes.musiclm.utils.dist import local_zero_first

def pad_crop(sequence, seq_len, dtype, padding_value=0):
    item_pad = torch.full((seq_len,), fill_value=padding_value, dtype=dtype)
    item_pad[:len(sequence)] = torch.as_tensor(sequence[:seq_len])
    return item_pad

class MCCInstrumentalBatchTransform():
    "Converts musiclm dataloader to work with bigmusic models"
    def __call__(self, item):
        wavs = item.pop('audio')
        if len(wavs.shape) == 3:
            wavs = wavs.squeeze(1)
        return { **item, 'style_audio': wavs, 'target_audio': wavs }
    

class MCCMetadataTextTransform():
    def _mcc_metadata_to_string(self, item, type="Vocal"):
        metadata = item['metadata']
        mood = metadata.get('final_mood')
        genre = metadata.get('final_genre')
        gender = metadata.get('merge_aed')
        text = ""
        if type == "Vocal":
            text = "A "
            if mood is not None and mood != 'nan':
                text += mood.lower() + " "
            if genre is not None and genre != 'nan':
                text += genre.lower() + " "
            text += "song"
            if gender is not None and gender != 'nan':
                if 'Female' in gender:
                    text += " with female vocal"
                elif 'Male' in gender:
                    text += " with male vocal"
            text += "."
        elif type == "Instrumental":
            text = ""
            if mood is not None and mood != 'nan':
                text += mood.lower() + " "
            if genre is not None and genre != 'nan':
                text += genre.lower() + " "
            text += "music."
        return text

    def __call__(self, item):
        metadata_string = self._mcc_metadata_to_string(item)
        return {
            **item, 'style_text': metadata_string
        }

class MetadataT5Transform(MCCMetadataTextTransform):
    def __init__(self, max_seq_len: int=50):
        self.text_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.max_seq_len = max_seq_len

    def __call__(self, item):
        if 'style_text' in item:
            metadata_string = item['style_text']
        elif 'metadata' in item:
            metadata_string = self._mcc_metadata_to_string(item)
        else:
            # Could not encode metadata. Return original item
            return item
        style_tokens = torch.LongTensor(self.text_tokenizer.encode(metadata_string, padding='max_length', max_length=self.max_seq_len))
        return {
            **item, 'style_tokens': style_tokens, 'style_text': metadata_string
        }

# Segment Transforms
class LyricsTokenTransform():
    def __init__(self, lyrics_tokenizer, pad_id, lyrics_max_seq_len: int, handler: Callable = wds.ignore_and_continue):
        self.lyrics_tokenizer = lyrics_tokenizer
        self.lyrics_max_seq_len = lyrics_max_seq_len
        self.pad_id = pad_id
        self.handler = handler

    def __call__(self, item):
        try:
            lyrics_text = item['lyrics']
            lyrics_tokens = self.lyrics_tokenizer(lyrics_text)['input_ids']
            if len(lyrics_tokens) > self.lyrics_max_seq_len:
                return None
        except Exception as e:
            self.handler(e)
            return None
        
        lyrics_tokens = pad_crop(torch.tensor(lyrics_tokens), self.lyrics_max_seq_len, torch.int, padding_value=self.pad_id)
        return { **item, 'lyrics_tokens': lyrics_tokens }

    @classmethod
    def init_cmu_tokenizer(cls, lyrics_max_seq_len, allow_unknown=False, **kwargs):
        cmu_tokenizer = CMUPhonemeTokenizer(allow_unknown=allow_unknown)
        return LyricsTokenTransform(cmu_tokenizer, cmu_tokenizer.pad_id, lyrics_max_seq_len, **kwargs)

    @classmethod
    def init_espeak_tokenizer(cls, lyrics_max_seq_len, **kwargs):
        with local_zero_first():
            espeak_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
            espeak_tokenizer._add_tokens(["<n>"])
        import logging, phonemizer
        # To silence espeak logging warnings: "WARNING - words count mismatch on 100.0% of the lines"
        phonemizer.logger.get_logger().setLevel(logging.ERROR)
        return LyricsTokenTransform(espeak_tokenizer, espeak_tokenizer.pad_token_id, lyrics_max_seq_len, **kwargs)


class AddConditionsTransform():
    def __init__(self, conditions=""):
        self.conditions = conditions

    def __call__(self, item):
        return { **item, 'conditions': self.conditions }

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
