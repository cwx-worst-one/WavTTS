from typing import Any, Dict, List, Generator, Optional, Callable
import io
import torch
import webdataset as wds
from torchaudio_augmentations import Compose
from dataclasses import dataclass
import math
import random

from recipes.musiclm.transforms.audio import (
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
    ReadMP3,
    crop_1d,
    NormalizeAudio
)
from recipes.musiclm.transforms.base import TransformBase
from recipes.bigmusic.datasets.tokenizers.phoneme_tokenizer import PhonemeTokenizer
from transformers import T5Tokenizer

class LyricsSegmentTransforms(TransformBase):
    def __init__(
        self,
        sample_rate: int,
        sample_duration: int,
        audio_format: str,
        audio_keys: dict,
        max_num_segments = 6, 
        shuffle_segments: bool = True,
        url2index = None,
        handler: Callable = wds.ignore_and_continue,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.audio_keys = audio_keys
        self.sample_duration = sample_duration
        self.handler = handler
        self.max_num_segments = max_num_segments
        self.shuffle_segments = shuffle_segments

        self.read_mp3 = ReadMP3(self.sample_rate, self.audio_format, fast=(self.audio_format=='mp3'))
        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.url2index = url2index
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [
                self.read_mp3,
                self.to_tensor,
                self.audio_dim, 
                self.normalize_audio_fp32,
            ]
        )

    def process_segment(self, segment, audio_wavs):
        audio_duration = audio_wavs['target_audio'].shape[-1] // self.sample_rate
        target_duration = self.sample_duration
        if segment.end > audio_duration + 2:
            return None

        cropped_segments = { name: crop_audio_to_segment(segment, audio, self.sample_rate, target_duration) for name, audio in audio_wavs.items() }
        lyrics_text = segment.text.strip()
        cropped_segments['lyrics'] = lyrics_text
        return cropped_segments

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        # Extract audio Wavs
        cached_wavs = {}
        audio_wavs = {}
        for name, audio_key in self.audio_keys.items():
            if audio_key in cached_wavs:
                audio = cached_wavs[audio_key]
            else:
                try:
                    audio = self.base_transform(io.BytesIO(x[audio_key]))[0, :]
                    cached_wavs[audio_key] = audio
                    assert len(audio.shape) == 1, 'Invalid audio shape'
                except Exception as e:
                    print(f"[MP3 decoding error] - {name} - {self.url2index} - {e}")
                    self._update_stats(skipped=True)
                    self.handler(e)
                    return
            audio_wavs[name] = audio
        del cached_wavs

        # Hack: Resso MSS does not contain mixture. For mixture data, we must add acc + vocals
        if 'target_audio' not in audio_wavs:
            assert 'mulan_audio' in audio_wavs and 'vocal_audio' in audio_wavs, 'Must provide target audio or mulan and vocal audio'
            audio_wavs['target_audio'] = audio_wavs['mulan_audio'] + audio_wavs['vocal_audio']
        
        # Normalize wavs based on target_audio
        target_audio = audio_wavs['target_audio']
        for name, audio_wav in audio_wavs.items():
            audio_wavs[name] = self.normalize_audio(audio_wav, target_audio)

        # Extract segment information
        index_data = x['__index_data__']
        lyrics_json = index_data['lyrics']['utterances']
        segments: List[Segment] = lyrics_to_segments(lyrics_json, maximum_clipped_length=self.sample_duration)
        if self.shuffle_segments:
            random.shuffle(segments)
        segment_count = 0

        # Output clips
        for segment in segments:
            item = self.process_segment(segment, audio_wavs)
            if item is None: continue
            if 'metadata' in index_data:
                item['metadata'] = index_data['metadata']
            yield item
            segment_count += 1
            if self.max_num_segments and segment_count >= self.max_num_segments:
                break
        self._update_stats(segment_count == 0)

def crop_audio_to_segment(segment, audio, sample_rate, target_duration):
    sample_start = int(segment.start*sample_rate)
    target_samples = int(target_duration*sample_rate)
    assert len(audio.shape) == 1, 'Invalid audio shape'
    return pad_to_seq_length(audio[sample_start:], target_samples, audio.dtype)

def pad_to_seq_length(seq, seq_len, dtype, padding_value=0):
    pad_x = torch.full((seq_len,), fill_value=padding_value, dtype=dtype)
    pad_x[:seq.size(0)] = seq[:seq_len]
    return pad_x

# Segment clipping

@dataclass
class Segment():
    start: float
    end: float
    text: str
    duration: float
        
    @classmethod
    def from_dict(cls, json_dict):
        start = json_dict['start_time']/1000.0
        end = json_dict['end_time']/1000.0
        duration = end - start
        if json_dict['start_time'] == -2:
            start = -2
            end = -2
            duration = 0
        text = json_dict['text']
        return Segment(start, end, text, duration)

    def has_valid_time(self):
        return self.start >= 0

def lyrics_to_segments(lyrics, maximum_clipped_length=10, minimum_voice_duration=3, 
                       new_line_token=" <n> ", fixed_duration=True):
    if not lyrics: return []
    if 'start_time' not in lyrics[0]:
        # convert force alignment lyrics to line format
        lyrics = word_format_to_line_format(lyrics)
        
    segments = []
    segment = None

    lyrics.append({
        'start_time': 10000*1000, 
        'end_time': 10001*1000,
        'text': 'STOP_PLACEHOLDER', 
        'words': [],
    })
    for i, current_diction in enumerate(lyrics):
        current_segment = Segment.from_dict(current_diction)
        if len(current_segment.text.strip()) == 0: continue
        if current_segment.start < 0: continue
            
        # Case #1: overflow. Append segment. Create new
        if segment and (current_segment.end - segment.start > maximum_clipped_length):
            if fixed_duration:
                # append words from current segment.
                target_end_time = segment.start + maximum_clipped_length
                extended_segment, _ = _words_to_segment(current_diction['words'], segment.start, target_end_time)
                if extended_segment:
                    segment.end = extended_segment.end
                    segment.text += new_line_token + extended_segment.text
                    segment.duration += extended_segment.duration                    
                segment.end = target_end_time
            else:
                segment.end = min(segment.start + maximum_clipped_length, current_segment.start)

            
            if segment.duration > minimum_voice_duration:
                segments.append(segment)

            segment = None
        
        # If current segment is long. skip: really long line. Break into multiple segments
        if current_segment and current_segment.duration > maximum_clipped_length:
            cached_index = 0
            for i in range(math.ceil(current_segment.duration / maximum_clipped_length)):
                start = current_segment.start + i * maximum_clipped_length
                end = current_segment.start + (i+1) * maximum_clipped_length
                clipped_segment, cached_index = _words_to_segment(current_diction['words'], start, end, cached_index)
                if clipped_segment and clipped_segment.duration <= maximum_clipped_length and clipped_segment.duration > minimum_voice_duration:
                    segments.append(clipped_segment)

            # reset everything
            segment = None
            current_segment = None

        if segment is None:
            segment = current_segment
        else:
            segment.end += current_segment.end
            segment.text = segment.text + new_line_token + current_segment.text
            segment.duration += current_segment.duration
    return segments

def _words_to_segment(words, start_time, end_time, start_index=0):
    segment = None
    end_index = start_index
    idx = 0

    for idx, word in enumerate(words[start_index:]):
        word = Segment.from_dict(word)
        end_index = start_index + idx
        if word.start >= 0 and word.start < start_time: # keep words < 0 as those are punctuation
            continue
        if word.end > end_time:
            break

        if segment:
            if word.has_valid_time():
                segment.end = word.end
                segment.duration = segment.end - segment.start
            segment.text += word.text
        elif segment is None and word.has_valid_time():
            segment = word
    return segment, end_index-1

# Converts word format (outputs from force alignment), to line-by-line format
PAD_TIME = -2
def _strip_non_words(words):
    words = words.copy()
    while words and (words[-1]['start_time'] < 0 or words[-1]['confidence'] < 0.05):
        words.pop()
    while words and (words[0]['start_time'] < 0 or words[0]['confidence'] < 0.05):
        words.pop(0)
    return words

def word_format_to_line_format(lyrics):
    words = lyrics[0]['words'].copy()
    words.append({ 'text': '\n' }) # dummy placeholder
    lines = []
    current_line = []
    for i, word_json in enumerate(words):
        lyrics = word_json['text']
        if '\n' in lyrics:
            current_line = _strip_non_words(current_line)
            if len(current_line) == 0: continue
            line = {
                'start_time': current_line[0]['start_time'],
                'end_time': current_line[-1]['end_time'],
                'text': ''.join([w['text'] for w in current_line]),
                'words': current_line
            }
            lines.append(line)
            current_line = []
        else:
            current_line.append(word_json)
    return lines

