from typing import Any, Dict, List, Generator, Optional, Callable
import io
import torch
import webdataset as wds
from torchaudio_augmentations import Compose
from dataclasses import dataclass
import math
import librosa
import numpy as np

from recipes.musiclm.transforms.audio import (
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
    ReadMP3,
    crop_1d,
    NormalizeAudio
)
from recipes.musiclm.transforms.base import TransformBase

class LyricsTransforms(TransformBase):
    def __init__(
        self,
        sample_rate: int,
        sample_duration: int,
        audio_format: str,
        audio_keys: dict,
        segment_transforms: list = None,
        fixed_duration = True,
        max_num_segments = None, 
        url2index = None,
        handler: Callable = wds.ignore_and_continue,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.audio_keys = audio_keys
        self.sample_duration = sample_duration
        self.fixed_duration = fixed_duration
        self.handler = handler
        self.max_num_segments = max_num_segments

        self.read_mp3 = ReadMP3(self.sample_rate, self.audio_format)
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
        if segment_transforms is None:
            self.segment_transform = lambda x: x # identity
        else:
            self.segment_transform = Compose(segment_transforms)

    def process_segment(self, segment, audio_wavs):
        audio_duration = audio_wavs['target_audio'].shape[-1] // self.sample_rate
        duration = self.sample_duration if self.fixed_duration else None
        if segment.end > audio_duration + 2:
            return None

        cropped_segments = { name: crop_audio_to_segment(segment, audio, self.sample_rate, duration) for name, audio in audio_wavs.items() }
        lyrics_text = segment.text.strip()

        return {
            "target_audio": cropped_segments['target_audio'],
            "mulan_audio": cropped_segments.get('mulan_audio'),
            'vocal_audio': cropped_segments.get('vocal_audio'),
            "lyrics": lyrics_text
        }

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
        lyrics_json = x['__index_data__']['lyrics']['utterances']
        segments: List[Segment] = lyrics_to_segments(lyrics_json, maximum_clipped_length=self.sample_duration, fixed_duration=self.fixed_duration)
        segment_count = 0

        # Output clips
        for segment in segments:
            item = self.process_segment(segment, audio_wavs)
            item = self.segment_transform(item)
            if item is None: continue

            yield item
            segment_count += 1
            if self.max_num_segments and segment_count >= self.max_num_segments:
                break
        self._update_stats(segment_count == 0)

def crop_audio_to_segment(segment, audio, sample_rate, duration=None):
    sample_start = int(segment.start*sample_rate)
    if duration is not None:
        num_samples = int(duration*sample_rate)
    else:
        num_samples = int((segment.end-segment.start)*sample_rate)

    # must min to audio size, as segments pads +2 seconds
    num_samples = min(audio.size(-1)-sample_start, num_samples)
    if len(audio.shape) == 1:
        return audio[sample_start : sample_start + num_samples]
    cropped_audio = crop_1d(audio, sample_start, num_samples)
    return cropped_audio

class LyricsTokenTransform():
    def __init__(self, lyrics_tokenizer, lyrics_max_seq_len: int, handler: Callable = wds.ignore_and_continue):
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
        return { **item, 'lyrics_tokens': lyrics_tokens }

class LyricsChromaTransform():
    "Transform for conditioning on vocal chromagram"
    def __init__(self, sample_rate=24_000, sample_duration=10):
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration

    # 10sec = 59 timesteps. In general, multiple duration by 6 to get max_len
    @classmethod
    def get_chromagram(cls, y, sr, hop_length=2**12, n_fft=2**14, max_len=60):
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
        cropped_vocals = item['vocal_audio']
        vocal_chroma = LyricsChromaTransform.get_chromagram(cropped_vocals.numpy(), self.sample_rate, max_len=int(self.sample_duration*6))
        return { **item, 'vocal_chroma': vocal_chroma }

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
    while words and words[-1]['start_time'] < 0:
        words.pop()
    while words and words[0]['start_time'] < 0:
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

