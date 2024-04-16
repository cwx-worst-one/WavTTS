from typing import Any, Dict, List, Generator, Optional, Callable, Tuple
import io
import torch
import webdataset as wds
from torchaudio_augmentations import Compose
from dataclasses import dataclass
import math
import random
import numpy as np
import json
from string import punctuation, whitespace
from copy import deepcopy
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
)
from recipes.musiclm.transforms.audio import (
    ReadMP3,
    FastNormalizeAudio
)
from recipes.musiclm.transforms.base import TransformBase
from recipes.bigmusic.datasets.transforms.lyrics_section_alignment import song_structure_from_metadata, group_by_section_start, get_section_aligned_segment
from recipes.bigmusic.datasets.transforms.lyrics_filters import is_invalid_song

import operator
from functools import reduce
import traceback

punctuation_whitespace = set(punctuation + whitespace)

class ReadMP3BytesIO(ReadMP3):
    def __call__(self, mp3: bytes) -> np.ndarray:
        return super().__call__(io.BytesIO(mp3))

class LyricsSegmentTransforms(TransformBase):
    def __init__(
        self,
        sample_rate: int,
        sample_duration: list,
        audio_format: str,
        audio_keys: dict,
        max_num_segments = 10,
        shuffle_segments: bool = True,
        url2index = None,
        min_song_confidence: int=0.8,
        min_segment_confidence: int=0.8,
        handler: Callable = wds.warn_and_continue,
        normalize_audio: bool = False
    ) -> None:
        super().__init__(log_interval=100)
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.audio_keys = audio_keys
        self.url2index = url2index
        if not isinstance(sample_duration, list):
            self.sample_duration = [sample_duration]
        else:
            self.sample_duration = list(sorted(sample_duration))
        self.handler = handler
        self.max_num_segments = max_num_segments
        self.shuffle_segments = shuffle_segments
        self.min_song_confidence = min_song_confidence
        self.min_segment_confidence = min_segment_confidence
        self.normalize_audio = normalize_audio

        if self.audio_format == 'npy':
            read_mp3 = lambda x: x
        else:
            read_mp3 = ReadMP3BytesIO(self.sample_rate, self.audio_format, fast=(self.audio_format=='mp3'))

        self.base_transform = Compose(
            [
                read_mp3,
                ToTensor(),
                SetAudioDimensions(), 
                NormalizeAudioToFloat32(),
            ]
        )
        self.normalize_audio_tfm = FastNormalizeAudio()

    def process_segment(self, segment: 'Segment', audio_wavs):
        audio_duration = audio_wavs['target_audio'].shape[-1] / self.sample_rate
        start, end, duration = segment.get_instrumental_range(max(self.sample_duration))
        sample_start = int(start*self.sample_rate)
        target_samples = int(duration*self.sample_rate)
        if math.ceil(segment.end) > math.ceil(audio_duration) or target_samples < 0:
            raise Exception(f"Invalid segment split for segment {segment}, audio: {audio_duration}, samples: {target_samples}")
        cropped_segments = { name: crop_pad_to_seq_length(audio[sample_start:], target_samples, audio.dtype) for name, audio in audio_wavs.items() }
        if cropped_segments['target_audio'].sum() == 0:
            raise Exception("Processed segment is silent, skipping segment")
        lyrics_text = segment.text.strip()
        if lyrics_text is None: return None
        if lyrics_text and segment.phoneme is None: return None
        cropped_segments['lyrics'] = lyrics_text
        cropped_segments['phoneme'] = segment.phoneme
        if cropped_segments['phoneme'] is None:
            cropped_segments['phoneme'] = ''
        return cropped_segments

    def extract_audio_wavs(self, x:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cached_wavs = {}
        audio_wavs = {}
        for name, audio_key in self.audio_keys.items():
            if audio_key in cached_wavs:
                audio = cached_wavs[audio_key]
            else:
                try:
                    audio = self.base_transform(x[audio_key])[0, :]
                except Exception as e:
                    raise Exception(f"[MP3 decoding error] - {name} - {self.url2index} - {e}")
                cached_wavs[audio_key] = audio
                assert len(audio.shape) == 1, 'Invalid audio shape'
                assert audio.shape[0] > self.sample_rate
            audio_wavs[name] = audio
        del cached_wavs

        # Hack: Resso MSS does not contain mixture. For mixture data, we must add acc + vocals
        if 'target_audio' not in audio_wavs:
            assert 'style_audio' in audio_wavs and 'vocal_audio' in audio_wavs, 'Must provide target audio or mulan and vocal audio'
            audio_wavs['target_audio'] = audio_wavs['style_audio'] + audio_wavs['vocal_audio']
        
        # Normalize wavs based on target_audio
        if self.normalize_audio:
            target_audio = audio_wavs['target_audio']
            for name, audio_wav in audio_wavs.items():
                audio_wavs[name] = self.normalize_audio_tfm(audio_wav, target_audio)
        return audio_wavs


    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        # 1. filter lyrics and metadata. 2. extract audio 2b. transform audio 3. clip audio to metadata 4. transform segment
        try:
            metadata, lyrics = extract_metadata_and_utterances(x)
        except Exception as e:
            self._update_stats(skipped=True, message="Error extracting metadata")
            self.handler(e)
            return
        
        is_invalid, reason = is_invalid_song(metadata, lyrics, confidence_threshold=self.min_song_confidence)
        if is_invalid:
            self._update_stats(skipped=True, message=reason)
            return

        # grouping = 'fixed_length' if len(self.sample_duration) == 1 else 'variable_length'
        grouping = 'fixed_length' if len(self.sample_duration) == 1 else 'section_start'
        segments: List[Segment] = lyrics_to_segments(lyrics, metadata, 
            target_durations=self.sample_duration,
            shuffle_start=self.shuffle_segments,
            min_confidence=self.min_segment_confidence,
            grouping=grouping
        )
        if self.shuffle_segments:
            random.shuffle(segments)
        segment_count = 0

        if len(segments) == 0: 
            self._update_stats(skipped=True, message="Filter low confident segments. 0 segments above threshold")
            return
        try:
            audio_wavs = self.extract_audio_wavs(x)
            # Extract segment information
        except Exception as e:
            self._update_stats(skipped=True, message="Error extracting audio.")
            self.handler(e)
            return
        
        # Output clips
        for segment in segments:
            try:
                item = self.process_segment(segment, audio_wavs)
            except Exception as e:
                print('Exception in process_segment', e)
                print(traceback.format_exc())
                self.handler(e)
                continue
            if item is None: continue
            item['metadata'] = metadata
            yield item
            segment_count += 1
            if self.max_num_segments and segment_count >= self.max_num_segments:
                break
        if segment_count == 0:
            self._update_stats(skipped=True, message="Error processing segments. 0 segments processed")
        else:
            self._update_stats(skipped=False)

def extract_metadata_and_utterances(item):
    # webdataset case
    if '__index_data__' in item:
        index_data = item['__index_data__']
        if 'metadata' in index_data:
            metadata = index_data['metadata']
        else:
            metadata = index_data
    # parquet case
    elif 'meta' in item:
        index_data = json.loads(item['meta'])
        metadata = index_data
    else:
        raise Exception('Unable to locate metadata')

    # # special case for spotify genre 3733 lyrics. where force aligned lyrics gives wrong confidence
    if 'downloaded_lyrics' in metadata:
        total_confidence = 1
        try: total_confidence = metadata['lyrics']['result'][0]['confidence']
        except: pass
        if total_confidence > 100:
            lyrics = metadata.pop('lyrics')
            metadata['is_lyrics_gt'] = True
            utterances = lyrics['result'][0]['utterances']
            words = line_format_to_force_aligned_word_format(utterances)
            lyrics = force_aligned_word_format_to_line_format([{'words': words}])
            return metadata, lyrics

    # special case for gt lyrics (9M gt)
    if 'lyrics_gt' in metadata:
        lyrics = metadata.pop('lyrics_gt')
        metadata['is_lyrics_gt'] = True
        if 'utterances' in lyrics:
            lyrics = force_aligned_word_format_to_line_format(lyrics['utterances'])
            return metadata, lyrics
        return metadata, lyrics
    
    if 'lyrics' in metadata:
        lyrics = metadata.pop('lyrics')
    elif 'lyrics' in index_data:
        lyrics = index_data.pop('lyrics')
    else:
        lyrics = None

    utterances = None
    if lyrics and 'utterances' in lyrics:
        # v1 (asr, no punctuation)
        utterances = lyrics['utterances']
    elif lyrics and 'result' in lyrics:
        # v2 (asr + punctuation)
        utterances = lyrics['result'][0]['utterances']

    # Hacky fix: Billboard-V2 has fields stored outside metadata.
    for k in ['tags', 'genre', 'style']:
        if k in index_data:
            metadata[k] = index_data[k]

    return metadata, utterances
    
def crop_pad_audio_to_segment(segment: 'Segment', audio, sample_rate, target_duration=None):
    sample_start = int(segment.start*sample_rate)
    if target_duration is None:
        target_samples = int(segment.duration*sample_rate)
    else:
        target_samples = int(target_duration*sample_rate)
    assert len(audio.shape) == 1, 'Invalid audio shape'
    return crop_pad_to_seq_length(audio[sample_start:], target_samples, audio.dtype)

def crop_pad_to_seq_length(seq: torch.Tensor, target_seq_len, dtype=None, padding_value=0, start=0):
    *dims, input_seq_len = seq.shape
    dtype = seq.dtype if dtype is None else dtype
    pad_x = torch.full((*dims, target_seq_len), fill_value=padding_value, dtype=dtype, device=seq.device)
    seq_slice = seq[..., start:start+target_seq_len]
    pad_x[..., :seq_slice.shape[-1]] = seq_slice
    return pad_x

def random_crop_pad_to_seq_length(seq: torch.Tensor, target_seq_len, dtype=None, padding_value=0):
    *dims, input_seq_len = seq.shape
    start_idx = random.randint(0, max(0, input_seq_len - target_seq_len))
    return crop_pad_to_seq_length(seq, target_seq_len=target_seq_len, dtype=dtype, padding_value=padding_value, start=start_idx)

@dataclass
class Segment():
    start: float
    end: float
    text: str
    confidences: List[float]
    words: List[str]
    phoneme: str
    inst_start: float = None
    inst_end: float = None
    target_duration: float = None

    @classmethod
    def from_utt(cls, json_dict):
        if json_dict['start_time'] is None:
            start, end = None, None
        elif json_dict['start_time'] == -2:
            start, end = -2, -2
        else:
            start = json_dict['start_time']/1000.0
            end = json_dict['end_time']/1000.0
        text = json_dict['text']
        if 'additions' in json_dict:
            confidences = [float(json_dict['additions']['confidence'])]
        elif 'confidence' in json_dict:
            confidences = [float(json_dict['confidence'])]
        elif 'attribute' in json_dict:
            confidences = [float(json_dict['attribute']['confidence'])]
        else:
            confidences = [1]
        words = deepcopy(json_dict.get('words', []))
        if words is None:
            words = []
            print('WARNING: words is None. Double check if lyrics exists', text)
        phoneme = json_dict.get('phoneme', '')
        return Segment(start, end, text, confidences, words, phoneme, start, end)

    @classmethod
    def from_word_segment(cls, ws: 'WordSegment'):
        return Segment(ws.start, ws.end, ws.text, ws.confidences, ws.words, ws.phoneme, ws.inst_start, ws.inst_end)

    @classmethod
    def segments_from_utterances(cls, utterances: List['Segment'], song_duration=None):
        segments = [Segment.from_utt(utt) for utt in utterances]
        valid_segments = [s for s in segments if s.text.strip()]
        Segment.update_segments_with_instrumental_timings(valid_segments, song_duration)
        return valid_segments

    @classmethod
    def update_segments_with_instrumental_timings(cls, segments, song_duration=None):
        for idx in range(len(segments)):
            s_cur = segments[idx]
            s_cur.inst_start = segments[idx-1].end if idx > 0 else 0
            if idx < len(segments)-1:
                s_cur.inst_end = segments[idx+1].start
            elif song_duration is not None:
                s_cur.inst_end = song_duration
            else: # inst_end is already set to end
                pass
            # Note: we should technically set the word timings, but ignoring that uncommon case for now

    def get_instrumental_range(self, target_duration=None, limit_duration=8):
        if target_duration is None: target_duration = self.target_duration
        if self.inst_start is None or self.inst_end is None or target_duration is None:
            return self.start, self.end, self.duration

        start = self.start
        end = self.end
        inst_start = min(self.inst_start, self.start)
        inst_end = max(self.inst_end, self.end)
        if limit_duration is not None:
            inst_start = max(inst_start, start - limit_duration)
            inst_end = min(inst_end, end + limit_duration)
        
        # target_duration = self.target_duration
        min_start = max(end - target_duration, inst_start) # -8 (-8, 0.)  t = 30. 0 (17, 0) t = 5, 
        min_start = min(min_start, start)
        max_start = start
        random_start = random.uniform(min_start, max_start)
        random_end = min(random_start + target_duration, inst_end)
        random_end = max(end, random_end)
        assert min_start >= inst_start and max_start <= start
        assert math.ceil(random_end) >= math.ceil(end) and math.ceil(random_end) <= math.ceil(inst_end)
        return random_start, random_end, random_end-random_start

    @property
    def duration(self): return self.end - self.start
    
    @property
    def confidence(self): 
        cs = [c for c in self.confidences if c > 0]
        if not cs: return 0
        return sum(cs) / len(cs)

    def has_valid_time(self):
        return self.start >= 0
    
    def __repr__(self) -> str:
        return f"[{self.start} - {self.end}] ({round(self.duration, 2)}, {round(self.confidence, 2)}) - {self.text}"
    
    def __add__(self, other: 'Segment', new_line_token=" <n> "):
        if self.phoneme is None or other.phoneme is None:
            phoneme = None
        else:
            phoneme = self.phoneme + new_line_token + other.phoneme
        return Segment(
            start=self.start,
            end=other.end,
            text=self.text + "\n" + other.text, # TODO: figure out if \n or <n> is better
            # text=self.text + new_line_token + other.text,
            confidences=self.confidences + other.confidences,
            words=self.words + [WordSegment.new_line_dict()] + other.words,
            phoneme=phoneme,
            inst_start=self.inst_start,
            inst_end=other.inst_end
        )

    def filter_invalid_words(self, words):
        words = words.copy()
        while words and (words[-1].start < 0):
            words.pop()
        while words and (words[0].start < 0):
            words.pop(0)
        return words

    def split_at(self, duration) -> Tuple['Segment', 'Segment']:
        if self.duration <= duration: return deepcopy(self), None
        words = [WordSegment.from_word_dict(word) for word in self.words] # TODO: set left right instrumental
        valid_words = self.filter_invalid_words(words)
        target_time = self.start + duration

        for idx, word in enumerate(valid_words):
            if word.end > target_time: break
        target_segment = Segment.from_word_segment(reduce(operator.add, self.filter_invalid_words(valid_words[:idx])))
        remaining_segment = Segment.from_word_segment(reduce(operator.add, self.filter_invalid_words(valid_words[idx:]))) if len(valid_words[idx:]) else None
        return target_segment, remaining_segment

    def split_to_segments(self, target_duration):
        long_segment = deepcopy(self)
        split_segments = []
        while long_segment.duration > target_duration:
            target_segment, long_segment = long_segment.split_at(target_duration)
            split_segments.append(target_segment)
            if long_segment is None: break
        return split_segments


class WordSegment(Segment):
    @staticmethod
    def new_line_dict():
        return {'text': "\n",
        'confidence': 0.0,
        'start_time': 0,
        'end_time': 0,
        'phoneme': " <n> ",
        'normalized_text': None}

    @classmethod
    def from_word_dict(cls, json_dict):
        if json_dict['start_time'] is None:
            start, end = None, None
        elif json_dict['start_time'] == -2:
            start, end = -2, -2
        else:
            start = json_dict['start_time']/1000.0
            end = json_dict['end_time']/1000.0

        text = json_dict['text']
        if 'additions' in json_dict:
            confidences = [float(json_dict['additions']['confidence'])]
        elif 'confidence' in json_dict:
            confidences = [float(json_dict['confidence'])]
        elif 'attribute' in json_dict:
            confidences = [float(json_dict['attribute']['confidence'])]
        else:
            confidences = [1]
        words = [json_dict] # for words, we set it to itself
        phoneme = json_dict.get('phoneme', '')
        return WordSegment(start, end, text, confidences, words, phoneme, start, end)

    def is_punctuation(self):
        is_punc = self.text in punctuation_whitespace or (self.text.startswith(" <") and self.endswith("> "))
        is_invalid_start = self.start < 0 or self.end < 0
        if is_invalid_start:
            # print('Invalid start and end', self.start, self.end, self.text)
            return True
        return is_punc

    def __add__(self, other: 'WordSegment'):
        if self.phoneme is None or other.phoneme is None:
            phoneme = None
        else:
            phoneme = self.phoneme + other.phoneme

        # punctuation does not have correct timings
        if other.is_punctuation():
            start, end = self.start, self.end
            inst_start, inst_end = self.inst_start, self.inst_end
        elif self.is_punctuation():
            start, end = other.start, other.end
            inst_start, inst_end = other.inst_start, other.inst_end
        else:
            start, end = self.start, other.end
            inst_start, inst_end = self.inst_start, other.inst_end

        confidences = self.confidences + other.confidences
        if end != max(self.end, other.end): 
            print('Invalid end', end, self.end, other.end, inst_start, inst_end, self, other)
            raise Exception()
        
        return WordSegment(
            start=start,
            end=end,
            text=self.text + other.text,
            confidences=confidences,
            words=self.words + other.words,
            phoneme=phoneme,
            inst_start=inst_start,
            inst_end=inst_end
        )

def lyrics_to_segments(lyrics, metadata=None, target_durations=(20,25,30), min_confidence=0.75, 
                       shuffle_start=False, grouping='variable_length'):
    if not lyrics: return []
    if 'duration' in metadata:
        song_duration = metadata['duration']
        song_duration = song_duration / 1000 if song_duration and song_duration > 1000 else song_duration # change from milliseconds to seconds
    elif 'full_duration' in metadata:
        song_duration = metadata['full_duration']
        if song_duration is not None: song_duration = song_duration/1000
    else:
        print('Warning: no duration found in metadata. Segments will not have instrumental ending')
    segments = Segment.segments_from_utterances(lyrics, song_duration)
    song_structure = song_structure_from_metadata(metadata, segments)

    target_segments = []
    if song_structure and max(target_durations) <= 30: # for 30s, return section
        target_segments = group_by_section_start(song_structure, segments, max(target_durations))
        max_misaligned = None if metadata.get('is_lyrics_gt', False) else 5
        target_segments = [get_section_aligned_segment(segment, song_structure, max_misaligned=max_misaligned) for segment in target_segments]
    elif song_structure and max(target_durations) > 30: # for 2 min, return full song + variable segments
        grouped_segments = group_by_section_start(song_structure, segments, max(target_durations))
        max_misaligned = None if metadata.get('is_lyrics_gt', False) else 5

        start_idx = random.randint(0, 2) if shuffle_start else 0
        variable_segments = group_by_variable_length(segments[start_idx:], target_durations, shuffle_start)

        target_segments = grouped_segments[:1] + variable_segments
        target_segments = [get_section_aligned_segment(segment, song_structure, max_misaligned=max_misaligned) for segment in target_segments]
    elif grouping == 'fixed_length':
        target_segments = group_by_fixed_length(segments, max(target_durations))
    else: # 'variable length' or song_structure is None
        start_idx = random.randint(0, 2) if shuffle_start else 0
        target_segments = group_by_variable_length(segments[start_idx:], target_durations, shuffle_start)

    # filter by confidence and max duration
    def is_valid_segment(segment: Segment, min_confidence, target_durations):
        if segment is None: return False
        if segment.confidence <= min_confidence: return False
        if len(target_durations) == 1: return segment.duration <= target_durations[0]
        inst_duration = segment.inst_end - segment.inst_start
        return inst_duration >= min(target_durations) and segment.duration <= max(target_durations)

    valid_target_segments = [t for t in target_segments if is_valid_segment(t, min_confidence, target_durations)]


    if len(valid_target_segments) == 0:
        if len(target_segments) != 0:
            print('WARNING no segments processed. Check filters', shuffle_start, len(target_segments), len(valid_target_segments), target_segments)
        if shuffle_start and max(target_durations) > 90:
            # Hack to improve 2 minute long segment slicing. Try re-running again without shuffling start time
            return lyrics_to_segments(lyrics, metadata, target_durations=target_durations, min_confidence=min_confidence, shuffle_start=False, grouping=grouping)
    return valid_target_segments


## Different grouping algorithms ## 
def group_by_fixed_length(segments: List[Segment], target_duration):
    if len(segments) == 0: return []
    grouped_segments = []
    base_segment: Segment = None
    for idx, s in enumerate(segments):
        if base_segment is None: # initial
            base_segment = s
            continue
        if base_segment.duration > target_duration: # target reached. append and reset
            target_segment, remainder = base_segment.split_at(target_duration)
            target_segment.end = base_segment.start + target_duration
            grouped_segments.append(target_segment)
            base_segment = s
        else: # extend
            base_segment += s
    if base_segment.duration > target_duration: # target reached. append and reset
        target_segment, remainder = base_segment.split_at(target_duration)
        target_segment.end = base_segment.start + target_duration
        grouped_segments.append(target_segment)
    """
    Group segments by variable length.
    :param segments: List of segments to group.
    :param target_durations: List of target durations to group by.
    :param shuffle_start: Whether to shuffle the start of the segments.
    :return: List of grouped segments.
    """
    return grouped_segments

def group_by_variable_length(segments: List[Segment], target_durations, shuffle_start=False):
    if len(segments) == 0: return []
    grouped_segments = []
    base_segment: Segment = None

    target_duration_length = random.choice(target_durations) if shuffle_start else max(target_durations)
    for idx, s in enumerate(segments):
        if base_segment is None: # initial
            base_segment = s
            continue
        if s.end - base_segment.start > target_duration_length:
            base_segment.target_duration = target_duration_length # append for instrumental support
            grouped_segments.append(base_segment)
            base_segment = s
            target_duration_length = random.choice(target_durations) # reset target
        else:
            base_segment += s
    # add last segment
    if base_segment.duration > target_duration_length:
        base_segment.target_duration = target_duration_length # append for instrumental support
        grouped_segments.append(base_segment)
    return grouped_segments


# Converts word format (outputs from force alignment), to line-by-line format
PAD_TIME = -2
def _strip_non_words(words):
    words = words.copy()
    while words and (words[-1]['start_time'] < 0 or words[-1]['confidence'] < 0.05):
        words.pop()
    while words and (words[0]['start_time'] < 0 or words[0]['confidence'] < 0.05):
        words.pop(0)
    return words

def line_format_to_force_aligned_word_format(lyrics):
    words = []
    for l in lyrics:
        words.append({ 'text': '\n' })
        words.extend(l['words'])
    return words

def force_aligned_word_format_to_line_format(lyrics):
    words = lyrics[0]['words'].copy()
    words.append({ 'text': '\n' }) # dummy placeholder
    lines = []
    current_line = []
    for i, word_json in enumerate(words):
        lyrics = word_json['text']
        if '\n' in lyrics:
            current_line = _strip_non_words(current_line)
            if len(current_line) == 0: continue

            confidences = [w['confidence'] for w in current_line if w['text'] not in punctuation_whitespace and w['start_time'] >= 0]
            confidence = sum(confidences) / len(confidences) if confidences else 0
            line = {
                'start_time': current_line[0]['start_time'],
                'end_time': current_line[-1]['end_time'],
                'text': ''.join([w['text'] for w in current_line]),
                'confidence': confidence,
                'phoneme': ''.join([w.get('phoneme', '') for w in current_line]), # TODO: maybe need to set this to None
                'words': current_line
            }
            lines.append(line)
            current_line = []
        else:
            current_line.append(word_json)
    return lines
