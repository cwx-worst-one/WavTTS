from typing import Any, Dict, List, Generator, Optional, Callable
import io
import torch
import webdataset as wds
from torchaudio_augmentations import Compose
from dataclasses import dataclass
import math
import random
import numpy as np
import json
from typing import Tuple
from string import punctuation, whitespace
from copy import deepcopy
import re
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
from recipes.bigmusic.datasets.transforms.lyrics_section_alignment import get_section_aligned_text, merge_intervals

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
    ) -> None:
        super().__init__(log_interval=500)
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
        self.normalize_audio = FastNormalizeAudio()

    def process_segment(self, segment, audio_wavs, metadata, process_music_structure=False):
        audio_duration = audio_wavs['target_audio'].shape[-1] / self.sample_rate
        if segment.end > audio_duration:
            return None
        cropped_segments = { name: crop_pad_audio_to_segment(segment, audio, self.sample_rate) for name, audio in audio_wavs.items() }
        if 'music_structure' in metadata and process_music_structure: # TODO: (AS) enable this in config
            music_structures = merge_intervals(metadata['music_structure'][0])
            lyrics_text = get_section_aligned_text(segment, music_structures)
        else:
            lyrics_text = segment.text.strip()
        if lyrics_text is None: return None
        cropped_segments['lyrics'] = lyrics_text
        cropped_segments['phoneme'] = segment.phoneme
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
        target_audio = audio_wavs['target_audio']
        for name, audio_wav in audio_wavs.items():
            audio_wavs[name] = self.normalize_audio(audio_wav, target_audio)
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

        fixed_duration = len(self.sample_duration) == 1 # if only one duration is provided. Fix it to that duration
        shuffle_start = self.shuffle_segments and len(lyrics) > 8
        shuffle_lengths = self.shuffle_segments
        include_intro = True # max(self.sample_duration) > 90 # include intro for 2 min training
        segments: List[Segment] = lyrics_to_segments(
            lyrics, 
            fixed_duration=fixed_duration,
            min_duration=min(self.sample_duration),
            max_duration=max(self.sample_duration),
            shuffle_start=shuffle_start,
            shuffle_lengths=shuffle_lengths,
            min_confidence=self.min_segment_confidence,
            include_intro=include_intro
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
                item = self.process_segment(segment, audio_wavs, metadata)
            except Exception as e:
                print('Exception in process_segment', e)
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

    if 'lyrics_gt' in metadata:
        lyrics = metadata.pop('lyrics_gt')
        if 'utterances' in lyrics: # For 9M gt data
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

    return metadata, utterances
    
def crop_pad_audio_to_segment(segment, audio, sample_rate, target_duration=None):
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
    duration: float
    confidence: float
    words: [str]
    phoneme: str
        
    @classmethod
    def from_dict(cls, json_dict):
        if json_dict['start_time'] is None:
            start, end, duration = None, None, None
        elif json_dict['start_time'] == -2:
            start, end, duration = -2, -2, 0
        else:
            start = json_dict['start_time']/1000.0
            end = json_dict['end_time']/1000.0
            duration = end - start
        text = json_dict['text']
        if 'additions' in json_dict:
            confidence = float(json_dict['additions']['confidence'])
        elif 'confidence' in json_dict:
            confidence = float(json_dict['confidence'])
        else:
            confidence = 1
        words = deepcopy(json_dict.get('words', []))
        if words is None:
            words = []
            print('WARNING: words is None. Double check if lyrics exists', text)
        phoneme = json_dict.get('phoneme', '')
        return Segment(start, end, text, duration, confidence, words, phoneme)

    @classmethod
    def from_word_dict(cls, json_dict):
        if json_dict['start_time'] is None:
            start, end, duration = None, None, None
        elif json_dict['start_time'] == -2:
            start, end, duration = -2, -2, 0
        else:
            start = json_dict['start_time']/1000.0
            end = json_dict['end_time']/1000.0
            duration = end - start

        text = json_dict['text']
        if 'additions' in json_dict:
            confidence = float(json_dict['additions']['confidence'])
        elif 'confidence' in json_dict:
            confidence = float(json_dict['confidence'])
        else:
            confidence = 1
        words = deepcopy([json_dict]) # for words, we set it to itself
        phoneme = json_dict.get('phoneme', '')
        return Segment(start, end, text, duration, confidence, words, phoneme)

    def has_valid_time(self):
        return self.start >= 0

def lyrics_to_segments(lyrics, min_duration=3, max_duration=10,
                       new_line_token=". ", fixed_duration=True, min_confidence=0.75, 
                       shuffle_start=False, shuffle_lengths=False, include_intro=False
    ):
    if not lyrics: return []
    segments = []
    segment = None

    lyrics.append({
        'start_time': 10000*1000, 
        'end_time': 10001*1000,
        'text': 'STOP_PLACEHOLDER', 
        'words': [],
    })

    start_idx = random.randint(0, 2) if shuffle_start else 0
    if shuffle_lengths and random.randint(0, 3) > 0 and not fixed_duration:
        target_duration_length = random.randint(int(min_duration), int(max_duration))
    else:
        target_duration_length = max_duration
    for i in range(start_idx, len(lyrics)):
        current_diction = lyrics[i]
        current_segment = Segment.from_dict(current_diction)
        if len(current_segment.text.strip()) == 0: continue # instrumental
        if current_segment.text and current_segment.phoneme is None: # error with phoneme - usually due to non-tts supported languages - arabic, etc.
            segment = None
            continue
        if current_segment.start is None or current_segment.start < 0 or (segment and current_segment.start < segment.end):
            segment = None
            continue
        
        # Case #1: overflow. Append segment. Create new
        if segment and (current_segment.end - segment.start) > target_duration_length:
            if fixed_duration and 'words' in current_diction:
                # append words from current segment.
                target_end_time = segment.start + target_duration_length
                extended_segment, _ = _words_to_segment(current_diction['words'], segment.start, target_end_time)
                if extended_segment:
                    segment.end = extended_segment.end
                    segment.text += new_line_token + extended_segment.text
                    segment.words += extended_segment.words
                    segment.phoneme += " <n> " + extended_segment.phoneme
                    segment.duration += extended_segment.duration                    
                segment.end = target_end_time
                segment.duration = segment.end - segment.start
            else:
                segment.end = max(segment.end, min(segment.start + target_duration_length, current_segment.start))
                segment.duration = segment.end - segment.start

            if (segment.duration >= min_duration):
                segments.append(segment)
            segment = None

        if current_segment.confidence < min_confidence:
            # low confidence segment. skip and reset
            segment = None
            continue
        
        # # Break long segments into multiple segments
        if current_segment and current_segment.duration > max_duration:
            # if subwords does not exist, we can't slice. skip segment
            if 'words' not in current_diction:
                segment = None
                continue
            cached_index = 0
            for i in range(math.ceil(current_segment.duration / max_duration)):
                start = current_segment.start + i * max_duration
                end = current_segment.start + (i+1) * max_duration
                clipped_segment, cached_index = _words_to_segment(current_diction['words'], start, end, cached_index)
                if clipped_segment is None: 
                    continue
                if fixed_duration and clipped_segment.duration <= max_duration:
                    clipped_segment.end = clipped_segment.start + max_duration
                    clipped_segment.duration = max_duration
                    segments.append(clipped_segment)
                elif clipped_segment.duration <= max_duration and clipped_segment.duration >= min_duration:
                    segments.append(clipped_segment)

            # reset everything
            segment = None
            continue

        # Create new segment
        if segment is None:
            segment = current_segment
            if shuffle_lengths and random.randint(0, 3) > 0:
                target_duration_length = random.randint(int(min_duration), int(max_duration))
            else:
                target_duration_length = max_duration

            # Set start to beginning of last segment to include instrumental sections
            previous_end_time = 0 if i == 0 else Segment.from_dict(lyrics[i-1]).end
            if include_intro and previous_end_time and segment.end - previous_end_time <= target_duration_length:
                segment.start = previous_end_time
                segment.duration = segment.end - segment.start
        else: # append to existing segment
            segment.end = current_segment.end
            segment.text = segment.text + new_line_token + current_segment.text
            segment.words += current_segment.words
            segment.phoneme += " <n> " + current_segment.phoneme
            segment.duration = segment.end - segment.start
    return segments

punctuation_whitespace = set(punctuation + whitespace)
def _words_to_segment(words, start_time, end_time, start_index=0):
    def is_word(word): return word.text not in punctuation_whitespace
    segment = None
    end_index = start_index
    idx = 0

    for idx, word in enumerate(words[start_index:]):
        word_segment = Segment.from_word_dict(word)
        end_index = start_index + idx
        if is_word(word_segment):
            if word_segment.start >= 0 and word_segment.start < start_time: # skip words less than start time
                continue
            if word_segment.end > end_time:
                break
        else: # keep punctuation
            pass

        if segment:
            if word_segment.has_valid_time() and word_segment.start >= start_time:
                segment.end = word_segment.end
                segment.duration = segment.end - segment.start
            segment.text += word_segment.text # need to append space 
            segment.words += word_segment.words
        elif segment is None and word_segment.has_valid_time() and word_segment.start >= start_time:
            segment = word_segment
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

# Filters
def is_audio_metrics_good(audio_metrics: Dict[str, Any]) -> Tuple[bool, str]:
    # Clipping
    clip = audio_metrics.get("clipping", {})
    if clip.get("rate", 0) >= 5e-5:
        return False
    for ch in ["left", "right"]:
        if clip.get(f"peak_rate_{ch}", 0) >= 0.05:
            return False
    # Loudness
    loudness = audio_metrics.get("loudness", {})
    if (
        loudness.get("integrated_loudness", -7) > -5
        or loudness.get("max_mom_loud", -7) >= 0
        or loudness.get("max_short_term_loud", -7) >= 0
    ):
        return False
    # RMS stats
    rms_stats = audio_metrics.get("rms_stats", {})
    if rms_stats.get("peak", 0) > 3:
        return False
    for ch in ["left", "right"]:
        if (
            rms_stats.get(f"{ch}_total", -10) > -5
            or rms_stats.get(f"{ch}_total", -10) < -40
            or rms_stats.get(f"normed_std_{ch}", -10) < -19.5
        ):
            return False
    # Cutoff frequency
    cutoff_freq = audio_metrics.get("cutoff_frequency", {})
    for ch in ["left", "right"]:
        if (
            cutoff_freq.get(f"rel_{ch}", 48000) < 15000
            and cutoff_freq.get(f"rel_{ch}_conf", 0) > 0.6
            and cutoff_freq.get(f"band_std_{ch}", 10) < 5
        ):
            return False
    # Phase
    phase = audio_metrics.get("phase_check", {})
    if (
        phase.get("has_phase_issue", False)
        or abs(phase.get("rms_downmix_diff", 0.1)) > 3
    ):
        return False
    return True


def is_valid_lyrics(lyrics, confidence_threshold=0.8):
    if lyrics is None:
        return False
    confidences = []
    for utterance in lyrics:
        if 'confidence' in utterance:
            confidence = float(utterance["confidence"])
        elif 'additions' in utterance:
            confidence = float(utterance["additions"]["confidence"])
        else:
            # some lyrics may not have confidence (force alignment). return True
            return True
        if confidence == 0:
            continue
        confidences.append(confidence)
    if len(confidences) == 0:
        return False
    return np.array(confidences).mean() > confidence_threshold

def is_invalid_song(metadata, lyrics, confidence_threshold):
    # Metadata filtering
    # (AS) TODO: figure out a better way for english and cn to coexist. However, it should have already been filtered out on dataset
    # if 'final_language' in metadata and metadata['final_language'] != 'English': 
    #     return True, "Filter invalid final_language"
    
    if not is_audio_metrics_good(metadata.get('audio_metrics', {})): 
        return True, "Filter invalid audio metrics"
    
    # Lyrics filtering
    if lyrics is None or len(lyrics) == 0:
        return True, "Filter invalid lyrics. None found"
    
    # Song title filtering
    if not is_valid_song_name(metadata):
        return True, "Filter live/instrumental song names."

    if not is_valid_lyrics(lyrics, confidence_threshold=confidence_threshold):
        return True, "Filter low confidence lyrics."
    
    
    return False, "Passed filters"


def is_valid_song_name(metadata):
    if 'meta_song_title' not in metadata: return True
    song_name = metadata['meta_song_title']
    album_name = metadata['meta_album_title']
    if not album_name: album_name = ''
    full_name = (str(song_name) + ' ' + str(album_name)).lower()
    if is_live(full_name): return False
    if is_instrumental(full_name): return False
    # if is_remix(full_name): return False # let's keep remixes for now
    return True

def is_live(song_name):
    if re.search(r'\(live\b', song_name):
        return True
    if re.search(r'- live\b', song_name):
        return True
    if re.search(r'\/ live\b', song_name):
        return True
    if re.search(r'\blive at ', song_name):
        return True
    if re.search(r'\blive from ', song_name):
        return True
    if 'recorded live ' in song_name:
        return True
    if ' live radio ' in song_name:
        return True
    if ' live lounge' in song_name:
        return True
    if ' live sets' in song_name:
        return True
    if ' live session' in song_name:
        return True
    if ' live performance' in song_name:
        return True
    if 'concert' in song_name:
        return True
    # chinese
    if '现场' in song_name: # live
        return True
    if '音乐会' in song_name: # concert
        return True
    return False

def is_instrumental(song_name):
    if 'instrumental' in song_name:
        return True
    return False

def is_remix(song_name):
    if 'remix' in song_name:
        return True
    if re.search(r'\bmix\b', song_name):
        return True
    if '混音' in song_name: # remix
        return True
    return False
