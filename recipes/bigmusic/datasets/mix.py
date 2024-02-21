import copy
from curses.ascii import ETB
from dataclasses import dataclass
import io
import random
import json
import math
from string import punctuation
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple
import re
import pytorch_lightning as pl
import torch
import ffmpeg
import numpy as np
import webdataset as wds
import random
import os
from functools import partial
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import BertTokenizer, Wav2Vec2PhonemeCTCTokenizer
from webdataset import WebDataset, shardlists
from webdataset.pipeline import DataPipeline
import logging, phonemizer
from recipes.bigmusic.datasets.index_lists import INDEX as BM_INDEX
from recipes.bigmusic.datasets.lyrics import DefaultDatasets, transform_dataset
from recipes.bigmusic.datasets.tokenizers.phoneme import MAX_PHONE_LEN
from recipes.bigmusic.datasets.transforms.lyrics import LyricsTokenTransform, rewrite_metadata, rewrite_playlist_labels
from recipes.bigmusic.datasets.mir_data_util import ARTIST_ID_MAP, ARTIST_ID_MAP_V2

from recipes.bigmusic.utils.format_utils import normalize_text, sa_music_tagging_to_style_text
from recipes.datasets.mcc.mix import (
    INDEX,
    WebDatasetBufferPreprocessor,
    BaseTransforms,
    DataModule
)
from recipes.datasets.mcc.sami_tokenizer import (
    add_singer_tag, 
    add_section_tag, 
    convert_labels_to_text_id, 
    extract_singer_tag,
    Phrase, 
)

from recipes.musiclm.utils.dist import local_zero_first
from recipes.musiclm.transforms.audio import (
    FastNormalizeAudio,
    LoudnessCheck,
    NormalizeAudio,
    ReadMP3,
)
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    Pad,
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
    RandomResizedCrop,
)
from samantha.utils.webdataset import return_self

MAX_STYLE_LEN = 16

# Yilin: No need to use line break for Chinese. A full stop `。` will be
# added 
# LINE_BREAK_PHONE_TOKEN = get_line_break_id()


def pad_crop(sequence, seq_len, dtype, padding_value=0):
    # in item_pad_idx, 0 indicates the values are padded.
    item_pad = torch.full((seq_len,), fill_value=padding_value, dtype=dtype)
    item_pad[:len(sequence)] = torch.as_tensor(sequence[:seq_len])
    item_pad_idx = torch.full((seq_len,), fill_value=0, dtype=int)
    item_pad_idx[:len(sequence)] = torch.ones_like(torch.as_tensor(sequence[:seq_len]), dtype=int)
    return item_pad, item_pad_idx

def collate_fn(batch: List[torch.Tensor], conditions="style_text,lyrics_tokens") -> Dict[str, torch.Tensor]:
    # collate_fn batches the examples based on the target_audio length.
    PHONE_PAD_ID = 0     
    max_phone_len = int(batch[0].get("max_phone_len", MAX_PHONE_LEN))
    max_length = max([x["target_audio"].shape[-1] for x in batch])
    random_pad = Pad(n_samples=max_length)
    default_lyrics_token = torch.full((max_phone_len,), PHONE_PAD_ID, dtype=torch.int)    
    default_speaker_id = ARTIST_ID_MAP_V2['zh_empty']
    target_audio = []   
    style_text = []    
    normalized_text = []
    lyrics_tokens = []
    speaker_id = []     
    dataset_name = []

    # DEBUG
    style_metadata = []
    song_id = []
    shard = []
    worker_id = []
    for idx in range(len(batch)):
        audio = batch[idx]["target_audio"]
        if audio.ndim == 1:
            audio = audio[None, :]
                  
        target_audio.append(random_pad(audio))

        style_label = batch[idx].get("style_text", None)
        if isinstance(style_label, Tuple):
            style_label = style_label[0]
        style_text.append(style_label)

        speaker_id.append(batch[idx].get("artist_id", default_speaker_id))
        
        construct = lambda key, default: \
            batch[idx][key].clone().detach() if key in batch[idx] and batch[idx][key] is not None \
                else default.clone().detach()
        
        text = batch[idx].get("normalized_text", None)
        if not text:
            text = batch[idx].get("lyrics", None)
        normalized_text.append(text)
        
        phoneme_tokens, _ = pad_crop(
            construct("lyrics_tokens", default_lyrics_token), 
            max_phone_len, torch.int, PHONE_PAD_ID)
        lyrics_tokens.append(phoneme_tokens)
        
        style_metadata.append(batch[idx].get("style_metadata", None))
        song_id.append(batch[idx].get("song_id", None))
        shard.append(batch[idx].get("shard", None))
        worker_id.append(batch[idx].get("worker_id", None))
        dataset_name.append(batch[idx].get("dataset_name", None))

    stacked_audio = torch.stack(target_audio, dim=0)
    if stacked_audio.dim() == 3:
        stacked_audio = stacked_audio.squeeze(1)    
    batch = {
        "target_audio": torch.stack(target_audio, dim=0),        
        "style_text": style_text,
        "normalized_text": normalized_text,
        "lyrics": normalized_text,
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "speaker_id": torch.as_tensor(speaker_id).unsqueeze(1),
        "conditions": conditions,
        "dataset_name": dataset_name,

        # DEBUG:
        "style_metadata": style_metadata,
        "song_id": song_id,
        "shard": shard,
        "worker_id": worker_id,
    }
    if "style_tag" in conditions or "style_audio" in conditions:
        batch["style_audio"] = stacked_audio
    return batch


def format_music_structure_tags(structure_tags):
    # This function converts the music_structure field in metadata into the format of [{'tag': tag, 'start_time': sec, 'end_time': sec}]
    # The music_structure are extracted from ground truth lyrics.
    if not structure_tags:
        return None
    reformated_structure_tags = []
    for structure_tag in structure_tags:
        tag_text = structure_tag['tag'].lower()
        if "chorus" in tag_text:
            structure_tag['tag'] = "chorus"
            reformated_structure_tags.append(structure_tag)
        if "verse" in tag_text:
            structure_tag['tag'] = "verse"
            reformated_structure_tags.append(structure_tag)
        if "section" in tag_text:
            structure_tag['tag'] = "section"
            reformated_structure_tags.append(structure_tag)
    return reformated_structure_tags


def format_deepchorus_structure_tags(deepchorus_tags):
    # This function converts the deepchorus field in metadata into the format of [{'tag': tag, 'start_time': sec, 'end_time': sec}]
    # The deepchorus tags are inferred based on the audio.
    structure_tags = []
    for segment in deepchorus_tags['segments']:
        structure_tag = {"tag": segment["label"]}
        structure_tag['start_time'] = segment["interval"][0]
        structure_tag['end_time'] = segment["interval"][1]
        structure_tags.append(structure_tag)
    return structure_tags


def format_utterances(utterances, time_in_sec=False, min_confidence=0.2):
    # This function cleans up the raw utterances data from metadata.
    # Extracts necessary information: [start_time_in_sec, end_time_in_sec, lyrics text with additional tags, precomputed phonemes]
    new_utterances = []    
    for i, u in enumerate(utterances):
        confidence = u.get('confidence', 1)
        if confidence < min_confidence:
            continue

        phone = u.get('phoneme', '')        
        phone = '' if not phone else phone

        if "lyrics" in u:
            u["text"] = u["lyrics"]
        
        utt_start = u.get('start_time', None)
        if utt_start is None:
            continue
        utt_start = math.floor(utt_start)   # Round down so we don't miss the first few phonemes
        utt_end = u.get('end_time', None)
        if utt_end is None:
            if i < (len(utterances) - 1):
                utt_end = max(math.floor(utterances[i+1].get('start_time', 0)), utt_start+1)
            else:
                continue
        utt_end = math.ceil(utt_end)        # Round up so we don't miss the last few phonemes
        if time_in_sec:
            new_utterances.append([utt_start, utt_end, u['text'], phone])
        else:
            # TODO (QQ) handle ms directly instead of converting to int.
            new_utterances.append([math.floor(utt_start/1000), math.ceil(utt_end/1000), u['text'], phone])


    # Sanity check utterances
    utterances = []
    for i, u in enumerate(new_utterances):
        if i > 0 and u[0] <= 0 and ":" in u[2]:
            # Handle the case of (0, 97, '合:对的人'), (-1, 97, '合:对的人')
            if i+1 < len(new_utterances) and u[1] > new_utterances[i+1][1]:
                # Handle the case of swap (0, 31, '合:色即是空 空即是色'), (25, 28, '选择就是选择无所谓对与错')
                u[0] = new_utterances[i+1][1]
                new_utterances[i] = new_utterances[i+1]
                new_utterances[i+1] = u            
            else:
                # Handle the case of (0, 25, '合:色即是空 空即是色'), (25, 28, '选择就是选择无所谓对与错')
                u[0] = min(new_utterances[max(i-1,0)][1], u[1]-1)
        if u[1] - u[0] > 0:
            # Only keep utterances that are longer than 1 sec.
            utterances.append(u)
    # Remove duplicate utterances, retain order.
    uniq_utt_idx = []
    for i in range(len(utterances)-1):
        if (utterances[i][0] == utterances[i+1][0]) and (utterances[i][1] == utterances[i+1][1]):
            continue
        uniq_utt_idx.append(i)
    utterances = [utterances[i] for i in uniq_utt_idx]
    
    if len(utterances) < 1:
        return []

    # Sanity check: utterances are non-overlapping.
    for i in range(len(utterances)-1):
        if utterances[i][1] > utterances[i+1][0] + 5:
            # logging.warning("utterances need to be non-overlapping")
            # TODO (QQ) need to handle time stamps of '合:色即是空 空即是色' more carefully.
            # print([x[:3] for x in utterances])
            return []
        
    return utterances


def extract_seg_from_utts(i, utterances, min_duration, max_duration, new_line_token, sec_start_idx=None):
    # Given the start time (and end time), combine the utterances in the window into a segment
    # Output is in the format [seg_start_sec, seg_end_sec, lyrics_text, precomputed_phonemes, seg_tag]
    s, cur_seg = i, []
    seg_tag = sec_start_idx[i] if sec_start_idx else "None"
    while i < len(utterances) and (utterances[i][1] - utterances[s][0] < min_duration):
        cur_seg.append(utterances[i])
        i += 1
    j = i
    while j < len(utterances) and utterances[j][1] - utterances[s][0] <= max_duration:
        j += 1
    if j == i:
        i = s + 1
        return (i, None)
    else:
        # Extract a segment with target duration uniformly distributed between min/max duration.
        k = random.randint(i+1, j)
        cur_seg.extend([utterances[kk] for kk in range(i, k)])
        seg_start = cur_seg[0][0]
        seg_end = cur_seg[-1][1]
        lyrics_text = ""
        for kk in range(s, k):
            if sec_start_idx and (kk in sec_start_idx):
                seg_tag = sec_start_idx[kk]
                lyrics_text = lyrics_text + ("[" + seg_tag + "]" + ". ")                
            lyrics_text = lyrics_text + utterances[kk][2] + ". "
        phoneme_sequence = new_line_token.join([u[3] for u in cur_seg])    
        new_seg = [
            seg_start, 
            seg_end, 
            lyrics_text,  # lyrics text
            phoneme_sequence,   # precomputed lyrics phonemes
            seg_tag]
        i = k
        return (i, new_seg)


@dataclass
class SongSlice:
    phrases: List[Phrase]
    # start: int
    # end: int

    @property
    def start(self) -> int:
        return self.phrases[0].start

    @property
    def end(self) -> int:
        return self.phrases[-1].end

    @property
    def duration(self) -> int:
        return self.end - self.start

    def is_time_span_valid(self, time_span: Tuple[int, int]) -> bool:
        start, end, duration = self.start, self.end, self.duration
        min_duration, max_duration = time_span
        return ((min_duration <= duration <= max_duration) and 
                start >= 0 and end >= 0 and start < end)


def remove_short_inst_phrases(song_slice: SongSlice) -> Optional[SongSlice]:
    """Remove short instrumental phrases, or verse/chorus/bridge without utterance to avoid adding unnessary section tags."""
    phrases=[
        p for p in song_slice.phrases
        if p.has_utterance or (p.section_tag not in ["verse", "chorus", "bridge"] and p.duration >= 2)
    ]
    return SongSlice(phrases=phrases) if phrases else None


def group_utterances_with_structure(
    utterances: List[Dict],
    min_duration: int,
    max_duration: int,
    structure_tags: Optional[Dict] = None
) -> List[SongSlice]:
    """Segment the full song into a list of SongSlice, each consists of multiple utterances."""
    def _format_utterances(utterances: Dict) -> List[Phrase]:
        formatted_us = format_utterances(utterances)
        return [
            Phrase.parse(text=nt, phonemes=phonemes, time_span=(start, end)) 
            for start, end, nt, phonemes in formatted_us
        ]

    def get_overlap(phrase_time_span: Tuple[int, int], section_time_span: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        left_overlap = max(phrase_time_span[0], section_time_span[0])
        right_overlap = min(phrase_time_span[1], section_time_span[1])
        if right_overlap <= left_overlap:
            return None
        return (left_overlap, right_overlap)

    def get_overlap_dur(phrase_time_span: Tuple[int, int], section_time_span: Tuple[float, float]) -> float:
        overlap = get_overlap(phrase_time_span, section_time_span)
        if overlap is None:
            return 0
        return overlap[1] - overlap[0]

    def get_section_span(structure_tag: Dict) -> Tuple[int, int]:
        return math.floor(structure_tag["start_time"]), math.ceil(structure_tag["end_time"])

    def add_section_tag(phrase: Phrase) -> Phrase:
        """Return a new phrase with a section tag based on the longest overlap"""
        if not structure_tags:
            return phrase
        overlaps = [
            get_overlap_dur(phrase.time_span, get_section_span(structure_tag)) 
            for structure_tag in structure_tags
        ]
        if not overlaps:
            return phrase
        idx = int(np.argmax(overlaps))
        return phrase._replace(section_tag=structure_tags[idx]["tag"])

    def get_inst_phrases(time_span: Tuple[int, int]) -> List[Phrase]:
        phrases = []
        for structure_tag in structure_tags:
            overlap = get_overlap(time_span, get_section_span(structure_tag))
            if not overlap:
                continue
            phrases.append(Phrase(section_tag=structure_tag["tag"], time_span=overlap))
        return phrases

    def insert_inst_phrases(phrases: List[Phrase], song_time_span: Tuple[int, int]) -> List[Phrase]:
        """Insert instrument phrases if a gap presents between two adjacent vocal phrases."""
        song_start, song_end = song_time_span
        if not phrases:
            return get_inst_phrases(song_time_span)
        new_phrases = []
        for idx, (curr_phrase, next_phrase) in enumerate(zip(phrases, phrases[1:] + [None])):
            # the gap between song start and the first phrase's start
            if idx == 0 and curr_phrase.start - song_start > 0:  # first phrase
                _phrases = get_inst_phrases((song_start, curr_phrase.start))
                new_phrases.extend(_phrases)
            # add the current phrase
            new_phrases.append(curr_phrase)
            # the gap between the current phrase and the next phrase
            if idx < len(phrases) - 1 and next_phrase.start - curr_phrase.end > 0:
                _phrases = get_inst_phrases((curr_phrase.end, next_phrase.start))
                new_phrases.extend(_phrases)
            # the gap between the last phrase's end and the song end
            elif idx == len(phrases) - 1 and song_end - curr_phrase.end > 0:  # last phrase
                _phrases = get_inst_phrases((curr_phrase.end, song_end))
                new_phrases.extend(_phrases)
        return new_phrases

    def split_inst_phrase(phrase: Phrase, split_t: int) -> List[Phrase]:
        # exceptions should not be possible, but just in case
        if phrase.has_utterance:
            raise ValueError("Invalid inst phrase.")
        start, end = phrase.time_span
        if split_t >= end:
            raise ValueError("Invalid desired_t for phrase split.")
        return [
            Phrase(section_tag=phrase.section_tag, time_span=(start, split_t)),
            Phrase(section_tag=phrase.section_tag, time_span=(split_t, end)),
        ]

    def get_phrase_group_dur(phrase_group: List[Phrase]) -> int:
        if not phrase_group:
            return 0
        return phrase_group[-1].end - phrase_group[0].start

    def get_section_start_ind(phrases: List[Phrase]) -> List[int]:
        sec_tags = [p.section_tag for p in phrases]
        ind = [0]
        for idx, (curr_tag, next_tag) in enumerate(zip(sec_tags, sec_tags[1:])):
            if next_tag != curr_tag:
                ind.append(idx + 1)
        return ind

    def get_song_slices(phrases: List[Phrase]) -> List[SongSlice]:
        """Group phrases into song slices (might chunk instrument phrases).
        The logic assumes there are short overlaps (~1s) between phrases, or some phrases might
        have gaps inbetween. Therefore, the most accurate group duration is (end - start).
        """
        section_start_ind = get_section_start_ind(phrases)
        song_slices = []
        # Start from the beginning of each section, obtain a chunk
        for start_idx in section_start_ind:
            phrases_in_proc = copy.deepcopy(phrases)[start_idx:]
            # the length will grow, the last index helps insert the last group
            idx, group = 0, []
            while idx <= len(phrases_in_proc):
                phrase = phrases_in_proc[idx] if idx < len(phrases_in_proc) else None
                # add a new song slice if unable to include the current one
                if (phrase is None) or (get_phrase_group_dur(group + [phrase]) > max_duration):
                    if group:
                        song_slices.append(SongSlice(phrases=group))
                        break
                # add phrase to group
                if phrase.has_utterance:
                    # Unable to chunk utterance, add directly
                    group.append(phrase)
                else:
                    if get_phrase_group_dur(group + [phrase]) <= max_duration:
                        group.append(phrase)
                    else:
                        # The corner case is to have an inst phrase that has a gap
                        # between it and its previous phrase, and the phrase's start
                        # is after the split_t. We need to handle two cases separately.
                        _start = group[0].start if group else phrase.start
                        split_t = math.floor(_start + max_duration)
                        # split_t < phrase.end is always valid, otherwise it would go to the above if branch.
                        if split_t > phrase.start:
                            inst_phrases = split_inst_phrase(phrase, split_t)
                            # split the current inst phrase into two, insert into the current index
                            phrases_in_proc[idx:idx+1] = inst_phrases
                            group.append(phrases_in_proc[idx])
                        else:  # impossible to utilize the current phrase in the current group
                            if group:
                                song_slices.append(SongSlice(phrases=group))
                                break
                idx += 1
        return song_slices

    song_time_span = (math.floor(structure_tags[0]["start_time"]), math.ceil(structure_tags[-1]["end_time"]))
    phrases = _format_utterances(utterances)
    phrases = list(map(add_section_tag, phrases))
    phrases = insert_inst_phrases(phrases, song_time_span)
    return get_song_slices(phrases)


# TODO (Yilin) Remove group_utterance, make group_utterence_with_structure handle no structure cases.
def group_utterances(utterances, min_duration, max_duration, time_in_sec=False,
                     new_line_token=' ', infer_structure_tags=False):
    # Segment the full song into segments, each segment consists of multiple utterances.
    utterances = format_utterances(utterances)
    if not utterances:
        return []
    sec_start_idx = {}        
    if infer_structure_tags:        
        # Infer sections based on separation of utterances.
        last_utt_end = -10
        MIN_NUM_CHAR = 3    # Minimum number of characters in an utterance to infer it's valid vocal.
        MIN_SEP_WITHIN_SEC = 4  # Max separation of two utterance to infer there is a section break.
        for i, utt in enumerate(utterances):
            if (utt[0] - last_utt_end >= MIN_SEP_WITHIN_SEC) and len(utt[2]) > MIN_NUM_CHAR:
                sec_start_idx[i] = "section"
            last_utt_end = utt[1]

    segs = []
    if sec_start_idx:
        # Extract segments starting from (inferred) starting points of sections.
        for sec_start_id, sec_tag in sec_start_idx.items():
            _, new_seg = extract_seg_from_utts(
                sec_start_id, utterances, min_duration, max_duration, new_line_token, sec_start_idx)
            if new_seg:
                segs.append(new_seg)
    else:
        # Group utterances into random non-overlapping segments.
        i = 0
        while i < len(utterances):
            i, new_seg = extract_seg_from_utts(i, utterances, min_duration, max_duration, new_line_token)
            if new_seg:
                segs.append(new_seg)

            # Move to the next vocal starting point
            while i < len(utterances):
                if len(utterances[i][2]) >= 2:
                    break
                i += 1
    return segs


########################## Chinese Datasets ######################


class VocalTransforms(BaseTransforms):
    name = "VocalTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        index_key: str = "__index_data__",
        min_duration: int = 1,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_field: str = "lyrics",
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        read_structure_tags: bool = False,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,   
        use_soda_gt_lyrics: bool = True,
        infer_structure_tags: bool = False,
        sinking_threshold: float = 0.51,
        remove_sinking: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.lyrics_confidence = lyrics_confidence
        self.audio_key = audio_key
        self.index_key = index_key  
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.infer_structure_tags = infer_structure_tags
        self.read_structure_tags = read_structure_tags
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.segment_max_phone_len = segment_max_phone_len
        self.use_soda_gt_lyrics = use_soda_gt_lyrics
        self.lyrics_field = lyrics_field       
        self.sinking_threshold = sinking_threshold
        self.remove_sinking = remove_sinking  

        assert self.tokenizer
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def is_confident_lyrics(self, utterance, threshold):
        conf, num_utt = 0., 0.
        for utt in utterance:
            if utt['text'].strip():
                confidence_score = utt.get("confidence", 1)
                conf += float(confidence_score)
                num_utt += 1
        if num_utt == 0:
            return False
        conf /= num_utt
        return True if conf > threshold else False

    def __call__(self, item: Dict[str, Any]) -> Generator:
        # Extract utterances
        meta = item[self.index_key]
        if isinstance(meta, str):
            meta = json.loads(meta)
        
        lyrics = meta.get(self.lyrics_field, None)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return            
        result = lyrics.get("result", None)
        if result is None or len(result) != 1:
            self._update_stats(skipped=True, message="No result")
            return
        utterances = result[0].get("utterances", None)
            
        if self.use_soda_gt_lyrics and "lyrics_gt" in meta:
            utterances = meta.get("lyrics_gt", None)

        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        # Check for lyrics confidence
        lyrics_connfidence_avg = lyrics.get("confidence_avg", None)
        if lyrics_connfidence_avg is None:
            if not self.is_confident_lyrics(utterances, self.lyrics_confidence):
                self._update_stats(skipped=True, message="Low confidence lyrics utt")
                return
        else:
            if lyrics_connfidence_avg < self.lyrics_confidence:
                self._update_stats(skipped=True, message="Low confidence lyrics sta avg")
                return

        # Extract structure tags
        structure_tags = []
        if self.read_structure_tags:
            # Use deepchorus only
            # music_structure_tags = meta.get("music_structure", None)
            deepchorus_tags = meta.get("deepchorus", None)
            if deepchorus_tags:
                structure_tags = format_deepchorus_structure_tags(deepchorus_tags)  
            # elif music_structure_tags:
            #     structure_tags = format_music_structure_tags(music_structure_tags)
            if not structure_tags:
                self._update_stats(skipped=True, message="No structure tags")
                return            

        # Extract segments from utterances
        new_line_token = " <n> "
        if isinstance(self.tokenizer, BertTokenizer):
            self.tokenizer.add_special_tokens({'additional_special_tokens': [new_line_token]})

        # Group utterances into segment groups        
        song_slices = group_utterances_with_structure(
            utterances, self.min_duration, self.max_duration, structure_tags=structure_tags
        )
        # if structure_tags:
        #     segments = group_utterances_with_structure(
        #         utterances, self.min_duration, self.max_duration,
        #         structure_tags=structure_tags)
        # else:
        #     segments = group_utterances(
        #         utterances, self.min_duration, self.max_duration,
        #         time_in_sec=False,
        #         new_line_token=new_line_token,                
        #         infer_structure_tags=self.infer_structure_tags)

        if self.segment_method == "first":
            song_slices = song_slices[:1]
        if self.segment_method == "verse_chorus_section":
            song_slices = [
                song_slice for song_slice in song_slices 
                if set(s.section_tag for s in song_slice.phrases).issubset(["verse", "chorus", "section"])
            ]
        if self.max_seg_per_track > 0:
            random.shuffle(song_slices)
            song_slices = song_slices[:self.max_seg_per_track]

        if len(song_slices) < 1:
            self._update_stats(skipped=True, message="No valid segment")
            return

        # Filter out invalid slices
        n_slices_pre_filter = len(song_slices)
        song_slices = [ss for ss in song_slices if ss.is_time_span_valid((self.min_duration, self.max_duration))]
        song_slices = list(filter(None, map(remove_short_inst_phrases, song_slices)))
        n_slices_post_filter = len(song_slices)
        if n_slices_pre_filter - n_slices_post_filter > 0:
            self._update_stats(
                skipped=False,
                message=f"Removed song slices: {n_slices_pre_filter-n_slices_post_filter}/{n_slices_pre_filter}"
            )

        # Filter out the whole song if there is any vocal phrase that does not have phonemes
        if any(phrase.text and not phrase.phonemes for song_slice in song_slices for phrase in song_slice.phrases):
            self._update_stats(skipped=True, message=f"No phoneme")
            return

        # Get style text            
        style_text = rewrite_metadata(meta)
        artist_id = ARTIST_ID_MAP_V2[str(meta.get("artist_id", "zh_empty"))]

        if "music_tagging" in meta:
            style_text, unfamiliar_tags, is_sinking = sa_music_tagging_to_style_text(meta["music_tagging"], self.sinking_threshold)
            if self.remove_sinking and is_sinking:
                self._update_stats(skipped=True, message="Sinking music")
                return
            if unfamiliar_tags:
                self._update_stats(skipped=False, message=f"Detected tag(s) not in SA vocab: {unfamiliar_tags}")
        elif "playlist_extra" in meta:
            label1 = meta["playlist_extra"].get("label1", "")
            label2 = meta["playlist_extra"].get("label2", "")
            style_text = rewrite_playlist_labels(label1, label2)
        elif "tags" in meta:
            # Try read English genre tag. Source: Q music tag, WYY tag, MCC tag.
            metadata = dict()
            tags = meta.get("tags")
            if isinstance(tags, str):
                tags = json.loads(tags)
            if "zq" in tags and tags["zq"][0] != 'Other':
                metadata['final_genre'] = tags["zq"][0]
            elif "wyy" in tags:
                metadata['final_genre'] = tags["wyy"][0]
            elif "merge_genre" in meta:
                mcc_genres = meta["merge_genre"]
                # print("tag, mcc meta", tags, mcc_genres)
                if isinstance(mcc_genres, str):
                    mcc_genres = mcc_genres.split(',')
                if isinstance(mcc_genres, List):
                    random.shuffle(mcc_genres)
                    metadata['final_genre'] = mcc_genres[0]
            style_text = rewrite_metadata(metadata)
        else:
            metadata = dict()
            if "merge_genre" in meta:
                mcc_genres = meta["merge_genre"]
                # print("mcc meta", mcc_genres)
                if isinstance(mcc_genres, str):
                    mcc_genres = mcc_genres.split(',')
                random.shuffle(mcc_genres)
                metadata['final_genre'] = mcc_genres[0]            
            # Try read MCC mood tag
            if "merge_mood" in meta:
                mcc_moods = meta["merge_mood"]
                # print("mcc meta", mcc_moods)
                random.shuffle(mcc_moods)
                metadata['final_mood'] = mcc_moods[0]
            style_text = rewrite_metadata(metadata)

        # Get track level audio
        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        # Yield one example per segment
        self._update_stats(skipped=False)
        for song_slice in song_slices:
            slice_start, slice_end = song_slice.start, song_slice.end
            start = int(slice_start * self.sample_rate)
            end = int(slice_end * self.sample_rate)
            clip = audio[:, start:end]

            # NOTE: `normalize_text` removes `:` for the singer tag.
            normalized_text = normalize_text(
                "\n".join([s.format_text() for s in song_slice.phrases]), enable_punctuation=True
            )

            if self.tokenizer == "tts_chinese_frontend_model":
                # TODO (QQ) call SamiOfflineTokenizer directly.
                text_tokens = []
                for _phrase in song_slice.phrases:
                    labels = [l for l in _phrase.phonemes.split("\n") if l] if _phrase.phonemes else None
                    if labels is None or (labels is not None and len(labels) > 0):
                        convert_result = convert_labels_to_text_id(labels, _phrase.prefix_tags)
                        if convert_result:
                            labels, _, _ = convert_result
                        else:
                            print("Invalid phone label to text id conversion", labels)
                            continue
                        line_phone_tokens = labels[0]
                    else:
                        continue
                    text_tokens.append(line_phone_tokens)
                    # Yilin: No need to use line break for Chinese. 
                    # The logic is synced with SamiTokenizer.
                    ## text_tokens.append(LINE_BREAK_PHONE_TOKEN)
                if len(text_tokens) > 0:
                    text_tokens = np.concatenate(text_tokens, axis=0)
                    text_tokens = torch.from_numpy(text_tokens).long()
                else:
                    text_tokens = None    
            else:
                text_tokens = self.tokenizer(
                    normalized_text, 
                    add_special_tokens=False,
                    return_tensors="pt")["input_ids"].squeeze(dim=0)
            if text_tokens is None:
                self._update_stats(skipped=True, message="No lyrics tokens.")
                continue
            
            yield {
                "target_audio": clip,                 
                "style_text": style_text,
                "artist_id": artist_id,
                "normalized_text": normalized_text,
                "lyrics_tokens": text_tokens,
                "max_phone_len": self.segment_max_phone_len,
            }        


class VocalDataset(WebPipeline):
    name = "VocalDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        region: str = "CN",
        dataset_names: List[str] = ["Soda"],
        dataset_weights: List[float] = [1],        
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 2,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.7,
        normalize_audio: bool = False,
        tokenizer: Any = BertTokenizer.from_pretrained("bert-base-chinese"),
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        use_soda_gt_lyrics: bool = True,
        sinking_threshold: float = 0.51,
        remove_sinking: bool = False,
        **kwargs,
    ):
        assert region in INDEX
        print(f"[{self.name}] initializing...")        

        transforms = VocalTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,            
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            use_soda_gt_lyrics=use_soda_gt_lyrics,
            sinking_threshold=sinking_threshold,
            remove_sinking=remove_sinking,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")

        if isinstance(dataset_names, list):            
            url2index = [INDEX[region][dataset_name] for dataset_name in dataset_names]
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=dataset_weights,
            )
        else:
            if dataset_names == "SodaTest":
                url2index = "hdfs://haruna/home/byte_speech_sv/data/soda_valid/val_url2index.txt"
                dataset = IndexedWebDataset(url2index=url2index, **kwargs)
            else:
                dataset = IndexedWebDataset(url2index=INDEX[region][dataset_names], **kwargs)

        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class VocalParquetDataset(WebPipeline):
    name = "VocalParqueDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        data_id: int = 365,  # QQ music: 365 WYY_music: TODO
        url_pattern: str = None,
        sample_rate: int = 24000,
        audio_key: str = "wav",
        index_key: str = "meta",
        min_duration: int = 10,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.7,
        normalize_audio: bool = False,
        tokenizer: Any = BertTokenizer.from_pretrained("bert-base-chinese"),
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        use_gt_lyrics: bool = True,
        infer_structure_tags: bool = False,
        read_structure_tags: bool = False,
        sinking_threshold: float = 0.51,
        remove_sinking: bool = False,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern, **kwargs)
        
        transforms = VocalTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            index_key=index_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,            
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            use_soda_gt_lyrics=use_gt_lyrics,
            infer_structure_tags=infer_structure_tags,
            read_structure_tags=read_structure_tags,
            sinking_threshold=sinking_threshold,
            remove_sinking=remove_sinking,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


########################## Data Modules ##########################


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        shuffle_buffer_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_dataset=None,
        validation_dataset=None,
        predict_dataset=None,
        collate_fn: Optional[Callable] = None,
        do_shuffle: bool = True,    # set to False if shuffling is already done at dataset level
    ):
        super().__init__()
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.do_shuffle = do_shuffle

    def train_dataloader(self):
        if self.do_shuffle:
            train_dataset = DataPipeline(
                self.train_dataset, wds.shuffle(self.shuffle_buffer_size)
            )
        else:
            train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if isinstance(self.validation_dataset, list):
            return [
                DataLoader(
                    val,
                    batch_size=None,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                )
                for val in self.validation_dataset
            ]
        else:
            return DataLoader(
                self.validation_dataset,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class MixVocalWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,             
        use_pipe: bool = True,
        tokenizer: str = "wordpiece",
        frame_rate: int = 25,
        normalize_audio: bool = False,
        region: str = "CN",
        conditions: str = "style_text,lyrics_tokens",
        wds_dataset_names: List[str] = [],
        wds_dataset_weights: List[int] = [],
        parquet_dataset_ids: List[int] = [],
        parquet_dataset_weights: List[int] = [],
        use_dynamic_batch: str = False,
        lyrics_confidence: float = 0.75,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        infer_structure_tags: bool = False,
        read_structure_tags: bool = False,
        max_seg_per_track: int = -1,
        buckets_in_sec: List[int] = [
            20,
            25,
            30,
        ],
        sinking_threshold: float = 0.51,
        remove_sinking: bool = False,
    ):        
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = partial(collate_fn, conditions=conditions)

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = "tts_chinese_frontend_model"
        elif tokenizer == "phoneme":
            with local_zero_first():
                self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                )
                self.tokenizer._add_tokens(["<n>"])
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None

        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["target_audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["target_audio"].shape[-1],  
            )
        self.wds_vocal_datasets = []
        wds_dataset_agg_weight = []
        if wds_dataset_names:
            wds_dataset_agg_weight = [sum(wds_dataset_weights)]
            self.wds_vocal_datasets = [VocalDataset(
                region=region,
                dataset_names=wds_dataset_names,
                dataset_weights=wds_dataset_weights,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=max_seg_per_track,
                segment_max_phone_len=segment_max_phone_len,
                use_soda_gt_lyrics=True,
                lyrics_confidence=lyrics_confidence,
                tokenizer=self.tokenizer,                
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,            
                handler=wds.warn_and_continue,
                sinking_threshold=sinking_threshold,
                remove_sinking=remove_sinking,
                )]
        self.parquet_vocal_datasets = []
        if parquet_dataset_ids:
            for parquet_id in parquet_dataset_ids:
                self.parquet_vocal_datasets.append(
                    VocalParquetDataset(
                        data_id=parquet_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        use_gt_lyrics=True,
                        lyrics_confidence=lyrics_confidence,
                        tokenizer=self.tokenizer,
                        infer_structure_tags=infer_structure_tags,
                        read_structure_tags=read_structure_tags,
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,            
                        handler=wds.warn_and_continue,
                        sinking_threshold=sinking_threshold,
                        remove_sinking=remove_sinking,                    
                        ))

        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=self.wds_vocal_datasets + self.parquet_vocal_datasets, 
                                 weights=wds_dataset_agg_weight + parquet_dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = [WebPipeline(
            VocalParquetDataset(
                data_id=1528,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=1,
                segment_max_phone_len=segment_max_phone_len,
                use_gt_lyrics=True,
                lyrics_confidence=lyrics_confidence,
                tokenizer=self.tokenizer,
                infer_structure_tags=infer_structure_tags,
                read_structure_tags=read_structure_tags,
                resampled=False,
                shardshuffle=True,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
                sinking_threshold=sinking_threshold,
                remove_sinking=remove_sinking,                    
                ),
            pipeline=[{"compose": [self.bucketize]}],
        )] 

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,  # TODO
            collate_fn=self.collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MixLangVocalWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,        
        use_pipe: bool = True,
        tokenizer: str = "tts_chinese_frontend_model",
        frame_rate: int = 25,
        normalize_audio: bool = False,
        zh_parquet_dataset_ids: List[int] = [],
        zh_parquet_dataset_weights: List[int] = [],
        en_parquet_dataset_ids: List[int] = [],        
        en_parquet_dataset_weights: List[int] = [],
        use_dynamic_batch: str = False,
        lyrics_confidence: float = 0.75,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        max_seg_per_track: int = -1,
        buckets_in_sec: List[int] = [
            20,
            25,
            30,
        ],
        sinking_threshold: float = 0.51,
        remove_sinking: bool = False,
    ):        
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = "tts_chinese_frontend_model"
        elif tokenizer == "phoneme":
            with local_zero_first():
                self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                )
                self.tokenizer._add_tokens(["<n>"])
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None

        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["target_audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["target_audio"].shape[-1],  
            )
        
        self.parquet_vocal_datasets = []
        if zh_parquet_dataset_ids:
            for parquet_id in zh_parquet_dataset_ids:
                self.parquet_vocal_datasets.append(
                    VocalParquetDataset(
                        data_id=parquet_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        use_gt_lyrics=True,
                        lyrics_confidence=lyrics_confidence,
                        tokenizer=self.tokenizer,                
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,            
                        handler=wds.warn_and_continue,
                        sinking_threshold=sinking_threshold,
                        remove_sinking=remove_sinking,                       
                        ))

        if en_parquet_dataset_ids:
            for parquet_id in en_parquet_dataset_ids:
                self.parquet_vocal_datasets.append(
                    VocalParquetDataset(
                        data_id=parquet_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        use_gt_lyrics=True,
                        lyrics_confidence=lyrics_confidence,
                        tokenizer=self.tokenizer,                
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,            
                        handler=wds.warn_and_continue,
                        sinking_threshold=sinking_threshold,
                        remove_sinking=remove_sinking,
                        ))

        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=self.parquet_vocal_datasets, 
                                 weights=zh_parquet_dataset_weights+en_parquet_dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = [WebPipeline(
            VocalDataset(
                region='CN',
                dataset_names="SodaTest",
                dataset_weights=1,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=1,
                segment_max_phone_len=segment_max_phone_len,
                tokenizer=self.tokenizer,                
                use_pipe=use_pipe,
                resampled=False,
                nodesplitter=return_self,                
                handler=wds.warn_and_continue,
                sinking_threshold=sinking_threshold,
                remove_sinking=remove_sinking,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )] 

        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,  # TODO
            collate_fn=collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class SftWebDataModule(DataModule):
    # This DataModule supports both Artist SFT and Lyrics SFT.
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        conditions: str = "style_text,speaker_id,lyrics_tokens",
        use_pipe: bool = True,
        tokenizer: str = "phoneme",
        frame_rate: int = 25,
        normalize_audio: bool = False,        
        parquet_datasets: dict = {},
        use_dynamic_batch: str = False,
        lyrics_field: str = "lyrics",
        lyrics_confidence: float = 0.7,
        read_structure_tags: bool = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        max_seg_per_track: int = -1,
        buckets_in_sec: List[int] = [
            20,
            25,
            30,
        ],
        sample_limit_per_file: int = 1000,
        sinking_threshold: float = 0.51,
        remove_sinking: bool = False,
    ):        
        print(conditions)
        print(parquet_datasets)
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = partial(collate_fn, conditions=conditions)
        assert parquet_datasets

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = "tts_chinese_frontend_model"
        elif tokenizer == "phoneme":
            with local_zero_first():
                self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                )
                self.tokenizer._add_tokens(["<n>"])
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None

        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["target_audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["target_audio"].shape[-1],  
            )
        self.parquet_vocal_datasets = {}
        for split, pd_id_weights in parquet_datasets.items():
            pds, pdws = [], []
            is_train = True if split == 'train' else False
            nodesplitter = shardlists.single_node_only if split == 'train' else return_self
            for pd_id, pd_weight in pd_id_weights:
                pdws.append(pd_weight)
                pds.append(
                    VocalParquetDataset(
                        data_id=pd_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        lyrics_field=lyrics_field,
                        lyrics_confidence=lyrics_confidence,
                        read_structure_tags=read_structure_tags,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        tokenizer=self.tokenizer,                
                        resampled=is_train,
                        shardshuffle=is_train,
                        nodesplitter=nodesplitter,
                        use_pipe=use_pipe,            
                        handler=wds.warn_and_continue,
                        sample_limit_per_file=sample_limit_per_file,
                        sinking_threshold=sinking_threshold,
                        remove_sinking=remove_sinking,
                        ))
            self.parquet_vocal_datasets[split] = WebPipeline(            
                MultiIterableDataset(datasets=pds, weights=pdws),
                pipeline=[{"compose": [self.bucketize]}],
            )
        
        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=self.parquet_vocal_datasets["train"],
            validation_dataset=self.parquet_vocal_datasets["test"],
            predict_dataset=self.parquet_vocal_datasets["train"],  # TODO
            collate_fn=self.collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class LyricsSftWebDataModule(SftWebDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            parquet_datasets={"train":[(1026, 1)], "test":[(1012, 1)]},
            **kwargs,
        )


class ArtistSftWebDataModule(SftWebDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            parquet_datasets={"train":[(1473, 1)], "test":[(1012, 1)]},
            **kwargs,
        )
