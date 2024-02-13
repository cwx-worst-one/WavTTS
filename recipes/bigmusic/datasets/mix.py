from curses.ascii import ETB
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

from recipes.bigmusic.utils.format_utils import normalize_text
from recipes.datasets.mcc.mix import (
    INDEX,
    WebDatasetBufferPreprocessor,
    BaseTransforms,
    DataModule
)
from recipes.datasets.mcc.sami_tokenizer import convert_labels_to_text_id, get_line_break_id

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
LINE_BREAK_PHONE_TOKEN = get_line_break_id()


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
            new_utterances.append([int(utt_start/1000), int(utt_end/1000), u['text'], phone])


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


def group_utterances_with_structure(
    utterances, min_duration, max_duration, new_line_token=' ', structure_tags=None):
    # Segment the full song into segments, each segment consists of multiple utterances.
    for structure_tag in structure_tags:        
        sec_start = int(structure_tag['start_time'])
        sec_end = int(structure_tag['end_time'])    
        if sec_start < 0 or sec_end < 0 or sec_start >= sec_end:
            continue
        sec_tag = structure_tag['tag']
        # TODO (QQ): (find the utterances included in the section)
    return group_utterances(utterances, min_duration, max_duration, time_in_sec=False,
                     new_line_token=new_line_token, infer_structure_tags=True)
    

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
    
    def _convert_to_msec(self, t):
        minute, second = [int(x) for x in t.split(':')]
        return 1000*(minute * 60 + second)

    def _is_valid_lyric_line(self, line, lyric_type="krc"):
        if lyric_type == "krc":
            time_char = line.split(']')
            if len(time_char) != 2:
                return False
            times = time_char[0][1:].split(',')
            if len(times) != 2:
                return False
            return True if times[0].isdigit() and times[1].isdigit() else False
        elif lyric_type == "lrc":
            time_char = line.split(']')
            if len(time_char) != 2:
                return False
            return True if len(time_char[0]) == 9 else False
        else:
            raise ValueError

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

        if self.use_soda_gt_lyrics and "lyrics_gt" in meta:
            utterances = meta.get("lyrics_gt", None)

        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        # Extract structure tags
        structure_tags = []
        if self.read_structure_tags:
            music_structure_tags = meta.get("music_structure", None)            
            structure_tags = format_music_structure_tags(music_structure_tags)
            deepchorus_tags = meta.get("deepchorus", None)            
            structure_tags = format_deepchorus_structure_tags(deepchorus_tags)

        # Extract segments from utterances
        new_line_token = " <n> "
        if isinstance(self.tokenizer, BertTokenizer):
            self.tokenizer.add_special_tokens({'additional_special_tokens': [new_line_token]})

        if structure_tags:
            segments = group_utterances_with_structure(
                utterances, self.min_duration, self.max_duration,                                        
                new_line_token=new_line_token,
                structure_tags=structure_tags)
        else:
            segments = group_utterances(
                utterances, self.min_duration, self.max_duration,
                time_in_sec=False,
                new_line_token=new_line_token,                
                infer_structure_tags=self.infer_structure_tags)
        if len(segments) < 1:
            self._update_stats(skipped=True, message="No valid segment")
            return

        if self.segment_method == "first":
            segments = segments[:1]
        if self.segment_method == "verse_chorus_section":
            segments = [s for s in segments if "chorus" in s[4].lower() or "verse" in s[4].lower() or "section" in s[4].lower()]
        if self.max_seg_per_track > 0:
            random.shuffle(segments)
            segments = segments[:self.max_seg_per_track]

        # Get style text            
        style_text = rewrite_metadata(meta)
        artist_id = ARTIST_ID_MAP_V2[str(meta.get("artist_id", "zh_empty"))]

        if "playlist_extra" in meta:
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
        for segment in segments:
            if segment[1] - segment[0] < 1:
                self._update_stats(skipped=True, message="Skip segment less than 1 sec")
                continue
            start = int(segment[0] * self.sample_rate)
            end = int(segment[1] * self.sample_rate)
            clip = audio[:, start:end]

            # At training and inference, the lyrics tokens should already contain structure tags.
            normalized_text = normalize_text(segment[2], enable_punctuation=True)

            if self.tokenizer == "tts_chinese_frontend_model":
                # TODO (QQ) call SamiOfflineTokenizer directly.
                # TODO (yilin)) handle the section tags and singer tags.
                lines = segment[3].split(new_line_token)
                text_tokens = []
                for line in lines:
                    labels = list(
                        filter(
                            lambda x: x != "", line.split("\n")
                        )
                    )
                    if labels:
                        convert_result = convert_labels_to_text_id(labels)
                        if convert_result:
                            labels, _, _ = convert_result
                        else:
                            print("Invalid phone label to text id conversion", labels)
                            continue
                        line_phone_tokens = labels[0]
                    else:
                        continue
                    text_tokens.append(line_phone_tokens)
                    text_tokens.append(LINE_BREAK_PHONE_TOKEN)                    
                if len(text_tokens) > 0:
                    if len(text_tokens) == 1: 
                        print(text_tokens)
                    text_tokens = np.concatenate(text_tokens, axis=0)
                    text_tokens = torch.from_numpy(text_tokens).long()
                else:
                    text_tokens = None    
            else:
                text_tokens = self.tokenizer(
                    normalized_text, 
                    add_special_tokens=False,
                    return_tensors="pt")["input_ids"].squeeze(dim=0)
            if text_tokens is None or text_tokens.size(-1) < 10:
                self._update_stats(skipped=True, message="Lyrics token too short.")
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
        wds_dataset_names: List[str] = ["Soda"],
        wds_dataset_weights: List[int] = [1],
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
                handler=wds.warn_and_continue
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
                        handler=wds.warn_and_continue                        
                        ))

        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=self.wds_vocal_datasets + self.parquet_vocal_datasets, 
                                 weights=wds_dataset_agg_weight + parquet_dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = [WebPipeline(
            VocalDataset(
                region=region,
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
                handler=wds.warn_and_continue
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
                        handler=wds.warn_and_continue                        
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
                        handler=wds.warn_and_continue                        
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
                handler=wds.warn_and_continue
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
