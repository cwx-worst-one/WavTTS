import io
import random
import json
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
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import BertTokenizer, Wav2Vec2PhonemeCTCTokenizer
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline
import logging, phonemizer
from recipes.bigmusic.datasets.tokenizers.phoneme import MAX_PHONE_LEN
from recipes.bigmusic.utils.format_utils import normalize_text
from recipes.bigmusic.datasets.transforms.lyrics_segment import (
    Segment,
    group_by_fixed_length,
    group_by_variable_length,
)
from recipes.datasets.mcc.mix import (
    INDEX,
    LibriTTSDataset,
    MCCInstrumentalDataset,
    MCCVocalDataset,
    WebDatasetBufferPreprocessor,
    BaseTransforms
)
from recipes.datasets.mcc.sami_tokenizer import convert_labels_to_text_id
from recipes.bigmusic.utils.format_utils import rewrite_metadata
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
    NormalizeAudioToFloat32,
    RandomPad,
    Pad,
    SetAudioDimensions,
    ToTensor,
    RandomResizedCrop,
)
from samantha.utils.webdataset import return_self
from transformers import Wav2Vec2PhonemeCTCTokenizer
import functools

MAX_STYLE_LEN = 16

def pad_crop(sequence, seq_len, dtype, padding_value=0):
    # in item_pad_idx, 0 indicates the values are padded.
    item_pad = torch.full((seq_len,), fill_value=padding_value, dtype=dtype)
    item_pad[:len(sequence)] = torch.as_tensor(sequence[:seq_len])
    item_pad_idx = torch.full((seq_len,), fill_value=0, dtype=int)
    item_pad_idx[:len(sequence)] = torch.ones_like(torch.as_tensor(sequence[:seq_len]), dtype=int)
    return item_pad, item_pad_idx

def ffmpeg_read_audio(audio_bin, sample_rate=24000):
    seg_bin, err = ffmpeg.input("pipe:").output("pipe:", loglevel="error", format="s16le", ar=sample_rate).run(input=audio_bin, quiet=True)
    return (np.frombuffer(seg_bin, dtype="int16") / 32768.0).astype(np.float32)

def select_tag_metadata_from_timestamps(index_data, start: int, end: int) -> Dict[str, List[str]]:
    mir_tags = index_data.get("tags")
    genres = []
    instruments = []
    vocals = []
    
    if len(mir_tags["starts"]):
        def closest_time(times, t):
            difference = lambda times: abs(times - t)
            return min(times, key=difference)

        mir_start = closest_time(mir_tags["starts"], start)
        mir_end = closest_time(mir_tags["ends"], end)
        mir_start_idx = mir_tags["starts"].index(mir_start)
        mir_end_idx = mir_tags["ends"].index(mir_end)

        for idx in range(mir_start_idx, mir_end_idx + 1):
            genres.extend(mir_tags["genre"][idx])
            instruments.extend(mir_tags["instrument"][idx])
            vocals.extend(mir_tags["vocal"][idx])
    return dict(genres=genres, instruments=instruments, vocals=vocals)

def join_words(words):
    result = []
    for ww in words:
        result += ww
    return result


def collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    # TODO: (QQ) make these constants configurable.
    SPEAKER_PAD_ID = 0
    PHONE_PAD_ID = 0 
    STYLE_PAD_ID = 0 
    max_phone_len = int(batch[0]["max_phone_len"])    
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    default_lyrics_token = torch.full((max_phone_len,), PHONE_PAD_ID, dtype=torch.int)
    default_style_token = torch.full((MAX_STYLE_LEN,), STYLE_PAD_ID, dtype=torch.int)
    default_speaker_id = torch.LongTensor([SPEAKER_PAD_ID])    
    audio = []
    style_text = []
    style_tokens = []
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
        audio.append(random_pad(batch[idx]["audio"]))
        
        if isinstance(batch[idx]["style_text"], Tuple):
            style_label = batch[idx]["style_text"][0]
        else:
            style_label = batch[idx]["style_text"]
        style_text.append(style_label)     
        
        construct = lambda key, default: \
            torch.tensor(batch[idx][key]) if key in batch[idx] and batch[idx][key] is not None \
                else default.clone().detach()
        
        normalized_text.append(batch[idx]["normalized_text"])
        
        phoneme_tokens, _ = pad_crop(
            construct("lyrics_tokens", default_lyrics_token), 
            max_phone_len, torch.int, PHONE_PAD_ID)
        lyrics_tokens.append(phoneme_tokens)
        
        speaker_id.append(construct("speaker_id", default_speaker_id))

        style_metadata.append(batch[idx].get("style_metadata", None))
        song_id.append(batch[idx].get("song_id", None))
        shard.append(batch[idx].get("shard", None))
        worker_id.append(batch[idx].get("worker_id", None))
        dataset_name.append(batch[idx].get("dataset_name", None))

    stacked_audio = torch.stack(audio, dim=0)
    if stacked_audio.dim() == 3:
        stacked_audio = stacked_audio.squeeze(1)
    return {
        "target_audio": torch.stack(audio, dim=0),
        # "style_audio": stacked_audio,
        "style_text": style_text,
        "normalized_text": normalized_text,
        "lyrics": normalized_text,
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "speaker_id": torch.stack(speaker_id),
        "conditions": "style_text,lyrics_tokens",
        "dataset_name": dataset_name,

        # DEBUG:
        "style_metadata": style_metadata,
        "song_id": song_id,
        "shard": shard,
        "worker_id": worker_id,

    }


def voiceclone_collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    SPEAKER_PAD_ID = 0
    PHONE_PAD_ID = 0 
    STYLE_PAD_ID = 0 
    max_phone_len = int(batch[0]["max_phone_len"])    
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    default_lyrics_token = torch.full((max_phone_len,), PHONE_PAD_ID, dtype=torch.int)
    default_style_token = torch.full((MAX_STYLE_LEN,), STYLE_PAD_ID, dtype=torch.int)
    default_speaker_id = torch.LongTensor([SPEAKER_PAD_ID])    
    audio = []
    vocal_audio = []
    style_text = []
    style_tokens = []
    normalized_text = []
    lyrics_tokens = []
    speaker_id = []     

    # DEBUG
    style_metadata = []
    song_id = []
    shard = []
    worker_id = []
    for idx in range(len(batch)):        
        audio.append(random_pad(batch[idx]["audio"]))

        #TODO if for voice clone -> crop; singsong -> pad, do it in dataloader?
        vocal_audio.append(batch[idx]["vocal_audio"])
        
        if isinstance(batch[idx]["style_text"], Tuple):
            style_label = batch[idx]["style_text"][0]
        else:
            style_label = batch[idx]["style_text"]
        style_text.append(style_label)     
        
        style_text_tokens, _ = pad_crop(
            torch.tensor(batch[idx].get("style_tokens", default_style_token.detach().clone())), 
            MAX_STYLE_LEN, torch.int, STYLE_PAD_ID)
        style_tokens.append(style_text_tokens)

        style_metadata.append(batch[idx].get("style_metadata", None))
        song_id.append(batch[idx].get("song_id", None))
        shard.append(batch[idx].get("shard", None))
        worker_id.append(batch[idx].get("worker_id", None))

        normalized_text.append(batch[idx]["normalized_text"])
        
        phoneme_tokens, _ = pad_crop(
            torch.tensor(batch[idx].get("lyrics_tokens", default_lyrics_token.detach().clone())), 
            max_phone_len, torch.int, PHONE_PAD_ID)
        lyrics_tokens.append(phoneme_tokens)
        
        speaker_id.append(batch[idx].get("speaker_id", default_speaker_id.detach().clone())) 

    stacked_audio = torch.stack(audio, dim=0)
    if stacked_audio.dim() == 3:
        stacked_audio = stacked_audio.squeeze(1)

    return {
        "target_audio": stacked_audio,
        "style_audio":  stacked_audio,
        "vocal_audio": torch.stack(vocal_audio, dim=0),
        "style_text": style_text,
        # "style_tokens": torch.stack(style_tokens),
        "normalized_text": normalized_text,
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "speaker_id": torch.stack(speaker_id),
        "conditions": "style_tag,lyrics_tokens,vocal_audio",

        # DEBUG:
        "style_metadata": style_metadata,
        "song_id": song_id,
        "shard": shard,
        "worker_id": worker_id,
    }


def singsong_collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:

    SPEAKER_PAD_ID = 0
    PHONE_PAD_ID = 0 
    STYLE_PAD_ID = 0 
    max_phone_len = int(batch[0]["max_phone_len"])    
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    default_lyrics_token = torch.full((max_phone_len,), PHONE_PAD_ID, dtype=torch.int)
    default_style_token = torch.full((MAX_STYLE_LEN,), STYLE_PAD_ID, dtype=torch.int)
    default_speaker_id = torch.LongTensor([SPEAKER_PAD_ID])    
    audio = []
    vocal_audio = []
    acc_audio = []
    style_text = []
    style_tokens = []
    normalized_text = []
    lyrics_tokens = []
    speaker_id = []     

    # DEBUG
    style_metadata = []
    song_id = []
    shard = []
    worker_id = []
    for idx in range(len(batch)):        
        audio.append(random_pad(batch[idx]["audio"]))
        vocal_audio.append(random_pad(batch[idx]["vocal_audio"]))
        
        if isinstance(batch[idx]["style_text"], Tuple):
            style_label = batch[idx]["style_text"][0]
        else:
            style_label = batch[idx]["style_text"]
        style_text.append(style_label)     
        
        style_text_tokens, _ = pad_crop(
            torch.tensor(batch[idx].get("style_tokens", default_style_token.detach().clone())), 
            MAX_STYLE_LEN, torch.int, STYLE_PAD_ID)
        style_tokens.append(style_text_tokens)

        normalized_text.append(batch[idx]["normalized_text"])
        
        phoneme_tokens, _ = pad_crop(
            torch.tensor(batch[idx].get("lyrics_tokens", default_lyrics_token.detach().clone())), 
            max_phone_len, torch.int, PHONE_PAD_ID)
        lyrics_tokens.append(phoneme_tokens)
        
        speaker_id.append(batch[idx].get("speaker_id", default_speaker_id.detach().clone()))        


        style_metadata.append(batch[idx].get("style_metadata", None))
        song_id.append(batch[idx].get("song_id", None))
        shard.append(batch[idx].get("shard", None))
        worker_id.append(batch[idx].get("worker_id", None))

    stacked_audio = torch.stack(audio, dim=0)
    if stacked_audio.dim() == 3:
        stacked_audio = stacked_audio.squeeze(1)

    return {
        "target_audio": stacked_audio,  # for singsong, this is pure BGM
        "style_audio": stacked_audio,
        "vocal_audio": torch.stack(vocal_audio, dim=0),
        "style_text": style_text,
        # "style_tokens": torch.stack(style_tokens),
        "normalized_text": normalized_text,
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "speaker_id": torch.stack(speaker_id),
        # "conditions": "style_tag,vocal_audio",  # for singsong, we don't need lyrics anymore, for inverse singsong, we need lyrics
        "conditions": "lyrics_tokens,vocal_audio",

        # DEBUG:
        "style_metadata": style_metadata,
        "song_id": song_id,
        "shard": shard,
        "worker_id": worker_id,
    }

def inverse_singsong_vc_collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:

    SPEAKER_PAD_ID = 0
    PHONE_PAD_ID = 0 
    STYLE_PAD_ID = 0 
    max_phone_len = int(batch[0]["max_phone_len"])    
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    default_lyrics_token = torch.full((max_phone_len,), PHONE_PAD_ID, dtype=torch.int)
    default_style_token = torch.full((MAX_STYLE_LEN,), STYLE_PAD_ID, dtype=torch.int)
    default_speaker_id = torch.LongTensor([SPEAKER_PAD_ID])    
    audio = []
    vocal_audio = []
    style_audio = []
    style_text = []
    style_tokens = []
    normalized_text = []
    lyrics_tokens = []
    speaker_id = []     

    # DEBUG
    style_metadata = []
    song_id = []
    shard = []
    worker_id = []
    for idx in range(len(batch)):        
        audio.append(random_pad(batch[idx]["audio"]))
        vocal_audio.append(random_pad(batch[idx]["vocal_audio"]))
        style_audio.append(random_pad(batch[idx]["style_audio"]))
        
        if isinstance(batch[idx]["style_text"], Tuple):
            style_label = batch[idx]["style_text"][0]
        else:
            style_label = batch[idx]["style_text"]
        style_text.append(style_label)     
        
        style_text_tokens, _ = pad_crop(
            torch.tensor(batch[idx].get("style_tokens", default_style_token.detach().clone())), 
            MAX_STYLE_LEN, torch.int, STYLE_PAD_ID)
        style_tokens.append(style_text_tokens)

        normalized_text.append(batch[idx]["normalized_text"])
        
        phoneme_tokens, _ = pad_crop(
            torch.tensor(batch[idx].get("lyrics_tokens", default_lyrics_token.detach().clone())), 
            max_phone_len, torch.int, PHONE_PAD_ID)
        lyrics_tokens.append(phoneme_tokens)
        
        speaker_id.append(batch[idx].get("speaker_id", default_speaker_id.detach().clone()))        


        style_metadata.append(batch[idx].get("style_metadata", None))
        song_id.append(batch[idx].get("song_id", None))
        shard.append(batch[idx].get("shard", None))
        worker_id.append(batch[idx].get("worker_id", None))

    stacked_audio = torch.stack(audio, dim=0)
    if stacked_audio.dim() == 3:
        stacked_audio = stacked_audio.squeeze(1)

    stacked_style_audio = torch.stack(style_audio, dim=0)
    if stacked_style_audio.dim() == 3:
        stacked_style_audio = stacked_style_audio.squeeze(1)

    return {
        "target_audio": stacked_audio,  # for singsong, this is pure BGM
        "style_audio": stacked_style_audio,
        "vocal_audio": torch.stack(vocal_audio, dim=0),
        "style_text": style_text,
        # "style_tokens": torch.stack(style_tokens),
        "normalized_text": normalized_text,
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "speaker_id": torch.stack(speaker_id),
        "conditions": "style_audio,lyrics_tokens,vocal_audio",

        # DEBUG:
        "style_metadata": style_metadata,
        "song_id": song_id,
        "shard": shard,
        "worker_id": worker_id,
    }

def vpp_collate_fn(batch: List[torch.Tensor], app_type: str) -> Dict[str, torch.Tensor]:

    SPEAKER_PAD_ID = 0
    PHONE_PAD_ID = 0 
    STYLE_PAD_ID = 0 
    max_phone_len = int(batch[0]["max_phone_len"])
    max_length = max([x["audio"].shape[-1] for x in batch])
    target_audio_pad = RandomPad(n_samples=max_length)
    
    acc_pad = target_audio_pad
    vocal_pad = target_audio_pad
    if app_type == "singsong":
        target_audio_pad = Pad(n_samples=max_length)
        vocal_max_length = max([x["vocal_audio"].shape[-1] for x in batch])
        vocal_pad = Pad(n_samples=vocal_max_length)
        acc_max_length = max([x["acc_audio"].shape[-1] for x in batch])
        acc_pad = Pad(n_samples=acc_max_length)

    elif app_type == "singsong_inverse":
        vocal_max_length = max([x["vocal_audio"].shape[-1] for x in batch])
        vocal_pad = RandomPad(n_samples=vocal_max_length)
        acc_max_length = max([x["acc_audio"].shape[-1] for x in batch])
        acc_pad = RandomPad(n_samples=acc_max_length)
    elif app_type == "singsong_inverse_vc":
        acc_max_length = max([x["acc_audio"].shape[-1] for x in batch])
        acc_pad = RandomPad(n_samples=acc_max_length)


    default_lyrics_token = torch.full((max_phone_len,), PHONE_PAD_ID, dtype=torch.int)
    default_style_token = torch.full((MAX_STYLE_LEN,), STYLE_PAD_ID, dtype=torch.int)
    default_speaker_id = torch.LongTensor([SPEAKER_PAD_ID])    
    audio = []
    vocal_audio = []
    acc_audio = []
    style_text = []
    style_tokens = []
    normalized_text = []
    lyrics_tokens = []
    speaker_id = []     

    # DEBUG
    style_metadata = []
    song_id = []
    shard = []
    worker_id = []
    uttid = []
    idx_seg = []
    
    for idx in range(len(batch)):       
        audio.append(target_audio_pad(batch[idx]["audio"]))


        if app_type == "singsong":
            vocal_audio.append(vocal_pad(batch[idx]["vocal_audio"]))
        elif app_type == "singsong_inverse":
            # not using it though
            vocal_audio.append(vocal_pad(batch[idx]["vocal_audio"]))
        else:
            # for all voice clone, just stack them!
            vocal_audio.append(batch[idx]["vocal_audio"])
        

        acc_audio.append(acc_pad(batch[idx]["acc_audio"]))


        if isinstance(batch[idx]["style_text"], Tuple):
            style_label = batch[idx]["style_text"][0]
        else:
            style_label = batch[idx]["style_text"]
        style_text.append(style_label)     
        
        style_text_tokens, _ = pad_crop(
            batch[idx].get("style_tokens", default_style_token.detach().clone()),
            MAX_STYLE_LEN, torch.int, STYLE_PAD_ID)
        style_tokens.append(style_text_tokens)

        normalized_text.append(batch[idx]["normalized_text"])
        
        phoneme_tokens, _ = pad_crop(
            batch[idx].get("lyrics_tokens", default_lyrics_token.detach().clone()),
            max_phone_len, torch.int, PHONE_PAD_ID)
        lyrics_tokens.append(phoneme_tokens)
        
        speaker_id.append(batch[idx].get("speaker_id", default_speaker_id.detach().clone()))        


        style_metadata.append(batch[idx].get("style_metadata", None))
        song_id.append(batch[idx].get("song_id", None))
        shard.append(batch[idx].get("shard", None))
        worker_id.append(batch[idx].get("worker_id", None))
        uttid.append(batch[idx].get("uttid", None))
        idx_seg.append(batch[idx].get("idx_seg", None))

    # TODO: Clarify the dimensions in audio, vocal_audio, and acc_audio. Squeeze here assumes certain redundant dimensions which may not be presented. (vibertthio)
    stacked_audio = torch.stack(audio, dim=0)
    if stacked_audio.dim() == 3:
        stacked_audio = stacked_audio.squeeze(1)

    stacked_vocal_audio = torch.stack(vocal_audio, dim=0)

    stacked_acc_audio = torch.stack(acc_audio, dim=0)
    if stacked_acc_audio.dim() == 3:
        stacked_acc_audio = stacked_acc_audio.squeeze(1)

    if app_type == "vclone":
        conditions = "acc_audio,lyrics_tokens,vocal_audio"
    elif app_type == "singsong":
        conditions = "noisy_vocal_audio"
    elif app_type == "singsong_inverse":
        conditions = "lyrics_tokens,acc_audio"
    elif app_type == "singsong_inverse_vc":
        conditions = "lyrics_tokens,acc_audio,vocal_audio"


    return {
        "target_audio": stacked_audio,  # for singsong, this is pure BGM
        "acc_audio": stacked_acc_audio,
        "vocal_audio": stacked_vocal_audio,
        "style_text": style_text,
        # "style_tokens": torch.stack(style_tokens),
        "normalized_text": normalized_text,
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "speaker_id": torch.stack(speaker_id),
        "conditions": conditions,

        # DEBUG:
        "style_metadata": style_metadata,
        "song_id": song_id,
        "shard": shard,
        "worker_id": worker_id,
        "uttid": uttid,
        "idx_seg": idx_seg,
    }

def get_collate_fn(app_type):
    collate_fn = functools.partial(vpp_collate_fn, app_type=app_type)
    return collate_fn


def group_utterances(utterances, min_duration, max_duration, time_in_sec=False, include_intro=False, new_line_token=' '):
    new_utterances = []
    # TODO (qq) Skip the first utterance because title is in QQ music. Fix it properly with VAD signal.
    for i, u in enumerate(utterances[1:]):
        phone = u.get('phoneme', '')
        phone = phone if phone else ''
        
        if "startTimeMs" in u:
            u["start_time"] = u["startTimeMs"]
            time_in_sec = False

        if "lyrics" in u:
            u["text"] = u["lyrics"]

        utt_start = u.get('start_time', None)
        if utt_start is None:
            continue
        utt_start = int(utt_start)
        utt_end = u.get('end_time', None)
        if utt_end is None:
            if i < (len(utterances) - 1):
                utt_end = max(int(utterances[i+1].get('start_time', 0)), utt_start)
            else:
                continue
        utt_end = int(utt_end)
        if time_in_sec:
            new_utterances.append((utt_start, utt_end, u['text'], phone))
        else:
            new_utterances.append((int(utt_start/1000), int(utt_end/1000), u['text'], phone))
    utterances = new_utterances
    
    if include_intro:
        u = utterances[0]
        if u[0] > 0:            
            utterances.insert(0, (0, u[0], "", ""))
    for i in range(len(utterances)-1):
        # Skip the entire track if there are overlapping utterances for more than 5 seconds overlapping.
        if utterances[i][1] > utterances[i+1][0] + 5:
            logging.warning("utterances need to be non-overlapping")
            return []
    utterances = [u for u in utterances if u[1]-u[0] > 0]
    segs = []
    close_end_index = 0
    start_index, cur_seg = close_end_index, []        
    while close_end_index < len(utterances):
        if utterances[close_end_index][1] - utterances[start_index][0] < min_duration:
            cur_seg.append(utterances[close_end_index])                
            close_end_index += 1
        else:
            far_end_index = close_end_index
            while far_end_index < len(utterances) and utterances[far_end_index][1] - utterances[start_index][0] <= max_duration:
                far_end_index += 1
            if far_end_index == close_end_index:
                close_end_index = start_index + 1
            else:
                end_index = random.randint(close_end_index+1, far_end_index)
                cur_seg.extend([utterances[pos] for pos in range(close_end_index, end_index)])
                segs.append([
                    cur_seg[0][0],  # new start
                    cur_seg[-1][1],  # new end
                    new_line_token.join([u[2] for u in cur_seg]),
                    new_line_token.join([u[3] for u in cur_seg])
                ])
                close_end_index = end_index
            # Move to the next vocal starting point
            while close_end_index < len(utterances):
                if len(utterances[close_end_index][2]) >= 2:
                    break
                close_end_index += 1
            if close_end_index < len(utterances):
                start_index, cur_seg = close_end_index, []        
    return segs


def group_utterances_vpp(utterances, min_duration, max_duration, time_in_sec=False, include_intro=False):
    if time_in_sec:
        utterances = [(int(u['start_time']), int(u['end_time']), u['text'], u['words']) for u in utterances]            
    else:
        utterances = [(int(u['start_time']/1000), int(u['end_time']/1000), u['text'], u['words']) for u in utterances]
    if include_intro:
        u = utterances[0]
        if u[0] > 0:            
            utterances.insert(0, (0, u[0], ""))    
    for i in range(len(utterances)-1):
        if utterances[i][1] > utterances[i+1][0]:
            logging.warning("utterances need to be non-overlapping")
            return []
    utterances = [u for u in utterances if u[1]-u[0] > 0]
    segs = []
    i = 0
    s, cur_seg = i, []        
    while i < len(utterances):
        if utterances[i][1] - utterances[s][0] < min_duration:
            cur_seg.append(utterances[i])                
            i += 1
        else:
            j = i
            while j < len(utterances) and utterances[j][1] - utterances[s][0] <= max_duration:
                j += 1
            if j == i:
                i = s + 1
            else:
                k = random.randint(i+1, j)
                cur_seg.extend([utterances[k] for k in range(i, k)])
                segs.append([cur_seg[0][0],     # start time 
                             cur_seg[-1][1],    # end time
                            ' '.join([u[2] for u in cur_seg]),          # join text
                            join_words([u[3] for u in cur_seg]),       # join words
                            ])
                i = k
            # Move to the next vocal starting point
            while i < len(utterances):
                if len(utterances[i][2]) >= 2:
                    break
                i += 1
            if i < len(utterances):
                s, cur_seg = i, []        
    return segs


########################## English Datasets ######################

class LibriTTSDataset(WebPipeline):
    name = "LibriTTS"
    data_sample_rate = 24000

    def __init__(
        self,
        urls="pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/train-clean-360/{00000..00007}.tar",
        sample_rate=24000,
        min_duration: int = 5,
        max_duration: int = 30,
        handler=wds.warn_and_continue,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration  
        self.handler = handler      
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"compose": [self.transform]}]

        with local_zero_first():
            self.phoneme_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        phonemizer.logger.get_logger().setLevel(logging.ERROR)

        super().__init__(dataset, pipeline)

    def transform(self, item_yielder) -> Dict[str, Any]:
        for item in item_yielder:
            try:
                audio = self.base_transform(item["audio.npy"])
            except Exception as e:
                self.handler(e)
                continue
            if audio.size(-1) < self.min_duration * self.sample_rate:
                continue
            if audio.size(-1) > self.max_duration * self.sample_rate:
                continue
            normalized_text = normalize_text(item["normalized_text.txt"])
            phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]                                
            yield {
                "audio": audio, 
                "normalized_text": normalized_text,
                "lyrics_tokens": phoneme_tokens,
                "style_text": "speech",
                "audio_type": "speech",
                "max_phone_len": MAX_PHONE_LEN, # TODO: (AS) remove hardcoded value
            }       

class LibrilightDataset(WebPipeline):
    name = "LibrilightASR"
    data_sample_rate = 24000

    
    def __init__(
        self,
        url2index: str = "hdfs://harunava/home/byte_speech_sv/data/speech/librilight_asr_npy/url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        handler=wds.warn_and_continue,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.handler = handler

        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        dataset = IndexedWebDataset(url2index=url2index, handler=handler, **kwargs)
        pipeline = ["decode", {"compose": [self.transform]}]

        with local_zero_first():
            self.phoneme_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        phonemizer.logger.get_logger().setLevel(logging.ERROR)

        super().__init__(dataset, pipeline)

    def transform(self, item_yielder) -> Dict[str, Any]:
        for item in item_yielder:
            try:
                audio = self.base_transform(item[self.audio_key])
            except Exception as e:
                self.handler(e)
                continue
            utterances = item["__index_data__"]
            if len(utterances) == 0:
                continue
            segments = group_utterances(utterances, self.min_duration, self.max_duration, time_in_sec=True)
            for segment in segments:                
                start = int(segment[0] * self.sample_rate)
                end = int(segment[1] * self.sample_rate)
                clip = audio[:, start:end]
                normalized_text = normalize_text(segment[2])
                phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]                                
                yield {
                    "audio": clip, 
                    "normalized_text": normalized_text,
                    "lyrics_tokens": phoneme_tokens,
                    "style_text": "speech",
                    "audio_type": "speech",
                    "max_phone_len": MAX_PHONE_LEN, # TODO: (AS) remove hardcoded value
                }               

class MCCInstrumentalDataset(WebPipeline):
    name = "DecoderMCCInstrumental"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores/npy_url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        max_style_token_seq_len: int = 16,
        loudness_ratio_threshold: float = 0.2,
        normalize_audio: bool = True,
        # filtering
        aed_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        handler=wds.warn_and_continue,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_style_token_seq_len = max_style_token_seq_len
        self.aed_filtered = aed_filtered
        self.audio_metrics_filtered = audio_metrics_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = exclude_licenses
        self.handler = handler
        self.is_loud = LoudnessCheck(
            sample_rate, min_volume_threshold, loudness_ratio_threshold
        )
        # self.t5_text_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        base_transforms = []
        if audio_key == "mp3":
            self.read_mp3 = ReadMP3(sample_rate)
            base_transforms.append(lambda x: self.read_mp3(io.BytesIO(x)))
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
            # base_transforms.append(NormalizeAudio())
        if self.data_sample_rate != sample_rate and audio_key != "mp3":
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)
        dataset = IndexedWebDataset(url2index=url2index, handler=handler, **kwargs)
        pipeline = ["decode", {"compose": [self.transform]}]
        super().__init__(dataset, pipeline)

    def is_audio_metrics_good(self, audio_metrics: Dict[str, Any]) -> Tuple[bool, str]:
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

    def is_metadata_good(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        if self.aed_filtered and not metadata.get("aed_filtered", False):
            return False
        if self.avoid_sound_effect and metadata.get("final_theme") == "Sound Effect":
            return False
        if len(self.exclude_licenses) > 0:
            for license in metadata.get("license_types", []):
                if license in self.exclude_licenses:
                    return False
        audio_metrics_is_good = self.is_audio_metrics_good(
            metadata.get("audio_metrics", {})
        )
        if self.audio_metrics_filtered and not audio_metrics_is_good:
            return False
        return True

    def transform(self, item_yielder) -> Dict[str, Any]:
        for item in item_yielder:
            style_text = rewrite_metadata(item["__index_data__"], type="Instrumental")            
            # style_tokens = torch.LongTensor(self.t5_text_tokenizer.encode(
            #     style_text, padding='max_length', max_length=self.max_style_token_seq_len))
            if not self.is_metadata_good(item["__index_data__"]):
                continue
            try:
                audio = self.base_transform(item[self.audio_key])
            except Exception as e:
                self.handler(e)
                continue
            if audio.size(-1) < self.max_duration * self.sample_rate:
                continue
            for i in range(4):            
                duration = (
                    random.randint(self.min_duration, self.max_duration) * self.sample_rate
                )
                start = random.randint(0, audio.size(-1) - duration)
                clip = audio[:, start : start + duration]
                if clip.dim() == 1:
                    clip = clip.unsqueeze(0)
                if not self.is_loud(clip):
                    continue
                yield {
                    "audio": clip,
                    "style_text": style_text,
                    # "style_tokens": style_tokens,
                    "normalized_text": "",
                    "max_phone_len": MAX_PHONE_LEN, # TODO: (AS) remove hardcoded value
                    }
  
class MCCVocalDataset(MCCInstrumentalDataset):
    name = "DecoderMCCVocal"    
    def __init__(
        self,
        url2index: str = "/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        segment_method: str = "random", # "first", "random"
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        max_seg_per_track: int = -1,
        **kwargs,
    ):
        with local_zero_first():
            self.phoneme_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        # To silence espeak logging warnings: "WARNING - words count mismatch on 100.0% of the lines". Must be set after tokenizer is initialized
        phonemizer.logger.get_logger().setLevel(logging.ERROR)

        super().__init__(
            url2index=url2index,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            **kwargs,
        )
        self.lyrics_confidence = lyrics_confidence
        self.segment_method = segment_method
        self.segment_max_phone_len = segment_max_phone_len
        self.include_intro = include_intro
        self.max_seg_per_track = max_seg_per_track

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        confidence = float(utterance["additions"]["confidence"])
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if confidence < self.lyrics_confidence:
            return False
        return True

    def is_confident_lyrics(self, utterance, threshold):
        conf = 0
        for utt in utterance:
            conf += float(utt["additions"]["confidence"])
        conf /= len(utterance)
        return True if conf > threshold else False

    def extract_metadata_and_utterances(self, index_data):
        if 'metadata' in index_data:
            metadata = index_data['metadata']
        else:
            metadata = index_data
            
        # Hiphop has format metadata: {..., lyrics: []}, THe rest has format { metadata: {}, lyrics: []}
        if 'lyrics' in metadata and metadata['lyrics'] is not None:
            utterances = metadata['lyrics'].get('utterances')
        elif 'lyrics' in index_data and index_data['lyrics'] is not None:
            utterances = index_data['lyrics'].get('utterances')
        else:
            utterances = None

        return metadata, utterances

    def transform(self, item_yielder) -> Dict[str, Any]:
        for item in item_yielder:
            metadata, utterances = self.extract_metadata_and_utterances(item['__index_data__'])
            if not self.is_metadata_good(metadata):
                continue
            if utterances is None:
                continue

            style_text = rewrite_metadata(metadata)
            # style_tokens = torch.LongTensor(self.t5_text_tokenizer.encode(
            #     style_text, padding='max_length', max_length=self.max_style_token_seq_len))
            try:
                audio = self.base_transform(item[self.audio_key])
            except Exception as e:
                self.handler(e)
                continue
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            if not self.is_confident_lyrics(utterances, 0.8):
                continue
            segments = group_utterances(utterances, self.min_duration, self.max_duration, 
                                        time_in_sec=False, include_intro=self.include_intro)
            if len(segments) < 1:
                continue
            if self.segment_method == "first":
                segments = segments[:1]
            elif self.max_seg_per_track > 0:
                random.shuffle(segments)
                segments = segments[:self.max_seg_per_track]
            for segment in segments:
                if segment[1] - segment[0] < 1:
                    print("short segment")
                    continue
                start = int(segment[0] * self.sample_rate)
                end = int(segment[1] * self.sample_rate)
                clip = audio[:, start:end]
                if clip.shape[-1] < self.sample_rate * self.min_duration // 2:
                    print('audio too short', clip.shape)
                    continue

                # TODO!!!!!
                # if not self.is_loud(clip):
                #     continue

                normalized_text = normalize_text(segment[2])
                phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]                                
                
                shard = self.get_shard_info(item)
                worker_id = self.get_worker_info()

                yield {
                    "audio": clip, 
                    # "style_tokens": style_tokens,
                    "style_text": style_text,
                    "normalized_text": normalized_text,
                    "lyrics_tokens": phoneme_tokens,
                    "max_phone_len": self.segment_max_phone_len,

                    # DEBUG
                    "song_id": metadata["meta_song_id"],
                    "style_metadata": metadata,
                    "shard": shard,
                    "worker_id": worker_id,
                }                


class ReadMP3BytesIO(ReadMP3):
    def __call__(self, mp3: bytes) -> np.ndarray:
        return super().__call__(io.BytesIO(mp3))


class SingsongDataset(MCCInstrumentalDataset):
    name = "DecoderSingsong"    
    def __init__(
        self,
        url2index: str = "/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "mp3",
        audio_map_keys: dict = {"acc_audio": "mss_acc", "vocal_audio": "mss_vocal"},
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        segment_method: str = "random", # "first", "random"
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        max_seg_per_track: int = -1,
        audio_format: str = "mp3",
        voice_clone_duration: int = -1,
        voice_clone_random_start: bool = False,
        app_type: str = "vclone",
        predict_mode: str = 'FULL',
        handler: Callable = wds.ignore_and_continue,
        **kwargs,
    ):
        with local_zero_first():
            self.phoneme_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        # To silence espeak logging warnings: "WARNING - words count mismatch on 100.0% of the lines". Must be set after tokenizer is initialized
        phonemizer.logger.get_logger().setLevel(logging.ERROR)       

        super().__init__(
            url2index=url2index,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            **kwargs,
        )
        self.lyrics_confidence = lyrics_confidence
        self.segment_method = segment_method
        self.segment_max_phone_len = segment_max_phone_len
        self.include_intro = include_intro
        self.max_seg_per_track = max_seg_per_track
        self.audio_format = audio_format
        self.voice_clone_duration = voice_clone_duration
        self.voice_clone_random_start = voice_clone_random_start
        self.app_type = app_type
        self.predict_mode = predict_mode
        self.audio_map_keys = audio_map_keys
        self.handler = handler


        # prepare base audio transform
        if self.audio_format == 'npy':
            self.read_mp3 = lambda x: x
        else:
            self.read_mp3 = ReadMP3BytesIO(self.sample_rate, self.audio_format, fast=(self.audio_format=='mp3'))
        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_audio_transform = Compose(
            [
                self.read_mp3,
                self.to_tensor,
                self.audio_dim, 
                self.normalize_audio_fp32,
            ]
        )
        self.base_vocal_audio_transform = Compose(
            [
                self.to_tensor,
                self.audio_dim, 
                self.normalize_audio_fp32,
            ]
        )

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        confidence = float(utterance["additions"]["confidence"])
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if confidence < self.lyrics_confidence:
            return False
        return True

    # def is_confident_lyrics(self, utterance, threshold):
    #     conf = 0
    #     for utt in utterance:
    #         conf += float(utt["additions"]["confidence"])
    #     conf /= len(utterance)
    #     return True if conf > threshold else False

    def is_confident_lyrics(self, utterance, threshold):        
        conf, num_utt = 0, 0
        for utt in utterance:
            # TODO(QQ update the filed
            if utt['text'].strip():
                conf += float(utt["confidence"])
                num_utt += 1
        conf /= num_utt
        return True if conf > threshold else False


    def audio_transform(self, item):
        # Extract audio Wavs

        # resso: style audio -> mss_acc, vocal_audio -> mss_vocal
        # karaoke:  style audio -> acc.mp3, target_audio -> full.mp3, vocal_audio -> vocal.mp3        
        cached_wavs = {}
        audio_wavs = {}

        for name, audio_key in self.audio_map_keys.items():
            if audio_key in cached_wavs:
                audio = cached_wavs[audio_key]
            else:
                try:
                    audio = self.base_audio_transform(item[audio_key])[0, :]
                    cached_wavs[audio_key] = audio
                    assert len(audio.shape) == 1, 'Invalid audio shape'
                except Exception as e:
                    print(f"[MP3 decoding error] - {name} - {e}")
                    # self._update_stats(skipped=True)
                    self.handler(e)
                    return
            audio_wavs[name] = audio
        del cached_wavs

        # Hack: Resso MSS does not contain mixture. For mixture data, we must add acc + vocals
        if 'target_audio' not in audio_wavs:
            assert 'acc_audio' in audio_wavs and 'vocal_audio' in audio_wavs, 'Must provide target audio or mulan and vocal audio'
            try:
                audio_wavs['target_audio'] = audio_wavs['acc_audio'] + audio_wavs['vocal_audio']
            except Exception as e:
                print(f"the size of style audio is different from vocal audio in Resso")
                self.handler(e)
                return
        
        if "other_vocal.npy" in item: # provide other song of same artist
            other_vocal = item["other_vocal.npy"]
            for i in range(other_vocal.shape[0]):
                other_vocal_prompt = other_vocal[i]
                other_vocal_prompt = self.base_vocal_audio_transform(other_vocal_prompt)[0, :]
                audio_wavs[f'other_vocal_{i}'] = other_vocal_prompt


        # Normalize wavs based on target_audio
        # target_audio = audio_wavs['target_audio']
        # for name, audio_wav in audio_wavs.items():
        #     audio_wavs[name] = self.normalize_audio(audio_wav, target_audio)
        return audio_wavs


    def find_nonrelevant_vocal(self, idx_seg, segments):
        num_candidates = len(segments)
        mid = num_candidates // 2
        if idx_seg < mid:
            possible_idx = idx_seg + mid
            if possible_idx >= num_candidates:
                possible_idx = num_candidates - 1
        else:
            possible_idx = idx_seg - mid
            if possible_idx < 0:
                possible_idx = 0

        start, end, lyrics, words = segments[possible_idx]
        runner = start
        other_lyrics_pieces = list()
        for word in words:
            runner = word['end_time']
            if runner - start < self.voice_clone_duration:
                other_lyrics_pieces.append(word['text'])
            else:
                break

        end = start + self.voice_clone_duration
        other_lyrics_pieces = " ".join(other_lyrics_pieces)
        return start, end, other_lyrics_pieces
            

        


    def transform(self, item_yielder) -> Dict[str, Any]:
        for item in item_yielder:
            if not self.is_metadata_good(item["__index_data__"]):
                continue
            lyrics = item["__index_data__"].get("lyrics", None)
            if lyrics is None:
                # for voice clone, we need to add lyrics
                if self.app_type == "vclone":
                    continue
                # for singsong it is not necessary to add lyrics
            style_text = rewrite_metadata(item["__index_data__"])


            # unify resso and karaoke 
            # resso: style audio -> mss_acc, vocal_audio -> mss_vocal
            # karaoke:  style audio -> acc.mp3, target_audio -> full.mp3, vocal_audio -> vocal.mp3
            audio_wavs = self.audio_transform(item)
            if audio_wavs is None:
                continue
            for name, audio in audio_wavs.items():
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                    audio_wavs[name] = audio

            # Now in audio wavs, there should be target_audio and vocal_audio, style audio

            utterances = item["__index_data__"]["lyrics"].get("utterances", None)
            if utterances is None:
                continue
                
            #TODO count of lyrics for MCC 1m; no confidence score for karaoke yet
            # if not self.is_confident_lyrics(utterances, 0.8):
            #     continue
            segments = group_utterances_vpp(utterances, self.min_duration, self.max_duration, 
                                        time_in_sec=False, include_intro=self.include_intro)
            index_data = item["__index_data__"]

            if len(segments) < 1:
                continue
            if self.segment_method == "first":
                segments = segments[:1]
            elif self.max_seg_per_track > 0:
                random.shuffle(segments)
                segments = segments[:self.max_seg_per_track]

            index_data = item["__index_data__"]
            try:
                metadata = index_data["metadata"]
            except:
                metadata = index_data
            for idx_seg, segment in enumerate(segments):
                if segment[1] - segment[0] < 1:
                    print("short segment")
                    continue
                start = int(segment[0] * self.sample_rate)
                end = int(segment[1] * self.sample_rate)
                current_start = start
                if end > audio_wavs['target_audio'].shape[1]:
                    continue  # somehow failed mss result
                clip = audio_wavs['target_audio'][:, start:end]


                # if self.app_type == "vclone":
                #     # non overlapping
                #     target_start = self.voice_clone_duration * self.sample_rate
                #     clip = audio_wavs['target_audio'][:, start+target_start:end]

                vocal_audio = audio_wavs['vocal_audio'][:, start:end]
                acc_audio = audio_wavs['acc_audio'][:, start:end]
                normalized_text = normalize_text(segment[2])
                phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]


                 # tailor for voice clone or singsong
                if self.app_type == 'vclone' and self.voice_clone_duration > 0:


                    # vocal_audio = audio_wavs['vocal_audio'][:, start: (start + self.voice_clone_duration * self.sample_rate)]
                    
                    # if there exists same artist of other song
                    # if "other_lyrcs" in metadata:
                    if 1 == 2:
                        other_lyrics = metadata['other_lyrcs']
                        num_candidates = len(other_lyrics)
                        idx = np.random.randint(num_candidates)

                        vocal_audio = audio_wavs[f'other_vocal_{idx}']

                        other_lyrics_pieces = other_lyrics[idx].strip() + "\n"
                        normalized_text = normalize_text(other_lyrics_pieces + segment[2])
                        phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]

                    # if no other song of same artist
                    # elif "other_lyrics" in metadata:
                    else:

                        # segment[3] is a list of words, each word contains text, start_time, end_time, attribute
                        start, end, other_lyrics_pieces = self.find_nonrelevant_vocal(idx_seg, segments)

                        start = int(start * self.sample_rate)
                        end = int(end * self.sample_rate)

                        vocal_audio = audio_wavs['vocal_audio'][:, start:end]
                        if vocal_audio.shape[-1] != self.voice_clone_duration * self.sample_rate:
                            continue


                        other_lyrics_pieces = other_lyrics_pieces.strip() + "\n"
                        normalized_text = normalize_text(other_lyrics_pieces + segment[2])
                        phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]

                elif self.app_type == 'singsong':
                    if self.predict_mode == "BGM":
                        clip = audio_wavs['acc_audio'][:, start:end]

                    vocal_audio = audio_wavs['vocal_audio'][:, start:end]
                    normalized_text = normalize_text(segment[2])
                    phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]
                elif self.app_type == 'singsong_inverse':
                    # for convenience, we don't change the name of variables
                    clip = audio_wavs['vocal_audio'][:, start:end]
                    if self.predict_mode == "FULL":
                        clip = audio_wavs['target_audio'][:, start:end]
                    acc_audio = audio_wavs['acc_audio'][:, start:end]
                    normalized_text = normalize_text(segment[2])
                    phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]

                elif self.app_type == "singsong_inverse_vc" and self.voice_clone_duration > 0:

                    clip = audio_wavs['vocal_audio'][:, start:end]
                    if self.predict_mode == "FULL":
                        clip = audio_wavs['target_audio'][:, start:end]

                    # segment[3] is a list of words, each word contains text, start_time, end_time, attribute
                    vp_start, vp_end, other_lyrics_pieces = self.find_nonrelevant_vocal(idx_seg, segments)

                    vp_start = int(vp_start * self.sample_rate)
                    vp_end = int(vp_end * self.sample_rate)
                    vocal_audio = audio_wavs['vocal_audio'][:, vp_start:vp_end]
                    if vocal_audio.shape[-1] != self.voice_clone_duration * self.sample_rate:
                        continue

                    acc_audio = audio_wavs['acc_audio'][:, start:end]

                    other_lyrics_pieces = other_lyrics_pieces.strip() + "\n"
                    normalized_text = normalize_text(other_lyrics_pieces + segment[2])
                    phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]                       


                        
                        # for continuation only
                        # vocal_audio = audio_wavs['vocal_audio'][:, start: (start + self.voice_clone_duration * self.sample_rate)]
                        # normalized_text = normalize_text(segment[2])
                        # phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]

                        # clip = audio_wavs['target_audio'][:, (start + self.voice_clone_duration * self.sample_rate):end]
                  
                
                shard = self.get_shard_info(item)
                worker_id = self.get_worker_info()
                try:
                    song_id = metadata["meta_song_id"]
                except:
                    song_id = item["__key__"]
 
                yield {
                    "audio": clip,   # target audio
                    "vocal_audio": vocal_audio, # TODO 30s clip_of_vocal_stem, same timestamp and duration as clip
                    "style_text": style_text,
                    "normalized_text": normalized_text,
                    "lyrics_tokens": phoneme_tokens,
                    "max_phone_len": self.segment_max_phone_len,
                    "acc_audio" : acc_audio,

                    # DEBUG
                    "song_id": song_id,
                    "style_metadata": metadata,
                    "shard": shard,
                    "worker_id": worker_id,
                }                


class VppParquetDataset(WebPipeline):
    name = "VppParqueDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        data_id: int = 95,  # US groupA MSS: 96 Karaoke train: 95 Karaoke val: 94 
        url_pattern: str = None,
        sample_rate: int = 24000,
        audio_key: str = "audio",
        index_key: str = "meta",
        min_duration: int = 10,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.6,
        normalize_audio: bool = False,
        tokenizer: Any = BertTokenizer.from_pretrained("bert-base-chinese"),
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        use_gt_lyrics: bool = True,
        vocal_key = 'vocal',
        style_key = 'acc',
        voice_clone_duration: int = -1,
        app_type = "vclone",
        singsong_predict_type = "BGM", # FULL
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern, extra_fields_in_data=["vocal", "acc"], **kwargs)
        
        transforms = VocalAppZhTransformsRefactored(
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
            include_intro=include_intro,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            use_soda_gt_lyrics=use_gt_lyrics,
            vocal_key=vocal_key,
            style_key=style_key,
            voice_clone_duration=voice_clone_duration,
            app_type=app_type,
            singsong_predict_type=singsong_predict_type,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class VocalZhParquetDataset(WebPipeline):
    name = "VocalZhParqueDataset"
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
        include_intro: bool = False,
        use_gt_lyrics: bool = True,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern, **kwargs)
        
        transforms = VocalZhTransforms(
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
            include_intro=include_intro,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            use_soda_gt_lyrics=use_gt_lyrics,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class SpeechZhTransforms(BaseTransforms):
    name = "SpeechZhTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 1,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key        
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.segment_max_phone_len = segment_max_phone_len

        assert self.tokenizer
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        
        lyrics = item["__index_data__"].get("text", None)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return
        
        if self.tokenizer == "tts_chinese_frontend_model":
            new_line_token = "\n"
        else:
            new_line_token = " "        
        
        if self.tokenizer == "tts_chinese_frontend_model":
            labels = list(
                filter(
                    lambda x: x != "", item["__index_data__"]["labels"].split("\n")
                )
            )
            if labels:
                convert_result = convert_labels_to_text_id(labels)
                if convert_result:
                    labels, _, _ = convert_result
                else:
                    self._update_stats("Invalid tacolab to text id conversion", labels)
                    return
                text_tokens = torch.from_numpy(labels[0]).long()
            else:
                self._update_stats(" skip empty labels")
                return
        else:
            encoded_text = self.tokenizer(
                lyrics, 
                add_special_tokens=False,
                return_tensors="pt")
            text_tokens = encoded_text["input_ids"].squeeze(dim=0)
        if text_tokens.size(-1) == 0:
            self._update_stats(skipped=True, message="Token zero length")
            return
        
        yield {
            "audio": audio,                 
            "style_text": "语音",
            "normalized_text": lyrics,
            "lyrics_tokens": text_tokens,
            "max_phone_len": self.segment_max_phone_len,
            "meta": item["__index_data__"]
        }





########################## SFT Datasets ######################

class BillboardDataset(WebPipeline):
    name = "DecoderBillboard"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/billboard_mss/train/mss_url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "full.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        max_style_token_seq_len: int = 16,
        normalize_audio: bool = True,
        segment_method: str = "random", # "first", "random"
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        max_seg_per_track: int = -1,
        handler=wds.warn_and_continue,
        **kwargs,
    ):
        with local_zero_first():
            self.phoneme_tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        # To silence espeak logging warnings: "WARNING - words count mismatch on 100.0% of the lines". Must be set after tokenizer is initialized
        phonemizer.logger.get_logger().setLevel(logging.ERROR)
        self.phoneme_tokenizer._add_tokens(["<n>"])        

        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_style_token_seq_len = max_style_token_seq_len
        self.segment_method = segment_method
        self.segment_max_phone_len = segment_max_phone_len
        self.include_intro = include_intro
        self.max_seg_per_track = max_seg_per_track
        self.handler = handler

        base_transforms = []
        if audio_key == "mp3":
            self.read_mp3 = ReadMP3(sample_rate)
            base_transforms.append(lambda x: self.read_mp3(io.BytesIO(x)))
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
            # base_transforms.append(NormalizeAudio())
        if self.data_sample_rate != sample_rate and audio_key != "mp3":
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)
        dataset = IndexedWebDataset(url2index=url2index, handler=handler, **kwargs)
        pipeline = ["decode", {"compose": [self.transform]}]
        super().__init__(dataset, pipeline)

    def is_confident_lyrics(self, utterance, threshold):
        conf = 0
        for utt in utterance:
            conf += float(utt["confidence"])
        conf /= len(utterance)
        return True if conf > threshold else False

    def transform(self, item_yielder) -> Dict[str, Any]:
        for item in item_yielder:
            try:
                audio = self.base_transform(item[self.audio_key])
            except Exception as e:
                self.handler(e)
                continue
            utterances = item["__index_data__"]['sa_lyrics']['result'][0].get("utterances", None)
            if utterances is None:
                continue
            if not self.is_confident_lyrics(utterances, 0.65):
                continue
                
            # TODO: Segmenting
            segments = group_utterances(utterances, self.min_duration, self.max_duration,
                                        time_in_sec=False, include_intro=self.include_intro)
            if len(segments) < 1:
                continue
            if self.segment_method == "first":
                segments = segments[:1]
            elif self.max_seg_per_track > 0:
                random.shuffle(segments)
                segments = segments[:self.max_seg_per_track]
            for segment in segments:                
                start = int(segment[0] * self.sample_rate)
                end = int(segment[1] * self.sample_rate)
                clip = audio[:, start:end]
                normalized_text = normalize_text(segment[2])
                phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]

                style_metadata = select_tag_metadata_from_timestamps(item["__index_data__"], start, end)
                style_text = rewrite_metadata(style_metadata, type="mir_tags")

                song_id = item["__key__"]

                shard = self.get_shard_info(item)
                worker_id = self.get_worker_info()
                
                yield {
                    "audio": clip, 
                    "style_text": style_text,
                    "style_metadata": style_metadata,
                    "normalized_text": normalized_text,
                    "lyrics_tokens": phoneme_tokens,
                    "song_id": song_id,
                    "max_phone_len": self.segment_max_phone_len,
                    "shard": shard,
                    "worker_id": worker_id,
                }                


class MusicCollectorTransforms(BaseTransforms):
    name = "MusicCollectorTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        dataset_name: str,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 1,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.lyrics_confidence = lyrics_confidence
        self.audio_key = audio_key        
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.segment_max_phone_len = segment_max_phone_len
        self.include_intro = include_intro

        assert self.tokenizer
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)


    def __call__(self, item: Dict[str, Any]) -> Generator:
       
        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        metadata = dict()
        index_data = item["__index_data__"]
        if "genre" in index_data:
            metadata['final_genre'] = index_data.get("genre") # TODO different genre taxonomy
        
        metadata['final_mood'] = item["__index_data__"].get("mood", None)        
        style_text = rewrite_metadata(metadata)


        audio_len = int(random.randint(self.min_duration, self.max_duration) * self.sample_rate)
        random_transform = Compose([
            RandomPad(audio_len),
            RandomResizedCrop(audio_len),
        ])
        clip = random_transform(audio)

        yield {
            "audio": clip,        
            "dataset_name": self.dataset_name,         
            "style_text": style_text,
            "normalized_text": "",
            "max_phone_len": self.segment_max_phone_len,
            "meta": item["__index_data__"]
        }        


class MusicCollectorDataset(WebPipeline):
    name = "MusicCollectorDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        region: str = "US",
        dataset_name: str = "Chinese",
        split: str = "train",
        weights: List[float] = None,
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
        include_intro: bool = False,
        **kwargs,
    ):
        assert region in INDEX
        assert dataset_name in INDEX[region]["MusicCollector"]
        print(f"[{self.name}] initializing...")        
        if split == "train":
            url2index = INDEX[region]["MusicCollector"][dataset_name]
        elif split == "test":
            # TODO (qq) sample 1 val for each dataset
            url2index = INDEX[region]["MusicCollector"][dataset_name] # TODO
            # url2index = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/kaiping/2023-09-20/val_url2index.txt"

        transforms = MusicCollectorTransforms(
            dataset_name=dataset_name,
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
            include_intro=include_intro,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")

        if isinstance(dataset_name, list):            
            url2index = [INDEX[region][dataset_name] for dataset_name in dataset_name]
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=weights,
            )
        else:
            if split == "train":
                dataset = IndexedWebDataset(url2index=INDEX[region][dataset_name], **kwargs)
            elif split == "test":
                # TODO (qq) sample 1 val for each dataset
                url2index = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/kaiping/2023-09-20/val_url2index.txt"
                dataset = IndexedWebDataset(url2index=INDEX[region][url2index], **kwargs)

        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class VocalAppZhTransforms(BaseTransforms):
    name = "VocalAppZhTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key = "audio",
        index_key = "__index_data__",
        min_duration: int = 1,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.5,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        segment_method = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        use_soda_gt_lyrics: bool = True,
        vocal_key = 'vocal',
        style_key = 'acc',
        voice_clone_duration: int = -1,
        app_type = "vclone",
        singsong_predict_type = "BGM" # FULL
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
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.segment_max_phone_len = segment_max_phone_len
        self.include_intro = include_intro
        self.use_soda_gt_lyrics = use_soda_gt_lyrics
        self.vocal_key = vocal_key
        self.style_key = style_key
        self.voice_clone_duration = voice_clone_duration
        self.app_type = app_type
        self.singsong_predict_type = singsong_predict_type


        assert self.tokenizer
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def is_confident_lyrics(self, utterance, threshold):
        # Compare mean of confidence of all utterances to threshold
        conf, num_utt = 0., 0.
        for utt in utterance:
            if utt['text'].strip():
                conf += float(utt["confidence"])
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
        if 'audio' not in item:
            item['audio'] = item['wav']

        # assume meta is presented as either string or dict
        if isinstance(meta, str):
            meta = json.loads(meta)

        # handle 2 formats of lyrics data in meta
        if self.use_soda_gt_lyrics and "lyrics_gt" in meta:
            lyrics_field = "lyrics_gt"
            utterances = meta.get(lyrics_field, None)
        else:
            lyrics_field = "lyrics"
            lyrics = meta.get(lyrics_field, None)
            if lyrics is None:
                self._update_stats(skipped=True, message="No lyrics")
                return
            
            if "utterances" in lyrics:
                utterances = lyrics.get('utterances', None)
            else:
                result = lyrics.get('result', None)
                if result is None or len(result) != 1:
                    self._update_stats(skipped=True, message="No result")
                    return
                utterances = result[0].get("utterances", None)

            # utterances = lyrics.get('utterances', None)
            # print(lyrics.keys())
            
            # if not self.is_confident_lyrics(utterances, self.lyrics_confidence):
            #     self._update_stats(skipped=True, message="Low confidence lyrics")
            #     return            

        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        # Extract segments from utterances
        new_line_token = " <n> "
        if isinstance(self.tokenizer, BertTokenizer):
            self.tokenizer.add_special_tokens({'additional_special_tokens': [new_line_token]})
        segments = group_utterances(utterances, self.min_duration, self.max_duration,
                                    time_in_sec=False, include_intro=self.include_intro,
                                    new_line_token=new_line_token)
        vocal_segments = group_utterances(utterances, self.voice_clone_duration-1, self.voice_clone_duration,
                                    time_in_sec=False, include_intro=self.include_intro,
                                    new_line_token=new_line_token)
        random.shuffle(vocal_segments)

        if len(segments) < 1:
            return
        if self.voice_clone_duration > 0 and len(vocal_segments) < 1:
            return

        if self.segment_method == "first":
            segments = segments[:1]
        elif self.max_seg_per_track > 0:
            random.shuffle(segments) # already shuffled, can't guarantee distant segment
            segments = segments[:self.max_seg_per_track]

        # Get style text
        metadata = dict()
        # Try read English genre tag. Source: Q music tag, WYY tag, MCC tag.
        tags = meta.get("tags", None)
        if tags:
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
        elif "merge_genre" in meta:
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
            acc = self.base_transform(item[self.style_key])
            vocal = self.base_transform(item[self.vocal_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        # Yield one example per segment
        for idx_seg, segment in enumerate(segments):
            if segment[1] - segment[0] < self.voice_clone_duration:
                print("not long enough for voice clone")
                continue
            start = int(segment[0] * self.sample_rate)
            end = int(segment[1] * self.sample_rate)
            clip = audio[:, start:end]
            vocal_audio = vocal[:, start: end]
            acc_audio = acc[:, start:end]
            other_phone = None
            original_text = segment[2]
            normalized_text = normalize_text(original_text, enable_punctuation=True)

            # prepare vocal audio and lyrics
            if self.app_type == 'vclone' and self.voice_clone_duration > 0:
                # start, end, other_lyrics_pieces = self.find_nonrelevant_vocal(idx_seg, segments)
                idx_seg = np.random.randint(len(vocal_segments))
                vocal_seg = vocal_segments[idx_seg]
                vocal_start, vocal_end, other_lyrics_pieces, other_phone = vocal_seg

                vocal_start = int(vocal_start * self.sample_rate)
                vocal_end = vocal_start + self.voice_clone_duration * self.sample_rate
                vocal_audio = vocal[:, vocal_start:vocal_end]
                vocal_lyrics = other_lyrics_pieces.strip() + "\n"
                normalized_text = normalize_text(vocal_lyrics + original_text, enable_punctuation=True)
                if vocal_audio.shape[-1] != self.voice_clone_duration * self.sample_rate:
                    continue

            elif self.app_type == "singsong":
                if self.singsong_predict_type == "BGM":
                    clip = acc[:, start:end]

            elif self.app_type == "singsong_inverse":
                clip = vocal[:, start: end]
                if self.singsong_predict_type == "FULL":
                    clip = audio[:, start:end]
                
            elif self.app_type == "singsong_inverse_vc" and self.voice_clone_duration > 0:
                clip = vocal[:, start:end]
                if self.singsong_predict_type == "FULL":
                    clip = audio[:, start:end]

                idx_seg = np.random.randint(len(vocal_segments))
                vocal_seg = vocal_segments[idx_seg]
                vocal_start, vocal_end, other_lyrics_pieces, other_phone = vocal_seg

                vocal_start = int(vocal_start * self.sample_rate)
                vocal_end = vocal_start + self.voice_clone_duration * self.sample_rate
                vocal_audio = vocal[:, vocal_start:vocal_end]
                vocal_lyrics = other_lyrics_pieces.strip() + "\n"
                normalized_text = normalize_text(vocal_lyrics + original_text, enable_punctuation=True)   

                if vocal_audio.shape[-1] != self.voice_clone_duration * self.sample_rate:
                    continue             

            
            if self.tokenizer == "tts_chinese_frontend_model":
                lines = segment[3].split(" <n> ")
                vp_len = 0             
                if self.app_type == "vclone" or self.app_type == "singsong_inverse_vc":
                    vocal_lines = other_phone.split(" <n> ")
                    lines = vocal_lines + lines
                    vp_len = len(vocal_lines)
                      
                text_tokens = []
                for i, line in enumerate(lines):
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
                    if i == vp_len - 1:
                        text_tokens.append(VP_END_PHONE_TOKEN)                    
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
            if text_tokens is None or text_tokens.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                continue
            
            yield {
                "audio": clip, 
                "vocal_audio": vocal_audio, 
                "acc_audio": acc_audio,               
                "style_text": style_text,
                "normalized_text": normalized_text,
                "lyrics_tokens": text_tokens,
                "max_phone_len": self.segment_max_phone_len,
                "meta": meta,
            } 



class VocalAppZhDataset(WebPipeline):
    name = "VocalAppZhDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        region = "CN",
        dataset_names: List[str] = ["Soda"],
        dataset_weights: List[float] = [1],        
        sample_rate: int = 24000,
        audio_key = "audio.npy",
        min_duration: int = 2,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.7,
        normalize_audio: bool = False,
        tokenizer: Any = BertTokenizer.from_pretrained("bert-base-chinese"),
        frame_rate: int = 25,
        segment_method = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        use_soda_gt_lyrics: bool = True,
        vocal_key = 'vocal.npy',
        style_key = 'acc.npy',
        voice_clone_duration: int = -1,
        app_type = "vclone",
        singsong_predict_type = "BGM", # FULL
        **kwargs,
    ):
        assert region in INDEX
        print(f"[{self.name}] initializing...")        

        transforms = VocalAppZhTransforms(
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
            include_intro=include_intro,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            use_soda_gt_lyrics=use_soda_gt_lyrics,
            vocal_key=vocal_key,
            style_key=style_key,
            voice_clone_duration=voice_clone_duration,
            app_type=app_type,
            singsong_predict_type=singsong_predict_type
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
            if dataset_names == "SodaMSSTest":
                url2index = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/soda/weituo_temp/soda_val.tsv"
                dataset = IndexedWebDataset(url2index=url2index, **kwargs)
            else:
                dataset = IndexedWebDataset(url2index=INDEX[region][dataset_names], **kwargs)

        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


# singsong refactor
class VocalAppZhTransformsRefactored(BaseTransforms):
    name = "VocalAppZhTransformsRefactored"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key = "audio",
        index_key = "__index_data__",
        min_duration: int = 1,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.5,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        segment_method = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        use_soda_gt_lyrics: bool = True,
        vocal_key = 'vocal',
        style_key = 'acc',
        voice_clone_duration: int = -1,
        app_type = "vclone",
        singsong_predict_type = "BGM" # FULL
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
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.segment_max_phone_len = segment_max_phone_len
        self.include_intro = include_intro
        self.use_soda_gt_lyrics = use_soda_gt_lyrics
        self.vocal_key = vocal_key
        self.style_key = style_key
        self.voice_clone_duration = voice_clone_duration
        self.app_type = app_type
        self.singsong_predict_type = singsong_predict_type


        assert self.tokenizer
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def is_confident_lyrics(self, utterance, threshold):
        # Compare mean of confidence of all utterances to threshold
        conf, num_utt = 0., 0.
        for utt in utterance:
            if utt['text'].strip():
                conf += float(utt["confidence"])
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
        if 'audio' not in item:
            item['audio'] = item['wav']

        # assume meta is presented as either string or dict
        if isinstance(meta, str):
            meta = json.loads(meta)

        # handle 2 formats of lyrics data in meta
        if self.use_soda_gt_lyrics and "lyrics_gt" in meta:
            lyrics_field = "lyrics_gt"
            utterances = meta.get(lyrics_field, None)
        else:
            lyrics_field = "lyrics"
            lyrics = meta.get(lyrics_field, None)
            if lyrics is None:
                self._update_stats(skipped=True, message="No lyrics")
                return
            
            if "utterances" in lyrics:
                utterances = lyrics.get('utterances', None)
            else:
                result = lyrics.get('result', None)
                if result is None or len(result) != 1:
                    self._update_stats(skipped=True, message="No result")
                    return
                utterances = result[0].get("utterances", None)

            # utterances = lyrics.get('utterances', None)
            # print(lyrics.keys())
            
            # if not self.is_confident_lyrics(utterances, self.lyrics_confidence):
            #     self._update_stats(skipped=True, message="Low confidence lyrics")
            #     return            

        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        # Extract segments from utterances
        new_line_token = " <n> "
        if isinstance(self.tokenizer, BertTokenizer):
            self.tokenizer.add_special_tokens({'additional_special_tokens': [new_line_token]})
        
        segments = []
        for utterance in utterances:
            segments.append(Segment.from_utt(utterance))
        segments = [[seg.start, seg.end, seg.text, seg.phoneme] for seg in segments]
        
        # segments = group_utterances(utterances, self.min_duration, self.max_duration,
        #                             time_in_sec=False, include_intro=self.include_intro,
        #                             new_line_token=new_line_token)
        vocal_segments = group_utterances(utterances, self.voice_clone_duration-1, self.voice_clone_duration,
                                    time_in_sec=False, include_intro=self.include_intro,
                                    new_line_token=new_line_token)
        random.shuffle(vocal_segments)

        if len(segments) < 1:
            return
        if self.voice_clone_duration > 0 and len(vocal_segments) < 1:
            return

        if self.segment_method == "first":
            segments = segments[:1]
        elif self.max_seg_per_track > 0:
            random.shuffle(segments) # already shuffled, can't guarantee distant segment
            segments = segments[:self.max_seg_per_track]

        # Get style text
        metadata = dict()
        # Try read English genre tag. Source: Q music tag, WYY tag, MCC tag.
        tags = meta.get("tags", None)
        if tags:
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
        elif "merge_genre" in meta:
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
            acc = self.base_transform(item[self.style_key])
            vocal = self.base_transform(item[self.vocal_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        # Yield one example per segment
        for idx_seg, segment in enumerate(segments):
            if segment[1] - segment[0] < self.voice_clone_duration:
                print("not long enough for voice clone")
                continue
            start = int(segment[0] * self.sample_rate)
            end = int(segment[1] * self.sample_rate)
            clip = audio[:, start:end]
            vocal_audio = vocal[:, start: end]
            acc_audio = acc[:, start:end]
            other_phone = None
            original_text = segment[2]
            normalized_text = normalize_text(original_text, enable_punctuation=True)

            # prepare vocal audio and lyrics
            if self.app_type == 'vclone' and self.voice_clone_duration > 0:
                # start, end, other_lyrics_pieces = self.find_nonrelevant_vocal(idx_seg, segments)
                idx_seg = np.random.randint(len(vocal_segments))
                vocal_seg = vocal_segments[idx_seg]
                vocal_start, vocal_end, other_lyrics_pieces, other_phone = vocal_seg

                vocal_start = int(vocal_start * self.sample_rate)
                vocal_end = vocal_start + self.voice_clone_duration * self.sample_rate
                vocal_audio = vocal[:, vocal_start:vocal_end]
                vocal_lyrics = other_lyrics_pieces.strip() + "\n"
                normalized_text = normalize_text(vocal_lyrics + original_text, enable_punctuation=True)
                if vocal_audio.shape[-1] != self.voice_clone_duration * self.sample_rate:
                    continue

            elif self.app_type == "singsong":
                if self.singsong_predict_type == "BGM":
                    clip = acc[:, start:end]

            elif self.app_type == "singsong_inverse":
                clip = vocal[:, start: end]
                if self.singsong_predict_type == "FULL":
                    clip = audio[:, start:end]
                
            elif self.app_type == "singsong_inverse_vc" and self.voice_clone_duration > 0:
                clip = vocal[:, start:end]
                if self.singsong_predict_type == "FULL":
                    clip = audio[:, start:end]

                idx_seg = np.random.randint(len(vocal_segments))
                vocal_seg = vocal_segments[idx_seg]
                vocal_start, vocal_end, other_lyrics_pieces, other_phone = vocal_seg

                vocal_start = int(vocal_start * self.sample_rate)
                vocal_end = vocal_start + self.voice_clone_duration * self.sample_rate
                vocal_audio = vocal[:, vocal_start:vocal_end]
                vocal_lyrics = other_lyrics_pieces.strip() + "\n"
                normalized_text = normalize_text(vocal_lyrics + original_text, enable_punctuation=True)   

                if vocal_audio.shape[-1] != self.voice_clone_duration * self.sample_rate:
                    continue             

            
            if self.tokenizer == "tts_chinese_frontend_model":
                lines = segment[3].split(" <n> ")
                vp_len = 0             
                if self.app_type == "vclone" or self.app_type == "singsong_inverse_vc":
                    vocal_lines = other_phone.split(" <n> ")
                    lines = vocal_lines + lines
                    vp_len = len(vocal_lines)
                      
                text_tokens = []
                for i, line in enumerate(lines):
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
                    if i == vp_len - 1:
                        text_tokens.append(VP_END_PHONE_TOKEN)                    
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
            if text_tokens is None or text_tokens.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                continue
            
            
            uttid = item['uttid']
            # breakpoint()
            # import torchaudio
            # torchaudio.save(filepath=f"./tmp/{uttid}.{idx_seg}.vocal.wav", src=vocal_audio, sample_rate=24000)
            # torchaudio.save(filepath=f"./tmp/{uttid}.{idx_seg}.acc.wav", src=acc_audio, sample_rate=24000)
            # torchaudio.save(filepath=f"./tmp/{uttid}.{idx_seg}.mix.wav", src=acc_audio+vocal_audio, sample_rate=24000)
            
            # breakpoint()
            yield {
                "audio": clip, 
                "vocal_audio": vocal_audio, 
                "acc_audio": acc_audio,               
                "style_text": style_text,
                "normalized_text": normalized_text,
                "lyrics_tokens": text_tokens,
                "max_phone_len": self.segment_max_phone_len,
                "meta": meta,
                "uttid": uttid,
                "idx_seg": idx_seg,
            }



########################## Chinese Datasets ######################

class VocalZhTransforms(BaseTransforms):
    name = "VocalZhTransforms"
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
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        use_soda_gt_lyrics: bool = True,
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
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.segment_max_phone_len = segment_max_phone_len
        self.include_intro = include_intro
        self.use_soda_gt_lyrics = use_soda_gt_lyrics

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
                conf += float(utt["confidence"])
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
        if self.use_soda_gt_lyrics and "lyrics_gt" in meta:
            lyrics_field = "lyrics_gt"            
            utterances = meta.get(lyrics_field, None)
        else:
            lyrics_field = "lyrics"
            lyrics = meta.get(lyrics_field, None)
            if lyrics is None:
                self._update_stats(skipped=True, message="No lyrics")
                return            
            result = lyrics.get("result", None)
            if result is None or len(result) != 1:
                self._update_stats(skipped=True, message="No result")
                return
            utterances = result[0].get("utterances", None)
            if not self.is_confident_lyrics(utterances, 0.8):
                self._update_stats(skipped=True, message="Low confidence lyrics")
                return            
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        # Extract segments from utterances
        new_line_token = " <n> "
        if isinstance(self.tokenizer, BertTokenizer):
            self.tokenizer.add_special_tokens({'additional_special_tokens': [new_line_token]})
        segments = group_utterances(utterances, self.min_duration, self.max_duration,
                                    time_in_sec=False, include_intro=self.include_intro,
                                    new_line_token=new_line_token)
        if len(segments) < 1:
            return
        if self.segment_method == "first":
            segments = segments[:1]
        elif self.max_seg_per_track > 0:
            random.shuffle(segments)
            segments = segments[:self.max_seg_per_track]

        # Get style text
        metadata = dict()
        # Try read English genre tag. Source: Q music tag, WYY tag, MCC tag.
        tags = meta.get("tags", None)
        if tags:
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
        elif "merge_genre" in meta:
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
        for segment in segments:
            if segment[1] - segment[0] < 1:
                print("short segment")
                continue
            start = int(segment[0] * self.sample_rate)
            end = int(segment[1] * self.sample_rate)
            clip = audio[:, start:end]

            normalized_text = segment[2]
            if self.tokenizer == "tts_chinese_frontend_model":
                lines = segment[3].split(" <n> ")                
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
            if text_tokens is None or text_tokens.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                continue
            
            yield {
                "audio": clip,                 
                "style_text": style_text,
                "normalized_text": normalized_text,
                "lyrics_tokens": text_tokens,
                "max_phone_len": self.segment_max_phone_len,
                "meta": meta,
            }        


class VocalZhDataset(WebPipeline):
    name = "VocalZhDataset"
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
        include_intro: bool = False,
        use_soda_gt_lyrics: bool = True,
        **kwargs,
    ):
        assert region in INDEX
        print(f"[{self.name}] initializing...")        

        transforms = VocalZhTransforms(
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
            include_intro=include_intro,
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
                url2index = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/kaiping/2023-09-20/val_url2index.txt"
                dataset = IndexedWebDataset(url2index=url2index, **kwargs)
            else:
                dataset = IndexedWebDataset(url2index=INDEX[region][dataset_names], **kwargs)

        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")



class VocalZhParquetDataset(WebPipeline):
    name = "VocalZhParqueDataset"
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
        include_intro: bool = False,
        use_gt_lyrics: bool = True,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern, **kwargs)
        
        transforms = VocalZhTransforms(
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
            include_intro=include_intro,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            use_soda_gt_lyrics=use_gt_lyrics,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class SpeechZhTransforms(BaseTransforms):
    name = "SpeechZhTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 1,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key        
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.segment_max_phone_len = segment_max_phone_len

        assert self.tokenizer
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        
        lyrics = item["__index_data__"].get("text", None)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return
        
        if self.tokenizer == "tts_chinese_frontend_model":
            new_line_token = "\n"
        else:
            new_line_token = " "        
        
        if self.tokenizer == "tts_chinese_frontend_model":
            labels = list(
                filter(
                    lambda x: x != "", item["__index_data__"]["labels"].split("\n")
                )
            )
            if labels:
                convert_result = convert_labels_to_text_id(labels)
                if convert_result:
                    labels, _, _ = convert_result
                else:
                    self._update_stats("Invalid tacolab to text id conversion", labels)
                    return
                text_tokens = torch.from_numpy(labels[0]).long()
            else:
                self._update_stats(" skip empty labels")
                return
        else:
            encoded_text = self.tokenizer(
                lyrics, 
                add_special_tokens=False,
                return_tensors="pt")
            text_tokens = encoded_text["input_ids"].squeeze(dim=0)
        if text_tokens.size(-1) == 0:
            self._update_stats(skipped=True, message="Token zero length")
            return
        
        yield {
            "audio": audio,                 
            "style_text": "语音",
            "normalized_text": lyrics,
            "lyrics_tokens": text_tokens,
            "max_phone_len": self.segment_max_phone_len,
            "meta": item["__index_data__"]
        }


class SpeechZhDataset(WebPipeline):
    name = "SpeechZhDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        region: str = "CN",
        dataset_names: List[str] = ["FanqieShort"],
        dataset_weights: List[float] = [1],        
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 2,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=BertTokenizer.from_pretrained("bert-large-uncased"),
        frame_rate: int = 25,
        segment_method: str = "random",
        max_seg_per_track: int = -1,
        segment_max_phone_len: int = 400,
        include_intro: bool = False,        
        **kwargs,
    ):
        assert region in INDEX
        print(f"[{self.name}] initializing...")        
        transforms = SpeechZhTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,            
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
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
            if dataset_names == "FanqieShort":
                # TODO (qq) sample 1 val for each dataset
                url2index = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/kaiping/2023-09-20/val_url2index.txt"
                dataset = IndexedWebDataset(url2index=INDEX[region][url2index], **kwargs)                
            else:
                dataset = IndexedWebDataset(url2index=INDEX[region][dataset_names], **kwargs)

        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")




#===================== Data Modules =====================

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
    ):
        super().__init__()
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset, wds.shuffle(self.shuffle_buffer_size)
        )
        return DataLoader(
            train_dataset_batched,
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


class SFTWebDataModule(DataModule):
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        sample_rate: int = 24000,
        batch_size: int = 2,
        buckets_in_sec: List[int] = [20, 25, 30],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        max_seg_per_track: int = 3,
        use_pipe: bool = False,
    ):    
        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],  
            )
        mcc1m_groupA_url2index_list = [
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-blues.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-childhood.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-classical.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-country.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-devotional.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-easy-listening.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-electronic.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-folk.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-hip-hop-rap.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-jazz.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-metal.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-pop.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-r-b-soul.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-reggae.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-rock.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-soundtrack.txt",
        ]
        datasets = [
            BillboardDataset(
                url2index='hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/billboard_mss/train/mss_url2idx.txt',
                resampled=True,
                shardshuffle=True,
                max_seg_per_track=max_seg_per_track,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )] + [
            MCCVocalDataset(
                url2index=url2index,
                sample_rate=sample_rate,
                resampled=True,
                shardshuffle=True,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                segment_method=segment_method,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                max_seg_per_track=max_seg_per_track,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            ) for url2index in mcc1m_groupA_url2index_list
            ]
        mcc1m_groupA_url2index_weights = [1/16] * len(mcc1m_groupA_url2index_list)
        weights = [1] + mcc1m_groupA_url2index_weights
        assert len(weights) == len(datasets)

        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=datasets, weights=weights),
            pipeline=[{"compose": [self.bucketize]}],
        )
   
        datasets = [
            BillboardDataset(
                url2index="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/billboard/test/url2index.txt",
                max_seg_per_track=1,
                use_pipe=False,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,                
            ),
            MCCVocalDataset(
                url2index="/mnt/bn/audio-diffusion/data/vocal_mcc_npy/pop_url2idx_val.txt",
                sample_rate=sample_rate,
                resampled=False,
                nodesplitter=return_self,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                segment_method=segment_method,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                max_seg_per_track=1,
                use_pipe=False,
                handler=wds.warn_and_continue,
            )
        ]

        val_weights = [0.5, 0.5] # TODO qq check weights
        validation_dataset = WebPipeline(      
            MultiIterableDataset(datasets=datasets, weights=val_weights),   
            # TODO (qq) change it to MTAT
            pipeline=[{"compose": [self.bucketize]}],
        )
        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,
            collate_fn=collate_fn,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch



class MixVocalZhWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,        
        use_pipe: bool = True,
        tokenizer: str = "wordpiece",
        frame_rate: int = 25,
        normalize_audio: bool = False,
        region: str = "CN",
        wds_dataset_names: List[str] = ["Soda"],
        wds_dataset_weights: List[int] = [1],
        parquet_dataset_ids: List[int] = [],
        parquet_dataset_weights: List[int] = [],
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
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
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
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
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],  
            )
        self.wds_vocal_datasets = []
        wds_dataset_agg_weight = []
        if wds_dataset_names:
            wds_dataset_agg_weight = [sum(wds_dataset_weights)]
            self.wds_vocal_datasets = [VocalZhDataset(
                region=region,
                dataset_names=wds_dataset_names,
                dataset_weights=wds_dataset_weights,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=max_seg_per_track,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                use_soda_gt_lyrics=True,
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
                    VocalZhParquetDataset(
                        data_id=parquet_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        include_intro=include_intro,
                        use_gt_lyrics=True,
                        tokenizer=self.tokenizer,                
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
            VocalZhDataset(
                region=region,
                dataset_names="SodaTest",
                dataset_weights=1,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=1,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
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


class MixZhWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,        
        use_pipe: bool = True,
        tokenizer: str = "wordpiece",
        frame_rate: int = 25,
        normalize_audio: bool = False,
        region: str = "CN",
        dataset_weights: List[int] = [1, 1, 1],
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
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
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
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
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],  
            )

        self.zh_gt_vocal_datasets = VocalZhDataset(
            region=region,
            dataset_names=["Soda"],
            dataset_weights=[1],
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            normalize_audio=normalize_audio,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            include_intro=include_intro,
            use_soda_gt_lyrics=True,
            tokenizer=self.tokenizer,                
            resampled=True,
            shardshuffle=True,
            use_pipe=use_pipe,
            handler=wds.warn_and_continue
            )        
        self.zh_sa_vocal_datasets = VocalZhDataset(
            region=region,
            dataset_names=["HotGalaxy", "MCCVocal-Zh-A", "MCCVocal-Zh-B"],
            dataset_weights=[1, 0.5, 0.2],
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            normalize_audio=normalize_audio,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            include_intro=include_intro,
            use_soda_gt_lyrics=False,
            tokenizer=self.tokenizer,                
            resampled=True,
            shardshuffle=True,
            use_pipe=use_pipe,
            handler=wds.warn_and_continue
            )
        self.zh_speech_datasets = SpeechZhDataset(
            region=region,
            dataset_names=["FanqieShort", "FanqieLong", "XimalayaShort", "XimalayaLong", "XiaoyuzhouShort", "XiaoyuzhouLong"],
            dataset_weights=[20, 20, 6, 3, 6, 3],
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            normalize_audio=normalize_audio,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            include_intro=include_intro,
            tokenizer=self.tokenizer,                
            resampled=True,
            shardshuffle=True,
            use_pipe=use_pipe,
            handler=wds.warn_and_continue
        )
        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=[self.zh_gt_vocal_datasets, self.zh_sa_vocal_datasets, self.zh_speech_datasets],
                                 weights=dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = [WebPipeline(
            VocalZhDataset(
                region=region,
                dataset_names="SodaTest",
                dataset_weights=1,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=1,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                use_soda_gt_lyrics=True,
                tokenizer=self.tokenizer,                
                use_pipe=use_pipe,
                resampled=False,
                nodesplitter=return_self,                
                handler=wds.warn_and_continue
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )] 

        super().__init__(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

class WebDataModule(DataModule):
    def __init__(
        self,
        train_dataset,
        validation_dataset,
        predict_dataset,
        sample_rate: int = 24000,
        batch_size: int = 2,
        buckets_in_sec: List[int] = [20, 25, 30],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        use_dynamic_batch: str = False,
        do_shuffle: bool = True,
    ):    
        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],  
            )
        super().__init__(
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=predict_dataset,
            collate_fn=collate_fn,
            do_shuffle=do_shuffle,
        )

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch

class SFTMCCVocalWebDataModule(WebDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        buckets_in_sec: List[int] = [20, 25, 30],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        max_seg_per_track: int = 3,
        use_pipe: bool = False,
    ):

        mcc1m_groupA_url2index_list = [
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-blues.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-childhood.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-classical.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-country.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-devotional.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-easy-listening.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-electronic.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-folk.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-hip-hop-rap.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-jazz.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-metal.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-pop.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-r-b-soul.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-reggae.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-rock.txt",
            "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/mcc1m_vocalA/vocal-A-soundtrack.txt",
        ]
        datasets = [
            MCCVocalDataset(
                url2index=url2index,
                sample_rate=sample_rate,
                resampled=True,
                shardshuffle=True,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                segment_method=segment_method,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                max_seg_per_track=max_seg_per_track,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            ) for url2index in mcc1m_groupA_url2index_list
        ]
        mcc1m_groupA_url2index_weights = [1/16] * len(mcc1m_groupA_url2index_list)
        assert len(mcc1m_groupA_url2index_weights) == len(datasets)

        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=datasets, weights=mcc1m_groupA_url2index_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )
   
        validation_dataset = MCCVocalDataset(
            url2index="/mnt/bn/audio-diffusion/data/vocal_mcc_npy/pop_url2idx_val.txt",
            sample_rate=sample_rate,
            resampled=False,
            nodesplitter=return_self,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            segment_method=segment_method,
            segment_max_phone_len=segment_max_phone_len,
            include_intro=include_intro,
            max_seg_per_track=1,
            use_pipe=False,
            handler=wds.warn_and_continue,
        )
        validation_dataset = WebPipeline(      
            validation_dataset,
            # TODO (qq) change it to MTAT
            pipeline=[{"compose": [self.bucketize]}],
        )
        
        predict_dataset = train_dataset
        super().__init__(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=train_dataset,  # TODO
            collate_fn=collate_fn,
        )

class SFTBilloardWebDataModule(WebDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        buckets_in_sec: List[int] = [20, 25, 30],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        max_seg_per_track: int = 3,
        use_pipe: bool = False,
        resampled: bool = True,
        shardshuffle: bool = True,
    ):
        train_dataset = BillboardDataset(
            url2index='hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/billboard_mss/train/mss_url2idx.txt',
            sample_rate=sample_rate,
            resampled=resampled,
            shardshuffle=shardshuffle,
            segment_method=segment_method,
            segment_max_phone_len=segment_max_phone_len,
            include_intro=include_intro,
            max_seg_per_track=max_seg_per_track,
            use_pipe=use_pipe,
            handler=wds.warn_and_continue,
        )

        train_dataset = WebPipeline(            
            train_dataset,
            pipeline=[{"compose": [
                wds.shuffle(shuffle_buffer_size),
                self.bucketize,
            ]}],
        )
   
        val_tar = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/billboard_mss/test/tars/00000.tar"
        val_idx = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/billboard_mss/test/indexes/00000.idx_merge.60"
        validation_dataset = BillboardDataset(
            url2index={val_tar: val_idx},
            sample_rate=sample_rate,
            segment_method=segment_method,
            segment_max_phone_len=segment_max_phone_len,
            include_intro=include_intro,
            max_seg_per_track=1,
            use_pipe=use_pipe,
            resampled=False,
            nodesplitter=return_self,
            handler=wds.warn_and_continue,                
        )

        validation_dataset = WebPipeline(      
            validation_dataset,
            pipeline=[{"compose": [self.bucketize]}],
        )
        
        predict_dataset = train_dataset
        super().__init__(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=predict_dataset,
            sample_rate=sample_rate,
            batch_size=batch_size,
            buckets_in_sec=buckets_in_sec,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            use_dynamic_batch=use_dynamic_batch,
            do_shuffle=False,
        )

class MusicCollectorWebDataModule(DataModule):
    def __init__(
        self,
        dataset_names: List[str],
        dataset_weights: List[int],
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,        
        use_pipe: bool = True,
        tokenizer: str = "wordpiece",
        frame_rate: int = 25,
        normalize_audio: bool = False,
        region: str = "US", # TODO
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
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
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
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
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],  
            )

        self.datasets = [MusicCollectorDataset(
            region=region,
            dataset_name=dataset_name,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            normalize_audio=normalize_audio,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            include_intro=include_intro,
            tokenizer=self.tokenizer,                
            frame_rate=frame_rate,
            resampled=True,
            shardshuffle=True,
            use_pipe=use_pipe,
            handler=wds.warn_and_continue
            ) for dataset_name in dataset_names]
        
        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=self.datasets, weights=dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = [WebPipeline(
            MusicCollectorDataset(
                region=region,
                dataset_name="Mandarin", # TODO
                split="test",
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=1,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                tokenizer=self.tokenizer,                
                frame_rate=frame_rate,
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


class SingsongWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        voice_clone_duration: int = -1,
        voice_clone_random_start: bool = False,
        app_type: str = "vclone",
        predict_mode: str = "FULL",
        buckets_in_sec: List[int] = [
            5,
            6,
            7,
            8,
            9,
            10,  
            11,
            12,
            13,
            14,
            15,          
        ],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = voiceclone_collate_fn,
        region: str = "groupA",  # groupA or groupA-Ka, Ka, CN 
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        max_seg_per_track: int = -1,
        exclude_licenses: List[str] = ["C"],
    ):

        # buckets samples related
        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        maximum_bucket_size = batch_size * sample_rate * buckets_in_sec[-1]
        if use_dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=maximum_bucket_size,
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],  
            )


        if region == "US":
            resso_train_index = "/mnt/bn/audio-diffusion/ashaw/webdataset/index_lists/resso_mss.tar_to_index.tsv"  # 683 tars        
            karaoke_train_index = "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/indexes_with_meta/karaoke_train.tar_to_index.tsv" # 131 tars
            karaoke_val_index = "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/indexes_with_meta/karaoke_valid.tar_to_index.tsv" 
            weights = [0.8, 0.2]
            url2idx_list = [resso_train_index, karaoke_train_index]
            audio_map_keys_list = [
                { 'acc_audio': 'mss_acc', 'vocal_audio': 'mss_vocal'},  # resso
                { 'acc_audio': 'acc.mp3', 'target_audio': 'full.mp3', 'vocal_audio': 'vocal.mp3' }  # karaoke
            ]
            audio_key = "mp3"
            audio_format = "mp3"
        elif region =="groupA" or region == "groupA-Ka" or region=="Ka":
            # dataset_list_path = "/mnt/bn/audio-diffusion/weituo/groupA-mss/mss_list_fix"
            dataset_list_path = "/mnt/bn/audio-diffusion/weituo/groupA-mss/mss_list_v1"
            with open(dataset_list_path, "r") as ff:
                datasets = ff.readlines()
            url2idx_list = [line.strip() for line in datasets]
            weights = [1.0 / len(datasets) ] * len(datasets)
            audio_map_keys_list = [
                { 'acc_audio': 'acc.npy', 'target_audio': 'audio.npy', 'vocal_audio': 'vocal.npy' }  # mcc grouop A
            ] * len(datasets)
            audio_key = "audio.npy"
            audio_format = "npy"
        else:
            raise ValueError

        datasets = [
            SingsongDataset(
                url2index=url2index_mapkey[0],
                audio_map_keys=url2index_mapkey[1],
                audio_key=audio_key,
                audio_format=audio_format,
                sample_rate=sample_rate,
                resampled=True,
                shardshuffle=True,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                segment_method=segment_method,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                max_seg_per_track=max_seg_per_track,
                use_pipe=False,
                voice_clone_duration=voice_clone_duration,
                voice_clone_random_start=voice_clone_random_start,
                app_type=app_type,
                predict_mode=predict_mode,
                handler=wds.ignore_and_continue,
            ) for url2index_mapkey in zip(url2idx_list, audio_map_keys_list)
        ]
        
        karaoke_dataset = SingsongDataset(
                url2index="/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/indexes_with_meta/karaoke_train.tar_to_index.tsv",
                # url2index= "/mnt/bn/audio-diffusion/weituo/webdataset/karaoke_artist.tsv",
                audio_map_keys={ 'acc_audio': 'acc.mp3', 'target_audio': 'full.mp3', 'vocal_audio': 'vocal.mp3' },
                audio_key="mp3",
                audio_format="mp3",
                sample_rate=sample_rate,
                resampled=True,
                shardshuffle=True,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                segment_method=segment_method,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                max_seg_per_track=max_seg_per_track,
                use_pipe=False,
                voice_clone_duration=voice_clone_duration,
                voice_clone_random_start=voice_clone_random_start,
                app_type=app_type,
                predict_mode=predict_mode,
                handler=wds.ignore_and_continue,
            )
        
        if region == 'groupA-Ka':
            datasets.append(karaoke_dataset)
            weights.append(2.0 / len(datasets))
        elif region == 'Ka':
            datasets = [karaoke_dataset]
            weights = [1.0]


        train_dataset = WebPipeline(            
            MultiIterableDataset(
                datasets=datasets, weights=weights                
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )

        karaoke_val_index = "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/indexes_with_meta/karaoke_valid.tar_to_index.tsv"
        groupA_pop_val = "/mnt/bn/audio-diffusion/weituo/voice_clone_pop_val.tsv"
        validation_dataset_ka = WebPipeline(
            SingsongDataset(
                url2index=karaoke_val_index,
                audio_map_keys={ 'acc_audio': 'acc.mp3', 'target_audio': 'full.mp3', 'vocal_audio': 'vocal.mp3' },
                sample_rate=sample_rate,
                audio_key = "mp3",
                audio_format = "mp3",
                resampled=False,
                nodesplitter=return_self,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                segment_method=segment_method,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                max_seg_per_track=1,
                exclude_licenses=["B", "C"],
                use_pipe=False,
                voice_clone_duration=voice_clone_duration,
                voice_clone_random_start=voice_clone_random_start,
                app_type=app_type,
                predict_mode=predict_mode,
                handler=wds.ignore_and_continue,                
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )


        validation_dataset_groupA = WebPipeline(
            SingsongDataset(
                url2index=groupA_pop_val,
                audio_map_keys={ 'acc_audio': 'acc.npy', 'target_audio': 'audio.npy', 'vocal_audio': 'vocal.npy' },
                sample_rate=sample_rate,
                audio_key = audio_key,
                audio_format = audio_format,
                resampled=False,
                nodesplitter=return_self,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                segment_method=segment_method,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                max_seg_per_track=1,
                exclude_licenses=["B", "C"],
                use_pipe=False,
                voice_clone_duration=voice_clone_duration,
                voice_clone_random_start=voice_clone_random_start,
                app_type=app_type,
                predict_mode=predict_mode,
                handler=wds.ignore_and_continue,                
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )

        if region == "groupA":
            validation_dataset = validation_dataset_groupA
        elif region == "Ka":
            validation_dataset = validation_dataset_ka
        elif region == "groupA-Ka":
            validation_dataset = [validation_dataset_ka, validation_dataset_groupA]


        collate_fn = functools.partial(vpp_collate_fn, app_type=app_type)

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



class MixVocalAppZhWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = voiceclone_collate_fn,        
        use_pipe: bool = True,
        tokenizer = "wordpiece",
        frame_rate: int = 25,
        normalize_audio: bool = False,
        region = "CN",
        wds_dataset_names: List[str] = [],
        wds_dataset_weights: List[int] = [1],
        parquet_dataset_ids: List[int] = [],
        parquet_dataset_weights: List[int] = [],
        use_dynamic_batch = False,
        segment_method = "random",
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        max_seg_per_track: int = -1,
        buckets_in_sec: List[int] = [
            20,
            25,
            30,
        ],
        vocal_key= 'vocal',
        style_key= 'acc',
        voice_clone_duration: int = 6,
        app_type= "vclone",
        singsong_predict_type= "BGM", # FUL
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
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
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
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:            
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],  
            )
        self.wds_vocal_datasets = []
        wds_dataset_agg_weight = []
        if wds_dataset_names:
            wds_dataset_agg_weight = [sum(wds_dataset_weights)]
            self.wds_vocal_datasets = [VocalAppZhDataset(
                region=region,
                dataset_names=wds_dataset_names,
                dataset_weights=wds_dataset_weights,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                normalize_audio=normalize_audio,
                segment_method=segment_method,
                max_seg_per_track=max_seg_per_track,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                use_soda_gt_lyrics=True,
                tokenizer=self.tokenizer,                
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,            
                handler=wds.warn_and_continue,
                vocal_key=vocal_key,
                style_key=style_key,
                voice_clone_duration=voice_clone_duration,
                app_type=app_type,
                singsong_predict_type=singsong_predict_type, # FUL
                )]
        self.parquet_vocal_datasets = []
        if parquet_dataset_ids:
            for parquet_id in parquet_dataset_ids:
                print(f"########  dataset id is {parquet_id}")
                self.parquet_vocal_datasets.append(
                    VppParquetDataset(
                        data_id=parquet_id,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        normalize_audio=normalize_audio,
                        segment_method=segment_method,
                        max_seg_per_track=max_seg_per_track,
                        segment_max_phone_len=segment_max_phone_len,
                        include_intro=include_intro,
                        use_gt_lyrics=True,
                        tokenizer=self.tokenizer,                
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,
                        vocal_key = vocal_key,
                        style_key = style_key,
                        voice_clone_duration = voice_clone_duration,
                        app_type = app_type,
                        singsong_predict_type = singsong_predict_type, # FULL            
                        handler=wds.warn_and_continue,                        
                        ))
        
        train_dataset = WebPipeline(            
            MultiIterableDataset(datasets=self.wds_vocal_datasets + self.parquet_vocal_datasets, 
                                 weights=wds_dataset_agg_weight + parquet_dataset_weights),
            pipeline=[{"compose": [self.bucketize]}],
        )

        # validation_dataset = [WebPipeline(
        #     VocalAppZhDataset(
        #         region=region,
        #         dataset_names="SodaMSSTest",
        #         dataset_weights=1,
        #         min_duration=buckets_in_sec[0],
        #         max_duration=buckets_in_sec[-1],
        #         normalize_audio=normalize_audio,
        #         segment_method=segment_method,
        #         max_seg_per_track=1,
        #         segment_max_phone_len=segment_max_phone_len,
        #         include_intro=include_intro,
        #         tokenizer=self.tokenizer,                
        #         use_pipe=use_pipe,
        #         resampled=False,
        #         nodesplitter=return_self,                
        #         handler=wds.warn_and_continue,
        #         vocal_key=vocal_key,
        #         style_key=style_key,
        #         voice_clone_duration=voice_clone_duration,
        #         app_type=app_type,
        #         singsong_predict_type=singsong_predict_type,
        #     ),
        #     pipeline=[{"compose": [self.bucketize]}],
        # )] 

        # karaoke_val_index = "/mnt/bn/audio-diffusion/ashaw/webdataset/karaoke/indexes_with_meta/karaoke_valid.tar_to_index.tsv"
        # validation_dataset_ka = WebPipeline(
        #     SingsongDataset(
        #         url2index=karaoke_val_index,
        #         audio_map_keys={ 'acc_audio': 'acc.mp3', 'target_audio': 'full.mp3', 'vocal_audio': 'vocal.mp3' },
        #         sample_rate=sample_rate,
        #         audio_key = "mp3",
        #         audio_format = "mp3",
        #         resampled=False,
        #         nodesplitter=return_self,
        #         min_duration=buckets_in_sec[0],
        #         max_duration=buckets_in_sec[-1],
        #         segment_method=segment_method,
        #         segment_max_phone_len=segment_max_phone_len,
        #         include_intro=include_intro,
        #         max_seg_per_track=1,
        #         exclude_licenses=["B", "C"],
        #         use_pipe=False,
        #         voice_clone_duration=voice_clone_duration,
        #         voice_clone_random_start=False,
        #         app_type=app_type,
        #         predict_mode=singsong_predict_type,
        #         handler=wds.ignore_and_continue,                
        #     ),
        #     pipeline=[{"compose": [self.bucketize]}],
        # )
        # validation_dataset = [validation_dataset_ka]

        karaoke_val = WebPipeline(VppParquetDataset(
            # data_id=94,  # i18n
            data_id=960,  # CN
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            normalize_audio=normalize_audio,
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            segment_max_phone_len=segment_max_phone_len,
            include_intro=include_intro,
            use_gt_lyrics=True,
            tokenizer=self.tokenizer,                
            resampled=False,
            nodesplitter=return_self,
            shardshuffle=True,
            use_pipe=use_pipe,            
            handler=wds.warn_and_continue                        
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        validation_dataset = [karaoke_val]


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


if __name__ == "__main__":
    # app_type = "vclone"
    # app_type = "singsong"
    app_type = "singsong_inverse"
    # app_type = "singsong_inverse_vc"
    
    # dataset = SingsongWebDataModule(
    #     sample_rate = 24000,
    #     batch_size = 8,
    #     app_type = app_type,
    #     voice_clone_duration=6,
    #     buckets_in_sec = [20, 22, 24, 26, 28, 30, 32],
    #     shuffle_buffer_size = 10,
    #     num_workers = 1,
    #     pin_memory = True,
    #     region = "groupA",
    #     use_dynamic_batch = False,
    #     segment_method  = "random",
    #     segment_max_phone_len  = 400,
    #     include_intro = False,
    #     max_seg_per_track = 3,
    #     exclude_licenses = ["C"],
    # )
    # dataloader = dataset.train_dataloader()
    # # dataloader = dataset.val_dataloader()
    # cnt = 0
    # for batch in dataloader:
    #     print(batch['vocal_audio'].size())
    #     print(batch['target_audio'].shape)
    #     print(batch['acc_audio'].shape)
    #     print(batch.keys())
    #     # if batch['vocal_audio'].shape != torch.Size([12, 1,144000]):
    #     #     print(batch['song_id'])
    #     #     print(batch['shard'])
    #     #     break
        
    #     cnt += 1
    #     if cnt > 5:
    #         break

    # app_type = "vclone"
    # dataset_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/soda/weituo_temp/url2index_l1out.txt"
    # val_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data/soda/weituo_temp/soda_val.tsv"


    dataset = MixVocalAppZhWebDataModule(
        sample_rate = 24000,
        batch_size = 2,
        shuffle_buffer_size = 10,
        num_workers = 1,
        pin_memory = True,
        collate_fn = functools.partial(vpp_collate_fn, app_type=app_type),        
        use_pipe = True,
        tokenizer = "phoneme",
        frame_rate = 25,
        normalize_audio = False,
        region = "CN",
        wds_dataset_names = [],
        wds_dataset_weights = [1],
        parquet_dataset_ids = [96],
        parquet_dataset_weights = [],
        use_dynamic_batch = False,
        segment_method = "random",
        segment_max_phone_len = 400,
        include_intro = False,
        max_seg_per_track = -1,
        buckets_in_sec = [
            20,
            25,
            30,
        ],
        vocal_key= 'vocal',
        style_key= 'acc',
        voice_clone_duration = 6,
        app_type= app_type,
        singsong_predict_type= "BGM", # FULL
    )
    dataloader = dataset.train_dataloader()
    # dataloader = dataset.val_dataloader()[0]
    cnt = 0
    for batch in dataloader:
        print(batch.keys())
        # print(batch['vocal_audio'].shape)
        # print(batch['acc_audio'].shape)
        # print(batch['target_audio'].shape)
        # print(cnt)
        # print(batch['vocal_phones'])
        # print(batch['lyrics_tokens'])

        # cnt += 1
        # if cnt > 20:
        #     break
        break
