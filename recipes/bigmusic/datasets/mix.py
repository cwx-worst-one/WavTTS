import io
import random
from string import punctuation
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
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
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline
import logging, phonemizer
from recipes.bigmusic.datasets.tokenizers.phoneme import MAX_PHONE_LEN
from recipes.bigmusic.utils.format_utils import normalize_text
from recipes.datasets.mcc.mix import (
    DataModule
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
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    SetAudioDimensions,
    ToTensor,
)
from samantha.utils.webdataset import return_self
from transformers import Wav2Vec2PhonemeCTCTokenizer

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

def rewrite_metadata(metadata, type="Vocal"):        
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
        
        # style_text_tokens, _ = pad_crop(
        #     torch.tensor(batch[idx].get("style_tokens", default_style_token.detach().clone())), 
        #     MAX_STYLE_LEN, torch.int, STYLE_PAD_ID)
        # style_tokens.append(style_text_tokens)

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

    return {
        "target_audio": torch.stack(audio, dim=0), 
        "style_text": style_text,
        # "style_tokens": torch.stack(style_tokens),
        "normalized_text": normalized_text,
        "lyrics": normalized_text,
        "lyrics_tokens": torch.stack(lyrics_tokens),
        "speaker_id": torch.stack(speaker_id),
        "conditions": "style_text,lyrics_tokens",

        # DEBUG:
        "style_metadata": style_metadata,
        "song_id": song_id,
        "shard": shard,
        "worker_id": worker_id,
    }

def group_utterances(utterances, min_duration, max_duration, time_in_sec=False, include_intro=False):
    if time_in_sec:
        utterances = [(int(u['start_time']), int(u['end_time']), u['text']) for u in utterances]            
    else:
        utterances = [(int(u['start_time']/1000), int(u['end_time']/1000), u['text']) for u in utterances]
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
                segs.append([cur_seg[0][0], cur_seg[-1][1], 
                            ' '.join([u[2] for u in cur_seg])])
                i = k
            # Move to the next vocal starting point
            while i < len(utterances):
                if len(utterances[i][2]) >= 2:
                    break
                i += 1
            if i < len(utterances):
                s, cur_seg = i, []        
    return segs


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

class MixWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
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
        collate_fn: Optional[Callable] = collate_fn,
        region: str = "US",
        use_dynamic_batch: str = False,
        segment_method: str = "random",
        segment_max_phone_len: int = 400,
        include_intro: bool = False,
        max_seg_per_track: int = -1,
        exclude_licenses: List[str] = ["C"],
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
        mcc_vocal_val_index = f'/mnt/bn/audio-diffusion/data/mcc60_slices/val_from_group_1.tsv' # Group A - Label 1 - genre balanced
        mcc_instrumental_index = "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/mega_index.with_ar_scores/npy_url2idx.txt"
        if region == "US":
            mcc_vocal_index = (
                "/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx.txt"
            )
        elif region == "CN":
            mcc_vocal_index = "recipes/datasets/mcc/mcc60m_index.txt"
            mcc_vocal_val_index = "recipes/datasets/mcc/mcc60m_index_val.txt"
            # mcc_instrumental_index = "recipes/datasets/mcc/mcc60m_index_val.txt"
        elif region == "groupA":
            mcc_vocal_index = "/mnt/bn/audio-diffusion/data/mcc60_slices/lyrics_npy_url2idx_groupA.txt"
        elif region == "groupB":
            mcc_vocal_index = "/mnt/bn/audio-diffusion/data/mcc60_slices/lyrics_npy_url2idx_groupB.txt"
        elif region == "1":
            mcc_vocal_index = "/mnt/bn/audio-diffusion/data/mcc60_slices/lyrics_npy_url2idx_1.txt"
        elif region == "3":
            mcc_vocal_index = "/mnt/bn/audio-diffusion/data/mcc60_slices/lyrics_npy_url2idx_3.txt"
        elif region == "4":
            mcc_vocal_index = "/mnt/bn/audio-diffusion/data/mcc60_slices/lyrics_npy_url2idx_4.txt"
        else:
            raise ValueError
        mcc_vocal = MCCVocalDataset(
            url2index=mcc_vocal_index,
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            segment_method=segment_method,
            segment_max_phone_len=segment_max_phone_len,
            include_intro=include_intro,
            max_seg_per_track=max_seg_per_track,
            exclude_licenses=exclude_licenses,
            use_pipe=use_pipe,
            handler=wds.reraise_exception,
        )
        # mcc_instrumental = MCCInstrumentalDataset(
        #     url2index=mcc_instrumental_index,
        #     sample_rate=sample_rate,
        #     resampled=True,
        #     shardshuffle=True,
        #     min_duration=buckets_in_sec[0],
        #     max_duration=buckets_in_sec[-1],
        #     exclude_licenses=exclude_licenses,
        #     use_pipe=False,
        #     handler=wds.reraise_exception,
        # )
        train_dataset = WebPipeline(            
            mcc_vocal,
            # MultiIterableDataset(
            #     datasets=[mcc_vocal, mcc_instrumental], weights=[5, 1]                
            # ),
            pipeline=[{"compose": [self.bucketize]}],
        )

        validation_dataset = WebPipeline(
            # TODO (qq) change it to MTAT
            MCCVocalDataset(
                url2index=mcc_vocal_val_index,
                sample_rate=sample_rate,
                resampled=False,
                nodesplitter=return_self,
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],
                segment_method=segment_method,
                segment_max_phone_len=segment_max_phone_len,
                include_intro=include_intro,
                max_seg_per_track=1,
                exclude_licenses=["B", "C"],
                use_pipe=use_pipe,
                handler=wds.reraise_exception,                
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )

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


class TTSWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        buckets_in_sec: List[int] = [
            20,
            25,
            30,
        ],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,        
        region: str = "CN",
        use_pipe: bool = True,        
    ):
        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=False,
            batch_size=batch_size,
            length_fn=lambda x: x["audio"].shape[-1],
        )
        if region == "US":
            librilight_url = "hdfs:////home/byte_speech_sv/data/speech/librilight_asr_npy/url2idx.txt"
            libritts_url = "pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/train-clean-360/{00000..00007}.tar"
            libritts_val_url = "hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar"
        elif region == "CN":
            librilight_url = "hdfs://haruna/home/byte_speech_sv/data/speech/librilight/index/url2idx.txt"
            libritts_url = "pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/train-clean-360/{00000..00007}.tar"
            libritts_val_url = "pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar"
        else:
            raise KeyError(f"Wrong region: {region}")
        librilight = LibrilightDataset(
            url2index=librilight_url,
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            use_pipe=use_pipe,
            handler=wds.warn_and_continue,
        )
        libritts = LibriTTSDataset(
            urls=libritts_url,
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],            
            handler=wds.warn_and_continue,
        )
        train_dataset = WebPipeline(
            MultiIterableDataset(
                datasets=[librilight, libritts], 
                weights=[50, 1]),   # natural weight 100:1
            pipeline=[{"compose": [self.bucketize]}],
        )
        validation_dataset = WebPipeline(
            LibriTTSDataset(
                urls=libritts_val_url,
                sample_rate=sample_rate,                        
                min_duration=buckets_in_sec[0],
                max_duration=buckets_in_sec[-1],                
                handler=wds.warn_and_continue,
                resampled=False,
                nodesplitter=return_self,                
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
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
