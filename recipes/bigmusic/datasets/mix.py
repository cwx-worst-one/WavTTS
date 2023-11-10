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
from recipes.bigmusic.datasets.transforms.lyrics_segment import force_aligned_word_format_to_line_format
from recipes.datasets.mcc.mix import (
    INDEX,
    LibriTTSDataset,
    MCCInstrumentalDataset,
    MCCVocalDataset,
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
    NormalizeAudioToFloat32,
    RandomPad,
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

def ffmpeg_read_audio(audio_bin, sample_rate=24000):
    seg_bin, err = ffmpeg.input("pipe:").output("pipe:", loglevel="error", format="s16le", ar=sample_rate).run(input=audio_bin, quiet=True)
    return (np.frombuffer(seg_bin, dtype="int16") / 32768.0).astype(np.float32)

def rewrite_metadata(metadata, type="Vocal"):
    mood = metadata.get('final_mood')
    genre = metadata.get('final_genre')
    gender = metadata.get('merge_aed')
    text = ""
    if type == "Vocal":
        text = "A"
        if mood is not None and mood != 'nan' and mood.strip():
            text += " " + mood.lower()
        if genre is not None and genre != 'nan' and genre.strip():
            text += " " + genre.lower()
        text += " song"
        if gender is not None and gender != 'nan':
            if 'Female' in gender:
                text += " with female vocal"
            elif 'Male' in gender:
                text += " with male vocal"
        text += "."
    elif type == "Instrumental":
        text = ""
        if mood is not None and mood != 'nan' and mood.strip():
            text += mood.lower() + " "
        if genre is not None and genre != 'nan' and genre.strip():
            text += genre.lower() + " "
        text += "music."
    elif type == "Speech":
        text = "Speech."
    elif type == "mir_tags":
        # NOTE: randomly shuffle to diversify prompt
        random.shuffle(metadata["genres"])
        random.shuffle(metadata["vocals"]) 

        def multiple_choices_text_processor(text: List[str]) -> str:
            if len(text) > 1:
                text = ", ".join(text[:-1]) + f" and {text[-1]}"
            elif len(text) == 1:
                text = text[0]
            else:
                text = ""
            return text.lower()

        # example: 'rock, pop and blues'
        genre_text = multiple_choices_text_processor(metadata["genres"])
        genre_text = genre_text.replace("_", " ") # rnb_soul -> rnb soul
        if mood is not None and mood != 'nan':
            genre_text = f"{mood.lower()} {genre_text}"

        gender = {
            "gender_male": "male",
            "gender_female": "female",
        }
        vocal_gender_text = multiple_choices_text_processor([gender.get(v, "") for v in metadata["vocals"] if "gender_" in v])
        text = f"""A {genre_text} song with {vocal_gender_text} vocal."""
    return text

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
                            new_line_token.join([u[2] for u in cur_seg]),
                            new_line_token.join([u[3] for u in cur_seg])])
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
            self.phoneme_tokenizer._add_tokens([" <n> "])
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
                start = int(segment[0] * self.sample_rate)
                end = int(segment[1] * self.sample_rate)

                if audio.ndim == 1:
                    audio = audio.unsqueeze(dim=0)
                
                clip = audio[:, start:end]
                if clip.shape[-1] < self.sample_rate * self.min_duration // 2:
                    print('audio too short', clip.shape)
                    continue

                if clip.shape[0] == 0 or clip.shape[1] == 0:
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
        self.phonemizer._add_tokens(["<n>"])        

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
            min_duration_sec = 15
            segments = [s for s in segments if s[1] - s[0] >= min_duration_sec]
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

                if audio.ndim == 1:
                    audio = audio.unsqueeze(dim=0)

                clip = audio[:, start:end]
                if clip.shape[0] == 0 or clip.shape[1] == 0:
                    continue

                normalized_text = normalize_text(segment[2])
                phoneme_tokens = self.phoneme_tokenizer(normalized_text)["input_ids"]

                style_metadata = select_tag_metadata_from_timestamps(item["__index_data__"], start, end)

                # skip if lyrics is too short
                if len(phoneme_tokens) < 5:
                    continue
                # skip when no genre detected
                if len(style_metadata["genres"]) == 0:
                    continue

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
