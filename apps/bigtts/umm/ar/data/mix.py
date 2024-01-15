import json
import logging
import math
import pickle
import random
import sys
from enum import IntEnum
from functools import partial
from string import punctuation
from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    Tuple, Union)

import numpy as np
import phonemizer
import pytorch_lightning as pl
import torch
import webdataset as wds
from apps.bigtts.umm.ar.data import INDEX
from apps.bigtts.umm.ar.data.phone_to_id import PhoneToId
from apps.bigtts.umm.ar.data.sami_tokenizer import convert_labels_to_text_id
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import BertTokenizer, Wav2Vec2PhonemeCTCTokenizer
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline

from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.utils import parse_data_urls
from samantha.dataio.webdataset import ra_wds
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (NormalizeAudioToFloat32, Pad, RandomPad,
                                       SetAudioDimensions, ToTensor)
from samantha.utils.audio import FastNormalizeAudio, LoudnessCheck
from samantha.utils.webdataset import return_self


class DatasetType(IntEnum):
    AUDIO = 1
    UMM_TOKEN = 2

def normalize_text(text):
    nlp_punctuation = punctuation.replace("'", "")
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    return text.translate(str.maketrans("", "", nlp_punctuation)).strip()


def collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    # text = []
    token = []
    # tag = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        # text.append(batch[idx]["text"])
        token.append(batch[idx].get("token", torch.zeros(0).long()))
        # tag.append(batch[idx]["tag"])
    return {
        "audio": torch.stack(audio, dim=0),
        # "text": text,
        "token": torch.nn.utils.rnn.pad_sequence(
            token, batch_first=True, padding_value=0
        ),
        # "tag": tag,
    }


def collate_audio_text(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    text = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        text.append(batch[idx]["text"])
    return {"audio": torch.stack(audio, dim=0), "text": text}


def collate_audio(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)
    audio = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
    return {"audio": torch.stack(audio, dim=0)}


class BaseTransforms:
    """Base class for all data transforms"""

    name = "BaseTransforms"

    def __init__(self, log_interval: int = 100):
        self.count = 0
        self.skipped = 0
        self.messages = {}
        self.log_interval = log_interval

    def _update_stats(self, skipped: bool, message: Optional[str] = None):
        self.count += 1
        if skipped:
            self.skipped += 1
        if message is not None:
            message = f"[{self.name}] {message}"
            if message not in self.messages:
                self.messages[message] = 0
            self.messages[message] += 1
        # Print
        if self.count > 0 and self.count % self.log_interval == 0:
            worker_id = torch.utils.data.get_worker_info()
            if worker_id is not None:
                worker_id = worker_id.id
            else:
                worker_id = "Undefined"
            print(
                f"[{worker_id}] "
                f"Skipped {self.skipped}/{self.count} items, "
                f"Messages: {self.messages}",
                file=sys.stderr,
                flush=True,
            )

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        raise NotImplementedError()


class MCCTransforms(BaseTransforms):
    name = "MCCTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
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
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ) -> None:
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.lyrics_confidence = lyrics_confidence
        self.aed_filtered = aed_filtered
        self.audio_metrics_filtered = audio_metrics_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = exclude_licenses
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.is_loud = LoudnessCheck(
            sample_rate, min_volume_threshold, loudness_ratio_threshold
        )
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def is_metadata_good(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        # Apply AED filtering if applicable
        if self.aed_filtered and not metadata.get("aed_filtered", False):
            return False, "Not AED Filtered"
        # Avoid sound effect if applicable
        if self.avoid_sound_effect and metadata.get("final_theme") == "Sound Effect":
            return False, "Sound Effect"
        # Apply license-based filtering if applicable
        if len(self.exclude_licenses) > 0:
            for license in metadata.get("license_types", []):
                if license in self.exclude_licenses:
                    return False, "Excluded License"
        # Apply AudioMetrics filtering if applicable
        if self.audio_metrics_filtered:
            is_good, msg = self.is_audio_metrics_good(metadata.get("audio_metrics", {}))
            if not is_good:
                return False, msg
        return True, None

    def is_audio_metrics_good(self, audio_metrics: Dict[str, Any]) -> Tuple[bool, str]:
        # Clipping
        clip = audio_metrics.get("clipping", {})
        if clip.get("rate", 0) >= 5e-5:
            return False, "clipping"
        for ch in ["left", "right"]:
            if clip.get(f"peak_rate_{ch}", 0) >= 0.05:
                return False, "clipping"
        # Loudness
        loudness = audio_metrics.get("loudness", {})
        if (
            loudness.get("integrated_loudness", -7) > -5
            or loudness.get("max_mom_loud", -7) >= 0
            or loudness.get("max_short_term_loud", -7) >= 0
        ):
            return False, "loudness"
        # RMS stats
        rms_stats = audio_metrics.get("rms_stats", {})
        if rms_stats.get("peak", 0) > 3:
            return False, "rms_stats"
        for ch in ["left", "right"]:
            if (
                rms_stats.get(f"{ch}_total", -10) > -5
                or rms_stats.get(f"{ch}_total", -10) < -40
                or rms_stats.get(f"normed_std_{ch}", -10) < -19.5
            ):
                return False, "rms_stats"
        # Cutoff frequency
        cutoff_freq = audio_metrics.get("cutoff_frequency", {})
        for ch in ["left", "right"]:
            if (
                cutoff_freq.get(f"rel_{ch}", 48000) < 15000
                and cutoff_freq.get(f"rel_{ch}_conf", 0) > 0.6
                and cutoff_freq.get(f"band_std_{ch}", 10) < 5
            ):
                return False, "cutoff_frequency"
        # Phase
        phase = audio_metrics.get("phase_check", {})
        if (
            phase.get("has_phase_issue", False)
            or abs(phase.get("rms_downmix_diff", 0.1)) > 3
        ):
            return False, "phase_check"
        return True, None

    def __call__(self, x: Dict[str, Any]) -> Generator:
        raise NotImplementedError()


class MCCVocalTransforms(MCCTransforms):
    name = "MCCVocalTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
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
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
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
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        is_good, message = self.is_metadata_good(item["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return
        lyrics = item["__index_data__"].get("lyrics", None)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return
        utterances = lyrics.get("utterances", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        random.shuffle(filtered_utterances)
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        for selected_utterance in filtered_utterances:
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            clip = audio[:, start:end]
            text = normalize_text(str(selected_utterance["text"]))
            output_dict = {"audio": clip, "text": text, "tag": "vocal"}
            if self.tokenizer is not None:
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                elif (
                    token.size(-1)
                    > math.floor(clip.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class MCCInstrumentalTransforms(MCCTransforms):
    name = "MCCInstrumentalTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )

    def __call__(self, item: Dict[str, Any]) -> Generator:
        is_good, message = self.is_metadata_good(item["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        if audio.size(-1) < self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return

        if self.max_num_crops is not None:
            max_num_crops = self.max_num_crops
        else:
            max_num_crops = audio.size(-1) // (self.max_duration * self.sample_rate)

        durations = []
        starts = []
        for _ in range(max_num_crops):
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            durations.append(duration)
            starts.append(random.randint(0, audio.size(-1) - duration))

        for i in range(max_num_crops):
            clip = audio[:, starts[i] : starts[i] + durations[i]]
            if not self.is_loud(clip):
                self._update_stats(skipped=True, message="Not loud enough")
                return
            output_dict = {"audio": clip, "text": "", "tag": "instrumental"}
            if self.tokenizer is not None:
                if self.tokenizer == "tts_chinese_frontend_model":
                    token = torch.zeros(0).long()
                else:
                    encoded_text = self.tokenizer(
                        "",
                        add_special_tokens=False,
                        # padding="longest",
                        return_tensors="pt",
                    )
                    token = encoded_text["input_ids"].squeeze(dim=0)
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class KaraokeTransforms(BaseTransforms):
    name = "KaraokeTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "full.npy",
        min_duration: float = 5,
        max_duration: float = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        lyrics = item["__index_data__"].get("lyrics", None)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return
        utterances = lyrics.get("utterances", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        random.shuffle(filtered_utterances)
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        for selected_utterance in filtered_utterances:
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            clip = audio[:, start:end]
            text = normalize_text(str(selected_utterance["text"]))
            output_dict = {"audio": clip, "text": text, "tag": "vocal"}
            if self.tokenizer is not None and callable(self.tokenizer):
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                elif (
                    token.size(-1)
                    > math.floor(clip.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class LibriLightASRTransforms(BaseTransforms):
    name = "LibriLightASRTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"])
        end_time = float(utterance["end_time"])
        delta = end_time - start_time
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        audio = self.base_transform(item[self.audio_key])
        utterances = item.get("__index_data__", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        random.shuffle(filtered_utterances)
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        for selected_utterance in filtered_utterances:
            start = int(float(selected_utterance["start_time"]) * self.sample_rate)
            end = int(float(selected_utterance["end_time"]) * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            clip = audio[:, start:end]
            text = normalize_text(str(selected_utterance["text"]))
            output_dict = {"audio": clip, "text": text, "tag": "vocal"}
            if self.tokenizer is not None:
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                elif (
                    token.size(-1)
                    > math.floor(clip.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class LibriTTSTransforms(BaseTransforms):
    name = "LibriTTSTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        audio = self.base_transform(item[self.audio_key])
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        text = normalize_text(item["normalized_text.txt"])
        output_dict = {"audio": audio, "text": text, "tag": "vocal"}
        if self.tokenizer is not None and callable(self.tokenizer):
            encoded_text = self.tokenizer(
                text,
                add_special_tokens=False,
                # padding="longest",
                return_tensors="pt",
            )
            token = encoded_text["input_ids"].squeeze(dim=0)
            if token.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                return
            elif (
                token.size(-1)
                > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
            ):
                self._update_stats(skipped=True, message="Token too long")
                return
            output_dict.update(token=token)
        yield output_dict
        self._update_stats(skipped=False)


class SpeechZhTransforms(BaseTransforms):
    name = "SpeechZhTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        audio = self.base_transform(item[self.audio_key])
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        text = item["__index_data__"].get("text", None)
        if text is None or len(text) == 0:
            self._update_stats(skipped=True, message="No text")
            return
        output_dict = {"audio": audio, "text": text, "tag": "speech"}
        if self.tokenizer is not None:
            if self.tokenizer == "tts_chinese_frontend_model":
                labels = list(
                    filter(
                        lambda x: x != "", item["__index_data__"]["labels"].split("\n")
                    )
                )
                labels, _, _ = convert_labels_to_text_id(labels)
                token = torch.from_numpy(labels[0]).long()
            else:
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)
            if token.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                return
            elif (
                token.size(-1)
                > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
            ):
                self._update_stats(skipped=True, message="Token too long")
                return
            output_dict.update(token=token)
        yield output_dict
        self._update_stats(skipped=False)


class DouyinMusicTransforms(BaseTransforms):
    name = "DouyinMusicTransforms"
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
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        audio_npy = item[self.audio_key]
        if audio_npy.shape[-1] < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio_npy.shape[-1] > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too long")
            return
        try:
            audio = self.base_transform(audio_npy)
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        text = item["__index_data__"].get("text", None)
        if text is None or len(text) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return

        output_dict = {"audio": audio, "text": text, "tag": "vocal"}
        if self.tokenizer is not None:
            encoded_text = self.tokenizer(
                text,
                add_special_tokens=False,
                # padding="longest",
                return_tensors="pt",
            )
            token = encoded_text["input_ids"].squeeze(dim=0)
            if token.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                return
            elif (
                token.size(-1)
                > math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
            ):
                self._update_stats(skipped=True, message="Token too long")
                return
            output_dict.update(token=token)
        yield output_dict
        self._update_stats(skipped=False)


class VocalZhTransforms(BaseTransforms):
    name = "VocalZhTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 1,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.lyrics_confidence = lyrics_confidence
        self.audio_key = audio_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate and self.audio_key.endswith("npy"):
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        confidence = float(utterance["confidence"])
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        if confidence < self.lyrics_confidence:
            return False
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        lyrics = item["__index_data__"].get("lyrics", None)
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

        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        random.shuffle(filtered_utterances)
        if self.max_num_crops is not None and self.max_num_crops > 0:
            filtered_utterances = filtered_utterances[: self.max_num_crops]

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        for selected_utterance in filtered_utterances:
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            clip = audio[:, start:end]
            text = normalize_text(str(selected_utterance["text"]))
            output_dict = {"audio": clip, "text": text, "tag": "vocal"}
            if self.tokenizer is not None:
                if self.tokenizer == "tts_chinese_frontend_model":
                    labels = list(
                        filter(
                            lambda x: x != "", selected_utterance["phoneme"].split("\n")
                        )
                    )
                    labels, _, _ = convert_labels_to_text_id(labels)
                    token = torch.from_numpy(labels[0]).long()
                else:
                    encoded_text = self.tokenizer(
                        text,
                        add_special_tokens=False,
                        # padding="longest",
                        return_tensors="pt",
                    )
                    token = encoded_text["input_ids"].squeeze(dim=0)
                if token.size(-1) == 0:
                    self._update_stats(skipped=True, message="Token zero length")
                    return
                elif (
                    token.size(-1)
                    > math.floor(clip.size(-1) / self.sample_rate) * self.frame_rate
                ):
                    self._update_stats(skipped=True, message="Token too long")
                    return
                output_dict.update(token=token)
            yield output_dict
            self._update_stats(skipped=False)


class WebDatasetBufferPreprocessor:
    def __init__(self, transforms: BaseTransforms):
        self.transforms = transforms

    def train_buffer_preprocessor(self, buffer: Generator) -> Generator:
        for item in buffer:
            yield from self.transforms(item)


class LibriLightDataset(WebPipeline):
    name = "LibriLight"
    data_sample_rate = 16000

    def __init__(
        self,
        urls,
        sample_rate=24000,
        min_duration: int = 5,
        max_duration: int = 30,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        self.base_transform = Compose(base_transforms)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"map": [self.transform]}]
        super().__init__(dataset, pipeline)

    def transform(self, item) -> Dict[str, Any]:
        audio = self.base_transform(item["audio.npy"])
        if audio.size(-1) < self.min_duration * self.sample_rate:
            return
        if audio.size(-1) > self.max_duration * self.sample_rate:
            return
        return {"audio": audio}


class LibriLightASRDataset(WebPipeline):
    name = "LibriLightASR"

    def __init__(
        self,
        url2index: str = "hdfs:///home/byte_speech_sv/data/speech/librilight_asr_npy/url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = LibriLightASRTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class BigTTSTransforms(BaseTransforms):
    name = "BigTTSTransforms"
    data_sample_rate = 24000

    def __init__(
            self,
            sample_rate: int = 24000,
            umm_token_freq: int = 40,
            audio_key: str = "wav",
            target_token_key: str = "umm_token",
            min_duration: int = 5,
            max_duration: int = 30,
            normalize_audio: bool = False,
            max_num_crops: int = None,
            tokenizer=None,
            phone2id=None,
            phone_tone_wordseg_dict=None,
            frame_rate: int = 25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.umm_token_freq = umm_token_freq
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.target_token_key = target_token_key
        self.max_num_crops = max_num_crops
        self.tokenizer = tokenizer
        self.phone2id = phone2id
        self.phone_tone_wordseg_dict = phone_tone_wordseg_dict
        self.frame_rate = frame_rate
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        # if self.data_sample_rate != sample_rate:
        #     base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        # if normalize_audio:
        #     base_transforms.append(FastNormalizeAudio())
        self.resampler = {}
        self.fast_normalizer = FastNormalizeAudio()
        self.normalize_audio = normalize_audio
        self.base_transform = Compose(base_transforms)
        self.lang2id = {
            "en": 0,
            "zh": 1,
            "zh_en": 2,
        }

    def phone_tone_wordseg_to_id(self, phone, tone, word_seg):
        text_id = phone.astype(np.int64) * 1_000_000 + tone.astype(np.int64) * 1_000 + word_seg.astype(np.int64)
        res = [0] * len(text_id)
        for i, t_id in enumerate(text_id):
            if t_id not in self.phone_tone_wordseg_dict:
                print(f"===>>> phone_tone_wordseg_dict OOV")
                # np.save('tmp/{}.npy'.format(t_id), t_id)
                # key_idx = np.asarray(list(self.phone_tone_wordseg_dict.keys()))
                # min_ind = np.argmin(np.abs(key_idx - t_id))
                # res[i] = self.phone_tone_wordseg_dict[key_idx[min_ind]]
                return None
            else:
                res[i] = self.phone_tone_wordseg_dict[t_id]
        return np.asarray(res).astype(np.int32)

    def preprocess_meta(self, sample):
        meta_obj = json.loads(sample["meta"])
        while not isinstance(meta_obj, dict):
            meta_obj = json.loads(meta_obj)
        item = {}
        item["labels"] = str(meta_obj.get("labels", ""))
        sample.update(item)
        return sample

    def do_resample(self, src_sample_rate, x):
        if src_sample_rate == self.sample_rate:
            return x

        if src_sample_rate not in self.resampler:
            self.resampler[src_sample_rate] = Resample(src_sample_rate, self.sample_rate)
        return self.resampler[src_sample_rate](x)

    def get_lang(self, tacolab):
        if len(tacolab[0].split('\t')) != 5:
            if tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword' or \
                tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit' or \
                tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\talignment' or \
                tacolab[0] == 'phn\ttone\tws\tpwpp\tsentype\tword\tunit\talignment':
                tacolab = tacolab[1:]
        prefix_phn_list = [x.split('\t')[0][:2] for x in tacolab]
        if 'C0' in prefix_phn_list:
            if 'E0' in prefix_phn_list:
                lang = 'zh_en'
            else:
                lang = 'zh'
        else:
            lang = 'en'
        return lang

    def __call__(self, item: Dict[str, Any]) -> Generator:

        if self.audio_key in item:
            audio = self.base_transform(item[self.audio_key])
            audio = self.do_resample(item["src_sample_rate"], audio)
            if self.normalize_audio:
                audio = self.fast_normalizer(audio)

            if audio.size(-1) < self.min_duration * self.sample_rate:
                self._update_stats(skipped=True, message="Audio too short")
                return
            if audio.size(-1) > self.max_duration * self.sample_rate:
                self._update_stats(skipped=True, message="Audio too long")
                return
        
        if self.target_token_key in item:
            target_token= torch.as_tensor(pickle.loads(item[self.target_token_key]), dtype=torch.long)

            if target_token.size(-1) < self.min_duration * self.umm_token_freq:
                self._update_stats(skipped=True, message=f"umm token too short")
                return
            if target_token.size(-1) > self.max_duration * self.umm_token_freq:
                self._update_stats(skipped=True, message=f"umm token too long")
                return

        text = normalize_text(item["text"])
        item = self.preprocess_meta(item)
        labels = item["labels"]
        # labels = item["labels"].decode()

        if labels is None:
            self._update_stats(skipped=True, message="Label is None")
            return

        output_dict = {"text": text, "tag": "vocal"}
        if self.audio_key in item:
            output_dict[self.audio_key] = audio
        if self.target_token_key in item:
            output_dict[self.target_token_key] = target_token

        if self.tokenizer is not None:
            if self.tokenizer == "sami":
                labels = list(filter(lambda x: x != "", labels.split('\n')))
                try:
                    if len(labels) < 2:
                        self._update_stats(skipped=True, message="Label length is shorter than 2")
                        return
                    if len(labels[-1].split('\t')) == 2:
                        last_duration = labels[-1].split('\t')[-1]
                        last_line = labels[-2]
                        labels = labels[:-2]
                        last_line = '\t'.join(last_line.split('\t')[:-1] + [last_duration])
                        labels.append(last_line)

                    # Get language ID
                    lang_key = self.get_lang(labels)
                    lang_id = self.lang2id[lang_key]
                    output_dict.update(lang=lang_id)

                    # Convert tacolabel to phone, tone, and wordseg ids
                    text_id_phones_tones = self.phone2id.convert_tacolab_to_text_id(labels)
                    if text_id_phones_tones is None:
                        self._update_stats(skipped=True, message="convert_tacolab_to_text_id failed")
                        return
                    else:
                        text_id, _, _, _, _ = text_id_phones_tones

                    # Map phone, tone, and wordseg to id
                    token = self.phone_tone_wordseg_to_id(text_id[0, :], text_id[1, :], text_id[2, :])
                    if token is None:
                        self._update_stats(skipped=True, message="phone_tone_wordseg_to_id failed")
                        return
                    else:
                        token = torch.from_numpy(token).long()

                    # phone, tone, word_seg
                    phone, tone, wordseg = text_id[0, :], text_id[1, :], text_id[2, :]
                    phone = torch.from_numpy(phone).long()
                    tone = torch.from_numpy(tone).long()
                    wordseg = torch.from_numpy(wordseg).long()
                    output_dict.update(phone=phone)
                    output_dict.update(tone=tone)
                    output_dict.update(wordseg=wordseg)

                except:
                    self._update_stats(skipped=True, message="Tacolabel process failed")
                    return
            else:
                encoded_text = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    # padding="longest",
                    return_tensors="pt",
                )
                token = encoded_text["input_ids"].squeeze(dim=0)

            if token.size(-1) == 0:
                self._update_stats(skipped=True, message="Token zero length")
                return
            else:
                
                if self.audio_key in item:
                    frame_num = math.floor(audio.size(-1) / self.sample_rate) * self.frame_rate
                elif self.target_token_key in item:
                    frame_num = math.floor(target_token.size(-1) / self.umm_token_freq) * self.frame_rate
                else:
                    frame_num = 0
                if ( token.size(-1) > frame_num):
                    self._update_stats(skipped=True, message="Token too long")
                    return
            output_dict.update(token=token)
        yield output_dict
        self._update_stats(skipped=False)


class BigTTSDataset(WebPipeline):
    name = "BigTTS"

    def __init__(
            self,
            data_id: int = 242, # 181: en_4.2wh, 182: en_4.2wh_cn_3.2wh
            # url_pattern: str = 'hdfs://haruna/home/byte_data_seed/lf_lq/speech/data_store/BigTTS/tts_Len_S11labs-rp2900_P1/package/wav_1.0_web_dataset_2/data/part=00003/shard-00010.tar', # for debug, datasets 181/182 are too large.
            url_pattern: str = None,
            sample_rate: int = 24000,
            umm_token_freq: int = 40,
            audio_key: str = "wav",
            target_token_key: str = "umm_token",
            min_duration: int = 5,
            max_duration: int = 30,
            normalize_audio: bool = False,
            max_num_crops: int = None,
            tokenizer=None,
            phone2id=None,
            phone_tone_wordseg_dict=None,
            frame_rate: int = 25,
            **kwargs,
    ):
        print(f"[{self.name}] [data_id: {data_id}] initializing...")
        # if url_pattern is None:
        #     urls = parse_data_urls(data_id=data_id)
        # else:
        #     urls = parse_data_urls(data_urls=url_pattern)
        # dataset = ra_wds.WebDataset(urls=urls, **kwargs)
        dataset = ParquetDataset(data_id=data_id, data_urls=url_pattern, **kwargs)
        transforms = BigTTSTransforms(
            sample_rate=sample_rate,
            umm_token_freq=umm_token_freq,
            audio_key=audio_key,
            target_token_key=target_token_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            phone2id=phone2id,
            phone_tone_wordseg_dict=phone_tone_wordseg_dict,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")



class LibriTTSDataset(WebPipeline):
    name = "LibriTTS"
    data_sample_rate = 24000

    def __init__(
        self,
        urls: str = "pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 0,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = LibriTTSTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MCCBaseDataset(WebPipeline):
    name = "MCCBaseDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = None,
        weights: List[float] = None,
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
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        if self.name.startswith("MCCVocal"):
            transforms = MCCVocalTransforms(
                sample_rate=sample_rate,
                audio_key=audio_key,
                min_duration=min_duration,
                max_duration=max_duration,
                min_volume_threshold=min_volume_threshold,
                loudness_ratio_threshold=loudness_ratio_threshold,
                lyrics_confidence=lyrics_confidence,
                normalize_audio=normalize_audio,
                aed_filtered=aed_filtered,
                audio_metrics_filtered=audio_metrics_filtered,
                avoid_sound_effect=avoid_sound_effect,
                exclude_licenses=exclude_licenses,
                max_num_crops=max_num_crops,
                tokenizer=tokenizer,
                frame_rate=frame_rate,
            )
        elif self.name.startswith("MCCInstrumental"):
            transforms = MCCInstrumentalTransforms(
                sample_rate=sample_rate,
                audio_key=audio_key,
                min_duration=min_duration,
                max_duration=max_duration,
                min_volume_threshold=min_volume_threshold,
                loudness_ratio_threshold=loudness_ratio_threshold,
                lyrics_confidence=lyrics_confidence,
                normalize_audio=normalize_audio,
                aed_filtered=aed_filtered,
                audio_metrics_filtered=audio_metrics_filtered,
                avoid_sound_effect=avoid_sound_effect,
                exclude_licenses=exclude_licenses,
                max_num_crops=max_num_crops,
                tokenizer=tokenizer,
                frame_rate=frame_rate,
            )
        else:
            raise KeyError("Invalid MCC dataset name")

        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        if isinstance(url2index, list):
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=weights,
            )
        else:
            dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MCCVocalDataset(MCCBaseDataset):
    name = "MCCVocal"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str],
        weights: List[int] = None,
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
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        super().__init__(
            url2index=url2index,
            weights=weights,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            **kwargs,
        )


class MCCVocalDatasetGroupAGenreBalanced(MCCBaseDataset):
    name = "MCCVocalGroupAGenreBalanced"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = [
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-blues.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-childhood.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-classical.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-country.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-devotional.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-easy-listening.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-electronic.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-folk.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-hip-hop-rap.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-jazz.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-metal.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-pop.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-r-b-soul.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-reggae.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-rock.txt",
            "hdfs://haruna/home/byte_speech_sv/data/indexes_vocal-A-1m/index_lists/vocal-A-soundtrack.txt",
        ],
        weights: List[int] = None,
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
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        super().__init__(
            url2index=url2index,
            weights=weights,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            **kwargs,
        )


class MCCInstrumentalDataset(MCCBaseDataset):
    name = "MCCInstrumental"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str],
        weights: List[int] = None,
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
        max_num_crops: int = None,
        **kwargs,
    ):
        super().__init__(
            url2index=url2index,
            weights=weights,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_volume_threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold,
            lyrics_confidence=lyrics_confidence,
            normalize_audio=normalize_audio,
            aed_filtered=aed_filtered,
            audio_metrics_filtered=audio_metrics_filtered,
            avoid_sound_effect=avoid_sound_effect,
            exclude_licenses=exclude_licenses,
            max_num_crops=max_num_crops,
            **kwargs,
        )


class KaraokeDataset(WebPipeline):
    name = "Karaoke"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = "hdfs:///home/byte_speech_sv/data/karaoke_for_singsong_npy/url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "full.npy",
        min_duration: int = 0,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = KaraokeTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class DouyinMusicDataset(WebPipeline):
    name = "DouyinMusic"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 0,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):

        print(f"[{self.name}] initializing...")
        transforms = DouyinMusicTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class VocalZhDataset(WebPipeline):
    name = "VocalZh"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = None,
        weights: List[float] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: int = 2,
        max_duration: int = 30,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.7,
        normalize_audio: bool = False,
        max_num_crops: int = None,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
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
            max_num_crops=max_num_crops,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        if isinstance(url2index, list):
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=weights,
            )
        else:
            dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class SpeechZhDataset(WebPipeline):
    name = "SpeechZhDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = None,
        weights: List[float] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 2,
        max_duration: float = 30,
        normalize_audio: bool = True,
        tokenizer=None,
        frame_rate: int = 25,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = SpeechZhTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        print(f"[{self.name}] MultiIterableDataset constructing...")
        if isinstance(url2index, list):
            dataset = MultiIterableDataset(
                datasets=[
                    IndexedWebDataset(url2index=url, **kwargs) for url in url2index
                ],
                weights=weights,
            )
        else:
            dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        print(f"[{self.name}] MultiIterableDataset constructed.")
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


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


class VocalWebDataModule(DataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 2,
        buckets_in_sec: List[int] = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
        ],
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: Tuple[int, int] = (1, 1),
        region: str = "US",
        use_pipe: bool = False,
        val_resampled: bool = False,
    ):
        buckets_samples = list(map(lambda i: i * sample_rate, buckets_in_sec))
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=False,
            batch_size=batch_size,
            length_fn=lambda x: x["audio"].shape[-1],
        )
        if region == "US":
            mcc_vocal_index = (
                "/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx.txt"
            )
            mcc_vocal_val_index = (
                "/mnt/bn/audio-diffusion/data/vocal_mcc_npy/lyrics_npy_url2idx_val.txt"
            )
        elif region == "CN":
            mcc_vocal_index = "recipes/datasets/mcc/mcc60m_index.txt"  # Not exist in original commit
            mcc_vocal_val_index = "recipes/datasets/mcc/mcc60m_index_val.txt"  # Not exist in original commit
        else:
            raise KeyError(f"Wrong region: {region}")
        mcc_vocal = MCCVocalDataset(
            url2index=mcc_vocal_index,
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            use_pipe=use_pipe,
            handler=wds.warn_and_continue,
        )
        libritts = LibriTTSDataset(
            urls="pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/train-clean-360/{00000..00007}.tar",
            sample_rate=sample_rate,
            resampled=True,
            shardshuffle=True,
            min_duration=buckets_in_sec[0],
            max_duration=buckets_in_sec[-1],
            handler=wds.warn_and_continue,
        )
        train_dataset = WebPipeline(
            MultiIterableDataset(datasets=[mcc_vocal, libritts], weights=weights),
            pipeline=[{"compose": [self.bucketize]}],
        )
        if val_resampled:
            validation_dataset = [
                WebPipeline(
                    MCCVocalDataset(
                        url2index=mcc_vocal_val_index,
                        sample_rate=sample_rate,
                        resampled=True,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                    ),
                    pipeline=[{"compose": [self.bucketize]}],
                ),
                WebPipeline(
                    LibriTTSDataset(
                        urls="pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar",
                        sample_rate=sample_rate,
                        resampled=True,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        handler=wds.warn_and_continue,
                    ),
                    pipeline=[{"compose": [self.bucketize]}],
                ),
            ]
        else:
            validation_dataset = [
                WebPipeline(
                    MCCVocalDataset(
                        url2index=mcc_vocal_val_index,
                        sample_rate=sample_rate,
                        nodesplitter=return_self,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                    ),
                    pipeline=[{"compose": [self.bucketize]}],
                ),
                WebPipeline(
                    LibriTTSDataset(
                        urls="pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar",
                        sample_rate=sample_rate,
                        nodesplitter=return_self,
                        min_duration=buckets_in_sec[0],
                        max_duration=buckets_in_sec[-1],
                        handler=wds.warn_and_continue,
                    ),
                    pipeline=[{"compose": [self.bucketize]}],
                ),
            ]

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
            if isinstance(item, Iterable):
                for i in item:
                    batch = self.batcher.collate_batch(i)
                    if batch is not None:
                        yield batch
            else:
                batch = self.batcher.collate_batch(item)
                if batch is not None:
                    yield batch


def bigmusic_collate_fn(
    target_audio_key: str,
    target_token_key: str,
    batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "audio" in batch[0]:
        max_length = max([x.get(target_audio_key).shape[-1] for x in batch])
        random_pad = RandomPad(n_samples=max_length)
        audio = []
        # text = []
        token = []
        # tag = []
        for idx in range(len(batch)):
            audio.append(random_pad(batch[idx].get(target_audio_key)))
            # text.append(batch[idx]["text"])
            token.append(batch[idx].get("token", torch.zeros(0).long()))
            # tag.append(batch[idx]["tag"])
        return {
            "target_audio": torch.stack(audio, dim=0),
            # "text": text,
            "conditions": "lyrics_tokens",
            "lyrics_tokens": torch.nn.utils.rnn.pad_sequence(
                token, batch_first=True, padding_value=0
            ),
            # "tag": tag,
        }
    else:
        max_length = max([x.get(target_token_key).shape[-1] for x in batch])
        zero_pad = Pad(n_samples=max_length)
        umm_token = []
        umm_token_length = []
        # text = []
        token = []
        # tag = []
        phone, tone, wordseg = [], [], []
        lang = []
        for idx in range(len(batch)):
            umm_token.append(zero_pad(batch[idx].get(target_token_key).unsqueeze(0)))
            umm_token_length.append(batch[idx].get(target_token_key).numel())
            # text.append(batch[idx]["text"])
            token.append(batch[idx].get("token", torch.zeros(0).long()))
            # tag.append(batch[idx]["tag"])
            phone.append(batch[idx].get("phone", torch.zeros(0).long()))
            tone.append(batch[idx].get("tone", torch.zeros(0).long()))
            wordseg.append(batch[idx].get("wordseg", torch.zeros(0).long()))

            lang.append(batch[idx]["lang"])
        res = {
            "target_ids": torch.cat(umm_token, dim=0),
            "target_ids_length": torch.tensor(umm_token_length),
            "lyrics_token_length": torch.tensor([x.numel() for x in token]),
            # "text": text,
            "conditions": "lyrics_tokens",
            "lyrics_tokens": torch.nn.utils.rnn.pad_sequence(
                token, batch_first=True, padding_value=0
            ),
            # "tag": tag,
            "phones": torch.nn.utils.rnn.pad_sequence(phone, batch_first=True, padding_value=0),
            "tones": torch.nn.utils.rnn.pad_sequence(tone, batch_first=True, padding_value=0),
            "wordsegs": torch.nn.utils.rnn.pad_sequence(wordseg, batch_first=True, padding_value=0),
            "lang": torch.tensor(lang),
        }
        # for k, v in res.items():
        #     if k == "conditions":
        #         continue
        #     print(f"key: {k}, value: {v}, value.size(): {v.size()}")
        return res


class MixWebDataModule(pl.LightningDataModule):

    def __init__(
        self,
        sample_rate: int = 24000,
        umm_token_freq: int = 40,
        batch_size: int = 2,
        min_duration: int = 5,
        max_duration: int = 30,
        max_num_crops: int = None,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = bigmusic_collate_fn,
        weights: List[int] = [1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        fast_dev: bool = False,
        small: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
        data_id: int = 242,
        dataset_type: DatasetType = DatasetType.AUDIO,
        skip_validation: bool = False,
        target_token_key: str = "umm_token",
        target_audio_key: str = "wav",
    ):
        super().__init__()
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = partial(collate_fn, target_audio_key, target_token_key)

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            self.phone2id = None
            self.phone_tone_wordseg_dict = None
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        elif tokenizer == "sami":
            self.tokenizer = "sami"
            self.phone2id = PhoneToId()
            self.phone_tone_wordseg_dict = torch.load('apps/bigtts/umm/ar/data/fronted_v3_2_dict.pyt')
            print(f"===>>> Using sami tokenizer")
            print(f"===>>> Size of phone_tone_wordseg_dict: {len(self.phone_tone_wordseg_dict)}")
        else:
            self.tokenizer = None
        self.frame_rate = frame_rate
        if dataset_type == DatasetType.AUDIO:
            assert batch_size >= min_duration * sample_rate
        else:
            assert batch_size >= min_duration * umm_token_freq
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        if dataset_type == DatasetType.AUDIO:
            buckets_samples = [x * sample_rate for x in buckets_samples]
        else:
            buckets_samples = [x * umm_token_freq for x in buckets_samples]

        print(f"[MixWebDataModule] dataset_type={dataset_type} data_id={data_id} target_token_key={target_token_key}")

        if dataset_type == DatasetType.AUDIO:
            length_fn = lambda x: x.get(target_audio_key).size(-1)
        else:
            length_fn = lambda x: x.get(target_token_key).size(-1)

        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=length_fn,
        )
        datasets = []
        if fast_dev:
            self.train_dataset = DataPipeline(
                LibriTTSDataset(
                    urls="pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/train-clean-360/{00000..00007}.tar",
                    sample_rate=sample_rate,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    tokenizer=self.tokenizer,
                    frame_rate=self.frame_rate,
                    resampled=True,
                    shardshuffle=True,
                    handler=wds.warn_and_continue,
                ),
                self.bucketize,
            )
        else:
            if weights[0] > 0:
                if small:
                    mcc_vocal = MCCVocalDatasetGroupAGenreBalanced(
                        sample_rate=sample_rate,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        max_num_crops=max_num_crops,
                        normalize_audio=normalize_audio,
                        tokenizer=self.tokenizer,
                        frame_rate=self.frame_rate,
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                    )
                else:
                    mcc_vocal = MCCVocalDataset(
                        url2index=INDEX[region]["MCCVocal"],
                        sample_rate=sample_rate,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        max_num_crops=max_num_crops,
                        normalize_audio=normalize_audio,
                        tokenizer=self.tokenizer,
                        frame_rate=self.frame_rate,
                        resampled=True,
                        shardshuffle=True,
                        use_pipe=use_pipe,
                        handler=wds.warn_and_continue,
                    )
                datasets.append(
                    DataPipeline(mcc_vocal, wds.shuffle(shuffle_buffer_size))
                )
            if weights[1] > 0:
                bigtts = BigTTSDataset(
                    data_id=data_id,
                    audio_key=target_audio_key,
                    target_token_key=target_token_key,
                    sample_rate=sample_rate,
                    umm_token_freq=umm_token_freq,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    max_num_crops=max_num_crops,
                    normalize_audio=normalize_audio,
                    tokenizer=self.tokenizer,
                    phone2id=self.phone2id,
                    phone_tone_wordseg_dict=self.phone_tone_wordseg_dict,
                    frame_rate=self.frame_rate,
                    resampled=True,
                    shardshuffle=True,
                    # use_pipe=use_pipe,
                    handler=wds.warn_and_continue,
                )
                datasets.append(
                    DataPipeline(bigtts, wds.shuffle(shuffle_buffer_size))
                )
            if weights[2] > 0:
                mcc_instrumental = MCCInstrumentalDataset(
                    url2index=INDEX[region]["MCCInstrumental"],
                    sample_rate=sample_rate,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    max_num_crops=max_num_crops,
                    normalize_audio=normalize_audio,
                    tokenizer=self.tokenizer,
                    frame_rate=self.frame_rate,
                    resampled=True,
                    shardshuffle=True,
                    use_pipe=use_pipe,
                    handler=wds.warn_and_continue,
                )
                datasets.append(
                    DataPipeline(mcc_instrumental, wds.shuffle(shuffle_buffer_size))
                )
            weights = [i for i in weights if i != 0]
            self.train_dataset = DataPipeline(
                MultiIterableDataset(
                    datasets=datasets, weights=[i for i in weights if i != 0]
                ),
                self.bucketize,
            )

        # karaoke = WebPipeline(
        #     KaraokeDataset(
        #         sample_rate=sample_rate,
        #         min_duration=min_duration,
        #         max_duration=max_duration,
        #         normalize_audio=False,
        #         tokenizer=self.tokenizer,
        #         frame_rate=self.frame_rate,
        #         resampled=False,
        #         use_pipe=use_pipe,
        #         nodesplitter=return_self,
        #         handler=wds.warn_and_continue,
        #     ),
        #     pipeline=[{"compose": [self.bucketize]}],
        # )
        if skip_validation:
            self.validation_dataset = []
        else:
            libritts = WebPipeline(
                LibriTTSDataset(
                    sample_rate=sample_rate,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    tokenizer=self.tokenizer,
                    frame_rate=self.frame_rate,
                    resampled=False,
                    nodesplitter=return_self,
                    handler=wds.warn_and_continue,
                ),
                pipeline=[{"compose": [self.bucketize]}],
            )
            # self.validation_dataset = [karaoke, libritts]
            self.validation_dataset = [libritts]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class DouyinMusicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        languages: List[str] = ["en"],
        sample_rate: int = 24000,
        batch_size: int = 2,
        min_duration: int = 5,
        max_duration: int = 30,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 50,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_audio_text,
        weights: List[int] = None,
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = "wordpiece",
        frame_rate: int = 25,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )

        datasets = [
            DataPipeline(
                DouyinMusicDataset(
                    url2index=INDEX[region]["DouyinMusic"][language],
                    sample_rate=sample_rate,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    normalize_audio=normalize_audio,
                    tokenizer=self.tokenizer,
                    frame_rate=self.frame_rate,
                    resampled=True,
                    shardshuffle=True,
                    use_pipe=use_pipe,
                    handler=wds.warn_and_continue,
                ),
                wds.shuffle(shuffle_buffer_size),
            )
            for language in languages
        ]
        if weights is None:
            weights = [1 for _ in range(len(datasets))]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(datasets=datasets, weights=weights), self.bucketize
        )

        karaoke = WebPipeline(
            KaraokeDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=False,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [karaoke]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MixZhWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 30 * 24000 * 2,
        min_duration: int = 2,
        max_duration: int = 30,
        max_num_crops: int = 3,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: List[int] = [1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
    ):
        super().__init__()
        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        elif tokenizer == "tts_chinese_frontend_model":
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
            collate_fn = collate_audio_text
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )
        datasets = []
        if weights[0] > 0:
            vocal = VocalZhDataset(
                url2index=[
                    INDEX[region]["MCCVocal-Zh-A"],
                    INDEX[region]["MCCVocal-Zh-B"],
                    INDEX[region]["MCCVocal-Zh-C"],
                    INDEX[region]["HotGalaxy"],
                    INDEX[region]["Soda"],
                ],
                weights=[1, 4, 80, 50, 50],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(vocal, wds.shuffle(shuffle_buffer_size)))
        if weights[1] > 0:
            speech = SpeechZhDataset(
                url2index=[
                    INDEX[region]["FanqieShort"],
                    INDEX[region]["FanqieLong"],
                    INDEX[region]["XimalayaShort"],
                    INDEX[region]["XimalayaLong"],
                    INDEX[region]["XiaoyuzhouShort"],
                    INDEX[region]["XiaoyuzhouLong"],
                ],
                weights=[20, 20, 6, 3, 6, 3],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(speech, wds.shuffle(shuffle_buffer_size)))
        if weights[2] > 0:
            mcc_instrumental = MCCInstrumentalDataset(
                url2index=INDEX[region]["MCCInstrumental"],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(
                DataPipeline(mcc_instrumental, wds.shuffle(shuffle_buffer_size))
            )
        weights = [i for i in weights if i != 0]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[i for i in weights if i != 0]
            ),
            self.bucketize,
        )

        karaoke = WebPipeline(
            KaraokeDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=False,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        libritts = WebPipeline(
            LibriTTSDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [karaoke, libritts]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


class MixLangVocalWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int = 24000,
        batch_size: int = 30 * 24000 * 2,
        min_duration: int = 2,
        max_duration: int = 30,
        max_num_crops: int = 3,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = collate_fn,
        weights: List[int] = [1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = None,
        frame_rate: int = 25,
    ):
        super().__init__()
        if tokenizer == "wordpiece":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        elif tokenizer == "phoneme":
            self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            phonemizer.logger.get_logger().setLevel(logging.ERROR)
        else:
            self.tokenizer = None
            collate_fn = collate_audio_text
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.frame_rate = frame_rate
        assert batch_size >= min_duration * sample_rate
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        self.batcher = BucketBatcher(
            buckets=buckets_samples,
            dynamic_batch=True,
            maximum_bucket_size=batch_size,
            length_fn=lambda x: x["audio"].size(-1),
        )
        datasets = []
        if weights[0] > 0:
            zh = VocalZhDataset(
                url2index=[
                    INDEX[region]["MCCVocal-Zh-A"],
                    INDEX[region]["MCCVocal-Zh-B"],
                    INDEX[region]["MCCVocal-Zh-C"],
                    INDEX[region]["HotGalaxy"],
                    INDEX[region]["Soda"],
                ],
                weights=[1, 4, 80, 50, 50],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(zh, wds.shuffle(shuffle_buffer_size)))
        if weights[1] > 0:
            mcc_vocal = MCCVocalDataset(
                url2index=INDEX[region]["MCCVocal"],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                max_num_crops=max_num_crops,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(mcc_vocal, wds.shuffle(shuffle_buffer_size)))
        weights = [i for i in weights if i != 0]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[i for i in weights if i != 0]
            ),
            self.bucketize,
        )

        karaoke = WebPipeline(
            KaraokeDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=False,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        libritts = WebPipeline(
            LibriTTSDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
        self.validation_dataset = [karaoke, libritts]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for val in self.validation_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch


if __name__ == "__main__":
    umm_token_freq = 40
    max_duration = 30
    batch_size = 4
    dm = MixWebDataModule(
            weights=[0, 1, 0],
            data_id=372, 
            umm_token_freq=umm_token_freq,
            max_duration=max_duration,
            batch_size=max_duration*umm_token_freq*batch_size, 
            tokenizer="sami",
            dataset_type=int(DatasetType.UMM_TOKEN),
            )
    for item in dm.train_dataloader():
        print("test", item.keys())
        # print("test", item["target_ids"].shape, item["target_ids_length"])
        break

