import json
import math
import os
import random
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import BertTokenizer
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline

from recipes.datasets.base import DataResult, BaseAudioTransform
from recipes.datasets.mcc import INDEX
from recipes.datasets.mcc.mix import BaseTransforms
from recipes.mir.datasets import SPOTIFY_ARTIST_GENDER
from recipes.musiclm.transforms.audio import FastNormalizeAudio, LoudnessCheck
from samantha.dataio.batching import BucketBatcher
from samantha.dataio.dataset import MultiIterableDataset
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    Pad,
    SetAudioDimensions,
    ToTensor,
)
from samantha.utils.hdfs_helper import hdfs_ls
from samantha.utils.webdataset import return_self

TASKS = {
    "understanding": {
        0: ["What is the genre of this music"],
        1: ["What is the mood of this music"],
        2: ["What is the theme of this music"],
        3: ["Describe the music"],
        4: ["Transcribe the lyrics of this song"],
        5: ["Transcribe this speech to text"],
        6: ["What is the tempo of this music"],
        7: ["What are the chords of this music"],
        8: ["What are the key and mode of this music"],
        9: ["What is the structure of this music"],
        10: ["What is the tempo of this music"],
        11: ["What is the dialect of this music"],
        12: ["Tag this music"],
        13: ["What is the gender of the singer of this song"]
    },
    "generation": {
        0: ["Generate an instrumental piece of music with the keywords of"],
        1: ["Generate a piece of song with the keywords of"],
        2: ["Generate a piece of song with the lyrics of"],
        3: ["Generate a piece of song with the keywords of", "and the lyrics of"],
        4: ["Convert this text to speech"],
    },
    "editing": {
        1: ["Add vocal to this accompaniment"],
        2: ["Add accompaniment to this vocal"],
        3: ["Remove vocal from this song"],
        4: ["Remove accompaniment from this song"],
    },
}

TAG_NAMES = (
        "guitar",
        "classical",
        "slow",
        "techno",
        "strings",
        "drums",
        "electronic",
        "rock",
        "fast",
        "piano",
        "ambient",
        "beat",
        "violin",
        "vocal",
        "synth",
        "female",
        "indian",
        "opera",
        "male",
        "singing",
        "vocals",
        "no vocals",
        "harpsichord",
        "loud",
        "quiet",
        "flute",
        "woman",
        "male vocal",
        "no vocal",
        "pop",
        "soft",
        "sitar",
        "solo",
        "man",
        "classic",
        "choir",
        "voice",
        "new age",
        "dance",
        "male voice",
        "female vocal",
        "beats",
        "harp",
        "cello",
        "no voice",
        "weird",
        "country",
        "metal",
        "female voice",
        "choral",
    )

MODE_MAJMIN = {
    "major": "major",
    "dorian": "minor",
    "phrygian": "minor",
    "lydian": "major",
    "mixolydian": "major",
    "minor": "minor",
    "locrian": "minor",
}

def prepare_mir_urls(directories, root):
    urls = []
    for d in directories:
        urls.extend(hdfs_ls(os.path.join(root, d)))
    urls = [f"pipe: hdfs dfs -cat {url}" for url in urls]
    return urls


def collate_fn(batch: List[torch.Tensor]):
    mono_lengths = []
    homo_lengths = []
    for item in batch:
        if (item["input_audio"] is None) != (item["target_audio"] is None):
            mono_lengths.append(item["audio_length"])
        else:
            homo_lengths.append(item["input_audio"].shape[-1])
            homo_lengths.append(item["target_audio"].shape[-1])
    if len(mono_lengths) > 0:
        mono_random_pad = Pad(n_samples=max(mono_lengths))
    if len(homo_lengths) > 0:
        homo_random_pad = Pad(n_samples=max(homo_lengths))

    mono_text = []
    homo_text = []
    mono_text_length = []
    homo_text_length = []
    mono_text_idx = 0
    homo_text_idx = 0
    mono_audio = []
    homo_audio = []
    mono_audio_idx = 0
    homo_audio_idx = 0
    mono_map = []
    homo_map = []

    for item in batch:
        index = [item["task_type"], item["task_id"]]
        if (item["input_audio"] is None) != (item["target_audio"] is None):
            if item["input_text"] is not None:
                mono_text.append(item["input_text"])
                mono_text_length.append(item["input_text"].size(-1))
                index.append(mono_text_idx)
                mono_text_idx += 1
            else:
                index.append(None)
            if item["input_audio"] is not None:
                mono_audio.append(mono_random_pad(item["input_audio"]))
                index.append(mono_audio_idx)
                mono_audio_idx += 1
            else:
                index.append(None)
            if item["target_text"] is not None:
                mono_text.append(item["target_text"])
                mono_text_length.append(item["target_text"].size(-1))
                index.append(mono_text_idx)
                mono_text_idx += 1
            else:
                index.append(None)
            if item["target_audio"] is not None:
                mono_audio.append(mono_random_pad(item["target_audio"]))
                index.append(mono_audio_idx)
                mono_audio_idx += 1
            else:
                index.append(None)
            mono_map.append(index)
        else:
            padded_audio = homo_random_pad(
                torch.cat([item["input_audio"], item["target_audio"]], dim=0)
            )
            input_audio = padded_audio[: item["input_audio"].size(0)]
            target_audio = padded_audio[-item["target_audio"].size(0) :]
            if item["input_text"] is not None:
                homo_text.append(item["input_text"])
                homo_text_length.append(item["input_text"].size(-1))
                index.append(homo_text_idx)
                homo_text_idx += 1
            else:
                index.append(None)
            homo_audio.append(input_audio)
            index.append(homo_audio_idx)
            homo_audio_idx += 1
            if item["target_text"] is not None:
                homo_text.append(item["target_text"])
                homo_text_length.append(item["target_text"].size(-1))
                index.append(homo_text_idx)
                homo_text_idx += 1
            else:
                index.append(None)
            homo_audio.append(target_audio)
            index.append(homo_audio_idx)
            homo_audio_idx += 1
            homo_map.append(index)

    if len(mono_map) > 0:
        mono_text = pad_sequence(mono_text, batch_first=True, padding_value=0)
        mono_audio = torch.stack(mono_audio, dim=0)
    if len(homo_map) > 0:
        homo_text = pad_sequence(homo_text, batch_first=True, padding_value=0)
        homo_audio = torch.stack(homo_audio, dim=0)

    return {
        "mono_text": mono_text,
        "homo_text": homo_text,
        "mono_audio": mono_audio,
        "homo_audio": homo_audio,
        "mono_text_length": mono_text_length,
        "homo_text_length": homo_text_length,
        "mono_map": mono_map,
        "homo_map": homo_map,
    }


class WebDatasetBufferPreprocessor:
    def __init__(self, transforms: BaseTransforms):
        self.transforms = transforms

    def train_buffer_preprocessor(self, buffer: Generator) -> Generator:
        for item in buffer:
            yield from self.transforms(item)


class MIRTempoTransforms(BaseTransforms):
    name = "MIRTempoTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 6
        try:
            audio = self.base_transform(item["audio.npy"])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return

        beats = [time[0] for time in item["beats.pickle"]]
        if self.split in ["train", "validation"]:
            sampled_duration = random.randint(
                self.min_duration * self.sample_rate,
                min(self.max_duration * self.sample_rate, audio.size(-1)),
            )
            sampled_start = random.randint(0, audio.size(-1) - sampled_duration)
            sampled_end = sampled_start + sampled_duration

            num_beats = 0
            for beat in beats:
                if (
                    sampled_start / self.sample_rate
                    <= beat
                    <= sampled_end / self.sample_rate
                ):
                    num_beats += 1
            tempo = int(num_beats / (sampled_duration / self.sample_rate / 60))
            input_audio = audio[:, sampled_start:sampled_end]
        else:
            tempo = int(len(beats) / (audio.size(-1) / self.sample_rate / 60))
            input_audio = audio

        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = f"{tempo}"
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]

        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug,
        }
        yield output_dict
        self._update_stats(skipped=False)


class MIRTempoDataset(WebPipeline):
    name = "MIRTempo"
    _root = "hdfs://haruna/home/byte_speech_sv/data/music/mir_benchmark/beat"
    _directories = {
        "train": [
            "ballroom_beat_24kHz/train",
            "beatles_beat_24kHz/train",
            "hainsworth_beat_24kHz/train",
            "hjdb_beat_24kHz/train",
            "karaoke_beat_24kHz/train",
            "rwc_beat_24kHz/train",
            "simac_beat_24kHz/train",
            "smc_beat_24kHz/train",
            "harmonix_beat_24kHz/train",
        ],
        "validation": [
            "gtzan_beat_24kHz/validation",
            "clip500_beat_24kHz/validation",
            "bytebeat_beat_24kHz/validation",
        ],
        "test": [
            "gtzan_beat_24kHz/test",
            "clip500_beat_24kHz/test",
            "bytebeat_beat_24kHz/test",
        ],
    }

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = MIRTempoTransforms(
            split=split,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        urls = prepare_mir_urls(self._directories[split], self._root)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MIRTagTransforms(BaseTransforms):
    name = "MIRTagTransforms"
    data_sample_rate = 24000
    _tag_names = TAG_NAMES

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 12
        try:
            audio = self.base_transform(item["audio.npy"])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return

        if self.split in ["train", "validation"]:
            sampled_duration = random.randint(
                self.min_duration * self.sample_rate,
                min(self.max_duration * self.sample_rate, audio.size(-1)),
            )
            sampled_start = random.randint(0, audio.size(-1) - sampled_duration)
            sampled_end = sampled_start + sampled_duration
            input_audio = audio[:, sampled_start:sampled_end]
        else:
            input_audio = audio
        tags = item["tag_binary.npy"].astype(bool).tolist()
        tags = [self._tag_names[i] for i, tag in enumerate(tags) if tag]
        random.shuffle(tags)
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = ", ".join(tags)
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]

        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug,
        }
        yield output_dict
        self._update_stats(skipped=False)


class MIRTagDataset(WebPipeline):
    name = "MIRTag"
    _root = "hdfs://haruna/home/byte_speech_sv/data/music/mir_benchmark/tagging/mtat_tagging_24kHz"
    _directories = {"train": ["train"], "validation": ["validation"], "test": ["test"]}

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = MIRTagTransforms(
            split=split,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        urls = prepare_mir_urls(self._directories[split], self._root)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MIRStructureTransforms(BaseTransforms):
    name = "MIRStructureTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

        self.seg_map = {
            "silence": "silence",
            "end": "end",
            "build": "verse",
            "fadein": "intro",
            "opening": "intro",
            "stutter": "chorus",
            "slow": "verse",
            "drumroll": "inst",
            "synth": "inst",
            "closing": "outro",
            "interlude": "inst",
            "mantra": "verse",
            "fade-out": "outro",
            "out": "outro",
            "guitar": "inst",
            "head": "inst",
            "loop": "inst",
        }

        self.substr_map = {
            "other": "other",
            "pre-chorus-and-chorus": "chorus",
            "verse-and-chorus": "chorus",
            "intro": "intro",
            "verse": "verse",
            "prechorus": "verse",
            "refrain": "chorus",
            "pre-chorus": "verse",
            "chorus": "chorus",
            "bridge": "bridge",
            "outro": "outro",
            "fadeout": "outro",
            "ending": "outro",
            "fadein": "intro",
            "inst": "inst",
            "solo": "inst",
            "break": "inst",
            "trans": "bridge",
            "gtr": "inst",
            "section": "verse",
            "riff": "inst",
            "rap": "verse",
            "coda": "outro",
            "interlude": "inst",
            "lead-in": "inst",
            "theme": "chorus",
            "development": "verse",
            "variation": "bridge",
            "impro": "inst",
            "guitar": "inst",
            "spoken": "inst",
            "trumpet": "inst",
            "applause": "inst",
            "voice": "inst",
            "stage": "inst",
            "banjo": "inst",
            "crowd": "inst",
            "pause": "inst",
            "tag": "inst",
        }

    def convert_structure_label(self, label):
        label = label.lower()
        for s in self.substr_map:
            if s in label:
                return self.substr_map[s]
        if label in self.seg_map:
            return self.seg_map[label]
        else:
            return "unknown"

    def select_segments(self, segments, intervals):
        selected_segments = []
        num = len(intervals)
        sampled_start_i = random.randint(0, num - 1)
        total_duration = 0
        for i in range(sampled_start_i, num):
            total_duration += intervals[i][1] - intervals[i][0]
            if total_duration > self.max_duration:
                break
            else:
                selected_segments.append((segments[i], intervals[i]))
        return selected_segments

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 9
        segments = item.get("labels.pickle", None)
        intervals = item.get("intervals.pickle", None)
        if (segments is None) or (intervals is None):
            self._update_stats(skipped=True, message="No segments or intervals")
            return
        if (len(segments) < 2) or (len(intervals) < 2):
            self._update_stats(skipped=True, message="At least two segments")
            return
        if len(segments) != len(intervals):
            self._update_stats(
                skipped=True, message="Segments and intervals are not matched"
            )
            return
        segments = [self.convert_structure_label(label) for label in segments]
        if self.split in ["train", "validation"]:
            selected_segments = self.select_segments(segments, intervals)
        else:
            selected_segments = list(zip(segments, intervals))
        if len(selected_segments) < 2:
            self._update_stats(skipped=True, message="Not enough selected segments")
            return
        start = int(selected_segments[0][1][0] * self.sample_rate)
        end = int(selected_segments[-1][1][1] * self.sample_rate)
        if (end - start) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return

        try:
            audio = self.base_transform(item["audio.npy"])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        input_audio = audio[:, start:end]
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = ", ".join([seg[0] for seg in selected_segments])
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]

        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug,
        }
        yield output_dict
        self._update_stats(skipped=False)


class MIRStructureDataset(WebPipeline):
    name = "MIRStructure"
    _root = "hdfs://haruna/home/byte_speech_sv/data/music/mir_benchmark/structure"
    _directories = {
        "train": ["billboard_structure_24kHz/train"],
        "validation": ["harmonix_structure_24kHz/validation"],
        "test": ["pop909_structure_24kHz/test"],
    }

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = MIRStructureTransforms(
            split=split,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        urls = prepare_mir_urls(self._directories[split], self._root)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MIRKeyTransforms(BaseTransforms):
    name = "MIRKeyTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        self.mode2name = {
            1: "Major",
            2: "Dorian",
            3: "Phrygian",
            4: "Lydian",
            5: "Mixolydian",
            6: "Minor",
            7: "Locrian",
        }
        self.key2name = {
            1: "C",
            2: "C#",
            3: "D",
            4: "D#",
            5: "E",
            6: "F",
            7: "F#",
            8: "G",
            9: "G#",
            10: "A",
            11: "A#",
            12: "B",
        }
        print(f"[{self.name}] initialized.")

    def filter_intervals(self, intervals):
        start_time = float(intervals[1][0])
        end_time = float(intervals[1][1])
        delta = end_time - start_time
        if self.min_duration > delta:
            return False
        if self.max_duration < delta:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 8
        keys = item.get("keys.pickle", None)
        intervals = item.get("intervals.pickle", None)
        if (
            (keys is None)
            or (intervals is None)
            or (len(keys) == 0)
            or (len(intervals) == 0)
        ):
            self._update_stats(skipped=True, message="No keys or Intervals")
            return
        if len(keys) != len(intervals):
            self._update_stats(
                skipped=True, message="Keys and Intervals are not matched"
            )
            return
        try:
            audio = self.base_transform(item["audio.npy"])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        for key, interval in zip(keys, intervals):
            selected_key = (key, interval)
            if self.min_duration > float(selected_key[1][1]) - float(
                selected_key[1][0]
            ) and self.split in ["train", "validation"]:
                self._update_stats(skipped=True, message="Too short")
                return
            duration_in_seconds = float(selected_key[1][1]) - float(selected_key[1][0])
            duration = int(duration_in_seconds * self.sample_rate)
            if self.max_duration < duration_in_seconds and self.split in [
                "train",
                "validation",
            ]:
                sampled_duration = random.randint(
                    self.min_duration * self.sample_rate,
                    min(duration, self.max_duration * self.sample_rate),
                )
                sampled_start = random.randint(
                    int(float(selected_key[1][0]) * self.sample_rate),
                    int(float(selected_key[1][0]) * self.sample_rate)
                    + (duration - sampled_duration),
                )
                clip = audio[:, sampled_start : sampled_start + sampled_duration]
            else:
                clip = audio[
                    :,
                    int(float(selected_key[1][0]) * self.sample_rate) : int(
                        float(selected_key[1][1]) * self.sample_rate
                    ),
                ]
            key_name = self.key2name.get(selected_key[0][0], None)
            mode_name = self.mode2name.get(selected_key[0][1], None)
            if (key_name is None) or (mode_name is None):
                self._update_stats(skipped=True, message="Invalid key or mode")
                return

            input_audio = clip
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].squeeze(dim=0)
            target_raw_text = f"{key_name} {mode_name}"
            target_text = self.tokenizer(
                target_raw_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]

            total_length = (
                math.ceil(audio_length / self.sample_rate * self.frame_rate)
                + (0 if input_text is None else input_text.shape[-1])
                + (0 if target_text is None else target_text.shape[-1])
            )
            if total_length > self.max_length and self.split in ["train", "validation"]:
                self._update_stats(skipped=True, message="Total length too large")
                return
            debug = (
                f"input_raw_text: {input_raw_text} | "
                + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
                + f"target_raw_text: {target_raw_text} | "
                + f"target_audio: {0 if target_audio is None else target_audio.size()}"
            )

            output_dict = {
                "input_text": input_text,
                "target_text": target_text,
                "input_audio": input_audio,
                "target_audio": target_audio,
                "audio_length": audio_length,
                "task_type": task_type,
                "task_id": task_id,
                "debug": debug,
            }
            yield output_dict
            self._update_stats(skipped=False)


class MIRKeyDataset(WebPipeline):
    name = "MIRKey"
    _root = "hdfs://haruna/home/byte_speech_sv/data/music/mir_benchmark/key"
    _directories = {
        "train": [
            "billboard_key_24kHz/train",
            "isophonic_key_24kHz/train",
            "leadsheet_key_24kHz/train",
            "karaoke_key_24kHz/train",
        ],
        "validation": ["leadsheet_key_24kHz/validation"],
        "test": ["giantstep_key_24kHz/test"],
    }

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = MIRKeyTransforms(
            split=split,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        urls = prepare_mir_urls(self._directories[split], self._root)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class GenderTransforms(BaseTransforms):
    name = "GenderTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 30,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def _get_gender(self, item):
        gender = SPOTIFY_ARTIST_GENDER[item["__index_data__"]["metadata"]["spotify_primary_artist_name"]]
        return gender

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 13
        gender = self._get_gender(item)
        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return
        if audio.size(-1) > self.min_duration * self.sample_rate and self.split in [
            "train",
            "validation",
        ]:
            duration = random.randint(
                self.min_duration * self.sample_rate,
                min(audio.size(-1), self.max_duration * self.sample_rate),
            )
            start = random.randint(0, audio.size(-1) - duration)
            audio = audio[:, start : start + duration]

        input_audio = audio
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = gender
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]

        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug,
        }
        yield output_dict
        self._update_stats(skipped=False)


class GenderDataset(WebPipeline):
    name = "Gender"

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = GenderTransforms(
            split=split,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        if split == "train":
            male = IndexedWebDataset(url2index="hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200_v2/24000hz/train/filtered/artist_gender/male/url2index.txt", **kwargs)
            female = IndexedWebDataset(url2index="hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200_v2/24000hz/train/filtered/artist_gender/female/url2index.txt", **kwargs)
        else:
            # Too many
            # male = IndexedWebDataset(url2index="hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/male/url2index.txt", **kwargs)
            # female = IndexedWebDataset(url2index="hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/female/url2index.txt", **kwargs)
            male = IndexedWebDataset(
                url2index={
                    "hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/male/0-00000.tar": "hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/male/0-00000.tar.index"
                },
                **kwargs,
            )
            female = IndexedWebDataset(
                url2index={
                    "hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/female/1-00000.tar": "hdfs://haruna/home/byte_speech_sv/data/music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/female/1-00000.tar.index",
                },
                **kwargs
            )
        dataset = MultiIterableDataset([male, female], weights=[7, 3])
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")

from dataclasses import dataclass

@dataclass
class DialectDataResult(DataResult):
    input_audio: torch.Tensor
    target_audio: torch.Tensor
    audio_length: int
    task_type: str
    target_raw_text: str
    task_id: int
    debug: str
    input_text: Optional[str] = None
    target_text: Optional[str] = None


class DialectTransforms(BaseTransforms):
    name = "DialectTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 30,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def _get_dialect(self, item):
        name = item["__dataset_name__"]
        if name.startswith("music_Sspotify_Mvocal_Lmand_Tdialect"):
            return "mandarin"
        elif name.startswith("music_Sspotify_Mvocal_Lcant_Tdialect"):
            return "cantonese"
        elif name.startswith("music_Sspotify_Mvocal_Lhokk_Tdialect"):
            return "hokkien"
        else:
            return None

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 11
        dialect = self._get_dialect(item)
        if dialect is None:
            self._update_stats(skipped=True, message=f"Unknown dialect")
            return
        try:
            audio = self.base_transform(item["wav"])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return
        if audio.size(-1) > self.min_duration * self.sample_rate and self.split in [
            "train",
            "validation",
        ]:
            duration = random.randint(
                self.min_duration * self.sample_rate,
                min(audio.size(-1), self.max_duration * self.sample_rate),
            )
            start = random.randint(0, audio.size(-1) - duration)
            audio = audio[:, start : start + duration]

        input_audio = audio
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = dialect
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]

        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        shard = item["__data_url__"]
        key = item["__key__"]
        yield DialectDataResult(
            input_text=input_text,
            target_text=target_text,
            input_audio=input_audio,
            target_audio=target_audio,
            audio_length=audio_length,
            task_type=task_type,
            task_id=task_id,
            target_raw_text=target_raw_text,
            debug=debug,
            shard=shard,
            key=key
        )
        self._update_stats(skipped=False)


class DialectDataset(WebPipeline):
    name = "Dialect"
    data_sample_rate = 24000
    _data_ids: Dict[str, int] = {
        "train": 359,
        "validation": 406,
        "test": 406,
    }

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        self.data_id = self._data_ids[split]

        print(f"[{self.name}] initializing...")
        transforms = DialectTransforms(
            split=split,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = ParquetDataset(data_id=self.data_id, **kwargs)
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")



@dataclass
class ChineseGenreDataResult(DataResult):
    input_text: str
    target_text: str
    input_audio: torch.Tensor
    target_audio: torch.Tensor
    audio_length: int
    task_type: str
    target_raw_text: str
    task_id: int
    debug: str


class ChineseGenreTransforms(BaseTransforms):
    name = "ChineseGenreTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 30,
        max_duration: float = 120,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        self.base_transform = BaseAudioTransform()
        print(f"[{self.name}] initialized.")

    def _get_dialect(self, item):
        name = item["__dataset_name__"]
        if name.startswith("music_Sspotify_Mvocal_Lmand_Tdialect"):
            return "mandarin"
        elif name.startswith("music_Sspotify_Mvocal_Lcant_Tdialect"):
            return "cantonese"
        elif name.startswith("music_Sspotify_Mvocal_Lhokk_Tdialect"):
            return "hokkien"
        else:
            return None

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 0 # genre (TODO switch with mood/scene)
        metadata = json.loads(item["meta"])

        genre_raw_text = metadata["bigmusic_genre_en"]
        mood_raw_text = metadata["bigmusic_mood_en"]
        scene_raw_text = metadata["bigmusic_scene_en"]

        if genre_raw_text is None:
            self._update_stats(skipped=True, message=f"Unknown genre")
            return
        try:
            audio = self.base_transform(item["wav"])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return
        if audio.size(-1) > self.min_duration * self.sample_rate and self.split in [
            "train",
            "validation",
        ]:
            duration = random.randint(
                self.min_duration * self.sample_rate,
                min(audio.size(-1), self.max_duration * self.sample_rate),
            )
            start = random.randint(0, audio.size(-1) - duration)
            audio = audio[:, start : start + duration]

        input_audio = audio
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = genre_raw_text
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]

        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        shard = item["__data_url__"]
        key = item["__key__"]
        yield DialectDataResult(
            # input_text=input_text,
            # target_text=target_text,
            input_audio=input_audio,
            target_audio=target_audio,
            audio_length=audio_length,
            task_type=task_type,
            task_id=task_id,
            target_raw_text=target_raw_text,
            debug=debug,
            shard=shard,
            key=key
        )
        self._update_stats(skipped=False)


class ChineseGenreDataset(WebPipeline):
    name = "ChineseGenre"
    data_sample_rate = 24000

    def __init__(
        self,
        data_id: int = 421,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = ChineseGenreTransforms(
            split=split,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = ParquetDataset(data_id=data_id, **kwargs)
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class CaptionTransforms(BaseTransforms):
    name = "CaptionTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def _get_text(self, item):
        meta = json.loads(item["meta"])
        return meta["text"].replace("\n", ". ")

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 3
        try:
            audio = self.base_transform(item["wav"])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return
        if audio.size(-1) > self.min_duration * self.sample_rate and self.split in [
            "train",
            "validation",
        ]:
            duration = random.randint(
                self.min_duration * self.sample_rate,
                min(audio.size(-1), self.max_duration * self.sample_rate),
            )
            start = random.randint(0, audio.size(-1) - duration)
            audio = audio[:, start : start + duration]
        try:
            text = self._get_text(item)
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading text: {e}")
            return

        input_audio = audio
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = text
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]

        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug,
        }
        yield output_dict
        self._update_stats(skipped=False)


class CaptionDataset(WebPipeline):
    name = "Caption"
    data_sample_rate = 24000

    def __init__(
        self,
        data_id: int = 400,
        split: str = "train",
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = CaptionTransforms(
            split=split,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = ParquetDataset(data_id=data_id, **kwargs)
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MusiccapsTransforms(BaseTransforms):
    name = "MusiccapsTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "test",
        sample_rate: int = 24000,
        min_duration: float = 1,
        max_duration: float = 10,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 3
        try:
            audio = self.base_transform(item["wav"])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return
        if audio.size(-1) > self.min_duration * self.sample_rate and self.split in [
            "train",
            "validation",
        ]:
            duration = random.randint(
                self.min_duration * self.sample_rate,
                min(audio.size(-1), self.max_duration * self.sample_rate),
            )
            start = random.randint(0, audio.size(-1) - duration)
            audio = audio[:, start : start + duration]
        text = item["text"]

        input_audio = audio
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = text
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]

        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug,
        }
        yield output_dict
        self._update_stats(skipped=False)


class MusiccapsDataset(WebPipeline):
    name = "Musiccaps"
    data_sample_rate = 24000

    def __init__(
        self,
        data_id: int = 428,
        split: str = "test",
        sample_rate: int = 24000,
        min_duration: float = 1,
        max_duration: float = 10,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = MusiccapsTransforms(
            split=split,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = ParquetDataset(data_id=data_id, **kwargs)
        pipeline = [{"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class MCCTransforms(BaseTransforms):
    name = "MCCTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 10,
        max_duration: float = 120,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
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
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
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
        if metadata.get("final_genre") is None:
            return False, "Genre none"
        elif metadata["final_genre"].strip() == "":
            return False, "Genre empty"
        if metadata.get("meta_song_author_name") is None:
            return False, "Author none"
        elif metadata["meta_song_author_name"].strip() == "":
            return False, "Author empty"
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
        min_duration: float = 10,
        max_duration: float = 120,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
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
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )

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
        task_type = "understanding"
        task_id = 4
        metadata = item["__index_data__"]
        is_good, message = self.is_metadata_good(metadata)
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return

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
        if len(filtered_utterances) == 0:
            self._update_stats(skipped=True, message="No utterances after filtering")
            return

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if audio.size(-1) < self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return

        random.shuffle(filtered_utterances)
        selected_utterance = filtered_utterances[0]
        start = int(float(selected_utterance["start_time"]) / 1000 * self.sample_rate)
        end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
        if end > audio.size(-1):
            self._update_stats(skipped=True, message="Lyrics timestamp out of range")
            return
        input_audio = audio[:, start:end]
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = str(selected_utterance["text"])
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]
        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )
        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug,
        }
        yield output_dict
        self._update_stats(skipped=False)


class MCCBaseDataset(WebPipeline):
    name = "MCCBaseDataset"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: Union[List[str], str] = None,
        weights: List[float] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 10,
        max_duration: float = 120,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
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
                tokenizer=tokenizer,
                frame_rate=frame_rate,
                max_length=max_length,
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

    def __init__(
        self,
        url2index: Union[List[str], str],
        weights: List[int] = None,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 10,
        max_duration: float = 120,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        lyrics_confidence: float = 0.8,
        normalize_audio: bool = True,
        aed_filtered: bool = False,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
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
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
            **kwargs,
        )


class LibriLightASRTransforms(BaseTransforms):
    name = "LibriLightASRTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
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
        task_type = "understanding"
        task_id = 5
        utterances = item.get("__index_data__", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return
        filtered_utterances = list(filter(self.filter_lyrics, utterances))
        if len(filtered_utterances) == 0:
            self._update_stats(skipped=True, message="No utterances after filtering")
            return

        try:
            audio = self.base_transform(item[self.audio_key])
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return

        random.shuffle(filtered_utterances)
        selected_utterance = filtered_utterances[0]
        start = int(float(selected_utterance["start_time"]) * self.sample_rate)
        end = int(float(selected_utterance["end_time"]) * self.sample_rate)
        if end > audio.size(-1):
            self._update_stats(skipped=True, message="Lyrics timestamp out of range")
            return

        input_audio = audio[:, start:end]
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = str(selected_utterance["text"])
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]
        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return
        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug,
        }
        yield output_dict
        self._update_stats(skipped=False)


class LibriLightASRDataset(WebPipeline):
    name = "LibriLightASR"

    def __init__(
        self,
        url2index: str = "hdfs:///home/byte_speech_sv/data/speech/librilight_asr_npy/url2idx.txt",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = LibriLightASRTransforms(
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class KaraokeTransforms(BaseTransforms):
    name = "KaraokeTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        audio_key: str = "full.npy",
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
        base_transforms = []
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)
        print(f"[{self.name}] initialized.")

    def filter_lyrics(self, utterance):
        start_time = float(utterance["start_time"]) / 1000
        end_time = float(utterance["end_time"]) / 1000
        delta = end_time - start_time
        if self.min_duration > delta and self.split in ["train", "validation"]:
            return False
        if self.max_duration < delta and self.split in ["train", "validation"]:
            return False
        if len(str(utterance["text"]).strip()) == 0:
            return False
        return True

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 4
        lyrics = item["__index_data__"].get("lyrics", None)
        if lyrics is None:
            self._update_stats(skipped=True, message="No lyrics")
            return
        utterances = lyrics.get("utterances", None)
        if utterances is None or len(utterances) == 0:
            self._update_stats(skipped=True, message="No utterances")
            return
        filtered_utterances = list(filter(self.filter_lyrics, utterances))
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
            input_audio = audio[:, start:end]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].squeeze(dim=0)
            target_raw_text = str(selected_utterance["text"])
            target_text = self.tokenizer(
                target_raw_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
            total_length = (
                math.ceil(audio_length / self.sample_rate * self.frame_rate)
                + (0 if input_text is None else input_text.shape[-1])
                + (0 if target_text is None else target_text.shape[-1])
            )
            if total_length > self.max_length and self.split in ["train", "validation"]:
                self._update_stats(skipped=True, message="Total length too large")
                return
            debug = (
                f"input_raw_text: {input_raw_text} | "
                + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
                + f"target_raw_text: {target_raw_text} | "
                + f"target_audio: {0 if target_audio is None else target_audio.size()}"
            )
            output_dict = {
                "input_text": input_text,
                "target_text": target_text,
                "input_audio": input_audio,
                "target_audio": target_audio,
                "audio_length": audio_length,
                "task_type": task_type,
                "task_id": task_id,
                "debug": debug,
            }
            yield output_dict
            self._update_stats(skipped=False)


class KaraokeDataset(WebPipeline):
    name = "Karaoke"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        audio_key: str = "full.npy",
        min_duration: int = 10,
        max_duration: int = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = KaraokeTransforms(
            split=split,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        if split == "train":
            url2index = "hdfs:///home/byte_speech_sv/data/karaoke_for_singsong_npy/url2idx.txt"
        else:
            url2index = "hdfs:///home/byte_speech_sv/data/karaoke_for_singsong_npy/url2idx_test.txt"
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = IndexedWebDataset(url2index=url2index, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class LibriTTSTransforms(BaseTransforms):
    name = "LibriTTSTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "test",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
    ):
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.frame_rate = frame_rate
        self.max_length = max_length
        self.tokenizer = tokenizer
        base_transforms = [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if self.data_sample_rate != sample_rate:
            base_transforms.append(Resample(self.data_sample_rate, sample_rate))
        if normalize_audio:
            base_transforms.append(FastNormalizeAudio())
        self.base_transform = Compose(base_transforms)

    def __call__(self, item: Dict[str, Any]) -> Generator:
        task_type = "understanding"
        task_id = 5
        audio = self.base_transform(item[self.audio_key])
        if audio.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Audio too short")
            return
        if audio.size(-1) > self.max_duration * self.sample_rate and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Audio too long")
            return
        input_audio = audio
        target_audio = None
        input_raw_text = TASKS[task_type][task_id][0]
        input_text = self.tokenizer(
            input_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        target_raw_text = item["normalized_text.txt"]
        target_text = self.tokenizer(
            target_raw_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].squeeze(dim=0)
        audio_length = input_audio.shape[-1]
        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length and self.split in ["train", "validation"]:
            self._update_stats(skipped=True, message="Total length too large")
            return

        debug = (
            f"input_raw_text: {input_raw_text} | "
            + f"input_audio: {0 if input_audio is None else input_audio.size()} | "
            + f"target_raw_text: {target_raw_text} | "
            + f"target_audio: {0 if target_audio is None else target_audio.size()}"
        )

        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug,
        }
        yield output_dict
        self._update_stats(skipped=False)


class LibriTTSDataset(WebPipeline):
    name = "LibriTTS"
    data_sample_rate = 24000

    def __init__(
        self,
        split: str = "validation",
        sample_rate: int = 24000,
        audio_key: str = "audio.npy",
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 3072,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = LibriTTSTransforms(
            split=split,
            sample_rate=sample_rate,
            audio_key=audio_key,
            min_duration=min_duration,
            max_duration=max_duration,
            normalize_audio=normalize_audio,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        if split == "train":
            urls = "pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/train-clean-360/{00000..00007}.tar"
        else:
            urls = "pipe: hdfs dfs -cat hdfs:///home/byte_speech_sv/data/speech/libritts/24000hz/test-clean/00000.tar"
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int,
        batch_size: int,
        min_duration: float = 5,
        max_duration: float = 120,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Callable = collate_fn,
        # tempo, tag, structure, key, gender, dialect, caption, asr-music, asr-speech
        weights: List[int] = [1, 1, 1, 1, 1, 1, 1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = "bert-base-uncased",
        frame_rate: int = 25,
        max_length: int = 3072,
        dynamic_batch: bool = True,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.frame_rate = frame_rate
        self.max_length = max_length
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        if dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=batch_size,
                length_fn=lambda x: x["audio_length"],
            )
        else:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio_length"],
            )
        datasets = []
        datasets_val = []
        if weights[0] > 0:
            tempo = MIRTempoDataset(
                split="train",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(tempo, wds.shuffle(shuffle_buffer_size)))
            tempo_val = MIRTempoDataset(
                split="validation",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets_val.append(tempo_val)
        if weights[1] > 0:
            tag = MIRTagDataset(
                split="train",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(tag, wds.shuffle(shuffle_buffer_size)))
            tag_val = MIRTagDataset(
                split="validation",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets_val.append(tag_val)
        if weights[2] > 0:
            structure = MIRStructureDataset(
                split="train",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(structure, wds.shuffle(shuffle_buffer_size)))
            structure_val = MIRStructureDataset(
                split="validation",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets_val.append(structure_val)
        if weights[3] > 0:
            key = MIRKeyDataset(
                split="train",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(key, wds.shuffle(shuffle_buffer_size)))
            key_val = MIRKeyDataset(
                split="validation",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets_val.append(key_val)
        if weights[4] > 0:
            gender = GenderDataset(
                split="train",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(gender, wds.shuffle(shuffle_buffer_size)))
            gender_val = GenderDataset(
                split="validation",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets_val.append(gender_val)
        if weights[5] > 0:
            dialect = DialectDataset(
                split="train",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(dialect, wds.shuffle(shuffle_buffer_size)))
            dialect_val = DialectDataset(
                split="validation",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets_val.append(dialect_val)
        if weights[6] > 0:
            caption = CaptionDataset(
                split="train",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=True,
                shardshuffle=True,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(caption, wds.shuffle(shuffle_buffer_size)))
            caption_val = CaptionDataset(
                split="validation",
                data_id=407,
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets_val.append(caption_val)
        if weights[7] > 0:
            mcc_vocal = MCCVocalDataset(
                url2index=INDEX[region]["MCCVocal"],
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(mcc_vocal, wds.shuffle(shuffle_buffer_size)))
            karaoke_val = KaraokeDataset(
                split="validation",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets_val.append(karaoke_val)
        if weights[8] > 0:
            librilight = LibriLightASRDataset(
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=True,
                shardshuffle=True,
                use_pipe=use_pipe,
                handler=wds.warn_and_continue,
            )
            datasets.append(DataPipeline(librilight, wds.shuffle(shuffle_buffer_size)))
            libritts_val = LibriTTSDataset(
                split="validation",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets_val.append(libritts_val)
        weights = [i for i in weights if i != 0]
        if sum(weights) > 0:
            self.train_dataset = DataPipeline(
                MultiIterableDataset(
                    datasets=datasets, weights=[i for i in weights if i != 0]
                ),
                self.bucketize,
            )

        self.validation_dataset = [WebPipeline(val, pipeline=[{"compose": [self.bucketize]}]) for val in datasets_val]

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


class TestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int,
        batch_size: int,
        min_duration: float = 5,
        max_duration: float = 120,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Callable = collate_fn,
        # tempo, tag, structure, key, gender, dialect, caption, asr-music, asr-speech
        weights: List[int] = [1, 1, 1, 1, 1, 1, 1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = "bert-base-uncased",
        frame_rate: int = 25,
        max_length: int = 3072,
        dynamic_batch: bool = True,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.frame_rate = frame_rate
        self.max_length = max_length
        buckets_samples = []
        sec = min_duration
        while sec <= max_duration:
            buckets_samples.append(sec)
            sec += math.ceil(sec * 0.1)
        if buckets_samples[-1] < max_duration:
            buckets_samples.append(max_duration)
        print(f"[Buckets] {len(buckets_samples)} {str(buckets_samples)}")
        buckets_samples = [x * sample_rate for x in buckets_samples]
        if dynamic_batch:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=True,
                maximum_bucket_size=batch_size,
                length_fn=lambda x: x["audio_length"],
            )
        else:
            self.batcher = BucketBatcher(
                buckets=buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio_length"],
            )
        datasets = []
        if weights[0] > 0:
            tempo = MIRTempoDataset(
                split="test",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets.append(tempo)
        if weights[1] > 0:
            tag = MIRTagDataset(
                split="test",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets.append(tag)
        if weights[2] > 0:
            structure = MIRStructureDataset(
                split="test",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets.append(structure)
        if weights[3] > 0:
            key = MIRKeyDataset(
                split="test",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets.append(key)
        if weights[4] > 0:
            gender = GenderDataset(
                split="test",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets.append(gender)
        if weights[5] > 0:
            dialect = DialectDataset(
                split="test",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets.append(dialect)
        if weights[6] > 0:
            caption = MusiccapsDataset(
                split="test",
                data_id=428,
                sample_rate=sample_rate,
                min_duration=1,
                max_duration=10,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets.append(caption)
        if weights[7] > 0:
            karaoke = KaraokeDataset(
                split="test",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                use_pipe=use_pipe,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets.append(karaoke)
        if weights[8] > 0:
            libritts = LibriTTSDataset(
                split="test",
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                normalize_audio=normalize_audio,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                shardshuffle=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            )
            datasets.append(libritts)
        self.test_dataset = [WebPipeline(d, pipeline=[{"compose": [self.bucketize]}]) for d in datasets]

    def test_dataloader(self):
        return [
            DataLoader(
                test,
                batch_size=None,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            for test in self.test_dataset
        ]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                yield batch
        for batch in self.batcher.collect_last_batch():
            if len(batch) > 0:
                yield batch
