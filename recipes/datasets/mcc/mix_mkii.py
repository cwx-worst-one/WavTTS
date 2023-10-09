import math
import random
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Tuple,
    Union,
)

import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchaudio_augmentations import Compose
from transformers import BertTokenizer
from webdataset import WebDataset
from webdataset.pipeline import DataPipeline

from recipes.datasets.mcc import INDEX
from recipes.musiclm.transforms.audio import FastNormalizeAudio, LoudnessCheck
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
from recipes.datasets.mcc.mix import BaseTransforms
from torch.nn.utils.rnn import pad_sequence


TASKS = {
    "understanding": [
        ["What is the genre of this music"],
        ["What is the mood of this music"],
        ["What is the theme of this music"],
        ["Describe the music"],
        ["Transcribe the lyrics of this song"],
        ["Transcribe this speech to text"],
    ],
    "generation": [
        ["Generate an instrumental piece of music with the keywords of"],
        ["Generate a piece of song with the keywords of"],
        ["Generate a piece of song with the lyrics of"],
        ["Generate a piece of song with the keywords of", "and the lyrics of"],
        ["Convert this text to speech"],
    ],
    "editing": [
        ["Add vocal to this accompaniment"],
        ["Add accompaniment to this vocal"],
        ["Remove vocal from this song"],
        ["Remove accompaniment from this song"],
    ]
}


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
        mono_random_pad = RandomPad(n_samples=max(mono_lengths))
    if len(homo_lengths) > 0:
        homo_random_pad = RandomPad(n_samples=max(homo_lengths))

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
            padded_audio = homo_random_pad(torch.cat([item["input_audio"], item["target_audio"]], dim=0))
            input_audio = padded_audio[:item["input_audio"].size(0)]
            target_audio = padded_audio[-item["target_audio"].size(0):]
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
        max_length: int = 2048,
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
        # if metadata.get("final_theme") is None:
        #     return False, "Theme none"
        # elif metadata["final_theme"].strip() == "":
        #     return False, "Theme empty"
        # if metadata.get("final_mood") is None:
        #     return False, "Mood none"
        # elif metadata["final_mood"].strip() == "":
        #     return False, "Mood empty"
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
        max_length: int = 2048,
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

        selected_task = torch.multinomial(torch.FloatTensor([1, 0, 0, 1, 1, 1, 1, 1]), 1).item()
        if selected_task == 0:
            task_type = "understanding"
            task_id = 0
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = audio[:, start : start + duration]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = str(metadata["final_genre"])
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 1:
            task_type = "understanding"
            task_id = 1
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = audio[:, start : start + duration]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = str(metadata["final_mood"])
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 2:
            task_type = "understanding"
            task_id = 2
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = audio[:, start : start + duration]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = str(metadata["final_theme"])
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 3:
            task_type = "understanding"
            task_id = 3
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = audio[:, start : start + duration]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_list = [
                str(metadata["final_genre"]),
                # str(metadata["final_mood"]),
                # str(metadata["final_theme"]),
                str(metadata["meta_song_author_name"]),
            ]
            random.shuffle(target_list)
            target_raw_text = ", ".join(target_list)
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 4:
            task_type = "understanding"
            task_id = 4
            random.shuffle(filtered_utterances)
            selected_utterance = filtered_utterances[0]
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            input_audio = audio[:, start : end]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = '"' + str(selected_utterance["text"]) + '"'
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 5:
            task_type = "generation"
            task_id = 1
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = None
            target_audio = audio[:, start : start + duration]
            keywords = [
                str(metadata["final_genre"]),
                # str(metadata["final_mood"]),
                # str(metadata["final_theme"]),
                str(metadata["meta_song_author_name"]),
            ]
            random.shuffle(keywords)
            keywords = TASKS[task_type][task_id][0] + " " + ", ".join(keywords)
            input_raw_text = keywords
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = target_audio.shape[-1]
        elif selected_task == 6:
            task_type = "generation"
            task_id = 2
            random.shuffle(filtered_utterances)
            selected_utterance = filtered_utterances[0]
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            input_audio = None
            target_audio = audio[:, start : end]
            input_raw_text = TASKS[task_type][task_id][0] + " " + '"' + str(selected_utterance["text"]) + '"'
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = target_audio.shape[-1]
        elif selected_task == 7:
            task_type = "generation"
            task_id = 3
            random.shuffle(filtered_utterances)
            selected_utterance = filtered_utterances[0]
            start = int(
                float(selected_utterance["start_time"]) / 1000 * self.sample_rate
            )
            end = int(float(selected_utterance["end_time"]) / 1000 * self.sample_rate)
            if end > audio.size(-1):
                self._update_stats(
                    skipped=True, message="Lyrics timestamp out of range"
                )
                return
            input_audio = None
            target_audio = audio[:, start : end]
            keywords = [
                str(metadata["final_genre"]),
                # str(metadata["final_mood"]),
                # str(metadata["final_theme"]),
                str(metadata["meta_song_author_name"]),
            ]
            random.shuffle(keywords)
            keywords = TASKS[task_type][task_id][0] + " " + ", ".join(keywords)
            lyrics = TASKS[task_type][task_id][1] + " " + '"' + selected_utterance["text"] + '"'
            input_raw_text = keywords + ", " + lyrics
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = target_audio.shape[-1]
        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length:
            self._update_stats(
                skipped=True, message="Total length too large"
            )
            return
        debug = (
            f'input_raw_text: {input_raw_text} | '
            + f'input_audio: {0 if input_audio is None else input_audio.size()} | '
            + f'target_raw_text: {target_raw_text} | '
            + f'target_audio: {0 if target_audio is None else target_audio.size()}'
        )
        output_dict = {
            "input_text": input_text,
            "target_text": target_text,
            "input_audio": input_audio,
            "target_audio": target_audio,
            "audio_length": audio_length,
            "task_type": task_type,
            "task_id": task_id,
            "debug": debug
        }
        yield output_dict
        self._update_stats(skipped=False)


class MCCInstrumentalTransforms(MCCTransforms):
    name = "MCCInstrumentalTransforms"
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
        aed_filtered: bool = True,
        audio_metrics_filtered: bool = True,
        avoid_sound_effect: bool = True,
        exclude_licenses: List[str] = ["C"],
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 2048,
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

    def __call__(self, item: Dict[str, Any]) -> Generator:
        metadata = item["__index_data__"]
        is_good, message = self.is_metadata_good(metadata)
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
        
        selected_task = torch.multinomial(torch.FloatTensor([1, 0, 0, 1, 1]), 1).item()
        if selected_task == 0:
            task_type = "understanding"
            task_id = 0
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = audio[:, start : start + duration]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = str(metadata["final_genre"])
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 1:
            task_type = "understanding"
            task_id = 1
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = audio[:, start : start + duration]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = str(metadata["final_mood"])
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 2:
            task_type = "understanding"
            task_id = 2
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = audio[:, start : start + duration]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = str(metadata["final_theme"])
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 3:
            task_type = "understanding"
            task_id = 3
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = audio[:, start : start + duration]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_list = [
                str(metadata["final_genre"]),
                # str(metadata["final_mood"]),
                # str(metadata["final_theme"]),
                str(metadata["meta_song_author_name"]),
            ]
            random.shuffle(target_list)
            target_raw_text = ", ".join(target_list)
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 4:
            task_type = "generation"
            task_id = 0
            duration = (
                random.randint(self.min_duration, self.max_duration) * self.sample_rate
            )
            start = random.randint(0, audio.size(-1) - duration)
            input_audio = None
            target_audio = audio[:, start : start + duration]
            keywords = [
                str(metadata["final_genre"]),
                # str(metadata["final_mood"]),
                # str(metadata["final_theme"]),
                str(metadata["meta_song_author_name"]),
            ]
            random.shuffle(keywords)
            input_raw_text = TASKS[task_type][task_id][0] + " " + ", ".join(keywords)
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = target_audio.shape[-1]
        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length:
            self._update_stats(
                skipped=True, message="Total length too large"
            )
            return
        debug = (
            f'input_raw_text: {input_raw_text} | '
            + f'input_audio: {0 if input_audio is None else input_audio.size()} | '
            + f'target_raw_text: {target_raw_text} | '
            + f'target_audio: {0 if target_audio is None else target_audio.size()}'
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


class KaraokeTransforms(BaseTransforms):
    name = "KaraokeTransforms"
    data_sample_rate = 24000

    def __init__(
        self,
        sample_rate: int = 24000,
        min_duration: float = 10,
        max_duration: float = 120,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 2048,
    ):
        print(f"[{self.name}] initializing...")
        super().__init__()
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
        try:
            full = self.base_transform(item["full.npy"])
            acc = self.base_transform(item["acc.npy"])
            vocal = self.base_transform(item["vocal.npy"])
            uni_len = min(full.shape[-1], acc.shape[-1], vocal.shape[-1])
            full = full[:, :uni_len]
            acc = acc[:, :uni_len]
            vocal = vocal[:, :uni_len]
        except Exception as e:
            self._update_stats(skipped=True, message=f"Error loading audio: {e}")
            return
        if full.size(-1) < self.min_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too short")
            return
        if full.size(-1) * 2 > self.max_duration * self.sample_rate:
            self._update_stats(skipped=True, message="Too long")
            return

        selected_task = torch.multinomial(torch.FloatTensor([1, 1, 1, 1]), 1).item()
        if selected_task == 0:
            task_type = "editing"
            task_id = 0
            input_audio = acc
            target_audio = full
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = input_audio.shape[-1] + target_audio.shape[-1]
        elif selected_task == 1:
            task_type = "editing"
            task_id = 1
            input_audio = vocal
            target_audio = full
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = input_audio.shape[-1] + target_audio.shape[-1]
        elif selected_task == 2:
            task_type = "editing"
            task_id = 2
            input_audio = full
            target_audio = acc
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = input_audio.shape[-1] + target_audio.shape[-1]
        elif selected_task == 3:
            task_type = "editing"
            task_id = 3
            input_audio = full
            target_audio = vocal
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = input_audio.shape[-1] + target_audio.shape[-1]
        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length:
            self._update_stats(
                skipped=True, message="Total length too large"
            )
            return
        debug = (
            f'input_raw_text: {input_raw_text} | '
            + f'input_audio: {0 if input_audio is None else input_audio.size()} | '
            + f'target_raw_text: {target_raw_text} | '
            + f'target_audio: {0 if target_audio is None else target_audio.size()}'
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
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 2048,
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
            self._update_stats(
                skipped=True, message="Lyrics timestamp out of range"
            )
            return

        selected_task = torch.multinomial(torch.FloatTensor([1, 1]), 1).item()
        if selected_task == 0:
            task_type = "understanding"
            task_id = 5
            input_audio = audio[:, start : end]
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = '"' + str(selected_utterance["text"]) + '"'
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 1:
            task_type = "generation"
            task_id = 4
            input_audio = None
            target_audio = audio[:, start : end]
            input_raw_text = TASKS[task_type][task_id][0] + " " + '"' + str(selected_utterance["text"]) + '"'
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = target_audio.shape[-1]
        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length:
            self._update_stats(
                skipped=True, message="Total length too large"
            )
            return
        debug = (
            f'input_raw_text: {input_raw_text} | '
            + f'input_audio: {0 if input_audio is None else input_audio.size()} | '
            + f'target_raw_text: {target_raw_text} | '
            + f'target_audio: {0 if target_audio is None else target_audio.size()}'
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
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 2048,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_key = audio_key
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate
        self.max_length = max_length
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
        
        selected_task = torch.multinomial(torch.FloatTensor([1, 1]), 1).item()
        if selected_task == 0:
            task_type = "understanding"
            task_id = 5
            input_audio = audio
            target_audio = None
            input_raw_text = TASKS[task_type][task_id][0]
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = '"' + item["normalized_text.txt"] + '"'
            target_text = self.tokenizer(
                target_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            audio_length = input_audio.shape[-1]
        elif selected_task == 1:
            task_type = "generation"
            task_id = 4
            input_audio = None
            target_audio = audio
            input_raw_text = TASKS[task_type][task_id][0] + " " +  '"' + item["normalized_text.txt"] + '"'
            input_text = self.tokenizer(
                input_raw_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].squeeze(dim=0)
            target_raw_text = ""
            target_text = None
            audio_length = target_audio.shape[-1]
        total_length = (
            math.ceil(audio_length / self.sample_rate * self.frame_rate)
            + (0 if input_text is None else input_text.shape[-1])
            + (0 if target_text is None else target_text.shape[-1])
        )
        if total_length > self.max_length:
            self._update_stats(
                skipped=True, message="Total length too large"
            )
            return
        debug = (
            f'input_raw_text: {input_raw_text} | '
            + f'input_audio: {0 if input_audio is None else input_audio.size()} | '
            + f'target_raw_text: {target_raw_text} | '
            + f'target_audio: {0 if target_audio is None else target_audio.size()}'
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


class WebDatasetBufferPreprocessor:
    def __init__(self, transforms: BaseTransforms):
        self.transforms = transforms

    def train_buffer_preprocessor(self, buffer: Generator) -> Generator:
        for item in buffer:
            yield from self.transforms(item)


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
        max_length: int = 2048,
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
    data_sample_rate = 24000

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
        max_length: int = 2048,
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


class MCCInstrumentalDataset(MCCBaseDataset):
    name = "MCCInstrumental"
    data_sample_rate = 24000

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
        max_length: int = 2048,
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
            max_length=max_length,
            **kwargs,
        )


class KaraokeDataset(WebPipeline):
    name = "Karaoke"
    data_sample_rate = 24000

    def __init__(
        self,
        url2index: str = "hdfs:///home/byte_speech_sv/data/karaoke_for_singsong_npy/url2idx.txt",
        sample_rate: int = 24000,
        min_duration: int = 0,
        max_duration: int = 30,
        normalize_audio: bool = False,
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 2048,
        **kwargs,
    ):
        print(f"[{self.name}] initializing...")
        transforms = KaraokeTransforms(
            sample_rate=sample_rate,
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
        tokenizer=None,
        frame_rate: int = 25,
        max_length: int = 2048,
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
        max_length: int = 2048,
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
            max_length=max_length,
        )
        preprocessor = WebDatasetBufferPreprocessor(transforms=transforms)
        dataset = WebDataset(urls=urls, **kwargs)
        pipeline = ["decode", {"compose": [preprocessor.train_buffer_preprocessor]}]
        super().__init__(dataset, pipeline)
        print(f"[{self.name}] initialized.")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate: int,
        batch_size: int,
        min_duration: int = 10,
        max_duration: int = 120,
        normalize_audio: bool = False,
        shuffle_buffer_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Callable = collate_fn,
        weights: List[int] = [1, 1, 1, 1],
        region: str = "CN",
        use_pipe: bool = False,
        tokenizer: str = "bert-base-multilingual-cased",
        frame_rate: int = 25,
        max_length: int = 2048,
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
        if weights[1] > 0:
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
        if weights[2] > 0:
            mcc_instrumental = MCCInstrumentalDataset(
                url2index=INDEX[region]["MCCInstrumental"],
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
            datasets.append(DataPipeline(mcc_instrumental, wds.shuffle(shuffle_buffer_size)))
        if weights[3] > 0:
            karaoke = KaraokeDataset(
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
            datasets.append(DataPipeline(karaoke, wds.shuffle(shuffle_buffer_size)))
        weights = [i for i in weights if i != 0]
        self.train_dataset = DataPipeline(
            MultiIterableDataset(
                datasets=datasets, weights=[i for i in weights if i != 0]
            ),
            self.bucketize,
        )

        libritts = WebPipeline(
            LibriTTSDataset(
                sample_rate=sample_rate,
                min_duration=1,
                max_duration=max_duration,
                tokenizer=self.tokenizer,
                frame_rate=self.frame_rate,
                max_length=max_length,
                resampled=False,
                nodesplitter=return_self,
                handler=wds.warn_and_continue,
            ),
            pipeline=[{"compose": [self.bucketize]}],
        )
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
