import io
import os
import pickle
import random
import sys
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from torchaudio_augmentations import Compose

from recipes.musiclm.transforms.audio import (
    LoudnessCheck,
    NormalizeAudio,
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    ReadMP3,
    SetAudioDimensions,
    ToTensor,
)
from recipes.musiclm.transforms.base import TransformBase


class TransformBase:
    """Base class for all data transforms"""

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


class MCCTransform(TransformBase):
    data_sample_rate = 24000

    def __init__(
        self,
        durations_in_sec: List[float],
        sample_rate: int,
        audio_key: str = "mp3",
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        aed_filtered: bool = False,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        max_num_crops: Optional[int] = None,  # if None, auto set based on audio length
        crop_step_size: Optional[int] = None,  # if None, auto set based on n_samples
    ) -> None:
        super().__init__()
        self.durations_in_sec = durations_in_sec
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.aed_filtered = aed_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = set(exclude_licenses)
        self.max_num_crops = max_num_crops
        if crop_step_size is None:
            crop_step_size = self.n_samples // 2
        assert crop_step_size <= self.n_samples
        self.crop_step_size = crop_step_size
        self.crop_wing_span = self.n_samples // self.crop_step_size

        self.is_loud = LoudnessCheck(
            self.sample_rate, self.min_volume_threshold, self.loudness_ratio_threshold
        )
        self.read_mp3 = ReadMP3(self.sample_rate)

        base_transforms = []
        if audio_key == "mp3":
            base_transforms.append(lambda x: self.read_mp3(io.BytesIO(x)))
        base_transforms += [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        if normalize_audio:
            base_transforms.append(NormalizeAudio())
        self.base_transform = Compose(base_transforms)

        self.random_pad = RandomPad(n_samples=n_samples)
        self.random_crop = RandomResizedCrop(n_samples=n_samples)

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
        # Deleted
        return True, None

    def get_window_ids(self, audio, x):
        metadata = x["__index_data__"]
        num_windows = 1 + (audio.size(1) - self.n_samples) // self.crop_step_size
        window_ids = list(range(num_windows))
        if self.ar_filtering is None:
            return audio, window_ids, num_windows
        elif "ar_data_quality" not in metadata:
            print(
                f"WARNING: ar_filtering is set but can't find ar_data_quality in metadata: {metadata}"
            )
            return audio, window_ids, num_windows
        else:
            # Only work for MCC40M numpy
            genre = "-".join(os.path.basename(x["__url__"]).split("-")[:-1])
            if genre not in self.ar_filtering.genres:
                print(
                    f"WARNING: can't find genre {genre} in ar_filtering's genres: {self.ar_filtering.genres}"
                )
                return audio, window_ids, num_windows
            ar_data_quality = metadata["ar_data_quality"]
            segment_config = ar_data_quality["segment_config"]
            segment_duration = segment_config["segment_duration"]
            segment_step_sec = segment_duration - segment_config["overlap"]
            sd = ar_data_quality["semantic_diversity"]
            sp = ar_data_quality["semantic_probs"]
            num_segments = len(sd)
            audio = audio[..., : segment_config["max_duration"] * self.sample_rate]
            num_windows = 1 + (audio.size(1) - self.n_samples) // self.crop_step_size
            window_length_sec = self.n_samples // self.sample_rate
            window_step_sec = self.crop_step_size // self.sample_rate
            window_ids = []
            for wid in range(num_windows):
                window_start_sec = wid * window_step_sec
                window_end_sec = window_start_sec + window_length_sec
                segment_start_id = window_start_sec // segment_step_sec
                for segment_end_id in range(segment_start_id, num_segments):
                    segment_end_sec = (
                        segment_duration + segment_end_id * segment_step_sec
                    )
                    if segment_end_sec >= window_end_sec:
                        break
                window_sd_score = np.mean(sd[segment_start_id : segment_end_id + 1])
                window_sp_score = np.mean(sp[segment_start_id : segment_end_id + 1])
                if self.ar_filtering.is_valid(genre, window_sd_score, window_sp_score):
                    window_ids.append(wid)
            return audio, window_ids, num_windows

    def __call__(self, x: Dict[str, Any]) -> Generator:
        n_samples = random.choice(self.durations_in_sec) * self.sample_rate
        is_good, message = self.is_metadata_good(x["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return

        try:
            audio = self.base_transform(x[self.audio_key])
        except Exception as e:
            print(f"[MP3 decoding error] {e}")
            self._update_stats(skipped=True, message="MP3 Decoding Error")
            return
        if audio.size(1) < n_samples:
            self._update_stats(skipped=True, message="Audio Too Short")
            return

        # Determine possible crop starting points
        audio, window_ids, num_windows = self.get_window_ids(audio, x)
        random.shuffle(window_ids)
        # Return up to max_num_crops
        max_num_crops = self.max_num_crops
        if max_num_crops is None:
            max_num_crops = max(1, audio.size(1) // self.n_samples)
        num_crops = 0
        taboo = set()
        for wid in window_ids:
            if wid in taboo:
                continue
            st_sample = wid * self.crop_step_size
            en_sample = st_sample + self.n_samples
            has_vocal = self.contains_vocal(
                vocal_segments,
                st_sample / self.sample_rate,
                en_sample / self.sample_rate,
            )
            if self.avoid_vocal and has_vocal:
                continue
            cropped_audio = audio[:, st_sample:en_sample]
            if not self.is_loud(cropped_audio):
                continue
            output = {
                "audio": cropped_audio,
                "has_vocal": has_vocal,
                "key": x["__key__"],
                "metadata": x["__index_data__"],
                "url": x["__url__"],
                "sample_start_pos": st_sample,
                "sample_rate": self.sample_rate,
            }
            yield output
            num_crops += 1
            if num_crops >= max_num_crops:
                break
            # Avoid overlapping segments
            for overlapped_wid in range(
                max(0, wid - self.crop_wing_span + 1),
                min(num_windows, wid + self.crop_wing_span),
            ):
                taboo.add(overlapped_wid)
        if num_crops == 0:
            self._update_stats(skipped=True, message="No Valid Crop")
        else:
            self._update_stats(skipped=False)
