from typing import Any, Dict, List, Generator, Optional, Tuple
import io
import torch
import random
from torchaudio_augmentations import Compose

from recipes.musiclm.transforms.audio import (
    NormalizeAudio,
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
    LoudnessCheck,
    ReadMP3,
)
from recipes.musiclm.transforms.base import TransformBase


class Segment:
    def __init__(self, st: float, en: float):
        self.st = st
        self.en = en

    def duration(self) -> float:
        return self.en - self.st

    def is_overlap(self, st: float, en: float) -> bool:
        return self.st <= en and st <= self.en

    def __repr__(self):
        return f"({self.st}, {self.en})"

    def __str__(self):
        return f"({self.st}, {self.en})"


class MusicLMTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        audio_key: str,
        sample_range_key: Optional[str] = None,
        min_volume_threshold: Optional[float] = 0.0,
        num_crops: int = 1,
        num_tries: int = 5,
    ) -> None:
        super().__init__()
        self.audio_key = audio_key
        self.sample_range_key = sample_range_key
        self.min_volume_threshold = min_volume_threshold
        self.num_crops = num_crops
        self.num_tries = num_tries

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.normalize_audio_fp32]
        )

        self.random_pad = RandomPad(n_samples=n_samples)
        self.random_crop = RandomResizedCrop(n_samples=n_samples)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        audio = self.base_transform(x[self.audio_key])

        if self.sample_range_key is not None:
            sample_range = self.base_transform(x[self.sample_range_key])
            audio = self.normalize_audio(audio, norm_tensor=sample_range)

        audio = self.random_pad(audio)
        # Return up to self.num_crops crops
        num_crops = 0
        num_tries = 0
        while num_crops < self.num_crops and num_tries < self.num_tries:
            cropped_audio = self.random_crop(audio)
            if (
                self.min_volume_threshold is None
                or torch.mean(torch.abs(cropped_audio)) >= self.min_volume_threshold
            ):
                yield {"audio": cropped_audio}
                num_crops += 1
            num_tries += 1
        self._update_stats(num_crops == 0)


class MCCTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        sample_rate: int,
        audio_key: str = "mp3",
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        aed_filtered: bool = False,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        avoid_vocal: bool = False,
        max_vocal_threshold: float = 0.25,
        max_num_crops: Optional[int] = None,    # if None, auto set based on audio length
        crop_step_size: Optional[int] = None,   # if None, auto set based on n_samples
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.aed_filtered = aed_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = set(exclude_licenses)
        self.avoid_vocal = avoid_vocal
        self.max_vocal_threshold = max_vocal_threshold
        self.max_num_crops = max_num_crops
        if crop_step_size is None:
            crop_step_size = self.n_samples // 2
        assert crop_step_size <= self.n_samples
        self.crop_step_size = crop_step_size
        self.crop_wing_span = self.n_samples // self.crop_step_size

        self.is_loud = LoudnessCheck(
            self.sample_rate,
            self.min_volume_threshold,
            self.loudness_ratio_threshold
        )
        self.read_mp3 = ReadMP3(self.sample_rate)
        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [
                self.read_mp3,
                self.to_tensor,
                self.audio_dim,
                self.normalize_audio_fp32
            ]
        )

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
        return True, None

    def get_vocal_data(self, metadata: Dict[str, Any]):
        thresh = 2  # 2 seconds
        trans_5stem = metadata.get("mir.json", {}).get("trans_5stem", {})
        vocal = trans_5stem.get("notes", {}).get("vocal", [])
        total_duration = trans_5stem.get("end_time", 0)
        if total_duration <= 0:
            return [], 0.0
        vocal_segments = []
        vocal_duration = 0.0
        curr_segment = None
        for x in vocal:
            st = x["start"]
            en = x["end"]
            if curr_segment is None:
                curr_segment = Segment(st, en)
            elif st - curr_segment.en <= thresh:
                curr_segment.en = en
            else:
                if curr_segment.duration() >= thresh:
                    vocal_segments.append(curr_segment)
                    vocal_duration += curr_segment.duration()
                curr_segment = Segment(st, en)
        if curr_segment is not None:
            if curr_segment.duration() >= thresh:
                vocal_segments.append(curr_segment)
                vocal_duration += curr_segment.duration()
        return vocal_segments, vocal_duration / total_duration

    def contains_vocal(
        self, vocal_segments: List[Segment], st: float, en: float
    ) -> bool:
        for vocal_segment in vocal_segments:
            if vocal_segment.is_overlap(st, en):
                #print(f"Skipped: st={st} en={en} vocal_segment={vocal_segment}")
                return True
        return False

    def __call__(self, x: Dict[str, Any]) -> Generator:
        is_good, message = self.is_metadata_good(x["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return
        vocal_segments = []
        if self.avoid_vocal:
            vocal_segments, vocal_ratio = self.get_vocal_data(x["__index_data__"])
            #print(f"vocal_segments: {vocal_segments}, vocal_ratio: {vocal_ratio}")
            if vocal_ratio > self.max_vocal_threshold:
                #print(f"Skipped: vocal_ratio={vocal_ratio}")
                self._update_stats(skipped=True, message="Has Vocal")
                return

        try:
            audio = self.base_transform(io.BytesIO(x[self.audio_key]))
        except Exception as e:
            print(f"[MP3 decoding error] {e}")
            self._update_stats(skipped=True, message="MP3 Decoding Error")
            return
        if audio.size(1) < self.n_samples * 0.95:
            self._update_stats(skipped=True, message="Audio Too Short")
            return
        else:            
            audio = self.random_pad(audio)
        
        # Determine possible crop starting points
        num_windows = 1 + (audio.size(1) - self.n_samples) // self.crop_step_size
        window_ids = list(range(num_windows))
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
            if self.avoid_vocal and self.contains_vocal(
                vocal_segments,
                st_sample / self.sample_rate,
                en_sample / self.sample_rate,
            ):
                continue
            cropped_audio = audio[:, st_sample : en_sample]
            if not self.is_loud(cropped_audio):
                continue
            genre = x["__url__"].split("/")[-2].split(".")[0]
            output = {
                "audio": cropped_audio,
                "clip_id": x["metadata.json"]["clip_id"],
                "meta_song_id": x["metadata.json"]["meta_song_id"],
                "genre": genre,
                "sample_start_pos": st_sample,
                "sample_rate": self.sample_rate
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
