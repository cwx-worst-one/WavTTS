from typing import Any, Dict, List, Generator, Optional, Tuple
import io
import torch
import random
import pickle
import os
import numpy as np
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

    def is_overlap(
        self,
        st: float,
        en: float,
        thresh: float = 0.1,
    ) -> bool:
        if self.st > en or st > self.en:
            return False
        if en - st <= 0:
            return False
        overlap_duration = min(self.en, en) - max(self.st, st)
        overlap_ratio = overlap_duration / (en - st)
        return overlap_ratio >= thresh

    def __repr__(self):
        return f"({self.st}, {self.en})"

    def __str__(self):
        return f"({self.st}, {self.en})"


class ARFiltering:
    def __init__(
        self,
        genre_stats_fname: str = "/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy.filtered+audio_metrics_good/genre_quantile_stats.pkl",
        semantic_diversity_range: Tuple[int, int] = (50, 100),
        semantic_probs_range: Tuple[int, int] = (25, 100),
    ):
        assert len(semantic_diversity_range) == 2 and semantic_diversity_range[0] < semantic_diversity_range[1]
        assert len(semantic_probs_range) == 2 and semantic_probs_range[0] < semantic_probs_range[1]
        for x in list(semantic_diversity_range) + list(semantic_probs_range):
            assert type(x) == int and x >= 0 and x <= 100, f"Invalid quantile: {x}"
        print(f"Loading genre stats from {genre_stats_fname}...")
        with open(genre_stats_fname, "rb") as f:
            genre_stats = pickle.load(f)
        print(f"...loaded stats for {len(genre_stats)} genres")
        self.genres = set(genre_stats.keys())
        self.semantic_diversity_range = {}
        self.semantic_probs_range = {}

        def get_value(scores, q):
            if q == 0:
                return 0.0
            elif q == 100:
                return 1.0
            else:
                # quantile starts from 1, so we need to offset the index
                return scores[q - 1]

        for genre in genre_stats:
            self.semantic_diversity_range[genre] = [
                get_value(genre_stats[genre]["semantic_diversity"], q) for q in semantic_diversity_range
            ]
            self.semantic_probs_range[genre] = [
                get_value(genre_stats[genre]["semantic_probs"], q) for q in semantic_probs_range
            ]
            print(f"{genre}: sd={self.semantic_diversity_range[genre]}, sp={self.semantic_probs_range[genre]}")

    def is_valid(self, genre, sd_score, sp_score):
        sd_min, sd_max = self.semantic_diversity_range[genre]
        sp_min, sp_max = self.semantic_probs_range[genre]
        return sd_score >= sd_min and sd_score <= sd_max and sp_score >= sp_min and sp_score <= sp_max


class MusicLMTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        audio_key: str,
        sample_range_key: Optional[str] = None,
        min_volume_threshold: Optional[float] = 0.0,
        normalize_audio: bool = True,
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
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.normalize_audio_fp32]
        )
        self.normalize_audio = normalize_audio
        self.audio_normalizer = NormalizeAudio()
        self.random_pad = RandomPad(n_samples=n_samples)
        self.random_crop = RandomResizedCrop(n_samples=n_samples)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        audio = self.base_transform(x[self.audio_key])

        sample_range = None
        if self.sample_range_key is not None:
            sample_range = self.base_transform(x[self.sample_range_key])
        if self.normalize_audio:
            audio = self.audio_normalizer(audio, norm_tensor=sample_range)

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
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        aed_filtered: bool = False,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        avoid_vocal: bool = False,
        max_vocal_threshold: float = 0.5,
        overlap_vocal_threshold: float = 0.1,
        audio_metrics_filtered: bool = False,
        ar_filtering: Optional[ARFiltering] = None,
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
        self.overlap_vocal_threshold = overlap_vocal_threshold
        self.audio_metrics_filtered = audio_metrics_filtered
        self.ar_filtering = ar_filtering
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

        base_transforms = []
        if audio_key == "mp3":
            base_transforms.append(lambda x: self.read_mp3(io.BytesIO(x)))
        base_transforms += [
            ToTensor(),
            SetAudioDimensions(),
            NormalizeAudioToFloat32(),
        ]
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
            loudness.get("integrated_loudness", -7) > -5 or
            loudness.get("max_mom_loud", -7) >= 0 or
            loudness.get("max_short_term_loud", -7) >= 0
        ):
            return False, "loudness"
        # RMS stats
        rms_stats = audio_metrics.get("rms_stats", {})
        if rms_stats.get("peak", 0) > 3:
            return False, "rms_stats"
        for ch in ["left", "right"]:
            if (
                rms_stats.get(f"{ch}_total", -10) > -5 or
                rms_stats.get(f"{ch}_total", -10) < -40 or
                rms_stats.get(f"normed_std_{ch}", -10) < -19.5
            ):
                return False, "rms_stats"
        # Cutoff frequency
        cutoff_freq = audio_metrics.get("cutoff_frequency", {})
        for ch in ["left", "right"]:
            if (
                cutoff_freq.get(f"rel_{ch}", 48000) < 15000 and
                cutoff_freq.get(f"rel_{ch}_conf", 0) > 0.6 and
                cutoff_freq.get(f"band_std_{ch}", 10) < 5
            ):
                return False, "cutoff_frequency"
        # Phase
        phase = audio_metrics.get("phase_check", {})
        if phase.get("has_phase_issue", False) or abs(phase.get("rms_downmix_diff", 0.1)) > 3:
            return False, "phase_check"
        return True, None

    def get_vocal_data(self, metadata: Dict[str, Any]):
        thresh = 2  # hardcode 2 seconds
        vad = metadata.get("vad", {})
        vad_segments = vad.get("segment", [])
        total_duration = vad.get("extra", {}).get("audio_duration_in_seconds", 0.0)
        if total_duration <= 0:
            return [], 0.0
        vocal_segments = []
        vocal_duration = 0.0
        curr_segment = None
        for x in vad_segments:
            st = x["start"] / 1000.0
            en = x["end"] / 1000.0
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
            if vocal_segment.is_overlap(st, en, thresh=self.overlap_vocal_threshold):
                #print(f"Skipped: st={st} en={en} vocal_segment={vocal_segment}")
                return True
        return False

    def get_window_ids(self, audio, x):
        metadata = x["__index_data__"]
        num_windows = 1 + (audio.size(1) - self.n_samples) // self.crop_step_size
        window_ids = list(range(num_windows))
        if self.ar_filtering is None:
            return audio, window_ids, num_windows
        elif "ar_data_quality" not in metadata:
            print(f"WARNING: ar_filtering is set but can't find ar_data_quality in metadata: {metadata}")
            return audio, window_ids, num_windows
        else:
            # Only work for MCC40M numpy
            genre = "-".join(os.path.basename(x["__url__"]).split("-")[:-1])
            if genre not in self.ar_filtering.genres:
                print(f"WARNING: can't find genre {genre} in ar_filtering's genres: {self.ar_filtering.genres}")
                return audio, window_ids, num_windows
            ar_data_quality = metadata["ar_data_quality"]
            segment_config = ar_data_quality["segment_config"]
            segment_duration = segment_config["segment_duration"]
            segment_step_sec = segment_duration - segment_config["overlap"]
            sd = ar_data_quality["semantic_diversity"]
            sp = ar_data_quality["semantic_probs"]
            num_segments = len(sd)
            audio = audio[..., :segment_config["max_duration"] * self.sample_rate]
            num_windows = 1 + (audio.size(1) - self.n_samples) // self.crop_step_size
            window_length_sec = self.n_samples // self.sample_rate
            window_step_sec = self.crop_step_size // self.sample_rate
            window_ids = []
            for wid in range(num_windows):
                window_start_sec = wid * window_step_sec
                window_end_sec = window_start_sec + window_length_sec
                segment_start_id = window_start_sec // segment_step_sec
                for segment_end_id in range(segment_start_id, num_segments):
                    segment_end_sec = segment_duration + segment_end_id * segment_step_sec
                    if segment_end_sec >= window_end_sec:
                        break
                window_sd_score = np.mean(sd[segment_start_id:segment_end_id + 1])
                window_sp_score = np.mean(sp[segment_start_id:segment_end_id + 1])
                if self.ar_filtering.is_valid(genre, window_sd_score, window_sp_score):
                    window_ids.append(wid)
            return audio, window_ids, num_windows

    def __call__(self, x: Dict[str, Any]) -> Generator:
        is_good, message = self.is_metadata_good(x["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return
        vocal_segments, vocal_ratio = self.get_vocal_data(x["__index_data__"])
        #print(f"vocal_segments: {vocal_segments}, vocal_ratio: {vocal_ratio}")
        if self.avoid_vocal and vocal_ratio > self.max_vocal_threshold:
            #print(f"Skipped: vocal_ratio={vocal_ratio}")
            self._update_stats(skipped=True, message="Has Vocal")
            return

        try:
            audio = self.base_transform(x[self.audio_key])
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
            cropped_audio = audio[:, st_sample : en_sample]
            if not self.is_loud(cropped_audio):
                continue
            output = {
                "audio": cropped_audio,
                "has_vocal": has_vocal,
                "key": x["__key__"],
                "metadata": x["__index_data__"],
                "url": x["__url__"],
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


class PGCTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        sample_rate: int,
        audio_key: str = "audio.npy",
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        max_num_crops: Optional[int] = None,    # if None, auto set based on audio length
        crop_step_size: Optional[int] = None,   # if None, auto set based on n_samples
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        assert self.sample_rate == 24000
        self.audio_key = audio_key
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
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
        base_transforms = [
            ToTensor(),
            SetAudioDimensions(),
            NormalizeAudioToFloat32(),
        ]
        if normalize_audio:
            base_transforms.append(NormalizeAudio())
        self.base_transform = Compose(base_transforms)

        self.random_pad = RandomPad(n_samples=n_samples)
        self.random_crop = RandomResizedCrop(n_samples=n_samples)

    def __call__(self, x: Dict[str, Any]) -> Generator:
        if x["__index_data__"]["theme"] == "Sound Effect":
            self._update_stats(skipped=True, message="Sound Effect")
            return
        audio = self.base_transform(x[self.audio_key])
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
            cropped_audio = audio[:, st_sample : en_sample]
            if not self.is_loud(cropped_audio):
                continue
            output = {
                "audio": cropped_audio,
                "meta": x["__index_data__"],
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
