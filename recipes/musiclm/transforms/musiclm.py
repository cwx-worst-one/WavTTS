from typing import Any, Dict, List, Generator, Optional, Tuple, Union
import io
import torch
import random
import pickle
import os
import json
import numpy as np
from torchaudio_augmentations import Compose
import subprocess
import pickle
import librosa
from scipy.stats import entropy

from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
)
from recipes.musiclm.transforms.audio import (
    NormalizeAudio,
    RandomPad,
    RandomResizedCrop,
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
        if genre_stats_fname.startswith("hdfs://"):
            p = subprocess.Popen(["hdfs", "dfs", "-cat", genre_stats_fname], stdout=subprocess.PIPE)
            pickle_bytes, _ = p.communicate()
            genre_stats = pickle.loads(pickle_bytes)
        else:
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
        n_samples: Union[int, List[int]],
        sample_rate: int,
        audio_key: str = "mp3",
        min_length_ratio: float = 0.8,
        normalize_audio: bool = True,
        min_volume_threshold: float = 0.05,
        loudness_ratio_threshold: float = 0.2,
        aed_filtered: bool = False,
        sstk_filtered: Optional[str] = None,
        avoid_sound_effect: bool = False,
        exclude_licenses: List[str] = [],
        avoid_vocal: bool = False,
        avoid_vocal_segments: bool = False,
        max_vocal_threshold: float = 0.5,
        overlap_vocal_threshold: float = 0.1,
        audio_metrics_filtered: bool = False,
        text_type: Optional[str] = None,
        max_num_crops: Optional[Union[int, List[int]]] = None,    # if None, auto set based on audio length
        crop_step_size: Optional[Union[int, List[int]]] = None,   # if None, auto set based on n_samples
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples if isinstance(n_samples, (list, tuple)) else [n_samples]
        self.sample_rate = sample_rate
        self.audio_key = audio_key
        self.min_volume_threshold = min_volume_threshold
        self.loudness_ratio_threshold = loudness_ratio_threshold
        self.aed_filtered = aed_filtered
        self.sstk_filtered = sstk_filtered
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = set(exclude_licenses)
        self.avoid_vocal = avoid_vocal
        self.avoid_vocal_segments = avoid_vocal_segments
        self.max_vocal_threshold = max_vocal_threshold
        self.overlap_vocal_threshold = overlap_vocal_threshold
        self.audio_metrics_filtered = audio_metrics_filtered
        self.text_type = text_type
        self.max_samples = max_samples

        if not isinstance(min_length_ratio, (list, tuple)):
            min_length_ratio = [min_length_ratio] * len(self.n_samples)
        assert len(min_length_ratio) == len(self.n_samples)
        self.min_length_ratio = min_length_ratio

        if max_num_crops is None:
            max_num_crops = [None] * len(self.n_samples)
        elif not isinstance(max_num_crops, (list, tuple)):
            max_num_crops = [max_num_crops] * len(self.n_samples)
        assert len(max_num_crops) == len(self.n_samples)
        self.max_num_crops = max_num_crops

        if crop_step_size is None:
            crop_step_size = [n // 2 for n in self.n_samples]
        elif not isinstance(crop_step_size, (list, tuple)):
            crop_step_size = [crop_step_size] * len(self.n_samples)
        assert len(crop_step_size) == len(self.n_samples)
        self.crop_step_size = crop_step_size

        self.is_loud = LoudnessCheck(
            self.sample_rate,
            self.min_volume_threshold,
            self.loudness_ratio_threshold,
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

    def is_metadata_good(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        # Apply AED filtering if applicable
        if self.aed_filtered and not metadata.get("aed_filtered", False):
            return False, "Not AED Filtered"
        # Apply SSTK filtering if applicable
        if self.sstk_filtered is not None:
            keywords = metadata.get("keywords", "")
            genres = metadata.get("genres", "")
            text = f"{keywords}, {genres}".lower()
            if any([w in text for w in self.sstk_filtered.split(",")]):
                return False, f"SSTK filtered ({self.sstk_filtered})"
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
        # Apply SFT filtering if the data is there
        if "filter_label" in metadata:
            if metadata["filter_label"]["high_quality"] != "yes":
                return False, "Not High Quality (SFT)"
            ## Note: only disabling high quality filter for v9
            # if metadata["filter_label"]["popular_potential"] != "yes":
            #     return False, "Not Popular (SFT)"
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

    def is_melody_good(self, audio):
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)
        chroma = librosa.feature.chroma_stft(
            y=audio.numpy(),
            sr=self.sample_rate,
            hop_length=self.sample_rate // 4,   # 0.25s
        )
        melody = chroma.argmax(axis=0)
        probs = np.bincount(melody) / len(melody)
        # Heuristics for now, sweep threshold carefully later
        if probs.max() >= 0.35 or entropy(probs) <= 1.7:
            return False
        return True

    def get_vocal_data(self, metadata: Dict[str, Any]):
        thresh = 2  # hardcode 2 seconds
        vad = metadata.get("vad", {})
        vad_segments = vad.get("segment", [])
        total_duration = vad.get("extra", {}).get("audio_duration_in_seconds", 0.0)
        if total_duration <= 0:
            return [], 0.0, False
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

        text = ""
        for key in ["description", "title", "keywords", "genres", "instruments"]:
            if key in metadata:
                text += str(key)
        has_vocal_metadata = 'vocal' in text.lower()
        return vocal_segments, vocal_duration / total_duration, has_vocal_metadata

    def contains_vocal(
        self, vocal_segments: List[Segment], st: float, en: float
    ) -> bool:
        for vocal_segment in vocal_segments:
            if vocal_segment.is_overlap(st, en, thresh=self.overlap_vocal_threshold):
                #print(f"Skipped: st={st} en={en} vocal_segment={vocal_segment}")
                return True
        return False

    def get_window_ids(self, audio, x, n_samples, crop_step_size):
        metadata = x["__index_data__"]
        num_windows = 1 + (audio.size(1) - n_samples) // crop_step_size
        window_ids = list(range(num_windows))
        return audio, window_ids, num_windows

    def get_text(self, metadata, text_type):
        if "human_label" in metadata:
            hvals = [l for l in metadata["human_label"].values() if isinstance(l, str) and len(l)]
            human_labels = ", ".join(hvals)
            return human_labels

        if text_type == "mixed":
            text_type = "long" if random.random() <= 0.5 else "short"
        if text_type == "long":
            if "description" in metadata:
                return metadata["description"]
            elif "DESCRIPTION" in metadata:
                return metadata["DESCRIPTION"]
            else:
                raise ValueError(f"Can't find any long text: {metadata.keys()}")
        elif text_type == "short":
            if "keywords" in metadata:
                ary = metadata["keywords"].split(",")
            elif "KEYWORDS" in metadata:
                ary = metadata["KEYWORDS"].split(",")
            elif "TAGS" in metadata:
                ary = metadata["TAGS"].split(",")
            else:
                raise ValueError(f"Can't find any short text: {metadata.keys()}")
            return ", ".join([x.strip() for x in ary])
        elif text_type == "mcc":
            tags = []
            for key in ["top_level_genre", "sub_genre", "mood", "scenario", "instrument"]:
                ary = metadata["mcc_annotation"].get(key, "").split(",")
                tags.extend([x.strip() for x in ary if len(x.strip()) > 0])
            return ", ".join(tags)
        elif text_type == "sstk_dropout":
            text_fields = {}
            for key in ["description", "title", "keywords", "genres", "instruments"]:
                v = metadata.get(key)
                if v is None or v == "\\N" or len(v.strip()) == 0:
                    continue
                text_fields[key] = v.strip()
            if len(text_fields) == 0:
                return ""
            if random.random() <= 0.3:
                if random.random() < 0.8 and "description" in text_fields:
                    return text_fields["description"]
                elif "title" in text_fields:
                    return text_fields["title"]
                
            def sample_pct(arr, dropout=0.5, min_examples=1):
                random.shuffle(arr)
                if len(arr) * dropout <= min_examples:
                    return arr[:min_examples]
                return [a for idx, a in enumerate(arr) if random.random() >= dropout]

            keywords = []
            if "keywords" in text_fields:
                kw = [t.strip() for t in text_fields["keywords"].split(",")]
                kw = sample_pct(kw, 0.5, min_examples=6)
                keywords.extend(kw)
            if "genres" in text_fields:
                g = [t.strip() for t in text_fields["genres"].split(",")]
                g = sample_pct(g, 0.3, min_examples=1)
                keywords.extend(g)
            if "instruments" in text_fields:
                i = [t.strip() for t in text_fields["instruments"].split(",")]
                i = sample_pct(i, 0.3, min_examples=1)
                keywords.extend(i)
            keywords = list(set(keywords))
            random.shuffle(keywords)
            if random.random() < 0.5:
                keywords = [k.lower() for k in keywords]
            else:
                keywords = [k.capitalize() for k in keywords]
            if random.random() < 0.5:
                return ", ".join(keywords)
            else:
                return " ".join(keywords)
        elif text_type in {"sstk_concat", "sstk_random"}:
            text_fields = {}
            for key in ["description", "keywords", "genres", "instruments"]:
                v = metadata.get(key)
                if v is None or v == "\\N" or len(v.strip()) == 0:
                    continue
                text_fields[key] = v.strip()
            if len(text_fields) == 0:
                return ""
            if text_type == "sstk_random":
                long_text_available = "description" in text_fields
                short_text_available = any([k in text_fields for k in ["keywords", "genres", "instruments"]])
                if long_text_available and not short_text_available:
                    text_fields = [text_fields["description"]]
                elif short_text_available and not long_text_available:
                    text_fields = [text_fields[k] for k in ["keywords", "genres", "instruments"] if k in text_fields]
                elif random.random() <= 0.5:
                    text_fields = [text_fields["description"]]
                else:
                    text_fields = [text_fields[k] for k in ["keywords", "genres", "instruments"] if k in text_fields]
            else:
                text_fields = list(text_fields.values())
            random.shuffle(text_fields)
            return ", ".join(text_fields)
        elif text_type == "human_label":
            if "human_label" in metadata and str(metadata["human_label"]["label_genre"]) != "nan":
                human_labels = metadata["human_label"]
                text_fields = {}
                for key in ["label_genre", "label_mood", "label_instrument"]:
                    v = human_labels.get(key)
                    if v is None or v == "\\N" or len(v.strip()) == 0:
                        continue
                if len(text_fields) == 0:
                    return ""
                selected_fields = random.sample(text_fields.keys(), k=random.randint(1, 3)) # Number of text keys to use can be configured
                text_fields = [text_fields[k] for k in selected_fields]
                return ",".join(text_fields)
            else:
                text_fields = {}
                for key in ["description", "keywords", "genres", "instruments"]:
                    v = metadata.get(key)
                    if v is None or v == "\\N" or len(v.strip()) == 0:
                        continue
                    text_fields[key] = v.strip()
                if len(text_fields) == 0:
                    return ""
                if text_type == "sstk_random":
                    long_text_available = "description" in text_fields
                    short_text_available = any([k in text_fields for k in ["keywords", "genres", "instruments"]])
                    if long_text_available and not short_text_available:
                        text_fields = [text_fields["description"]]
                    elif short_text_available and not long_text_available:
                        text_fields = [text_fields[k] for k in ["keywords", "genres", "instruments"] if k in text_fields]
                    elif random.random() <= 0.5:
                        text_fields = [text_fields["description"]]
                    else:
                        text_fields = [text_fields[k] for k in ["keywords", "genres", "instruments"] if k in text_fields]
                else:
                    text_fields = list(text_fields.values())
                random.shuffle(text_fields)
                return ", ".join(text_fields)
        else:
            raise ValueError(f"Unknown text type: {text_type}")

    def maybe_convert_parquet(self, x: Dict[str, Any]):
        if "__index_data__" in x:   # WebDataset, nothing to do
            return x
        x[self.audio_key] = x["wav"]
        del x["wav"]
        x["__index_data__"] = json.loads(x["meta"])
        del x["meta"]
        x["__url__"] = x["__data_url__"]
        del x["__data_url__"]
        return x

    def __call__(self, x: Dict[str, Any]) -> Generator:
        x = self.maybe_convert_parquet(x)
        is_good, message = self.is_metadata_good(x["__index_data__"])
        if not is_good:
            self._update_stats(skipped=True, message=message)
            return
        vocal_segments, vocal_ratio, has_vocal_metadata = self.get_vocal_data(x["__index_data__"])
        #print(f"vocal_segments: {vocal_segments}, vocal_ratio: {vocal_ratio}")
        if self.avoid_vocal and (vocal_ratio > self.max_vocal_threshold or has_vocal_metadata):
            #print(f"Skipped: vocal_ratio={vocal_ratio}")
            self._update_stats(skipped=True, message="Has Vocal")
            return

        try:
            audio = self.base_transform(x[self.audio_key])
        except Exception as e:
            print(f"[MP3 decoding error] {e}")
            self._update_stats(skipped=True, message="MP3 Decoding Error")
            return

        # Choose n_samples at random
        candidates = [
            (n, m, c) for n, m, c, l in zip(
                self.n_samples,
                self.max_num_crops,
                self.crop_step_size,
                self.min_length_ratio,
            ) if audio.size(1) >= n * l and (self.max_samples is None or n <= self.max_samples)
        ]
        if len(candidates) == 0:
            self._update_stats(skipped=True, message="Audio Length Not Suitable")
            return
        random.shuffle(candidates)
        n_samples, max_num_crops, crop_step_size = candidates[0]
        crop_wing_span = n_samples // crop_step_size
        audio = RandomPad(n_samples=n_samples)(audio)
        
        # Determine possible crop starting points
        audio, window_ids, num_windows = self.get_window_ids(
            audio, x, n_samples=n_samples, crop_step_size=crop_step_size
        )
        random.shuffle(window_ids)
        # Return up to max_num_crops
        if max_num_crops is None:
            max_num_crops = max(1, audio.size(1) // n_samples)
        num_crops = 0
        taboo = set()
        for wid in window_ids:
            if wid in taboo:
                continue
            st_sample = wid * crop_step_size
            en_sample = st_sample + n_samples
            has_vocal = self.contains_vocal(
                vocal_segments,
                st_sample / self.sample_rate,
                en_sample / self.sample_rate,
            )
            if self.avoid_vocal_segments and has_vocal:
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
                "start_time": st_sample // self.sample_rate,
                "sample_start_pos": st_sample,
                "sample_rate": self.sample_rate,
            }
            if self.text_type is not None:
                output["text"] = self.get_text(x["__index_data__"], self.text_type)
            yield output
            num_crops += 1
            if num_crops >= max_num_crops:
                break
            # Avoid overlapping segments
            for overlapped_wid in range(
                max(0, wid - crop_wing_span + 1),
                min(num_windows, wid + crop_wing_span),
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
