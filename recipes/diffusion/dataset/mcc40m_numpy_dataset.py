from typing import Any, Dict, List, Generator, Optional, Tuple
import hashlib
import random

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import webdataset as wds
from pathlib import Path
from webdataset.pipeline import DataPipeline
from torch.utils.data import IterableDataset
from samantha.dataio.webdataset.extension import IndexedWebDataset
from recipes.soundstream.dataset.utils import collate_fn, fix_hash
from samantha.dataio.dataset import MultiIterableDataset

from recipes.musiclm.transforms.musiclm import Segment
from recipes.musiclm.transforms.audio import (
    NormalizeAudio,
    LoudnessCheck,
)

SAMPLE_RATE = 24000
# down sample to ~1000
GENRE_WEIGHTING = {
    'alternative-hip-hop': 1, 
    'alternative-indie': 1,
    'blues': 1,
    'childhood': 1,
    'chinese-opera': 1,
    'chinese-style': 1,
    'classical': 0.3,
    'country': 1,
    'devotional': 1,
    'easy-listening': 0.3,
    'electronic': 0.1,
    'experimental': 1,
    'folk': 1,
    'hip-hop-rap': 1,
    'indie-folk': 1,
    'jazz': 1,
    'k-pop': 1,
    'latin': 1,
    'metal': 1,
    'new-age': 0.4,
    'others': 1,
    'pop': 1,
    'r-b-soul': 1,
    'reggae': 1,
    'rock': 1,
    'soundtrack': 1,
    'techno': 1,
    'traditional-chinese-folk': 1,
    'trance': 1,
    'world-music': 1,
}

class MCC40MNumpyDataset(IterableDataset):
    def __init__(
        self,
        genre,
        source_path='/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy/npy_url2idx.txt',
        music_len=30720,
        normalize_audio=True,
        min_volume_threshold=0.05,
        loudness_ratio_threshold=0.2,
        aed_filtered: bool = True,
        max_vocal_threshold: float = 0.25,
        exclude_licenses: List[str] = ["C"],
        avoid_sound_effect=True,
        avoid_vocal=True,
        mode="train",
    ):
        assert mode in ["train"], "doens't support val mode"

        self.music_len = music_len
        self.normalize_audio = normalize_audio
        if self.normalize_audio:
            self.norm_audio = NormalizeAudio()
        
        self.aed_filtered = aed_filtered
        self.max_vocal_threshold = max_vocal_threshold
        self.avoid_vocal = avoid_vocal
        self.avoid_sound_effect = avoid_sound_effect
        self.exclude_licenses = exclude_licenses

        self.is_loud = LoudnessCheck(
            SAMPLE_RATE,
            threshold=min_volume_threshold,
            loudness_ratio_threshold=loudness_ratio_threshold
        )
        self.mode = mode

        genere_map = {}
        with open(source_path, "r") as f:
            for line in f:
                if type(line) == bytes:
                    line = line.decode("utf-8")
                ary = line.strip().split("\t")
                # get genre
                stem = str(Path(ary[0]).stem)
                stem = stem[:stem.rfind('-')]
                
                if stem not in genere_map:
                    genere_map[stem] = {ary[0]: ary[1]}
                else:
                    genere_map[stem][ary[0]] = ary[1]

        self.dataset = (
            IndexedWebDataset(
                genere_map[genre],
                resampled=True,
            )
            .decode()
            .map(self._process_audio)
        )

    def __len__(self):
        if self.mode == "val":
            return self.num_steps_per_val

    def _get_vocal_data(self, metadata: Dict[str, Any]):
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
    
    def _contains_vocal(
        self, vocal_segments: List[Segment], st: float, en: float
    ) -> bool:
        for vocal_segment in vocal_segments:
            if vocal_segment.is_overlap(st, en):
                #print(f"Skipped: st={st} en={en} vocal_segment={vocal_segment}")
                return True
        return False

    def _is_metadata_good(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
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

    def _process_audio(self, data):
        # meta filter
        is_good, message = self._is_metadata_good(data["__index_data__"])
        if not is_good:
            return None
        if self.avoid_vocal:
            vocal_segments, vocal_ratio = self._get_vocal_data(data["__index_data__"])
            if vocal_ratio > self.max_vocal_threshold:
                return

        audio = data["audio.npy"]
        audio = (audio / 32768.0).astype("float32")
        audio = torch.from_numpy(audio).float()[None,]
        if self.normalize_audio:
            audio = self.norm_audio(audio)
        
        if self.mode == "train":
            if audio.shape[-1] < self.music_len:
                audio = F.pad(
                    audio, ((0, self.music_len - audio.shape[-1])), "constant"
                )
                start_idx = 0
            else:
                start_idx = random.randint(0, audio.shape[-1] - self.music_len)
        else:
            mid_point = int(audio.shape[-1] // 2)
            if mid_point + self.music_len > audio.shape[-1]:
                audio = F.pad(
                    audio,
                    ((0, mid_point + self.music_len - audio.shape[-1])),
                    "constant",
                )
                start_idx = 0
            else:
                start_idx = mid_point

        audio = audio[..., start_idx : start_idx + self.music_len]

        if self.avoid_vocal and self._contains_vocal(
                vocal_segments,
                start_idx / SAMPLE_RATE,
                (start_idx + self.music_len) / SAMPLE_RATE,
            ):
                return None

        if not self.is_loud(audio):
            return None

        data["audio"] = audio[None,]
        data["music_id"] = fix_hash(data["__key__"])
        return data

    def __iter__(self):
        return iter(self.dataset)


class GenreBalanceMCC40MNumpyDataset(MultiIterableDataset):
    def __init__(
        self,
        num_samples,
        seed,
        source_path='/mnt/bn/audio-diffusion/data/non_vocal_mcc_npy/npy_url2idx.txt',
        music_len=30720,
        normalize_audio=True,
        min_volume_threshold=0.05,
        loudness_ratio_threshold=0.2,
        aed_filtered: bool = True,
        max_vocal_threshold: float = 0.25,
        exclude_licenses: List[str] = ["C"],
        avoid_sound_effect=True,
        avoid_vocal=True,
        mode="train",
    ):
        datasets = [
            MCC40MNumpyDataset(
                genre=genre,
                source_path=source_path,
                music_len=music_len,
                normalize_audio=normalize_audio,
                min_volume_threshold=min_volume_threshold,
                loudness_ratio_threshold=loudness_ratio_threshold,
                max_vocal_threshold=max_vocal_threshold,
                exclude_licenses=exclude_licenses,
                avoid_sound_effect=avoid_sound_effect,
                avoid_vocal=avoid_vocal,
                mode=mode,
            )
            for genre in GENRE_WEIGHTING.keys()
        ]
        super().__init__(
            datasets=datasets,
            num_samples=num_samples,
            weights=[v for v in GENRE_WEIGHTING.values()],
            seed=seed,
        )

if __name__ == "__main__":
    dataset = GenreBalanceMCC40MNumpyDataset(
        num_samples=1000,
        seed=0,
        music_len=240000,
        normalize_audio=True,
    )
    loader = wds.WebLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=4)
    for batch in tqdm(loader):
        print(batch['audio'].shape, batch['music_id'])
        # assert ["mp3" in item for item in batch]
        import soundfile as sf
        for i in range(batch['audio'].shape[0]):
            sf.write(f'test/test_{i}.wav', batch['audio'][i].numpy().T, 24000)
        assert 1==2