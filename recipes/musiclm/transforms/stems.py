import random
from typing import Dict, Generator, Optional

import torch
import torch.nn as nn
from torchaudio_augmentations import Compose

from recipes.musiclm.transforms.audio import (
    NormalizeAudio,
    NormalizeAudioToFloat32,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
    amplitude_to_db,
    crop_1d,
    get_random_idx,
    pad_1d,
    rms,
)
from recipes.musiclm.transforms.base import TransformBase


class StemsTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        source_audio_key: str,
        source_sample_range_key: str,
        target_audio_key: str,
        target_sample_range_key: str,
        min_db: float,
        relative_db: float,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.source_audio_key = source_audio_key
        self.source_sample_range_key = source_sample_range_key
        self.target_audio_key = target_audio_key
        self.target_sample_range_key = target_sample_range_key
        self.min_db = min_db
        self.relative_db = relative_db

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.normalize_audio_fp32]
        )
        # self.random_pad = RandomPad(n_samples=n_samples)
        # self.random_crop = RandomResizedCrop(n_samples=n_samples)

    def preprocess_audio(
        self, x: Dict[str, torch.Tensor], audio_key: str, sample_range_key: str
    ) -> torch.Tensor:
        audio = self.base_transform(x[audio_key])
        sample_range = self.base_transform(x[sample_range_key])
        return self.normalize_audio(audio, norm_tensor=sample_range)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        norm_source_audio = self.preprocess_audio(
            x, self.source_audio_key, self.source_sample_range_key
        )
        norm_target_audio = self.preprocess_audio(
            x, self.target_audio_key, self.target_sample_range_key
        )

        src_len = norm_source_audio.shape[-1]
        target_len = norm_target_audio.shape[-1]

        if target_len > src_len:
            norm_source_audio = pad_1d(
                norm_source_audio, start_idx=0, n_samples=target_len
            )
        elif src_len > target_len:
            norm_target_audio = pad_1d(
                norm_target_audio, start_idx=0, n_samples=src_len
            )

        max_samples = norm_target_audio.shape[-1]
        if max_samples < self.n_samples:
            rand_pad_idx = get_random_idx(self.n_samples - max_samples)
            norm_target_audio = pad_1d(norm_target_audio, rand_pad_idx, self.n_samples)
            norm_source_audio = pad_1d(norm_source_audio, rand_pad_idx, self.n_samples)

        max_samples = norm_target_audio.shape[-1]
        rand_crop_idx = get_random_idx(max_samples - self.n_samples)
        norm_target_audio = crop_1d(norm_target_audio, rand_crop_idx, self.n_samples)
        norm_source_audio = crop_1d(norm_source_audio, rand_crop_idx, self.n_samples)

        db_src = amplitude_to_db(rms(norm_source_audio).max())
        db_target = amplitude_to_db(rms(norm_target_audio).max())

        skipped = False
        if (
            db_src < self.min_db
            or db_target < self.min_db
            or (db_src - db_target >= self.relative_db)
        ):
            skipped = True

        if not skipped:
            yield {
                "target_audio.npy": norm_target_audio,
                "source_audio.npy": norm_source_audio,
            }
        self._update_stats(skipped=skipped)


class StemGenerationTransforms(TransformBase):
    def __init__(
        self,
        n_samples: int,
        sample_rate: int,
        stem_key: str = "stems.npy",
        json_key: str = "metadata.json",
        text_emb_key: str = None,
        min_volume_threshold: Optional[float] = None,
        num_mixes=2,
        max_stems_per_mix=[4, 1],
    ) -> None:
        super().__init__()
        self.stem_key = stem_key
        self.text_emb_key = text_emb_key
        self.json_key = json_key
        self.min_volume_threshold = min_volume_threshold

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.normalize_audio_fp32]
        )

        self.random_crop = RandomResizedCrop(n_samples=n_samples)
        self.n_samples = n_samples
        self.sample_rate = sample_rate

        self.num_mixes = num_mixes
        self.max_stems_per_mix = max_stems_per_mix
        if len(max_stems_per_mix) != num_mixes:
            raise Exception(
                f"Num_mixes ({num_mixes}) does not match length of max_stems list({len(max_stems_per_mix)})"  # noqa
            )

    def _extract_mixes(self, stems, num_stems):
        mixes = torch.zeros(self.num_mixes, stems.shape[1])
        rand_idxs = torch.randperm(num_stems)
        mix_idx_log = []
        for i, max_stems_in_mix_channel in enumerate(self.max_stems_per_mix):
            target_stems_in_mix_channel = random.randint(1, max_stems_in_mix_channel)
            target_stems_in_mix_channel = min(
                target_stems_in_mix_channel,
                rand_idxs.shape[0] - (self.num_mixes - 1 - i),
            )
            mix_idxs = rand_idxs[0:target_stems_in_mix_channel]
            rand_idxs = rand_idxs[target_stems_in_mix_channel:]
            mix_idx_log.append(mix_idxs)
            mixes[i, :] = stems[mix_idxs, :].sum(dim=0, keepdim=True)
        return mixes, mix_idx_log

    def __call__(self, x: Dict[str, torch.Tensor]) -> Generator:
        try:
            stems = self.base_transform(x[self.stem_key])
        except KeyError:
            print("stems.npy missing, skipping")
            self._update_stats(skipped=True)
            return

        norm_stems = self.normalize_audio(stems)

        try:
            norm_stems = self.random_crop(norm_stems)
        except ValueError:
            print("File too short. Skipping")
            self._update_stats(skipped=True)
            return

        vocal_mask = []
        instrument_categories = x[self.json_key]["38_stem_instrument_names"]
        for instrument in x[self.json_key]["instrument_names"]:
            is_not_vocal = "vocal" not in instrument.lower()
            vocal_mask.append(is_not_vocal)
        norm_stems = norm_stems[vocal_mask, :]
        instrument_categories = [
            i for (i, v) in zip(instrument_categories, vocal_mask) if v
        ]

        silence_mask = norm_stems.abs().max(dim=1).values > self.min_volume_threshold
        norm_stems = norm_stems[silence_mask, :]
        instrument_categories = [
            i for (i, v) in zip(instrument_categories, silence_mask) if v
        ]

        num_stems = norm_stems.shape[0]
        if num_stems < self.num_mixes:
            print(f"Found {num_stems} stems but needed {self.num_mixes}. Skipping")
            self._update_stats(skipped=True)
            return

        mixes, mix_idxs = self._extract_mixes(norm_stems, num_stems)

        target_idx = mix_idxs[-1][0]
        target_category = instrument_categories[target_idx][0]
        ret = {"target_category": torch.tensor(target_category).unsqueeze(0)}

        for i in range(self.num_mixes):
            signal = mixes[i, :].unsqueeze(0)
            ret[f"mix_{i}"] = signal

        if self.text_emb_key is not None:
            emb = torch.tensor(x[self.text_emb_key])
            emb = emb.mean(dim=0).view(-1, 1)
            emb = nn.functional.normalize(emb, dim=0)
            ret["text_emb"] = emb

        yield ret
        self._update_stats(skipped=False)
