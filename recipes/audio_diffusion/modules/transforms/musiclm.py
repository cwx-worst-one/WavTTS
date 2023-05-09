import random
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchaudio_augmentations import Compose

from recipes.audio_diffusion.modules.transforms.audio import (
    NormalizeAudio,
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
)


class Text2SemanticTransforms(nn.Module):
    def __init__(
        self,
        length_seconds: int = 10,
        semantic_frame_rate: int = 25,
        text_frame_rate: int = 1,
        semantic_key: str = "acc_semantic_tok.npy",
        text_key: str = "acc_audio_emb.npy",
        use_text_emb_rvq: bool = False,
    ) -> None:
        super().__init__()
        self.length_seconds = length_seconds
        self.semantic_frame_rate = semantic_frame_rate
        self.n_semantic_frames = self.semantic_frame_rate * self.length_seconds
        self.text_frame_rate = text_frame_rate
        self.semantic_key = semantic_key
        self.text_key = text_key
        self.use_text_emb_rvq = use_text_emb_rvq
        assert self.text_frame_rate == 1 and self.semantic_frame_rate == 25

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}
        semantic_tokens = torch.from_numpy(x[self.semantic_key]).long()
        text_embs = torch.from_numpy(x[self.text_key]).float()
        if self.n_semantic_frames > semantic_tokens.size(-1):
            output["__skip__"] = True
            return output

        # This assumes text_frame_rate is 1
        semantic_chunks = semantic_tokens.unfold(
            0, self.n_semantic_frames, self.semantic_frame_rate
        )

        idx = random.randint(0, min(semantic_chunks.size(0), text_embs.size(0)) - 1)
        output["semantic_tokens"] = semantic_chunks[idx]
        if self.use_text_emb_rvq:
            raise NotImplementedError("use_text_emb_rvq has not been supported yet")
        else:
            output["text_input"] = text_embs[idx : idx + 1]
        return output


class CoarseTransforms(nn.Module):
    def __init__(
        self,
        length_seconds: int = 10,
        audio_sample_rate: int = 24000,
        semantic_frame_rate: int = 25,
        text_frame_rate: int = 1,
        audio_key: str = "acc.npy",
        sample_range_key: str = "acc_sample_range.npy",
        semantic_key: str = "acc_semantic_tok.npy",
        text_key: str = "acc_audio_emb.npy",
        use_text_emb_rvq: bool = False,
        min_volume_threshold: Optional[float] = 0.01,
    ) -> None:
        super().__init__()
        self.length_seconds = length_seconds
        self.audio_sample_rate = audio_sample_rate
        self.semantic_frame_rate = semantic_frame_rate
        self.text_frame_rate = text_frame_rate

        self.n_audio_samples = self.audio_sample_rate * self.length_seconds
        self.n_semantic_frames = self.semantic_frame_rate * self.length_seconds

        self.audio_key = audio_key
        self.sample_range_key = sample_range_key
        self.semantic_key = semantic_key
        self.text_key = text_key
        self.use_text_emb_rvq = use_text_emb_rvq
        assert self.text_frame_rate == 1 and self.semantic_frame_rate == 25

        self.min_volume_threshold = min_volume_threshold

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.normalize_audio_fp32]
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}

        sample_range = self.base_transform(x[self.sample_range_key])
        audio = self.normalize_audio(
            self.base_transform(x[self.audio_key]), norm_tensor=sample_range
        )
        semantic_tokens = torch.from_numpy(x[self.semantic_key]).long()
        text_embs = torch.from_numpy(x[self.text_key]).float()
        if self.n_semantic_frames > semantic_tokens.size(
            -1
        ) or self.n_audio_samples > audio.size(-1):
            output["__skip__"] = True
            return output

        if (
            self.min_volume_threshold is not None
            and torch.max(torch.abs(audio)) < self.min_volume_threshold
        ):
            output["__skip__"] = True
            return output

        # This assumes text_frame_rate is 1
        audio_chunks = audio.unfold(1, self.n_audio_samples, self.audio_sample_rate)
        semantic_chunks = semantic_tokens.unfold(
            0, self.n_semantic_frames, self.semantic_frame_rate
        )

        idx = random.randint(
            0, min(audio_chunks.size(1), semantic_chunks.size(0), text_embs.size(0)) - 1
        )

        output["audio"] = audio_chunks[:, idx, :]
        output["semantic_tokens"] = semantic_chunks[idx, :]
        if self.use_text_emb_rvq:
            raise NotImplementedError("use_text_emb_rvq has not been supported yet")
        else:
            output["text_input"] = text_embs[idx : idx + 1]
        return output


class FineTransforms(nn.Module):
    def __init__(
        self,
        length_seconds: int = 10,
        audio_sample_rate: int = 24000,
        audio_key: str = "acc.npy",
        sample_range_key: str = "acc_sample_range.npy",
        min_volume_threshold: Optional[float] = 0.01,
    ) -> None:
        super().__init__()
        self.length_seconds = length_seconds
        self.audio_sample_rate = audio_sample_rate

        self.n_audio_samples = self.audio_sample_rate * self.length_seconds

        self.audio_key = audio_key
        self.sample_range_key = sample_range_key

        self.min_volume_threshold = min_volume_threshold

        self.to_tensor = ToTensor()
        self.audio_dim = SetAudioDimensions()
        self.normalize_audio_fp32 = NormalizeAudioToFloat32()
        self.normalize_audio = NormalizeAudio()
        self.base_transform = Compose(
            [self.to_tensor, self.audio_dim, self.normalize_audio_fp32]
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}

        sample_range = self.base_transform(x[self.sample_range_key])
        audio = self.normalize_audio(
            self.base_transform(x[self.audio_key]), norm_tensor=sample_range
        )
        if self.n_audio_samples > audio.size(-1):
            output["__skip__"] = True
            return output

        if (
            self.min_volume_threshold is not None
            and torch.max(torch.abs(audio)) < self.min_volume_threshold
        ):
            output["__skip__"] = True
            return output

        audio_chunks = audio.unfold(1, self.n_audio_samples, self.audio_sample_rate)

        idx = random.randint(0, audio_chunks.size(1) - 1)

        output["audio"] = audio_chunks[:, idx, :]
        return output
