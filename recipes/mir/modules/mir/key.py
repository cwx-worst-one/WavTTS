from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from recipes.mir.eval.key_detection.constants import (
    KEY_MAJMIN_NAME,
    KEYMODE_MAJMIN_ID,
    VOCAL_KEYMODE_MAJMIN_ID,
)
from recipes.mir.eval.key_detection.core import loss, val_test_step

from recipes.mir.datasets.mir.key import KeyDataResult
from samantha.models.base import DefaultTrainingBaseModule, LossDict, ModelIdentifier
from samantha.models.external.zoo import MODEL_ZOO
from samantha.utils.context import evaluate_model


@dataclass
class MIRKeyConfig:
    model_name: str
    sample_rate: int
    sample_len: int
    label_hop: int
    finetune_foundation: bool = False
    n_embd_head: int = 1024
    resample: int = 0
    learning_rate: float = 1.0e-4
    scheduler_patience: int = 20
    scheduler_decay_factor: float = 0.5


@dataclass
class MIRKeyResult:
    key_logits: torch.Tensor
    x: torch.Tensor
    loss: Optional[LossDict] = None


class MIRKeyHead(nn.Module):
    output_dim = 25

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        n_embd: int,
        resample: int,
        foundation_model_id: ModelIdentifier,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.resample = resample
        self.foundation_model_id = foundation_model_id

        if self.resample > 0:
            avg_pool = nn.AdaptiveAvgPool1d(resample)
        else:
            avg_pool = nn.Identity()

        self.pooling = nn.Sequential(
            avg_pool,
            Rearrange("b c t -> b t c"),
        )

        out_key = nn.Sequential(
            nn.Linear(input_dim, n_embd),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_embd, self.output_dim),
        )
        self.heads = nn.ModuleDict(
            {
                "key": out_key,
            }
        )

    @property
    def _inputs(self) -> torch.Tensor:
        return torch.randn(1, self.input_dim, self.seq_len)

    def _verify_input(self, x: torch.Tensor) -> None:
        if x.shape[1] != self.input_dim or x.shape[2] != self.seq_len:
            raise Exception("Shapes do not match")

    def export(self):
        with evaluate_model(self):
            traced_model = torch.jit.script(self, (self._inputs))
        return traced_model

    def forward(self, x: torch.Tensor) -> MIRKeyResult:
        # self._verify_input(x)
        x = self.pooling(x)
        logits = self.heads.key(x)
        return MIRKeyResult(key_logits=logits, x=x)


class MIRKey(DefaultTrainingBaseModule):
    _hop_factor: int = 6  # TODO why 6?

    def __init__(
        self,
        config: MIRKeyConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.musicfm = MODEL_ZOO[config.model_name](
            finetune=self.config.finetune_foundation
        )

        self.heads = MIRKeyHead(
            self.musicfm.output_dim,
            self.max_seq_len,
            self.config.n_embd_head,
            self.config.resample,
            self.musicfm.identifier,
        )

    @property
    def max_seq_len(self):
        return int(self.musicfm.frame_rate * self.config.sample_len)

    def forward(self, audio: torch.Tensor) -> MIRKeyResult:
        result = self.musicfm.forward(audio)
        return self.heads(result.x)

    def step(self, batch: KeyDataResult, return_loss: bool) -> MIRKeyResult:
        result = self.forward(batch.audio)
        if return_loss:
            result.loss = loss(result.key_logits, batch.key_label)
        return result

    def _val_test_step(self, batch: KeyDataResult):
        # prepare data
        sample = batch.audio

        duration_samples = int(self.config.sample_rate * self.config.sample_len)
        sample = torch.nn.functional.pad(sample, (0, duration_samples))

        # TODO
        # stride the audio similar to the label hop size
        # 30 second audio (b x 1 x duration) -> (b * (duration / hop) x 1 x label_hop)
        sample_hop = int(
            self.config.sample_len / self._hop_factor * self.config.sample_rate
        )
        sample = sample[0].unfold(-1, duration_samples, sample_hop)[0]

        result = self.forward(sample)
        return val_test_step(
            key_logits=result.key_logits,
            key_labels=batch.key_label,
            duration=self.config.sample_len,
            label_hop=self.config.label_hop,
            hop_factor=self._hop_factor,
            key_map=KEYMODE_MAJMIN_ID,
        )

    def validation_step(self, batch: KeyDataResult, batch_idx: int) -> torch.Tensor:
        scores = self._val_test_step(batch)
        self.log_step(
            scores,
            tag="valid",
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
            sync_dist=True,
        )

    def test_step(self, batch: KeyDataResult, batch_idx: int) -> torch.Tensor:
        scores = self._val_test_step(batch)
        self.log_step(
            scores,
            tag="test",
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
            sync_dist=True,
        )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            fused=False,
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     patience=self.config.scheduler_patience,
        #     factor=self.config.scheduler_decay_factor,
        #     verbose=True,
        #     mode="min",
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            # "monitor": "train/loss",
        }
