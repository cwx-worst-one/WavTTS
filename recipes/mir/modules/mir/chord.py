from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from recipes.mir.eval.chord.core import loss, val_test_step

from recipes.mir.datasets.mir.chord import ChordDataResult
from samantha.models.base import DefaultTrainingBaseModule, LossDict, ModelIdentifier
from samantha.models.external.zoo import MODEL_ZOO
from samantha.utils.context import evaluate_model



@dataclass
class MIRChordConfig:
    model_name: str
    sample_rate: int
    finetune_foundation: bool = False
    n_embd_head: int = 1024
    sample_len: float = 12.0
    hop_length: int = 750
    label_hop: float = 0.0625
    resample: int = 192  # 0
    learning_rate: float = 1.0e-4
    scheduler_patience: int = 20
    scheduler_decay_factor: float = 0.5


@dataclass
class MIRChordResult:
    root_logits: torch.Tensor
    triad_logits: torch.Tensor
    x: torch.Tensor
    loss: Optional[LossDict] = None


class MIRChordHead(nn.Module):
    chord_root_output_dim = 13
    chord_triad_output_dim = 7

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

        head_chord_root = nn.Sequential(
            nn.Linear(self.input_dim, self.n_embd),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.n_embd, self.chord_root_output_dim),
        )
        head_chord_triad = nn.Sequential(
            nn.Linear(self.input_dim, self.n_embd),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.n_embd, self.chord_triad_output_dim),
        )
        self.heads = nn.ModuleDict(
            {
                "chord_root": head_chord_root,
                "chord_triad": head_chord_triad,
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

    def forward(self, x: torch.Tensor):
        # self._verify_input(x)
        x = self.pooling(x)
        root_logits = self.heads.chord_root(x)
        triad_logits = self.heads.chord_triad(x)
        return MIRChordResult(
            root_logits=root_logits,
            triad_logits=triad_logits,
            x=x,
        )


class MIRChord(DefaultTrainingBaseModule):
    _hop_factor: int = 6  # TODO

    def __init__(
        self,
        config: MIRChordConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.musicfm = MODEL_ZOO[config.model_name](
            finetune=self.config.finetune_foundation
        )
        self.heads = MIRChordHead(
            self.musicfm.output_dim,
            self.max_seq_len,
            self.config.n_embd_head,
            self.config.resample,
            self.musicfm.identifier,
        )

    @property
    def max_seq_len(self):
        return int(self.musicfm.frame_rate * self.config.sample_len)

    def forward(self, audio: torch.Tensor) -> MIRChordResult:
        result = self.musicfm.forward(audio)
        return self.heads(result.x)

    def step(self, batch: ChordDataResult, return_loss: bool) -> MIRChordResult:
        result = self.forward(batch.audio)
        if return_loss:
            result.loss = loss(
                root_logits=result.root_logits,
                root_labels=batch.chord_root,
                triad_logits=result.triad_logits,
                triad_labels=batch.chord_triad,
                chord_note=batch.chord_note,
                chord_ignore=batch.chord_ignore,
            )
        return result

    def _val_test_step(
        self, batch: ChordDataResult
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        sample = batch.audio
        duration_samples = int(self.config.sample_rate * self.config.sample_len)
        sample = torch.nn.functional.pad(sample, (0, duration_samples))
        sample_hop = int(
            self.config.sample_len / self._hop_factor * self.config.sample_rate
        )
        sample = sample[0].unfold(-1, duration_samples, sample_hop)[0]

        result = self.forward(sample)
        return val_test_step(
            root_logits=result.root_logits,
            root_labels=batch.chord_root,
            triad_logits=result.triad_logits,
            triad_labels=batch.chord_triad,
            chord_ignore=batch.chord_ignore,
            duration=self.config.sample_len,
            label_hop=self.config.label_hop,
            hop_factor=self._hop_factor,
        )

    def validation_step(self, batch: ChordDataResult, batch_idx: int) -> None:
        partial_metrics, overall_metrics = self._val_test_step(batch)

        self.log_step(
            partial_metrics,
            tag="valid",
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log_step(
            overall_metrics,
            tag="valid",
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
            sync_dist=True,
        )

    def test_step(self, batch: ChordDataResult, batch_idx: int) -> None:
        partial_metrics, overall_metrics = self._val_test_step(batch)

        self.log_step(
            partial_metrics,
            tag="test",
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log_step(
            overall_metrics,
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
