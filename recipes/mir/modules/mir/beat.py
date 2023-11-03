from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from recipes.mir.eval.beat.core import loss, val_test_step

from recipes.mir.datasets.mir.beat import BeatDataResult
from samantha.models.base import DefaultTrainingBaseModule, LossDict, ModelIdentifier
from samantha.models.external.zoo import MODEL_ZOO
from samantha.utils.context import evaluate_model


@dataclass
class MIRBeatConfig:
    model_name: str
    sample_rate: int
    sample_len: float = 6
    label_hop: float = 0.02
    finetune_foundation: bool = False
    n_embd_head: int = 512
    resample: int = 300
    learning_rate: float = 1.0e-4
    scheduler_patience: int = 20
    scheduler_decay_factor: float = 0.5
    do_tempo_loss: bool = True


@dataclass
class MIRBeatResult:
    beat_logits: torch.Tensor  # inherited from sami_ai_models, TODO refactor
    tempo_logits: torch.Tensor  # inherited from sami_ai_models, TODO refactor
    x: torch.Tensor
    loss: Optional[LossDict] = None


class MIRBeatHead(nn.Module):
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

        head_beat = nn.Sequential(
            nn.Linear(input_dim, n_embd),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_embd, 3),
        )
        head_tempo = nn.Sequential(
            nn.Linear(input_dim, n_embd),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_embd, 300),
        )

        self.heads = nn.ModuleDict(
            {
                "beat": head_beat,
                "tempo": head_tempo,
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
        beat_logits = self.heads.beat(x)
        tempo_logits = self.heads.tempo(x.mean(dim=1))
        return MIRBeatResult(
            beat_logits=beat_logits,
            tempo_logits=tempo_logits,
            x=x,
        )


class MIRBeat(DefaultTrainingBaseModule):
    _hop_length: int = 240
    _beat_window_length: int = 2
    _n_beats: int = 3
    _n_tempo: int = 300
    _hop_factor: int = 1
    _f_measure_threshold: float = 0.07
    _use_tempo_prior: bool = False

    def __init__(
        self,
        config: MIRBeatConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.musicfm = MODEL_ZOO[config.model_name](
            finetune=self.config.finetune_foundation
        )

        self.heads = MIRBeatHead(
            self.musicfm.output_dim,
            self.max_seq_len,
            self.config.n_embd_head,
            self.config.resample,
            self.musicfm.identifier,
        )

    @property
    def max_seq_len(self):
        return self.musicfm.frame_rate * self.config.sample_len

    def forward(self, audio: torch.Tensor) -> MIRBeatResult:
        result = self.musicfm.forward(audio)
        return self.heads(result.x)

    def step(self, batch: BeatDataResult, return_loss: bool) -> MIRBeatResult:
        result = self.forward(batch.audio)
        if return_loss:
            beat_preds = [result.beat_logits]  # TODO refactor
            tempo_preds = [result.tempo_logits]
            result.loss = loss(
                beat=batch.beat_labels,
                hop_length=self._hop_length,
                aug_hop_size=self._hop_length,  # TODO
                beat_preds=beat_preds,
                beat_window_length=self._beat_window_length,
                n_beats=self._n_beats,
                n_tempo=self._n_tempo,
                tempo_preds=tempo_preds,
                tempo_target=batch.tempo,
                do_tempo_loss=self.config.do_tempo_loss,
            )
        return result

    def _val_test_step(self, batch: BeatDataResult):
        inputs = {}

        sample = batch.audio
        duration_samples = int(self.config.sample_rate * self.config.sample_len)
        sample = torch.nn.functional.pad(sample, (0, duration_samples))
        sample_hop = int(
            self.config.sample_len / self._hop_factor * self.config.sample_rate
        )
        sample = sample[0].unfold(-1, duration_samples, sample_hop)[0]

        result = self.forward(sample)
        return val_test_step(
            beat_logits=result.beat_logits,
            tempo_logits=result.tempo_logits,
            beat_labels=batch.beat_labels,
            tempo_labels=batch.tempo,
            orig_beats=batch.orig_beats,
            n_beats=self._n_beats,
            duration=self.config.sample_len,
            label_hop=self.config.label_hop,
            dataset_name=batch.dataset_name[0],
            use_tempo_prior=self._use_tempo_prior,
        )

    def validation_step(self, batch: BeatDataResult, batch_idx: int) -> None:
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

    def test_step(self, batch: BeatDataResult, batch_idx: int) -> None:
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
            # TODO: weight_decay is set to 0?
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
            # "monitor": "loss/train",
        }
