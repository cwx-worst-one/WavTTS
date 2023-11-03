from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from einops.layers.torch import Rearrange

from recipes.datasets.billboard import BillboardDataResult
from samantha.models.base import DefaultTrainingBaseModule, LossDict, ModelIdentifier
from samantha.models.zoo import MODEL_ZOO
from samantha.utils.context import evaluate_model



@dataclass
class MIRGenderConfig:
    model_name: str
    sample_len: float = 30.0
    finetune_foundation: bool = False
    n_embd_head: int = 64
    learning_rate: float = 1.0e-4
    scheduler_patience: int = 20
    scheduler_decay_factor: float = 0.5


@dataclass
class MIRGenderResult:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: Optional[LossDict] = None


class MIRGenderHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        foundation_model_id: ModelIdentifier,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.foundation_model_id = foundation_model_id
        self.pooling = nn.Sequential()

        head_gender = nn.Sequential(
            nn.Linear(input_dim, 2),
        )
        self.heads = nn.ModuleDict(
            {
                "gender": head_gender,
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
        x = x.mean(dim=-1)  # adaptive?
        logits = self.heads.gender(x)
        return MIRGenderResult(logits=logits, hidden_states=x)


class MIRGender(DefaultTrainingBaseModule):
    def __init__(
        self,
        config: MIRGenderConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = MODEL_ZOO[config.model_name](
            finetune=self.config.finetune_foundation
        )

        self.heads = MIRGenderHead(
            self.model.output_dim,
            self.max_seq_len,
            self.model.identifier,
        )

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.male_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.female_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.pr_auc = torchmetrics.AveragePrecision(
            task="multiclass",
            num_classes=2,
            average="macro",
            thresholds=None,
        )

    @property
    def max_seq_len(self):
        return int(self.model.frame_rate * self.config.sample_len)

    def forward(
        self,
        audio: torch.Tensor,
    ) -> MIRGenderResult:
        result = self.model.forward(audio)
        return self.heads(result.x)

    @staticmethod
    def loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossDict:
        loss = F.cross_entropy(logits, targets)
        return {"loss": loss}

    def step(self, batch: BillboardDataResult, return_loss: bool) -> MIRGenderResult:
        result = self.forward(batch.audio)
        if return_loss:
            result.loss = self.loss(result.logits, batch.artist_gender_label)
        return result

    def training_step(self, batch: BillboardDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        pred = result.logits.argmax(dim=-1)
        self.accuracy(pred, batch.artist_gender_label)
        self.pr_auc(result.logits, batch.artist_gender_label)
        self.log(
            "accuracy/train",
            self.accuracy,
            prog_bar=True,
            rank_zero_only=True,
            on_step=True,
            on_epoch=True,
        )
        self.log("pr_auc/train", self.pr_auc, on_step=True, on_epoch=True)
        self.log_step(
            result.loss,
            tag="train",
            prog_bar=True,
            rank_zero_only=True,
            on_step=True,
            sync_dist=True,  # TODO
        )
        return result.loss["loss"]

    def validation_step(
        self, batch: BillboardDataResult, batch_idx: int
    ) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        pred = result.logits.argmax(dim=-1)
        self.accuracy(pred, batch.artist_gender_label)
        self.pr_auc(result.logits, batch.artist_gender_label)
        self.log(
            "accuracy/valid",
            self.accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log("pr_auc/valid", self.pr_auc, on_step=True, on_epoch=True)
        self.log_step(result.loss, tag="valid", prog_bar=True, rank_zero_only=True)
        return result.loss["loss"]

    def on_test_start(self) -> None:
        self.pr_auc.reset()

    def test_step(self, batch: BillboardDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        pred = result.logits.argmax(dim=-1)

        male_mask = batch.artist_gender_label == 0
        female_mask = batch.artist_gender_label == 1

        if len(batch.artist_gender_label[male_mask]):
            self.male_accuracy(pred[male_mask], batch.artist_gender_label[male_mask])

        if len(batch.artist_gender_label[female_mask]):
            self.female_accuracy(
                pred[female_mask], batch.artist_gender_label[female_mask]
            )

        self.accuracy(pred, batch.artist_gender_label)
        self.pr_auc(result.logits, batch.artist_gender_label)
        self.log(
            "accuracy/test",
            self.accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "male_accuracy/test",
            self.male_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "female_accuracy/test",
            self.female_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log("pr_auc/test", self.pr_auc, on_step=True, on_epoch=True)
        self.log_step(result.loss, tag="test", prog_bar=True, rank_zero_only=True)
        return result.loss["loss"]

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),  # TODO optimizers for foundation/head models
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
