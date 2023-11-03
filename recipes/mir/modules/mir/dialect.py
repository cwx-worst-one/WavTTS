from dataclasses import dataclass
from typing import Any, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from recipes.mir.datasets.mix import DialectDataResult
from samantha.models.base import DefaultTrainingBaseModule, LossDict, ModelIdentifier
from samantha.models.zoo import MODEL_ZOO
from samantha.utils.context import evaluate_model



@dataclass
class DialectConfig:
    model_name: str
    duration: float = 30.0
    finetune_foundation: bool = False
    n_embd_head: int = 64
    learning_rate: float = 1.0e-4
    scheduler_patience: int = 20
    scheduler_decay_factor: float = 0.5


@dataclass
class DialectModelResult:
    logits: torch.Tensor
    x: torch.Tensor
    loss: Optional[LossDict] = None


class DialectHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        foundation_model_id: ModelIdentifier,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.foundation_model_id = foundation_model_id
        head_dialect = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.heads = nn.ModuleDict(
            {
                "dialect": head_dialect,
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
        x = x.mean(dim=1)
        logits = self.heads.dialect(x)
        return DialectModelResult(logits=logits, x=x)


class Dialect(DefaultTrainingBaseModule):
    _classes: Dict[str, str] = {
        "mandarin": 0,
        "cantonese": 1,
        "hokkien": 2,
    }

    n_classes: int = len(_classes)

    def __init__(
        self,
        config: DialectConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = MODEL_ZOO[config.model_name](
            finetune=self.config.finetune_foundation
        )

        self.heads = DialectHead(
            self.model.output_dim,
            self.n_classes,
            self.max_seq_len,
            self.model.identifier,
        )

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes)
        self.pr_auc = torchmetrics.AveragePrecision(
            task="multiclass",
            num_classes=self.n_classes,
            average="macro",
            thresholds=None,
        )

    @property
    def max_seq_len(self):
        return int(self.model.frame_rate * self.config.duration)

    def dialect_name2ids(self, dialect_names: List[str]) -> torch.Tensor:
        return torch.tensor([self._classes[t] for t in dialect_names], device=self.device)

    def forward(
        self,
        audio: torch.Tensor,
    ) -> DialectModelResult:
        result = self.model.forward(audio)
        return self.heads(result.hidden_states)

    def loss(
        self,
        logits: torch.Tensor,
        dialect_names: List[str],
    ) -> LossDict:
        targets = self.dialect_name2ids(dialect_names)
        loss = F.cross_entropy(logits, targets)
        return {"loss": loss}

    def step(self, batch: DialectDataResult, return_loss: bool) -> DialectModelResult:
        result = self.forward(batch.input_audio)
        if return_loss:
            result.loss = self.loss(result.logits, batch.target_raw_text)
        return result

    def training_step(self, batch: DialectDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        pred = result.logits.argmax(dim=-1)

        targets = self.dialect_name2ids(batch.target_raw_text)
        self.accuracy(pred, targets)
        self.pr_auc(result.logits, targets)
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
        self, batch: DialectDataResult, batch_idx: int
    ) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        pred = result.logits.argmax(dim=-1)

        targets = self.dialect_name2ids(batch.target_raw_text)

        self.accuracy(pred, targets)
        self.pr_auc(result.logits, targets)
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
        self.accuracy.reset()
        self.pr_auc.reset()

    def test_step(self, batch: DialectDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)

        pred = result.logits.argmax(dim=-1)

        targets = self.dialect_name2ids(batch.target_raw_text)

        self.accuracy(pred, targets)
        self.pr_auc(result.logits, targets)
        self.log(
            "accuracy/test",
            self.accuracy,
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
        return {
            "optimizer": optimizer,
        }
