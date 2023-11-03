from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from recipes.mir.datasets.mir.tagging import TaggingDataResult
from samantha.models.base import DefaultTrainingBaseModule, LossDict, ModelIdentifier
from samantha.models.external.zoo import MODEL_ZOO
from samantha.utils.context import evaluate_model


@dataclass
class MIRTaggingConfig:
    model_name: str
    sample_len: float = 29.1
    finetune_foundation: bool = False
    n_embd_head: int = 512
    learning_rate: float = 1.0e-4
    scheduler_patience: int = 20
    scheduler_decay_factor: float = 0.5


@dataclass
class MIRTaggingResult:
    logits: torch.Tensor
    x: torch.Tensor
    loss: Optional[LossDict] = None


class MIRTaggingHead(nn.Module):
    output_dim = 50

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        n_embd: int,
        foundation_model_id: ModelIdentifier,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.foundation_model_id = foundation_model_id

        head_tagging = nn.Sequential(
            nn.Linear(self.input_dim, self.n_embd),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.n_embd, self.output_dim),
        )

        self.heads = nn.ModuleDict(
            {
                "tagging": head_tagging,
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
        x = x.mean(dim=-1)
        logits = self.heads.tagging(x)
        return MIRTaggingResult(logits=logits, x=x)


class MIRTagging(DefaultTrainingBaseModule):
    def __init__(
        self,
        config: MIRTaggingConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.musicfm = MODEL_ZOO[config.model_name](
            finetune=self.config.finetune_foundation
        )

        self.heads = MIRTaggingHead(
            self.musicfm.output_dim,
            self.max_seq_len,
            self.config.n_embd_head,
            self.musicfm.identifier,
        )

        self.roc_auc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=self.heads.output_dim,
            average="macro",
            thresholds=None,
        )
        self.pr_auc = torchmetrics.AveragePrecision(
            task="multilabel",
            num_labels=self.heads.output_dim,
            average="macro",
            thresholds=None,
        )

    @property
    def max_seq_len(self):
        return int(self.musicfm.frame_rate * self.config.sample_len)

    def forward(self, audio: torch.Tensor) -> MIRTaggingResult:
        result = self.musicfm.forward(audio)
        return self.heads(result.x)

    @staticmethod
    def loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossDict:
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        return {"loss": loss}

    def step(self, batch: TaggingDataResult, return_loss: bool) -> MIRTaggingResult:
        result = self.forward(batch.audio)
        if return_loss:
            result.loss = self.loss(result.logits, batch.tag)
        return result

    def validation_step(self, batch: TaggingDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.log_step(result.loss, tag="valid", prog_bar=True, rank_zero_only=True)

        self.roc_auc(result.logits, batch.tag)
        self.pr_auc(result.logits, batch.tag)
        self.log(
            "valid/roc_auc", self.roc_auc, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "valid/pr_auc", self.pr_auc, prog_bar=True, on_step=False, on_epoch=True
        )
        return result.loss["loss"]

    def on_test_start(self) -> None:
        self.roc_auc.reset()
        self.pr_auc.reset()

    def test_step(self, batch: TaggingDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.roc_auc(result.logits, batch.tag)
        self.pr_auc(result.logits, batch.tag)
        self.log("test/roc_auc", self.roc_auc, on_step=True, on_epoch=True)
        self.log("test/pr_auc", self.pr_auc, on_step=True, on_epoch=True)
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
