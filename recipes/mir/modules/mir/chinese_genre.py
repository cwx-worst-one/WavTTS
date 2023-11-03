from dataclasses import dataclass
from typing import Any, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from recipes.mir.datasets.mix import ChineseGenreDataResult
from samantha.models.base import DefaultTrainingBaseModule, LossDict, ModelIdentifier
from samantha.models.zoo import MODEL_ZOO
from samantha.utils.context import evaluate_model



@dataclass
class ChineseGenreConfig:
    model_name: str
    duration: float = 30.0
    finetune_foundation: bool = False
    n_embd_head: int = 64
    learning_rate: float = 1.0e-4
    scheduler_patience: int = 20
    scheduler_decay_factor: float = 0.5


@dataclass
class ChineseGenreModelResult:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: Optional[LossDict] = None



class ChineseGenreHead(nn.Module):
    _classes: Dict[str, int] = {
        "mandopop": 0,
        "cantopop": 1,
        "rock": 2,
        "dj": 3,
        "hanmai": 4,
        "nostalgia_pop": 5,
        "blues": 6,
        "house": 7,
        "hokkien_pop": 8,
        "guofeng": 9,
        "children": 10,
        "vietnamese_drum": 11,
        "jazz": 12,
        "folk": 13,
        "edm": 14,
        "metal": 15,
        "new_age": 16,
        "chinese_opera": 17,
        "guofeng_edm": 18,
        "religion": 19
    }
    _idx2class = {v: k for k, v in _classes.items()}


    def __init__(
        self,
        input_dim: int,
        n_embd: int,
        seq_len: int,
        foundation_model_id: ModelIdentifier,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.foundation_model_id = foundation_model_id
        head_chinese_genre = nn.Sequential(
            nn.Linear(input_dim, n_embd),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_embd, self.n_classes),
        )
        self.heads = nn.ModuleDict(
            {
                "chinese_genre": head_chinese_genre,
            }
        )

    def genre_name_to_ids(self, genre_names: List[str], device) -> torch.Tensor:
        return torch.tensor([self._classes[t] for t in genre_names], device=device)

    def genre_ids_to_names(self, genre_ids: torch.Tensor) -> List[str]:
        return [self._idx2class[i.item()] for i in genre_ids]

    @property
    def n_classes(self) -> int:
        return len(self._classes)

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
        x = x.mean(dim=1)  # adaptive?
        logits = self.heads.chinese_genre(x)
        return ChineseGenreModelResult(logits=logits, hidden_states=x)


class ChineseGenre(DefaultTrainingBaseModule):

    def __init__(
        self,
        config: ChineseGenreConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = MODEL_ZOO[config.model_name](
            finetune=self.config.finetune_foundation
        )

        self.heads = ChineseGenreHead(
            self.model.output_dim,
            self.config.n_embd_head,
            self.max_seq_len,
            self.model.identifier,
        )

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.heads.n_classes)
        self.pr_auc = torchmetrics.AveragePrecision(
            task="multiclass",
            num_classes=self.heads.n_classes,
            average="macro",
            thresholds=None,
        )

    @property
    def max_seq_len(self):
        return int(self.model.frame_rate * self.config.duration)

    def forward(
        self,
        audio: torch.Tensor,
    ) -> ChineseGenreModelResult:
        result = self.model.forward(audio)
        return self.heads(result.hidden_states)

    def loss(
        self,
        logits: torch.Tensor,
        genre_names: List[str],
    ) -> LossDict:
        targets = self.heads.genre_name_to_ids(genre_names, device=logits.device)
        loss = F.cross_entropy(logits, targets)
        return {"loss": loss}

    def step(self, batch: ChineseGenreDataResult, return_loss: bool) -> ChineseGenreModelResult:
        result = self.forward(batch.input_audio)
        if return_loss:
            result.loss = self.loss(result.logits, batch.target_raw_text)
        return result

    def training_step(self, batch: ChineseGenreDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        pred = result.logits.argmax(dim=-1)

        targets = self.heads.genre_name_to_ids(batch.target_raw_text, device=pred.device)
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
        self, batch: ChineseGenreDataResult, batch_idx: int
    ) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        pred = result.logits.argmax(dim=-1)

        targets = self.heads.genre_name_to_ids(batch.target_raw_text, device=pred.device)

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

    def test_step(self, batch: ChineseGenreDataResult, batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)

        pred = result.logits.argmax(dim=-1)

        targets = self.heads.genre_name_to_ids(batch.target_raw_text, device=pred.device)

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
