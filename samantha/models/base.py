from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
from functools import cached_property
from typing import IO, Any, Dict, NamedTuple, Optional, Sequence, Union

import torch
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_helpers import _restricted_classmethod
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm
from typing_extensions import Self

from samantha.utils.hdfs_tools import hdfs_get_cache
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)

LossDict = Union[Dict[str, torch.Tensor], Dict[str, float]]


class ModelIdentifier(NamedTuple):
    name: str
    mean: float
    std: float


class LightningModuleBase(LightningModule):
    def __init__(self, ignore: Optional[Union[Sequence[str], str]] = None):
        super().__init__()
        self.save_hyperparameters(ignore=ignore, logger=False)

    @property
    @torch.jit.unused
    def batch_size(self) -> int:
        return self.trainer.datamodule.batch_size

    @property
    @torch.jit.unused
    def world_size(self) -> int:
        return self.trainer.world_size

    @property
    @torch.jit.unused
    def global_batch_size(self) -> int:
        return self.world_size * self.batch_size

    @abstractmethod
    def prepare_model(self, stage: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def forward(self, batch: Any):
        pass

    @abstractmethod
    def training_step(self):
        pass

    @property
    @torch.jit.unused
    def extra_hparams(self) -> Dict[str, Any]:
        hparams = vars(self.config)
        hparams.update(
            {"batch_size": self.batch_size, "global_batch_size": self.global_batch_size}
        )
        return hparams

    @cached_property
    def identifier(self):
        mean = []
        std = []
        for param in tqdm(self.parameters(), "Computing model identfier..."):
            mean.append(param.mean())
            std.append(param.std())

        return ModelIdentifier(
            name=self.__class__.__name__,
            mean=torch.stack(mean).sum().item(),
            std=torch.stack(std).sum().item(),
        )

    def summarize(self, max_depth: int = 1):
        return summarize(self, max_depth=max_depth)

    # def on_train_start(self) -> None:
    #     if self.logger is not None:
    #         self.logger.log_hyperparams(
    #             self.extra_hparams, {"loss/train": torch.inf, "loss/valid": torch.inf}
    #         )

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        return super().validation_step(*args, **kwargs)

    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        return super().test_step(*args, **kwargs)

    def on_test_epoch_end(self) -> None:
        return super().on_test_epoch_end()

    @abstractmethod
    def configure_optimizers(self) -> Any:
        pass

    # Any extra hooks go here

    def log_step(
        self,
        loss_dict: LossDict,
        tag: str,
        prog_bar: bool,
        rank_zero_only: bool,
        sync_dist: bool = False,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        batch_size = loss_dict.get("batch_size", None)
        loss_dict = {f"{k}/{tag}": v for k, v in loss_dict.items()}
        self.log_dict(
            loss_dict,
            prog_bar=prog_bar,
            rank_zero_only=rank_zero_only,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )

    @abstractmethod
    def preprocess_step(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_inputs(self, batch: Any):
        pass

    @_restricted_classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        cache: bool = False,
        cache_dir: str = ".cache",
        **kwargs: Any,
    ) -> Self:
        if cache:
            checkpoint_path = hdfs_get_cache(checkpoint_path, cache_dir=cache_dir)
            logger.info(f"Loading checkpoint from cache: {checkpoint_path}")
        return super().load_from_checkpoint(checkpoint_path=checkpoint_path, **kwargs)


@dataclass
class TrainingResultBase:
    loss: Optional[LossDict] = None


class DefaultTrainingBaseModule(LightningModuleBase):
    @abstractmethod
    def step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, return_loss: bool
    ) -> TrainingResultBase:
        pass

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        result = self.step(batch, batch_idx, return_loss=True)
        self.log_step(
            result.loss,
            tag="train",
            prog_bar=True,
            rank_zero_only=True,
            on_step=True,
            sync_dist=False,
        )
        return result.loss["loss"]

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        result = self.step(batch, batch_idx, return_loss=True)
        self.log_step(
            result.loss, tag="valid", prog_bar=True, rank_zero_only=True, sync_dist=True
        )
        return result.loss["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        result = self.step(batch, batch_idx, return_loss=True)
        self.log_step(
            result.loss, tag="test", prog_bar=True, rank_zero_only=True, sync_dist=True
        )
        return result.loss["loss"]

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, return_loss=False)
