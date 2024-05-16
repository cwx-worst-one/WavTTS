import os
from abc import abstractmethod, abstractproperty
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from samantha.components.attention import MultiHeadAttention, SeerAttention
from samantha.dataio.webdataset.writer import IndexShardWriter
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    def init_cache(
        self, init_values: Optional[torch.Tensor] = None, cache: Optional[dict] = None
    ) -> Tuple[dict, List]:
        """The `MultiHeadAttention` module optionally accepts `kv_cache` which
        stores the key and value tensors calculated for the previous positions.
        This method returns a dictionary that stores all caches, and the necessary
        hooks for the key and value projection modules that save the intermediate
        tensors to be reused during later calculations.

        The `self.hooks` contain a list of PyTorch RemovableHandle objects to stop
        the hooks from being called. This is done in `self.deinit_cache`.

        Args:
            init_values (Optional[torch.Tensor], optional):
                Tensor containing values to initialize the k/v cache with
                (i.e., single forward pass). Defaults to None.

            cache (Optional[dict], optional):
                Existing k/v cache. Defaults to None.

        Returns:
            Tuple[dict, List]:
                A dictionary object mapping the key/value projection modules
                to its cache
        """
        self.hooks = []
        cache = {**cache} if cache is not None else {}

        def save_to_cache(module, _, output):
            if module not in cache:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, (MultiHeadAttention, SeerAttention)):
                layer._use_cache = True
                self.hooks.append(layer.to_k.register_forward_hook(save_to_cache))
                self.hooks.append(layer.to_v.register_forward_hook(save_to_cache))

        self.apply(install_hooks)

        if init_values is not None:
            self.forward(init_values)

        return cache

    def deinit_cache(self) -> None:
        def unset_cache(layer: nn.Module):
            if isinstance(layer, (MultiHeadAttention, SeerAttention)):
                layer._use_cache = False

        self.apply(unset_cache)

        for h in self.hooks:
            h.remove()
        self.hooks = []


LossDict = Union[Dict[str, torch.Tensor], Dict[str, float]]


class ModelIdentifier(NamedTuple):
    name: str
    mean: float
    std: float


class LightningModuleBase(LightningModule):
    def __init__(self, ignore_hparams: List[str] = []):
        super().__init__()
        self.save_hyperparameters(ignore=ignore_hparams)

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
    def forward(self):
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

    def preprocess_dataloader(
        self, data_loader, directory: str, rank: int, padding_strategy: str
    ):
        pattern = os.path.join(directory, str(rank), "%05d.tar")

        # TODO retain shard variety -> audio is larger, around ~400 tracks per 8GB
        # mel take up much less space, so I'm setting a manual maxcount instead

        # maxsize = (1 << 32) * 2  # 8GiB, maximum size of each shard
        maxcount = 1000
        writer = IndexShardWriter(pattern, maxcount=maxcount)

        logger.info(f"Writing shards to: {pattern}")

        audio_key = "audio"

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch_keys = list(batch.keys())

            batch_size = batch[audio_key].shape[0]

            batch.audio = batch.audio.to(self.device)
            inputs = self.get_inputs(batch)

            # write to new items
            batch_keys.remove(audio_key)
            for idx in range(batch_size):
                obj = {}
                index = {}

                # write new index
                for k in batch_keys:
                    if batch[k] is not None:
                        if isinstance(batch[k], str):
                            index[k] = batch[k]
                        else:
                            index[k] = batch[k][idx]

                # write new tar
                for k, v in asdict(inputs).items():
                    numpy_key = f"{k}.npy"
                    if isinstance(v, torch.Tensor):
                        if v.ndim:
                            obj[numpy_key] = v[idx].detach().cpu().numpy()
                        else:
                            # scalars
                            obj[numpy_key] = np.array([v.item()])

                unique_id = str(uuid4())
                obj["__key__"] = unique_id
                writer.write(obj, index)
        writer.close()

    def preprocess_save_fp(self, pl_datamodule, root_dir: str):
        return os.path.join(
            root_dir, f"{pl_datamodule.__class__.__name__}_{self.__class__.__name__}"
        )

    def preprocess(self, pl_datamodule, root_dir: str, rank: int):
        fp = self.preprocess_save_fp(pl_datamodule, root_dir)
        pl_datamodule.setup(stage="fit")
        self = self.to("cuda")

        self.preprocess_dataloader(
            pl_datamodule.train_dataloader(),
            f"{fp}/train",
            rank,
            pl_datamodule.padding_strategy,
        )
        self.preprocess_dataloader(
            pl_datamodule.val_dataloader(),
            f"{fp}/validation",
            rank,
            pl_datamodule.padding_strategy,
        )  # TODO: check if this does all validation?
        self.preprocess_dataloader(
            pl_datamodule.test_dataloader(),
            f"{fp}/test",
            rank,
            pl_datamodule.padding_strategy,
        )


@dataclass
class TrainingResultBase:
    loss: Optional[LossDict] = None


class DefaultTrainingBaseModule(LightningModuleBase):
    @abstractmethod
    def step(
        self, batch: Dict[str, torch.Tensor], return_loss: bool
    ) -> TrainingResultBase:
        pass

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
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
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.log_step(result.loss, tag="valid", prog_bar=True, rank_zero_only=True)
        return result.loss["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        result = self.step(batch, return_loss=True)
        self.log_step(result.loss, tag="test", prog_bar=True, rank_zero_only=True)
        return result.loss["loss"]

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        return self.step(batch, return_loss=False)


class GenerativeBaseModule(DefaultTrainingBaseModule):
    def __init__(self):
        super().__init__()

    @abstractproperty
    def max_seq_len(self) -> int:
        pass

    @property
    def extra_hparams(self):
        hparams = super().extra_hparams
        hparams.update({"global_token_size": self.global_token_size})
        return hparams

    @property
    def global_token_size(self) -> int:
        return self.global_batch_size * self.max_seq_len

    @property
    def _tokens_seen(self) -> int:
        try:
            return self.global_token_size * self.global_step
        except Exception as e:
            print(e)
            return 0
