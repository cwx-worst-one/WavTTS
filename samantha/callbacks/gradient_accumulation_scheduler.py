from typing import Any, Dict

import pytorch_lightning as pl
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_utilities.core.rank_zero import rank_zero_warn
from pytorch_lightning import Callback
from pytorch_lightning.utilities.imports import _LIGHTNING_COLOSSALAI_AVAILABLE
from pytorch_lightning.utilities.model_helpers import is_overridden


class GradientAccumulationScheduler(Callback):
    r"""Change gradient accumulation factor according to scheduling.

    Args:
        scheduling (Dict[int, int]): scheduling in format {step/epoch: acc_factor}
        interval_type (str): change accumulate_grad_batches by step or epoch.

    Note:
        The argument scheduling is a dictionary. Each key represent an epoch or a step
        and its associated accumulation factor value.
        Warning: Epoch/Step are zero-indexed c.f it means if you want to change
        the accumulation factor after 4 epochs, set
        ``GradientAccumulationScheduler(scheduling={4: factor})``.
        For more info check the example below.

    Raises:
        TypeError:
            If ``scheduling`` is an empty ``dict``,
            or not all keys and values of ``scheduling`` are integers.
        IndexError:
            If ``minimal_epoch`` or ``minimal_step`` is less than 0.

    User may config the callback via yaml

    .. code-block::yaml

        trainer: !new:pytorch_lightning.Trainer
            callbacks:
                - !new:samantha.callbacks.GradientAccumulationScheduler
                  scheduling:
                      10000: 2
                      20000: 4
                      40000: 8
                  interval_type: step

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from samantha.callbacks import GradientAccumulationScheduler

        # from epoch 5, it starts accumulating every 2 batches. Here we have 4 instead
        # of 5 because epoch (key) should be zero-indexed.
        >>> epoch_accumulator = GradientAccumulationScheduler(
        >>>     scheduling={4: 2}, interval_type="epoch"
        >>> )
        >>> trainer = Trainer(callbacks=[epoch_accumulator])
        >>> # for step based accumulator
        >>> step_accumulator = GradientAccumulationScheduler(
        >>>     scheduling={4000: 2}, interval_type="step"
        >>> )

    """

    def __init__(self, scheduling: Dict[int, int], interval_type: str = "epoch"):
        super().__init__()
        if not scheduling:  # empty dict error
            raise TypeError("Empty dict cannot be interpreted correct")

        if interval_type not in ["step", "epoch"]:
            raise MisconfigurationException(
                f"Interval type should be either step or epoch, but got {interval_type}"
            )
        if any(not isinstance(key, int) or key < 0 for key in scheduling):
            raise MisconfigurationException(
                f"Interval should be an int greater than or equal to 0."
                f" Got {list(scheduling.keys())}."
            )

        if any(
            not isinstance(value, int) or value < 1 for value in scheduling.values()
        ):
            raise MisconfigurationException(
                f"Accumulation factor should be an int greater than 0."
                f" Got {list(scheduling.values())}."
            )

        minimal_interval = min(scheduling.keys())
        if minimal_interval < 0:
            raise IndexError(
                f"Interval indexing from 1, interval {minimal_interval} cannot be"
                f" interpreted correct"
            )
        if (
            minimal_interval != 0
        ):  # if user didn't define first epoch accumulation factor
            scheduling.update({0: 1})

        self.scheduling = scheduling
        self.intervals = sorted(scheduling.keys())
        self.interval_type = interval_type

    def going_to_accumulate_grad_batches(self) -> bool:
        return any(v > 1 for v in self.scheduling.values())

    def get_accumulate_grad_batches(self, interval: int) -> int:
        accumulate_grad_batches = 1
        for iter_interval in reversed(self.intervals):
            if interval >= iter_interval:
                accumulate_grad_batches = self.scheduling[iter_interval]
                break
        return accumulate_grad_batches

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Performs a configuration validation before training starts and raises errors
        for incompatible settings."""

        if not pl_module.automatic_optimization:
            raise RuntimeError(
                """Automatic gradient accumulation and the `GradientAccumulationScheduler` # noqa
                is not supported for manual optimization. Please remove the callback or
                switch to automatic optimization."""
            )

        overridden_optimizer_step = is_overridden("optimizer_step", pl_module)
        overridden_optimizer_zero_grad = is_overridden("optimizer_zero_grad", pl_module)
        going_to_accumulate_grad_batches = self.going_to_accumulate_grad_batches()
        has_overridden_optimization_functions = (
            overridden_optimizer_step or overridden_optimizer_zero_grad
        )
        if has_overridden_optimization_functions and going_to_accumulate_grad_batches:
            rank_zero_warn(
                "When using `Trainer(accumulate_grad_batches != 1)` and overriding"
                " `LightningModule.optimizer_{step,zero_grad}`, the hooks will not be"
                " called on every batch"
                " (rather, they are called on every optimization step)."
            )

        # local import to avoid circular import
        from pytorch_lightning.accelerators import IPUAccelerator
        from pytorch_lightning.strategies import DeepSpeedStrategy

        unsupported_accelerators = (IPUAccelerator,)
        unsupported_strategies = [DeepSpeedStrategy]
        if _LIGHTNING_COLOSSALAI_AVAILABLE:
            from lightning_colossalai import ColossalAIStrategy

            unsupported_strategies.append(ColossalAIStrategy)

        if isinstance(trainer.accelerator, unsupported_accelerators):
            raise RuntimeError(
                f"The `{type(trainer.accelerator).__name__}` does not support"
                f" `accumulate_grad_batches` changing between epochs/steps."
            )
        if isinstance(trainer.strategy, tuple(unsupported_strategies)):
            raise RuntimeError(
                f"The `{type(trainer.strategy).__name__}` does not support"
                f" `accumulate_grad_batches` changing between epochs/steps."
            )
        if trainer.accumulate_grad_batches != 1:
            raise ValueError(
                "You have set `accumulate_grad_batches` and are using the"
                " `GradientAccumulationScheduler` callback. Either remove"
                " `accumulate_grad_batches` from the Trainer or remove the callback."
            )

    def on_train_batch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        if self.interval_type == "step":
            interval = trainer.global_step
        else:
            interval = trainer.current_epoch
        trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(interval)
