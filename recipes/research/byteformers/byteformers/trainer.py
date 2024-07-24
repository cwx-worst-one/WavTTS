import os
import time
from functools import partial
from typing import Any, Iterable, List, Literal, Mapping, Optional, Tuple, Union, cast

import lightning as L
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning_utilities import apply_to_collection
from tqdm import tqdm

from .benchmarks.utils import flops_achieved


class FabricTrainer:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        logger: Optional[Union[Logger, List[Logger]]] = None,
        accumulate_grad_batches: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        gradient_clip_val: Optional[Union[int, float]] = None,
        max_epochs: int = 1,
        max_steps: Optional[int] = None,
        checkpoint_dir: str = "",
    ) -> None:
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=logger,
        )

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        # gradient clipping
        if gradient_clip_val is not None and not isinstance(
            gradient_clip_val, (int, float)
        ):
            raise TypeError(
                "`gradient_clip_val` should be an int or a float. Got"
                f" {gradient_clip_val}."
            )

        self.global_step = 0
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val
        self.should_stop = False
        self.current_epoch = 0
        self.checkpoint_dir = checkpoint_dir
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self.fast_dev_run = False

        self.limit_train_batches = limit_train_batches

    def fit(
        self,
        pl_module: L.LightningModule,
        train_dataloader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        ckpt_path: Optional[str] = None,
    ):
        self.fabric.launch()

        # TODO: Do we want to seed all models similarly?
        self.fabric.seed_everything(42 + self.fabric.global_rank)

        # setup dataloaders
        self.train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader)

        pl_module = self.fabric.setup_module(pl_module)
        optimizer, lr_scheduler = self._parse_optimizers_schedulers(
            pl_module.configure_optimizers()
        )
        optimizer = self.fabric.setup_optimizers(optimizer)

        self.state = {"model": pl_module, "optim": optimizer, "scheduler": lr_scheduler}

        self.fabric.call("on_train_start", trainer=self, pl_module=pl_module)
        while not self.should_stop:
            self.train_loop(
                pl_module,
                optimizer,
                lr_scheduler=lr_scheduler,
                limit_batches=self.limit_train_batches,  # , scheduler_cfg=scheduler_cfg
            )

            self.step_scheduler(
                pl_module, lr_scheduler, level="epoch", current_value=self.current_epoch
            )

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

        # reset for next fit call
        self.should_stop = False

    def train_loop(
        self,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        limit_batches: Union[int, float] = float("inf"),
        lr_scheduler: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ] = None,
    ):
        """The training loop running a single training epoch.
        Args:
            pl_module: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            limit_batches: Limits the batches during this training epoch.
                If greater then the number of batches in the
                ``self.train_dataloader``, this has no effect.
            lr_scheduler: The learning rate scheduler configuration.
        """
        self.fabric.call("on_train_epoch_start", trainer=self, pl_module=pl_module)
        iterable = self.progbar_wrapper(
            self.train_dataloader,
            total=min(len(self.train_dataloader), limit_batches),
            desc=f"Epoch {self.current_epoch}",
        )

        current_tflops = None
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this
            # epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                self.fabric.call(
                    "on_train_epoch_end", trainer=self, pl_module=pl_module
                )
                return

            start_time = time.time()

            should_optim_step = (
                self.global_step
                and self.global_step % self.accumulate_grad_batches == 0
            )
            if should_optim_step:
                self.fabric.call(
                    "on_before_optimizer_step",
                    trainer=self,
                    pl_module=pl_module,
                    optimizer=optimizer,
                )

                optimizer.zero_grad()
                optimizer.step(
                    partial(
                        self.training_step,
                        pl_module=pl_module,
                        batch=batch,
                        batch_idx=batch_idx,
                    )
                )
                self.fabric.call(
                    "on_before_zero_grad",
                    trainer=self,
                    pl_module=pl_module,
                    optimizer=optimizer,
                )
            else:
                # gradient accumulation -> no optimizer step
                self.training_step(pl_module, batch, batch_idx)

            self.fabric.call(
                "on_train_batch_end",
                trainer=self,
                pl_module=pl_module,
                outputs=self._current_train_return,
                batch=batch,
                batch_idx=batch_idx,
            )
            end_time = time.time() - start_time

            if batch_idx % 10 == 0:
                current_tflops = (
                    flops_achieved(end_time, batch[0].shape[0], pl_module.total_flops)
                    / 10**12
                )
                # info = torch.cuda.mem_get_info()
                # print(
                #     info[0] / info[1],
                #     end_time,
                #     pl_module.model.get_num_params(False),
                #     flops_achieved(end_time, batch[0].shape[0], pl_module.total_flops)
                #     / 10**12,
                # )

            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(
                    pl_module,
                    lr_scheduler,
                    level="step",
                    current_value=self.global_step,
                )

            # add output values to progress bar
            self._format_iterable(iterable, self._current_train_return, "train")
            if current_tflops is not None:
                self._format_iterable(iterable, {"TFLOPs": current_tflops}, "train")

            # only increase global step if optimizer stepped
            self.global_step += 1

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

    def training_step(
        self, pl_module: L.LightningModule, batch: Any, batch_idx: int
    ) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer
        step is called separately, as this
        is given as a closure to the optimizer step.
        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch
        """

        src, targets = batch
        logits = pl_module(src)
        loss = pl_module.loss(logits, targets)
        nll = loss["nll"]

        self.fabric.log_dict(loss, step=self.global_step)

        self.fabric.call(
            "on_before_backward", trainer=self, pl_module=pl_module, loss=loss
        )
        self.fabric.backward(nll)
        self.fabric.call("on_after_backward", trainer=self, pl_module=pl_module)

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(
            loss, dtype=torch.Tensor, function=lambda x: x.detach()
        )
        return nll

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``
        Args:
            state: A mapping containing model, optimizer and lr scheduler.
        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        self.fabric.save(
            os.path.join(self.checkpoint_dir, f"step-{self.global_step:04d}.ckpt"),
            state,
        )

    def step_scheduler(
        self,
        pl_module: L.LightningModule,
        lr_scheduler: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.
        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``,
            else holds the ``global_step``
        """

        # no scheduler
        if lr_scheduler is None:
            return

        # wrong interval (step vs. epoch)
        if lr_scheduler["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, lr_scheduler["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update(
                {"train_" + k: v for k, v in self._current_train_return.items()}
            )

        # if isinstance(self._current_val_return, torch.Tensor):
        #     possible_monitor_vals.update("val_loss", self._current_val_return)
        # elif isinstance(self._current_val_return, Mapping):
        #     possible_monitor_vals.update(
        #         {"val_" + k: v for k, v in self._current_val_return.items()}
        #     )

        try:
            monitor = possible_monitor_vals[
                cast(Optional[str], lr_scheduler["monitor"])
            ]
        except KeyError as e:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {lr_scheduler['monitor']} is invalid. Possible values are"
                f" {possible_keys}."
            ) from e

        # rely on model hook for actual step
        pl_module.lr_scheduler_step(lr_scheduler["scheduler"], monitor)

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.
        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the
            number of batches was limited.
        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def _parse_optimizers_schedulers(self, configure_optim_output) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
    ]:
        """Recursively parses the output of
        :meth:`lightning.pytorch.LightningModule.configure_optimizers`.
        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to
                :meth:`lightning.pytorch.LightningModule.configure_optimizers`
        """
        _lr_sched_defaults = {
            "interval": "epoch",
            "frequency": 1,
            "monitor": "train_nll",
        }

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        elif isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        elif isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        elif isinstance(configure_optim_output, (list, tuple)):
            if all(
                [
                    isinstance(_opt_cand, L.fabric.utilities.types.Optimizable)
                    for _opt_cand in configure_optim_output
                ]
            ):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            elif all(
                [
                    isinstance(
                        _lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping)
                    )
                    for _lr_cand in configure_optim_output
                ]
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return (
                        None,
                        self._parse_optimizers_schedulers(configure_optim_output[0])[1],
                    )

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar,
        candidates: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ],
        prefix: str,
    ):
        """Adds values as postfix string to progressbar.
        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable
                (every other rank).

            candidates: the values to add as postfix strings to the progressbar.

            prefix: the prefix to add to each of these values.
        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(
                candidates, torch.Tensor, lambda x: x.item()
            )
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
