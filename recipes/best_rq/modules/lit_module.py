import torch

import pytorch_lightning as pl
from pytorch_lightning.profilers import PassThroughProfiler

import warnings

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class BestRq(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        checkpointing=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.val_outputs = dict()
        if checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2), sync_dist=True)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def _shared_step(self, batch):
        model_outputs = self.model(batch)
        return model_outputs["backward_loss"], model_outputs["acc"] * 100

    def training_step(self, batch, batch_idx):
        loss, accu = self._shared_step(batch)
        self.log_dict({"tr_loss": loss, "accu": accu}, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accu = self._shared_step(batch)
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        self.val_outputs[dataloader_idx].append((loss, accu))

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            loss = 0
            accu = 0
            for l, a in outputs:
                loss += l
                accu += a
            loss /= len(outputs)
            accu /= len(outputs)

            self.log_dict(
                {
                    f"val_loss_{dataloader_idx}": loss,
                    f"val_accu_{dataloader_idx}": accu,
                },
                prog_bar=True,
                sync_dist=True,
            )
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class WarmupCosine(_LRScheduler):
    def __init__(
        self,
        optimizer,
        init_lr,
        warmup_steps,
        cycle_steps,
        min_lr,
        last_epoch=-1,
        verbose=False,
    ):
        self.optimizer = optimizer

        if not isinstance(init_lr, list) and not isinstance(init_lr, tuple):
            self.init_lrs = [init_lr] * len(optimizer.param_groups)
        else:
            if len(init_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected {} init_lrs, but got {}".format(
                        len(optimizer.param_groups), len(init_lr)
                    )
                )
            self.init_lrs = list(init_lr)

        if not isinstance(min_lr, list) and not isinstance(min_lr, tuple):
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        else:
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected {} min_lrs, but got {}".format(
                        len(optimizer.param_groups), len(min_lr)
                    )
                )
            self.min_lrs = list(min_lr)

        # TODO: fix hyper bugs
        self.init_lrs = [eval(lr) if type(lr) == str else lr for lr in self.init_lrs]
        self.min_lrs = [eval(lr) if type(lr) == str else lr for lr in self.min_lrs]

        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict["last_epoch"]

    def cal_lr(self, init_lr, min_lr):
        if self.last_epoch <= self.warmup_steps:
            lr = max(0, self.last_epoch) / self.warmup_steps * init_lr
        else:
            lr = (init_lr - min_lr) * (
                1
                + np.cos(
                    np.pi
                    * min(self.last_epoch - self.warmup_steps, self.cycle_steps)
                    / self.cycle_steps
                )
            ) / 2 + min_lr
        return lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        lrs = [
            self.cal_lr(init_lr, min_lr)
            for init_lr, min_lr in zip(self.init_lrs, self.min_lrs)
        ]
        return lrs
