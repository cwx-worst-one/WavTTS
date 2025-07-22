import warnings

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl

def get_model_parameters_with_lr(
        pl_module: pl.LightningModule, 
    ):
    ''' each basestage module has a lr_ratio, and this process assign the true_lr for the corresponding module parameters
        true_lr = lr * lr_ratio, enabling different learning rates for different modules.  '''

    optimizer_groups = []
    lr = pl_module.optimizer_cls.keywords['lr']
    for stage in pl_module.model.stages:
        if stage.validattr('lr_ratio'):
            iter_parameters = foundation_parameters(stage)
            optimizer_groups.append({"params": iter_parameters, "lr": lr * stage.lr_ratio})
        if hasattr(stage, 'spans'): 
            # include parameters from span_models
            for span in stage.spans:
                if span.validattr('lr_ratio'):
                    optimizer_groups.append({"params": span.parameters(), "lr": lr * span.lr_ratio})
        if stage.validattr('insert_modules'):
            # include parameters from insert_modules
            for module in stage.insert_modules:
                if module.validattr('lr_ratio'):
                    optimizer_groups.append({"params": module.parameters(), "lr": lr * module.lr_ratio})
    lr_groups = [g['lr'] for g in optimizer_groups]

    if not optimizer_groups: # for some reasons, no parameters are appended
        optimizer_groups = pl_module.model.parameters()
        lr_groups = None

    print("####", optimizer_groups)
    return optimizer_groups, lr_groups

def foundation_parameters(stage):
    for name, params in stage.named_parameters():
        if "insert_modules" not in name: # exclude params from insert_modules
            yield params

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


class DecayCosine(_LRScheduler):
    def __init__(
        self, optimizer, init_lr, cycle_steps, min_lr, last_epoch=-1, verbose=False
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

        self.cycle_steps = cycle_steps
        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict["last_epoch"]

    def cal_lr(self, init_lr, min_lr):
        lr = (init_lr - min_lr) * (
            1
            + np.cos(np.pi * min(self.last_epoch, self.cycle_steps) / self.cycle_steps)
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
