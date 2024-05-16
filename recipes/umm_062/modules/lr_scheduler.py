import warnings

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


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
