import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler


class TriStageLR(_LRScheduler):
    """
    Warm up LR linearly from `lr * init_lr_scale` to `lr` for the first
    `warmup_steps` iters. Then hold LR constant at `lr` for `hold_steps` iters.
    Then decay LR exponentially to `lr * final_lr_scale` in `decay_steps`.
    After that, LR is kept constant at `lr * final_lr_scale`.

    Modeled after Fairseq's TriStageLRSchedule:
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py
    """

    def __init__(
        self,
        optimizer,
        init_lr_scale=0.01,
        final_lr_scale=0.01,
        warmup_steps=0,
        hold_steps=0,
        decay_steps=0,
        last_epoch=-1,
        verbose=False,
    ):
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps
        # For some reason these can't be initialized after parent constructor
        # Will construct them as singletons as a workaround
        self.init_lrs = None
        self.final_lrs = None
        self.warmup_rates = None
        assert (
            self.warmup_steps + self.hold_steps + self.decay_steps > 0
        ), "please specify steps"
        super(TriStageLR, self).__init__(optimizer, last_epoch, verbose)

    @property
    def hparams(self):
        return {
            "init_lr_scale": self.init_lr_scale,
            "final_lr_scale": self.final_lr_scale,
            "warmup_steps": self.warmup_steps,
            "hold_steps": self.hold_steps,
            "decay_steps": self.decay_steps,
            "decay_factor": self.decay_factor,
        }

    def get_init_lrs(self):
        if self.init_lrs is None:
            self.init_lrs = [lr * self.init_lr_scale for lr in self.base_lrs]
        return self.init_lrs

    def get_final_lrs(self):
        if self.final_lrs is None:
            self.final_lrs = [lr * self.final_lr_scale for lr in self.base_lrs]
        return self.final_lrs

    def get_warmup_rates(self):
        if self.warmup_rates is None:
            self.warmup_rates = [
                (lr - init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
                for lr, init_lr in zip(self.base_lrs, self.get_init_lrs())
            ]
        return self.warmup_rates

    def _decide_stage(self, update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset

    def _get_lr(self, stage, steps_in_stage):
        if stage == 0:
            ret = [
                init_lr + warmup_rate * steps_in_stage
                for init_lr, warmup_rate in zip(
                    self.get_init_lrs(), self.get_warmup_rates()
                )
            ]
        elif stage == 1:
            ret = self.base_lrs
        elif stage == 2:
            ret = [
                lr * math.exp(-self.decay_factor * steps_in_stage)
                for lr in self.base_lrs
            ]
        elif stage == 3:
            ret = self.get_final_lrs()
        else:
            raise ValueError(f"Undefined stage: {stage}")
        return ret

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        stage, steps_in_stage = self._decide_stage(self.last_epoch)
        return self._get_lr(stage, steps_in_stage)

    def _get_closed_form_lr(self):
        stage, steps_in_stage = self._decide_stage(self.last_epoch)
        return self._get_lr(stage, steps_in_stage)
