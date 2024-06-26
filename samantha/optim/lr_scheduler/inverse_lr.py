from torch.optim.lr_scheduler import _LRScheduler


class InverseLR(_LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        inv_gamma=1.0,
        power=1.0,
        warmup=0.0,
        final_lr=0.0,
        last_epoch=-1,
        verbose=False,
    ):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0.0 <= warmup < 1:
            raise ValueError("Invalid value for warmup")
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings

            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [
            warmup * max(self.final_lr, base_lr * lr_mult) for base_lr in self.base_lrs
        ]
