from torch.optim.lr_scheduler import _LRScheduler

class DinoLRScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_steps, hold_steps, decay_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps
        super(DinoLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1
        if current_step < self.warmup_steps:
            lr = current_step / self.warmup_steps * self.base_lr
        elif current_step < self.warmup_steps + self.hold_steps:
            lr = self.base_lr
        else:
            gamma = 0.1 ** (1 / self.decay_steps)
            lr = self.base_lr * gamma ** (current_step - self.warmup_steps - self.hold_steps)

        return [lr for _ in self.base_lrs]

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        print("do not update!")
        # self.__dict__.update(state_dict)
