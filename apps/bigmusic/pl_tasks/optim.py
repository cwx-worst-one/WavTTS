import math
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup_lrdecay(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_decay_rate: float = 0.87,
    last_epoch: int = -1,
    lr_end: float = 1e-7,
):
    # breakpoint()
    initial_lr = optimizer.param_groups[0]['lr']

    def lr_lambda(current_step):
        lr_decay_steps = int(lr_decay_rate * num_training_steps)

        if current_step < num_warmup_steps:
            # warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        if current_step >= lr_decay_steps:
            return lr_end / initial_lr

        # cosine decay
        decay_ratio = (current_step - num_warmup_steps) / (lr_decay_steps - num_warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr_end / initial_lr + coeff * (1 - lr_end / initial_lr)
    return LambdaLR(optimizer, lr_lambda, last_epoch)
