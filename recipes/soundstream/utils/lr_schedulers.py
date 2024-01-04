def lr_lambda(current_step, warmup_steps, decay_factor, decay_interval):

    if warmup_steps and current_step < warmup_steps:
        # Linear warm-up
        return float(current_step) / float(warmup_steps)
    else:
        # Exponential decay
        # return decay_factor ** (min(3, current_step // decay_interval))
        return decay_factor ** (current_step // decay_interval)
