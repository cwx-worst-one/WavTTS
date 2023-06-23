from time import perf_counter

from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class Timer(Callback):

    r"""Logs the time taken to complete a training step.
    Record the time taken to:
    1. load the data
    2. compute the forward pass
    3. compute the backward pass
    4. optimizer step
    """

    @rank_zero_only
    def on_train_epoch_start(self, *_, **__):
        self.reset_timer()

    @rank_zero_only
    def on_train_batch_start(self, *_, **__):
        self.update_timer("data")

    @rank_zero_only
    def on_before_backward(self, *_, **__):
        self.update_timer("fwd")

    @rank_zero_only
    def on_before_optimizer_step(self, *_, **__):
        self.update_timer("bwd")

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *_, **__):
        self.update_timer("opt")

        current_step, interval = trainer.global_step, trainer.log_every_n_steps
        if self.last_step == current_step or current_step % interval != 0:
            return

        self.last_step = current_step
        self.log_timer(pl_module, interval)

    @rank_zero_only
    def log_timer(self, pl_module, interval=1.0):
        record = {k: v / interval for k, v in self.record.items()}
        time_dict = {
            "timer/data": round(record["data"], 3),
            "timer/fwd": round(record["fwd"], 3),
            "timer/bwd": round(record["bwd"], 3),
            "timer/opt": round(record["opt"], 3),
        }

        pl_module.log_dict(time_dict, prog_bar=True)
        self.reset_timer()

    def reset_timer(self):
        self.tik = perf_counter()
        self.last_step = -1
        self.record = {"data": 0, "fwd": 0, "bwd": 0, "opt": 0}

    def update_timer(self, stage):
        tok = perf_counter()
        self.record[stage] += tok - self.tik
        self.tik = tok
