from time import perf_counter

from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class StepTimeLogger(Callback):
    r"""Logs the time taken to complete a training step.

    Record the time taken to:
    1. load the data
    2. compute the forward pass
    3. compute the backward pass
    4. optimizer step
    """

    def __init__(self):
        self.batch_start_time = perf_counter()
        self.after_data_load_time = perf_counter()
        self.after_forward_time = perf_counter()
        self.after_backward_time = perf_counter()

    @rank_zero_only
    def on_train_batch_start(self, *args, **kwargs):
        self.after_data_load_time = perf_counter()

    @rank_zero_only
    def on_before_backward(self, *args, **kwargs):
        self.after_forward_time = perf_counter()

    @rank_zero_only
    def on_before_optimizer_step(self, *args, **kwargs):
        self.after_backward_time = perf_counter()

    @rank_zero_only
    def on_train_batch_end(self, *args, **kwargs):
        next_batch_start_time = perf_counter()
        time_dict = {
            "data_load_time": self.after_data_load_time - self.batch_start_time,
            "forward_time": self.after_forward_time - self.after_data_load_time,
            "backward_time": self.after_backward_time - self.after_forward_time,
            "optimizer_step_time": next_batch_start_time - self.after_backward_time,
        }
        self.log_dict(time_dict, prog_bar=True)
        self.batch_start_time = next_batch_start_time
