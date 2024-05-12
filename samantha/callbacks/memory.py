import time
from typing import Union

import torch
from pytorch_lightning import Callback
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info


class CUDACallback(Callback):
    """
    This callback tracks the memory usage and running time for a single epoch.
    """

    def _reduce(self, trainer, value: Union[int, float], reduce_op="mean"):
        # trainer.strategy.reduce only accept tensor
        tensor = torch.Tensor([value]).cuda()
        reduced_tensor = trainer.strategy.reduce(tensor, reduce_op=reduce_op)
        return reduced_tensor[0]

    def on_train_epoch_start(self, trainer, pl_module):
        if isinstance(trainer.strategy, DeepSpeedStrategy):
            # Reset the memory use counter
            torch.cuda.reset_peak_memory_stats(self._root_cpu(trainer))
            torch.cuda.synchronize(self._root_cpu(trainer))
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if isinstance(trainer.strategy, DeepSpeedStrategy):
            torch.cuda.synchronize(self._root_cpu(trainer))
            max_memory = (
                torch.cuda.max_memory_allocated(self._root_cpu(trainer)) / 2**20
            )
            max_memory = self._reduce(trainer, max_memory)
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")

        epoch_time = time.time() - self.start_time
        epoch_time = self._reduce(trainer, epoch_time)
        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")

    def _root_cpu(self, trainer):
        return trainer.strategy.root_device.index


class MemoryCallback(Callback):

    MB = 1024**2

    def on_train_batch_end(self, trainer, pl_module, *_) -> None:
        max_memory_allocated = torch.cuda.max_memory_allocated() / self.MB
        max_memory_reserved = torch.cuda.max_memory_reserved() / self.MB

        pl_module.log_dict(
            {
                "gpu/max_memory_allocated": max_memory_allocated,
                "gpu/max_memory_reserved": max_memory_reserved,
            },
            prog_bar=False,
            rank_zero_only=True,
        )

    def on_before_backward(self, trainer, pl_module, *_):
        memory_allocated = torch.cuda.memory_allocated() / self.MB
        pl_module.log_dict(
            {"gpu/before_bwd_memory_allocated": memory_allocated},
            prog_bar=False,
            rank_zero_only=True,
        )

    def on_before_optimizer_step(self, trainer, pl_module, *_):

        memory_allocated = torch.cuda.memory_allocated() / self.MB
        pl_module.log_dict(
            {"gpu/before_opt_memory_allocated": memory_allocated},
            prog_bar=False,
            rank_zero_only=True,
        )
