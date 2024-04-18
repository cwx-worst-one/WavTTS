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
    def on_train_batch_end(self, trainer, pl_module, *_) -> None:
        MB = 1024**2
        max_memory_allocated = torch.cuda.max_memory_allocated() / MB
        max_memory_reserved = torch.cuda.max_memory_reserved() / MB
        memory_allocated = torch.cuda.memory_allocated() / MB
        memory_reserved = torch.cuda.memory_reserved() / MB

        pl_module.log_dict(
            {
                "gpu/max_memory_allocated": max_memory_allocated,
                "gpu/max_memory_reserved": max_memory_reserved,
                "gpu/memory_allocated": memory_allocated,
                "gpu/memory_reserved": memory_reserved,
            },
            prog_bar=False,
            rank_zero_only=True,
        )
