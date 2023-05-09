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
