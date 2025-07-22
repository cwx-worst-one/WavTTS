import time
import samantha
from mariana.utils.audio.audio_logger import AudioLogger
logger = AudioLogger()

from pytorch_lightning import Callback


class DataLoaderTimer(Callback):
    def __init__(self, log_every_n_batches=100):
        super().__init__()
        self.batch_start_time = None
        self.batch_times = []
        self.log_every_n_batches = log_every_n_batches

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
         # This is *before* the batch is moved to the device.
        self.batch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.batch_start_time is not None:
            batch_time = time.perf_counter() - self.batch_start_time
            self.batch_times.append(batch_time)

            if (batch_idx + 1) % self.log_every_n_batches == 0:
                avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
                logger.info(f"Average batch processing time (last {self.log_every_n_batches} batches): {avg_batch_time:.4f}s")
                self.batch_times = []  # Reset the list