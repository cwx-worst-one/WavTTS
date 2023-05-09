import pytorch_lightning as pl
from arnold_profiler.switcher import ArnoldTorchProfiler


class ArnoldProfiler(pl.Callback):
    def __init__(self, prof_start_step=20, prof_steps=10):
        super().__init__()
        self.profiler = ArnoldTorchProfiler(prof_start_step, prof_steps)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self.profiler.check_and_profile()
