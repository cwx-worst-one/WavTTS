from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar


class TQDMProgress(TQDMProgressBar):
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        super().on_train_epoch_start(trainer=trainer)
        self.train_progress_bar.initial = trainer.fit_loop.epoch_loop.batch_idx
