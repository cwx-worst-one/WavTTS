import json
import logging

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


class UnusedParamsFinder(Callback):
    r"""Find unused model parameters during training."""

    def __init__(self, warning_times=1) -> None:
        super().__init__()
        self.warning_times = warning_times

    @rank_zero_only
    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.should_warn():
            unused_params = [
                name for name, p in pl_module.named_parameters() if p.grad is None
            ]

            logger.warning(
                "The following parameters are not participate model training, please"
                f" make sure it's on purpose. \n{json.dumps(unused_params, indent=2)}"
            )
            self.warning_times -= 1

    @rank_zero_only
    def should_warn(self):
        return self.warning_times > 0
