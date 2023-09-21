import os

from pytorch_lightning import Callback


class PreemtibleBaseHandler(Callback):
    def __init__(self):
        pass

    @property
    def is_preemtible(self):
        return os.getenv("ARNOLD_MONITOR_PREEMPTABLE", False)

    @property
    def ckpt_trial_ids(self):
        return os.getenv("ARNOLD_CKPT_TRIAL_ID", [])

    def get_last_ckpt_trial_id(self):
        trial_ids = self.ckpt_trial_ids
        if len(trial_ids):
            return trial_ids[-1]
        return None


class PreemtibleCheckpointHandler(PreemtibleBaseHandler):
    def get_last_ckpt(self, ckpt_path: str):
        breakpoint()

    # def on_load_checkpoint(
    #     self,
    #     trainer: "pl.Trainer",
    #     pl_module: "pl.LightningModule",
    #     checkpoint: Dict[str, Any],
    # ) -> None:
    #     return super().on_load_checkpoint(trainer, pl_module, checkpoint)
