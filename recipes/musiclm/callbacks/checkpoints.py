
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint

class ModelCheckpointSaveStep0(ModelCheckpoint):
    def _should_skip_saving_checkpoint(self, trainer) -> bool:
        return True

    @rank_zero_only
    def on_train_start(self, trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        ckpt_file_path = self.format_checkpoint_name({}, 'step=0')
        self._save_checkpoint(trainer, ckpt_file_path)
