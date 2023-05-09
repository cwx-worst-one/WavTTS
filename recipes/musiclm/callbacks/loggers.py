import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class LogTotalTokensCallback(pl.Callback):
    def __init__(self) -> None:
        self.total_tokens = 0

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        world_size = trainer.world_size
        batch_size = trainer.datamodule.batch_size
        n_tokens = pl_module.max_flattened_seq_len
        effective_batch_size_tokens = world_size * batch_size * n_tokens

        self.total_tokens += effective_batch_size_tokens

        self.log("tokens", float(self.total_tokens))


class LogGradientScalerCallback(pl.Callback):
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            pl_module.global_step
            and pl_module.global_step % trainer.log_every_n_steps == 0
        ):
            precision_plugin = trainer.strategy.precision_plugin
            if precision_plugin.scaler is not None:
                self.log("gradient_scale", precision_plugin.scaler.get_scale())
