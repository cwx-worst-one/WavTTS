import lightning.pytorch as pl

from samantha.byteformers import GradientNoiseScale


class GradientNoiseScaleLogger(pl.Callback):
    def __init__(
        self,
        model,
        batch_size_small,
        n_batches,
        beta: float = 0.99,
        fdsp_strategy: bool = False,
        cpu_offload: bool = False,
        is_pipe_parallel: bool = False,
    ):
        self.gns = GradientNoiseScale(
            model,
            batch_size_small,
            n_batches,
            beta=beta,
            fdsp_strategy=fdsp_strategy,
            cpu_offload=cpu_offload,
            is_pipe_parallel=is_pipe_parallel,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.gns.update()
        if (
            trainer.global_step
            and trainer.global_step % self.gns.n_batches == 0
            and self.gns.noise_scale is not None
        ):
            pl_module.log("gradient_noise_scale", self.gns.noise_scale)
