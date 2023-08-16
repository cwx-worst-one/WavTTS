import pytorch_lightning as pl
import torch
import wandb


class LogAudio(pl.Callback):
    def __init__(self, n_examples: int = 4):
        super().__init__()
        self.n_examples = n_examples

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            batch_idxs = [i for i in range(self.n_examples)]
            model_output = pl_module.get_audio(batch)
            pred_audio, target_audio = (
                model_output["Reconstructed"],
                model_output["Original"],
            )
            pred_audio = pred_audio[batch_idxs].detach().cpu()
            target_audio = target_audio[batch_idxs].detach().cpu()

            for i, (pred, target) in enumerate(zip(pred_audio, target_audio)):
                pl_module.loggers[0].experiment.add_audio(
                    f"Reconstructed/item-{i}",
                    pred,
                    pl_module.global_step,
                    pl_module.extra_params.sample_rate,
                )
                pl_module.loggers[0].experiment.add_audio(
                    f"Original/item-{i}",
                    target,
                    pl_module.global_step,
                    pl_module.extra_params.sample_rate,
                )
                if len(pl_module.loggers) > 1:
                    pl_module.loggers[1].experiment.log(
                        {
                            "Reconstructed": wandb.Audio(
                                data_or_path=pred,
                                sample_rate=pl_module.extra_params.sample_rate,
                                caption=f"step-{pl_module.global_step:>06}-item-{i}",
                            ),
                            "Original": wandb.Audio(
                                data_or_path=target,
                                sample_rate=pl_module.extra_params.sample_rate,
                                caption=f"step-{pl_module.global_step:>06}-item-{i}",
                            ),
                        }
                    )
