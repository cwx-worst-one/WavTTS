import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from recipes.umm.modules.lit_module import SoundStorm


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


class LogGTAudio(pl.Callback):
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
            audio = batch["audio"][batch_idxs].detach().cpu()

            for i, (target) in enumerate(audio):
                pl_module.logger.experiment.add_audio(
                    f"val_{dataloader_idx}/Original-{i}",
                    target,
                    pl_module.global_step,
                    pl_module.extra_params.sample_rate,
                )


class AudioDemo(pl.Callback):
    def __init__(self, max_demos_per_step: int):
        super().__init__()
        self.max_demos_per_step = max_demos_per_step
        self.completed_demos = {}

    @rank_zero_only
    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer,
        pl_module: SoundStorm,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if pl_module.global_step not in self.completed_demos:
            self.completed_demos[pl_module.global_step] = 0

        if self.completed_demos[pl_module.global_step] >= self.max_demos_per_step:
            return

        input_dict = pl_module.prepare_feature(batch)
        audio = batch["audio"].squeeze(1)[: self.max_demos_per_step]
        semantic_tokens = input_dict["semantic_tokens"][: self.max_demos_per_step]

        num_iterations = [48, 32, 24, 16, 8, 4, 2, 2, 1, 1, 1, 1]
        score_strategies = [
            "random",
            "random",
            "random",
            "random",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
            "maskgit",
        ]
        temperatures = [1.0, 1.0, 0.95, 0.95, 0.9, 0.9, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4]

        sampled_audio_tokens = pl_module.sample(
            semantic_tokens=semantic_tokens,
            num_iterations=num_iterations,
            score_strategies=score_strategies,
            temperatures=temperatures,
        )

        sampled_audio = pl_module.token2audio(sampled_audio_tokens)

        self.log_audio(pl_module, sampled_audio.cpu(), audio.cpu())

    def log_audio(
        self,
        pl_module: pl.LightningModule,
        sampled_audio: torch.Tensor,
        audio: torch.Tensor,
    ) -> None:
        self.completed_demos[pl_module.global_step] += audio.shape[0]
        demo_id = self.completed_demos[pl_module.global_step]
        print(f"Generating demo {demo_id} for step {pl_module.global_step}")

        for i, (pred, target) in enumerate(zip(sampled_audio, audio)):
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
