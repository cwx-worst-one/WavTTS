import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class AcousticModelDemo(pl.Callback):
    def __init__(self, sample_rate: int, temperature: float, max_demos_per_step=2):
        super().__init__()
        self.sample_rate = sample_rate
        self.temperature = temperature
        self.max_demos_per_step = max_demos_per_step
        self.completed_demos = {}

    @rank_zero_only
    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.global_step == 0:
            return

        if module.global_step not in self.completed_demos:
            self.completed_demos[module.global_step] = 0

        if self.completed_demos[module.global_step] >= self.max_demos_per_step:
            return

        audio = batch[0]
        audio = audio[: self.max_demos_per_step]

        sampled_token_ids = module.sample_with_audio_conditioning(
            audio, temperature=self.temperature
        )

        with torch.no_grad():
            module.audio_model.eval()
            modeled_quantizers = module.input_quantizers
            modeled_token_labels = module.audio_model(audio)
            modeled_token_labels[:, modeled_quantizers] = sampled_token_ids

            decoded_modeled_preds = module.audio_model.decode(modeled_token_labels)

        self.completed_demos[module.global_step] += audio.shape[0]
        demo_id = self.completed_demos[module.global_step]
        print(f"Generating demo {demo_id} for step {module.global_step}")

        for idx, (a, modeled) in enumerate(zip(audio, decoded_modeled_preds)):
            module.logger.experiment.add_audio(
                f"validation/target_audio-{idx}",
                a,
                module.global_step,
                sample_rate=self.sample_rate,
            )
            module.logger.experiment.add_audio(
                f"validation/coarse_fine_sampled_audio-{idx}",
                modeled,
                module.global_step,
                sample_rate=self.sample_rate,
            )


class SingSongDemo(AcousticModelDemo):
    def __init__(self, use_cond=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cond = use_cond

    @rank_zero_only
    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.global_step == 0:
            return

        if module.global_step not in self.completed_demos:
            self.completed_demos[module.global_step] = 0

        if self.completed_demos[module.global_step] >= self.max_demos_per_step:
            return

        source_audio = batch[0]
        target_audio = batch[1]
        if self.use_cond:
            cond = batch[-1]
        else:
            cond = None
        pred_target_audio, _ = module.sample_with_audio_conditioning(
            source_audio, temperature=self.temperature, cond=cond
        )

        ground_truth = (source_audio + target_audio) / 2
        pred_mix = (source_audio + pred_target_audio) / 2

        self.completed_demos[module.global_step] += pred_mix.shape[0]
        demo_id = self.completed_demos[module.global_step]
        print(f"Generating demo {demo_id} for step {module.global_step}")

        for idx, (gt, pm, pta) in enumerate(
            zip(ground_truth, pred_mix, pred_target_audio)
        ):
            module.logger.experiment.add_audio(
                f"validation/target_audio_mix-{idx}",
                gt,
                module.global_step,
                sample_rate=self.sample_rate,
            )
            module.logger.experiment.add_audio(
                f"validation/coarse_sampled_mix-{idx}",
                pm,
                module.global_step,
                sample_rate=self.sample_rate,
            )
