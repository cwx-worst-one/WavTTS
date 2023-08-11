import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from recipes.soundstorm2.lightning.soundstorm import SoundStorm


class AudioDemo(pl.Callback):
    def __init__(self, sample_rate: int, max_demos_per_step: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_demos_per_step = max_demos_per_step
        self.completed_demos = {}

    @rank_zero_only
    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, module: SoundStorm, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.global_step == 0:
            return

        if module.global_step not in self.completed_demos:
            self.completed_demos[module.global_step] = 0

        if self.completed_demos[module.global_step] >= self.max_demos_per_step:
            return

        inputs = module.prepare_inputs(batch)
        audio = inputs["audio"][: self.max_demos_per_step]
        audio_tokens = inputs["audio_tokens"][: self.max_demos_per_step]

        if module.semantic_model is None:
            semantic_tokens = None
            seed_tokens = audio_tokens
            sampled_t = torch.randint(50, 100, (1,), device=module.device)
        else:
            semantic_tokens = inputs["semantic_tokens"][: self.max_demos_per_step]
            seed_tokens = None
            sampled_t = None

            # TODO: only batch size 1 works for now
            semantic_tokens = semantic_tokens[:1]

        iterations = [48, 32, 24, 16, 8, 4, 2, 2, 1, 1, 1, 1]
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
        guidance_scale = None
        temperatures = [1.0, 1.0, 0.95, 0.95, 0.9, 0.9, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4]

        sampled_audio_tokens, _ = module.iterative_decoding(
            semantic_tokens=semantic_tokens,
            max_seq_len=audio_tokens.shape[2],
            iterations=iterations,
            score_strategies=score_strategies,
            guidance_scale=guidance_scale,
            temperatures=temperatures,
            sampled_t=sampled_t,
            seed_tokens=seed_tokens,
            prefix_tokens=None,
        )

        with torch.no_grad():
            sampled_audio = module.audio_model.decode(sampled_audio_tokens)

        self.log_audio(module, sampled_audio, audio)

    def log_audio(
        self,
        module: pl.LightningModule,
        sampled_audio: torch.Tensor,
        audio: torch.Tensor,
    ) -> None:
        self.completed_demos[module.global_step] += audio.shape[0]
        demo_id = self.completed_demos[module.global_step]
        print(f"Generating demo {demo_id} for step {module.global_step}")

        for idx, (a, sampled) in enumerate(zip(audio, sampled_audio)):
            a = a.mean(dim=0, keepdim=True)
            sampled = sampled.mean(dim=0, keepdim=True)
            module.logger.experiment.add_audio(
                f"validation/target_audio-{idx}",
                a,
                module.global_step,
                sample_rate=self.sample_rate,
            )
            module.logger.experiment.add_audio(
                f"validation/sampled_audio-{idx}",
                sampled,
                module.global_step,
                sample_rate=self.sample_rate,
            )


# class FineAudioDemo(AudioDemo):
#     def __init__(self, keep_coarse_quant_idx: int, sample_rate: int, max_demos_per_step: int):
#         super().__init__(sample_rate, max_demos_per_step)
#         self.keep_coarse_quant_idx = keep_coarse_quant_idx

#     @rank_zero_only
#     @torch.no_grad()
#     def on_validation_batch_end(
#         self, trainer, module: SoundStorm, outputs, batch, batch_idx, dataloader_idx=0
#     ):
#         if trainer.global_step == 0:
#             return

#         if module.global_step not in self.completed_demos:
#             self.completed_demos[module.global_step] = 0

#         if self.completed_demos[module.global_step] >= self.max_demos_per_step:
#             return

#         _, batch_audio_tokens, batch_audio = module.prepare_inputs(batch)

#         for idx in range(self.max_demos_per_step):
#             audio = batch_audio[idx:idx+1]

#             # TODO: only batch size 1 works for now

#             iterations = [48, 32, 24, 16, 8, 4, 2, 2, 1, 1, 1, 1]
#             score_strategies = [
#                 "random",
#                 "random",
#                 "random",
#                 "random",
#                 "maskgit",
#                 "maskgit",
#                 "maskgit",
#                 "maskgit",
#                 "maskgit",
#                 "maskgit",
#                 "maskgit",
#                 "maskgit",
#             ]
#             temperatures = [1.0, 1.0, 0.95, 0.95, 0.9, 0.9, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4]

#             sampled_t = None

#             sampled_audio_tokens, _ = module.iterative_decoding(
#                 seed_tokens=seed_tokens,
#                 max_seq_len=seed_tokens.shape[2],
#                 iterations=iterations,
#                 score_strategies=score_strategies,
#                 temperatures=temperatures,
#                 sampled_t=sampled_t,
#             )

#             with torch.no_grad():
#                 sampled_audio = module.audio_model.decode(sampled_audio_tokens)

#             self.log_audio(module, sampled_audio, audio)
