import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from recipes.soundstorm.lightning.soundstorm import SoundStorm


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

        semantic_tokens, audio_tokens, audio = module.prepare_inputs(batch)
        audio = audio[: self.max_demos_per_step]
        semantic_tokens = semantic_tokens[: self.max_demos_per_step]

        # TODO: only batch size 1 works for now
        semantic_tokens = semantic_tokens[:1]

        iterations = [32, 32, 32, 32, 8, 8, 8, 8, 8, 8, 8, 8]
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

        sampled_t = None
        if module.hparams.audio_prompting:
            sampled_t = torch.randint(50, 100, (1,), device=module.device)

        sampled_audio_tokens, _ = module.iterative_decoding(
            semantic_tokens,
            max_seq_len=audio_tokens.shape[2],
            iterations=iterations,
            score_strategies=score_strategies,
            temperatures=[1.0] * len(iterations),
            sampled_t=sampled_t,
            guidance_scale=None,
        )

        with torch.no_grad():
            sampled_audio = module.audio_model.decode(sampled_audio_tokens)

        self.completed_demos[module.global_step] += audio.shape[0]
        demo_id = self.completed_demos[module.global_step]
        print(f"Generating demo {demo_id} for step {module.global_step}")

        for idx, (a, sampled) in enumerate(zip(audio, sampled_audio)):
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
