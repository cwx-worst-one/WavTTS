import os

import pytorch_lightning as pl
import torch
import torchaudio
from einops import rearrange
from frechet_audio_distance import FrechetAudioDistance
from pytorch_lightning.utilities.distributed import rank_zero_only


class GenerateDemo(pl.Callback):
    def __init__(
        self,
        num_demos,
        sample_rate,
        sample_length,
        encoder,
        max_demos_per_step=2,
        enable_vocal=False,
    ):
        super().__init__()
        self.num_demos = num_demos
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.encoder = encoder
        self.max_demos_per_step = max_demos_per_step
        self.completed_demos = {}
        self.reconstructed = set()
        self.enable_vocal = enable_vocal

    @rank_zero_only
    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, module, outputs, batch, batch_idx, dataloader_idx
    ):
        if module.global_step not in self.completed_demos:
            self.completed_demos[module.global_step] = 0
        if self.completed_demos[module.global_step] >= self.max_demos_per_step:
            return

        self.completed_demos[module.global_step] += 1
        demo_id = self.completed_demos[module.global_step]
        print(f"Generating demo {demo_id} for step {module.global_step}")

        cond_embeddings = batch[1]
        if self.num_demos > cond_embeddings.shape[0]:
            cond_embeddings = cond_embeddings.expand(self.num_demos, -1, -1)
        elif self.num_demos < cond_embeddings.shape[0]:
            cond_embeddings = cond_embeddings[: self.num_demos, ...]

        encoder = self.encoder.to(module.device)

        noise = torch.randn(
            [self.num_demos, module.model.num_signal_channels, self.sample_length],
            device=module.device,
        )
        demos = module.generate_samples(noise, cond_embeddings)
        demos = encoder.decode(demos)
        if self.enable_vocal:
            vocal = batch[2][0 : self.num_demos, ...]
            demos += vocal
        demos = (
            (rearrange(demos, "b d n -> d (b n)") / demos.abs().max()).detach().cpu()
        )

        module.logger.experiment.add_audio(
            f"validation/sampled_audio_{module.global_step}_{demo_id}",
            demos,
            module.global_step,
            sample_rate=self.sample_rate,
        )


class EvaluateDemo(pl.Callback):
    def __init__(
        self,
        eval_every_n_steps,
        num_demos,
        sample_rate,
        sample_length,
        stereo,
        fad_label_dir,
    ):
        super().__init__()
        self.eval_every_n_steps = eval_every_n_steps
        self.num_demos = num_demos
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.stereo = stereo
        self.demo_dir = "/tmp/audio-diffusion-demo"
        if not os.path.exists(self.demo_dir):
            os.makedirs(self.demo_dir)
        self.fad_label_dir = fad_label_dir

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (
            trainer.current_epoch == 0
            or (trainer.global_step - 1) % self.eval_every_n_steps != 0
        ):
            return

        num_channels = 2 if self.stereo else 1
        noise = torch.randn(
            [self.num_demos, num_channels, self.sample_length], device=module.device
        )
        demos = module.generate_samples(noise)
        demos = (demos / demos.abs().max()).detach().cpu()
        sub_step_dir = os.path.join(
            self.demo_dir, f"global_step_{trainer.global_step:08}"
        )
        os.makedirs(sub_step_dir, exist_ok=True)

        # Generate and save demo wav files
        for i in range(self.num_demos):
            torchaudio.save(
                os.path.join(sub_step_dir, f"demo_{i}.wav"), demos[i], self.sample_rate
            )
        frechet = FrechetAudioDistance(
            use_pca=False, use_activation=False, verbose=False
        )
        # Get FAD score
        fad_score = frechet.score(sub_step_dir, self.fad_label_dir)
        del frechet
        module.log("fad_score", fad_score, rank_zero_only=True)
