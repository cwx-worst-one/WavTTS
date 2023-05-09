import os

import pytorch_lightning as pl
import torch
import torchaudio
from einops import rearrange
from frechet_audio_distance import FrechetAudioDistance
from pytorch_lightning.utilities.distributed import rank_zero_only

# TODO: fix decoding


class GenerateDemo(pl.Callback):
    def __init__(
        self,
        num_demos,
        sample_rate,
        sample_length,
        encoder,
        start_channel,
        end_channel,
        max_demos_per_step=2,
        temperature=1.0,
    ):
        super().__init__()
        self.num_demos = num_demos
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.encoder = encoder
        self.start_channel = start_channel
        self.end_channel = end_channel
        self.max_demos_per_step = max_demos_per_step
        self.temperature = temperature
        self.completed_demos = {}
        self.reconstructed = set()

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

        semantic_acc = None
        encoder = self.encoder.to(module.device)
        if module.acoustic_token_model.use_t5:
            if module.acoustic_token_model.use_semantic_tokens:
                audio, audio_vocal, semantic_acc, _ = batch

                semantic_acc = torch.tile(semantic_acc, (self.num_demos, 1))
            else:
                audio, audio_vocal = batch
            vocal_encoded_refs = encoder.quantize(encoder.encode(audio_vocal))
            cond_embeddings = torch.tile(vocal_encoded_refs, (self.num_demos, 1, 1))
        else:
            audio, cond_embeddings = batch
            cond_embeddings = cond_embeddings.unsqueeze(dim=1)
        assert audio.shape[0] == 1, "Only support valid batch size 1 for now"

        with torch.no_grad():
            # 1 x num_channels x seq_len
            encoded_refs = encoder.quantize(encoder.encode(audio))
            # num_demos x num_channels x seq_len
            encoded_demos = module.generate_samples(
                cond_embeddings=cond_embeddings,
                labels=encoded_refs,
                temperature=self.temperature,
                num_outputs=self.num_demos,
                sequence_length=self.sample_length,
                cond_semantic=semantic_acc,
            )

        if demo_id not in self.reconstructed:
            reconstructed = encoder.decode(encoded_refs)
            reconstructed = (
                rearrange(reconstructed, "b d n -> d (b n)") / reconstructed.abs().max()
            )
            module.logger.experiment.add_audio(
                f"validation/reconstructed_audio_{module.global_step}_{demo_id}",
                reconstructed,
                module.global_step,
                sample_rate=self.sample_rate,
            )
            self.reconstructed.add(demo_id)

            if module.acoustic_token_model.use_t5:  # For SingSong, save vocal audio
                vocal_reconstructed = encoder.decode(vocal_encoded_refs)
                vocal_reconstructed = (
                    rearrange(vocal_reconstructed, "b d n -> d (b n)")
                    / vocal_reconstructed.abs().max()
                )
                fp = "validation/reconstructed_vocal_audio_"
                fp += f"{module.global_step}_{demo_id}"
                module.logger.experiment.add_audio(
                    fp,
                    vocal_reconstructed,
                    module.global_step,
                    sample_rate=self.sample_rate,
                )

                remixed_reconstructed = reconstructed + audio_vocal
                remixed_reconstructed = (
                    remixed_reconstructed / remixed_reconstructed.abs().max()
                )

                fp = "validation/reconstructed_audio_"
                fp += f"{module.global_step}_{demo_id}_REMIXED"
                module.logger.experiment.add_audio(
                    fp,
                    remixed_reconstructed,
                    module.global_step,
                    sample_rate=self.sample_rate,
                )

        for i in range(encoded_demos.shape[0]):
            # Reuse encoder_refs here
            encoded_refs[:, self.start_channel : self.end_channel, :] = encoded_demos[
                i : i + 1, ...
            ]
            demo = encoder.decode(encoded_refs)
            demo = rearrange(demo, "b d n -> d (b n)") / demo.abs().max()
            module.logger.experiment.add_audio(
                f"validation/sampled_audio_{module.global_step}_{demo_id}.{i + 1}",
                demo,
                module.global_step,
                sample_rate=self.sample_rate,
            )

            if module.acoustic_token_model.use_t5:  # For singsong, report remixed
                remixed = demo + audio_vocal
                remixed = remixed / remixed.abs().max()
                fp = "validation/sampled_audio_"
                fp += f"{module.global_step}_{demo_id}.{i + 1}_REMIXED"
                module.logger.experiment.add_audio(
                    fp, remixed, module.global_step, sample_rate=self.sample_rate
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
