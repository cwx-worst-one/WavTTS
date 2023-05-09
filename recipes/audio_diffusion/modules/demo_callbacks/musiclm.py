import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.utilities.distributed import rank_zero_only


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

    @rank_zero_only
    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.demo_controller["global_step"] != module.global_step:
            # Reset once a new validation epoch
            self.demo_controller["counter"] = 0
            self.demo_controller["global_step"] = module.global_step
        # Skip if the max number of demo reached in this validation epoch
        if self.demo_controller["counter"] > self.max_demos_per_step:
            return

        self.completed_demos[module.global_step] += 1
        demo_id = self.completed_demos[module.global_step]
        print(f"Generating demo {demo_id} for step {module.global_step}")

        audio, cond_embeddings = batch
        assert audio.shape[0] == 1, "Only support valid batch size 1 for now"
        cond_embeddings = cond_embeddings.unsqueeze(dim=1)

        encoder = self.encoder.to(module.device)
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
            )

        reconstructed = encoder.decode(encoded_refs)
        reconstructed = (
            rearrange(reconstructed, "b d n -> d (b n)") / reconstructed.abs().max()
        )
        module.logger.experiment.add_audio(
            f"reconstructed/{module.global_step}_{demo_id}",
            reconstructed,
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
                f"demo/{module.global_step}_{demo_id}.{i + 1}",
                demo,
                module.global_step,
                sample_rate=self.sample_rate,
            )
