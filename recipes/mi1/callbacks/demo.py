import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from recipes.mi1.scripts.inference import DiffusionModels

from recipes.mi1.models.mi1 import MI1Input, MI1

class AudioDemo(Callback):

    @rank_zero_only
    def __init__(self):
        self.diffusion_models = DiffusionModels()
        self.diffusion_models = self.diffusion_models.to("cuda")

    @rank_zero_only
    def on_validation_start(self, trainer: Trainer, pl_module: MI1) -> None:
        batch_size = 2
        # style_text = "A country song with female vocal."
        lyrics = """Imagine there's no heaven.
        It's easy if you try.
        No hell below us.
        Above us, only sky.
        Imagine all the people. 
        Livin' for today.
        oh oh oh oh oh.
        """

        inputs = MI1Input(
            lyrics=[lyrics] * batch_size,
        )

        audio_tokens = pl_module.generate(inputs, precision=torch.float16)

        audio = self.diffusion_models.tokens_to_audio(audio_tokens)

        for idx in range(len(audio)):
            pl_module.loggers[0].experiment.log(
                {
                    "Demo": wandb.Audio(
                        data_or_path=audio[idx].cpu(),
                        sample_rate=pl_module.config.sample_rate,
                        caption=f"step-{pl_module.global_step:>06}-item-{idx}",
                    ),
                }
            )