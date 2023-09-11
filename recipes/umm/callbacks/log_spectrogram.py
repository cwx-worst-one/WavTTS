import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from PIL import Image

from samantha.transforms.audio import plot_spectrogram


class LogSpectrogram(pl.Callback):
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
            log_dict = pl_module.get_spec(batch)

            fig, ax = plt.subplots(
                4, self.n_examples, figsize=(10 * self.n_examples, 10)
            )

            for i, (k, v) in enumerate(log_dict["mel"].items()):
                for mel, a in zip(v[:self.n_examples], ax[i]):
                    plot_spectrogram(mel.cpu(), plot_log=False, mel=True, title=k, ax=a)
            for i, (k, v) in enumerate(log_dict["chroma"].items()):
                for chroma, a in zip(v[:self.n_examples], ax[i + 2]):
                    plot_spectrogram(
                        chroma.cpu(), plot_log=False, mel=True, title=k, ax=a
                    )
            plt.tight_layout()
            fig.canvas.draw()
            pil_image = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )

            pl_module.logger.experiment.log(
                {
                    f"val_{dataloader_idx}": wandb.Image(
                        pil_image, caption=f"step-{pl_module.global_step:>06}"
                    )
                }
            )
            plt.close(fig)
