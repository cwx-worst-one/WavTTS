import random

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from samantha.transforms.audio import plot_spectrogram


class LogMelSpectrogram(pl.Callback):
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
            batch_idxs = [0, 1, 2, 3]  # TODO sample with replacement
            fig, ax = plt.subplots(
                2, self.n_examples, figsize=(20 * self.n_examples, 20)
            )

            pred_mel, target_mel = pl_module.get_mel(batch)
            pred_mel = pred_mel[batch_idxs]
            target_mel = target_mel[batch_idxs]

            for mel, a in zip(pred_mel, ax[0]):
                plot_spectrogram(
                    mel.cpu(), plot_log=False, mel=True, title="Reconstructed", ax=a
                )

            for mel, a in zip(target_mel, ax[1]):
                plot_spectrogram(
                    mel.cpu(), plot_log=False, mel=True, title="Target", ax=a
                )

            plt.tight_layout()
            pl_module.logger.experiment.add_figure(
                f"val_{dataloader_idx}/mel", fig, global_step=pl_module.global_step
            )


class LogMaskedMelSpectrogram(pl.Callback):
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
            log_dict = pl_module.get_mel(batch)
            batch_idxs = [
                i for i in range(self.n_examples)
            ]  # TODO sample with replacement
            fig, ax = plt.subplots(
                len(log_dict), self.n_examples, figsize=(20 * self.n_examples, 20)
            )

            for i, (k, v) in enumerate(log_dict.items()):
                for mel, a in zip(v[batch_idxs], ax[i]):
                    plot_spectrogram(mel.cpu(), plot_log=False, mel=True, title=k, ax=a)

            plt.tight_layout()
            pl_module.logger.experiment.add_figure(
                f"val_{dataloader_idx}/mel", fig, global_step=pl_module.global_step
            )


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
            batch_idxs = [i for i in range(self.n_examples)]
            fig, ax = plt.subplots(
                4, self.n_examples, figsize=(20 * self.n_examples, 20)
            )

            for i, (k, v) in enumerate(log_dict["mel"].items()):
                for mel, a in zip(v[batch_idxs], ax[i]):
                    plot_spectrogram(mel.cpu(), plot_log=False, mel=True, title=k, ax=a)
            for i, (k, v) in enumerate(log_dict["chroma"].items()):
                for chroma, a in zip(v[batch_idxs], ax[i + 2]):
                    plot_spectrogram(
                        chroma.cpu(), plot_log=False, mel=True, title=k, ax=a
                    )

            plt.tight_layout()
            pl_module.logger.experiment.add_figure(
                f"val_{dataloader_idx}/spec", fig, global_step=pl_module.global_step
            )
