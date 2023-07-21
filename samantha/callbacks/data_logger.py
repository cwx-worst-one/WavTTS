import logging
import os
from random import choices
from typing import Optional

import pytorch_lightning as pl
import torch
import torchaudio
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


class AudioDataLogger(pl.Callback):
    def __init__(
        self,
        sample_rate: int,
        n_examples: int,
        examples_per_batch: int,
        save_dir: Optional[str] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_examples = n_examples
        self.examples_per_batch = examples_per_batch
        self.save_dir = save_dir

    def save_hook(self, batch, batch_idx: int):
        pass

    def get_audios_from_batch(self, batch):
        audios = batch[0]
        return audios

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if pl_module.current_epoch == 0 and batch_idx < self.n_examples:
            audios = self.get_audios_from_batch(batch)
            audios = audios.cpu()
            batch_size = audios.shape[0]
            batch_idxs = choices(
                range(batch_size),
                k=self.examples_per_batch
                if self.examples_per_batch < batch_size
                else batch_size,
            )

            for idx in batch_idxs:
                audio = audios[idx]

                data_tag = f"data_batch_{batch_idx}/{idx}"

                logger.info(f"Adding {data_tag} to logger for preview")

                if audio.ndim == 1:
                    audio = audio.unsqueeze(dim=0)

                self.save_hook(pl_module, batch, idx, data_tag)

                audio_data_tag = data_tag + "_audio"
                pl_module.logger.experiment.add_audio(
                    audio_data_tag, audio, sample_rate=self.sample_rate
                )
                if self.save_dir is not None:
                    out_fp = os.path.join(
                        self.save_dir, audio_data_tag.replace("/", "-") + ".mp3"
                    )
                    torchaudio.save(out_fp, audio, self.sample_rate)
