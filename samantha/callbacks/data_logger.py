import logging
import os
from abc import abstractmethod
from random import choices
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


class DataTracker:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.dataframe = []

    def export(self, global_rank: int, iter_idx: int):
        df = pd.DataFrame(self.dataframe)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        fp = os.path.join(
            self.save_dir, f"DataCheck-rank-{global_rank}-step-{iter_idx}.p"
        )
        df.to_pickle(fp)

    def __call__(
        self,
        data: List[Dict[str, Any]],
        iter_idx: int,
        data_split: str,
        global_step: int,
        global_rank: int,
    ):
        for idx in range(len(data)):
            data[idx]["iter_idx"] = iter_idx
            data[idx]["data_split"] = data_split
            data[idx]["global_step"] = global_step
            data[idx]["global_rank"] = global_rank
        self.dataframe.extend(data)


class DataUsageLogger(pl.Callback):
    def __init__(self, save_dir: str, save_every_n_iter: int = 10000):
        trial_id = os.getenv("ARNOLD_TRIAL_ID", "merlin")
        self.save_dir = os.path.join(save_dir, trial_id)
        self.save_every_n_iter = save_every_n_iter
        self.data_tracker = DataTracker(self.save_dir)

    @abstractmethod
    def get_batch_data(self, batch: Any) -> Dict[str, Any]:
        pass

    @staticmethod
    def reorder_batch(
        batch: Dict[str, Union[List[Any], torch.Tensor]]
    ) -> List[Dict[str, Any]]:
        batch_size = len(batch[list(batch.keys())[0]])
        d = [{} for _ in range(batch_size)]
        for k in batch.keys():
            for idx in range(len(batch[k])):
                if len(batch[k]):
                    data = batch[k][idx]
                    if type(data) is torch.Tensor:
                        data = data.cpu()
                    d[idx][k] = data
        return d

    def log_data(self, pl_module, batch: Any, batch_idx: int, data_split: str) -> None:
        batch = self.get_batch_data(batch)
        data = self.reorder_batch(batch)
        self.data_tracker(
            data,
            iter_idx=batch_idx,
            data_split=data_split,
            global_step=pl_module.global_step,
            global_rank=pl_module.global_rank,
        )

        if batch_idx % self.save_every_n_iter == 0:
            self.data_tracker.export(pl_module.global_rank, batch_idx)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_data(pl_module, batch, batch_idx, "train")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_data(pl_module, batch, batch_idx, "validation")

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_data(pl_module, batch, batch_idx, "predict")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_data(pl_module, batch, batch_idx, "test")


class AudioDataLogger(pl.Callback):
    def __init__(
        self,
        sample_rate: int,
        check_interval: int,
        examples_per_batch: int,
        save_dir: Optional[str] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.check_interval = check_interval
        self.examples_per_batch = examples_per_batch
        self.save_dir = save_dir

    @abstractmethod
    def save_hook(self, pl_module, batch, batch_idx: int, data_tag: str) -> None:
        pass

    @abstractmethod
    def get_audios_from_batch(self, batch: Any) -> torch.Tensor:
        pass

    def log_audio(self, pl_module, batch, batch_idx: int, data_split: str):
        if batch_idx % self.check_interval == 0:
            audios = self.get_audios_from_batch(batch)
            audios = audios.cpu()
            batch_size = audios.shape[0]
            batch_idxs = choices(
                range(batch_size),
                k=(
                    self.examples_per_batch
                    if self.examples_per_batch < batch_size
                    else batch_size
                ),
            )

            for idx in batch_idxs:
                audio = audios[idx]

                data_tag = f"{data_split}/data_batch_{batch_idx}/{idx}"

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

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_audio(pl_module, batch, batch_idx, "train")

    @rank_zero_only
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_audio(pl_module, batch, batch_idx, "validation")

    @rank_zero_only
    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_audio(pl_module, batch, batch_idx, "predict")

    @rank_zero_only
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_audio(pl_module, batch, batch_idx, "test")
