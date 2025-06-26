from typing import Any

import cruise as crs
from cruise.trainer.callback import Callback
from cruise.utilities.types import STEP_OUTPUT
from cruise.trainer.logger.tracking import TrackingLogger
from cruise.utilities.rank_zero import rank_zero_debug, rank_zero_warn, rank_zero_only
from io import BytesIO
import torch
import numpy as np
import torchaudio
import wandb as tk
from bytedance.easycycle import get_current_region
import bytedtos
import pickle
from recipes.bigmusic.utils.upload import upload_obj_to_tos
import os
from tasks.audio.audio_trainer import AudioTrainer



class SpeechDataCollectCallback(Callback):
    def __init__(self, keys_to_collect=None, audio_key="target_audio", audio_duration_key="duration", sample_rate=24000, audio_upload_to_tos=True, max_items_to_save=32, every_n_train_steps=100):
        super().__init__()
        keys_to_collect = keys_to_collect or []
        if "uttid" not in keys_to_collect:
            keys_to_collect = ["uttid"] + keys_to_collect
        if audio_key in keys_to_collect:
            keys_to_collect.remove(audio_key)
        self.keys_to_collect = keys_to_collect
        self.audio_key = audio_key
        self.audio_upload_to_tos = audio_upload_to_tos
        self.audio_duration_key = audio_duration_key
        self.sample_rate = sample_rate
        self.tracking_logger = None
        self.max_items_to_save = max_items_to_save
        self.every_n_train_steps = every_n_train_steps
        self.project = ""
        self.exp = ""
        self.version = ""

    def on_train_batch_end(
        self,
        trainer: "crs.CruiseTrainer",
        crs_module: "crs.CruiseModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        # TODO: sync dist
        if trainer.global_rank != rank_zero_only.rank_to_log:
            return
        if trainer.global_step % self.every_n_train_steps != 0:
            return
        if self.tracking_logger is None:
            if isinstance(trainer, AudioTrainer):
                logger_collection = trainer.cruise_logger
            else:
                logger_collection = trainer.logger
            for logger in logger_collection._logger_iterable:
                if isinstance(logger, TrackingLogger):
                    self.tracking_logger = logger
                    self.project = logger.project()
                    self.name = logger.name()
                    self.version = logger.version()
                    break
        if not isinstance(self.tracking_logger, TrackingLogger):
            rank_zero_warn(f"{self.__class__.__name__} only support TrackingLogger ({self.tracking_logger})")
            return    

        # infer batch size
        batch_size = len(batch[self.keys_to_collect[0]])
        if self.audio_key:
            if self.audio_key not in batch:
                rank_zero_warn(f"{self.__class__.__name__} audio_key={self.audio_key} not in batch ({list(batch.keys())})")

        rows_to_save = []
        media_to_save = {}
        columns = self.keys_to_collect + ["audio"]
        b_indices_choosen = np.random.choice(batch_size, min(self.max_items_to_save, batch_size), replace=False)
        for bidx in b_indices_choosen:
            row = []
            for key in self.keys_to_collect:
                if key not in batch:
                    row.append(None)
                    continue

                value = batch[key][bidx]
                if isinstance(value, torch.Tensor):
                    serialized_value = pickle.dumps(value.detach().cpu())
                elif isinstance(value, np.ndarray):
                    serialized_value = pickle.dumps(value)
                else:
                    try:
                        serialized_value = str(value)
                    except:
                        rank_zero_warn(f"{self.__class__.__name__} failed to convert {key}={value} to string")
                        serialized_value = None
                row.append(serialized_value)
                
            if self.audio_key and self.audio_key in batch:
                uttid = batch["uttid"][bidx]
                audio_samples = batch[self.audio_key][bidx].detach().cpu()
                if batch.get(self.audio_duration_key) is not None:
                    duration = batch[self.audio_duration_key][bidx]
                    audio_samples = audio_samples[..., :int(duration * self.sample_rate)]
                if audio_samples.ndim == 1:
                    audio_samples = audio_samples.unsqueeze(0)
                
                if self.audio_upload_to_tos:
                    audio_bytes = self.get_audio_bytes(audio_samples, self.sample_rate)
                    audio_url = upload_obj_to_tos(audio_bytes, os.path.join(self.project, self.name, str(self.version), str(trainer.global_step), f"{uttid}.wav"), verbose=True)
                else:
                    audio_url = None
                    if isinstance(audio_samples, torch.Tensor):
                        audio_samples = audio_samples.cpu().numpy()
                    assert isinstance(audio_samples, np.ndarray)
                    media_to_save[f"global_step={trainer.global_step}/{uttid}"] = tk.Audio(audio_samples, sample_rate=self.sample_rate, caption=uttid)
            else:
                audio_url = None
                
            row.append(audio_url)
            rows_to_save.append(row)

        log_dict = {
            f"global_step={trainer.global_step}": tk.Table(columns=columns, data=rows_to_save, allow_mixed_types=True),
            **media_to_save
        }
        tk.log(log_dict, step=trainer.global_step, commit=False)
        rank_zero_debug(f"Logging {len(b_indices_choosen)} items to wandb in global_step={trainer.global_step}")

    @staticmethod
    def get_audio_bytes(audio_samples, sr=24000):
        assert isinstance(audio_samples, torch.Tensor)
        audio_bytes = BytesIO()
        torchaudio.save(audio_bytes, audio_samples, sr, format="wav")
        audio_bytes.seek(0)
        return audio_bytes.getvalue()