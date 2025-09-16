from asyncio.log import logger
import os
from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torchaudio.transforms import AmplitudeToDB
from tqdm import tqdm
from uuid import uuid4

from samantha.dataio.webdataset.writer import IndexShardWriter
from samantha.transforms.audio import MelSpectrogram, safe_log
from samantha.transforms.utils import get_transform_version
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)

class SpeechTransform(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        f_min: int,
        f_max: int,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            power=2,  # power
        )
        self.amplitude_to_db = AmplitudeToDB(stype="power")  # power
        self.register_buffer("mean", torch.zeros(n_mels))
        self.register_buffer("std_running", torch.ones(n_mels))
        self.register_buffer("std", torch.ones(n_mels))
        self.register_buffer("n_datapoints", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("total_samples", torch.zeros(1, dtype=torch.int32))

    def save_data_stats(self, fp: str, iter_idx: Optional[int] = None):
        iter_idx = "" if iter_idx is None else f"_{iter_idx:>06}"

        # override std
        self.std = torch.sqrt(self.std_running / (self.total_samples - 1))
        state_dict = deepcopy(self.state_dict())
        torch.save(state_dict, fp + iter_idx)

    def update(self, x):
        num_samples = x.size(0)
        self.total_samples += num_samples
        new_mean = x.mean(dim=0)
        delta = new_mean - self.mean
        self.mean += delta * num_samples / self.total_samples
        new_std = ((x - new_mean) * (x - self.mean)).sum(dim=0)
        self.std_running += new_std

    def load_from_checkpoint(self, fp: str, verify: bool = True):
        # if os.environ.get("BYTED_RAY_CLUSTER", "") != "":
        #     # special fix for ray env
        #     fp = os.path.join(os.path.dirname(__file__), "../../../", fp)
        if os.path.exists(fp):
            self.load_state_dict(torch.load(fp))
            # if verify:
            #     self.verify_data_stats()
        else:
            raise FileNotFoundError(
                "statistics are not yet computed, use `calculate_statistics`."
            )

    def forward(self, waveform: torch.Tensor, normalize: bool, mel: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mel is None or (type(mel) == list and mel[0] is None):
            if waveform.ndim != 2:
                waveform = waveform.squeeze(dim=1)
                # raise Exception("input tensor should be [batch_size, channels, time")

            mel = self.mel_transform(waveform)
            mel = mel[..., :-1]
            mel = self.amplitude_to_db(mel)  # alternatively, safe_log(mel)
            mel = rearrange(mel, "b n_mels time -> b time n_mels")

        if normalize:
            if self.n_datapoints == 0:
                raise Exception(
                    "statistics are not loaded, use `load_data_stats(fp)` first"
                )

            mel = mel - self.mean.float()
            mel = mel / self.std.float()
            # print("Normalizing")
        return mel

    def calculate_statistics(self, pl_datamodule, max_datapoints: int = -1):
        fp = f"{pl_datamodule.__class__.__name__}_{self.__class__.__name__}.stats.pt"

        train_loader = pl_datamodule.train_dataloader()
        
        pbar = tqdm(train_loader, total=max_datapoints)
        for batch_idx, batch in enumerate(pbar):
            if max_datapoints > 0 and self.n_datapoints >= max_datapoints:
                break
            
            # ! CHECK CAREFULLY           
            if "audio_24k" in batch:
                inputs = self.forward(batch["audio_24k"], normalize=False, mel=None)
            else:
                inputs = self.forward(batch["audio"], normalize=False, mel=batch["mel"])

            batch_size = inputs.shape[0]
            inputs = rearrange(inputs, "b time n_mels -> (b time) n_mels")
            self.update(inputs)

            self.n_datapoints += batch_size
            if batch_idx % 100 == 0:
                self.save_data_stats(fp, batch_idx)

            pbar.update(batch_size)

        self.save_data_stats(fp)
        # self.check_statistics(fp, pl_datamodule, max_batches=20)
        print("State dict:", self.state_dict())

    def check_statistics(self, fp: str, pl_datamodule, max_batches: int):
        self.load_from_checkpoint(fp)
        train_loader = pl_datamodule.train_dataloader()
        for batch_idx, batch in enumerate(tqdm(train_loader, total=max_batches)):
            if batch_idx == max_batches:
                break

            # zero mean, unit variance
            if "audio_24k" in batch:
                inputs = self.forward(batch["audio_24k"], normalize=True, mel=batch.get("mel"))
            else:
                inputs = self.forward(batch["audio"], normalize=True, mel=batch["mel"])
            print("Mean, std", inputs.mean(), inputs.std())
            torch.testing.assert_close(inputs.mean(), torch.tensor(0, dtype=torch.float32), atol=1e-2, rtol=0)
            torch.testing.assert_close(inputs.std(), torch.tensor(1, dtype=torch.float32), atol=1e-2, rtol=0)
