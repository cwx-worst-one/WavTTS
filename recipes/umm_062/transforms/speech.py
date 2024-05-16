import os
from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torchaudio.transforms import AmplitudeToDB
from tqdm import tqdm

from samantha.transforms.audio import MelSpectrogram, safe_log
from samantha.transforms.utils import get_transform_version


class ModelInputTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.datamodule_name = "Undefined"
        self.total_samples = 0.0

    @abstractmethod
    def calculate_statistics(self, pl_datamodule, max_datapoints: int):
        pass

    def update(self, x):
        num_samples = x.size(0)
        self.total_samples += num_samples
        new_mean = x.mean(dim=0)
        delta = new_mean - self.mean
        self.mean += delta * num_samples / self.total_samples
        new_std = ((x - new_mean) * (x - self.mean)).sum(dim=0)
        self.std += new_std

    def load_from_checkpoint(self, fp: str, verify: bool = True):
        if os.path.exists(fp):
            self.load_state_dict(torch.load(fp))
            # if verify:
            #     self.verify_data_stats()
        else:
            raise FileNotFoundError(
                "statistics are not yet computed, use `calculate_statistics`."
            )

    # def verify_data_stats(self):
    #     transform_version = get_transform_version(self)
    #     if self._data_stats["model_input_transform"] != transform_version:
    #         raise Exception("transform_version MD5 hash does not match")

    def save_data_stats(self, fp: str, iter_idx: Optional[int] = None):
        iter_idx = "" if iter_idx is None else f"_{iter_idx:>06}"

        # override std
        std = torch.sqrt(self.std / (self.total_samples - 1))
        state_dict = deepcopy(self.state_dict())
        state_dict["std"] = std
        torch.save(state_dict, fp + iter_idx)
        print(state_dict)

    def forward(self, x: torch.Tensor, normalize: bool) -> torch.Tensor:
        return x


class SpeechTransform(ModelInputTransform):
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
        self.register_buffer("std", torch.ones(n_mels))
        self.register_buffer("n_datapoints", torch.zeros(1, dtype=torch.int32))

    def forward(self, waveform: torch.Tensor, normalize: bool) -> torch.Tensor:
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
        return mel

    def calculate_statistics(self, pl_datamodule, max_datapoints: int):
        fp = f"{pl_datamodule.__class__.__name__}_{self.__class__.__name__}.stats.pt"

        train_loader = pl_datamodule.train_dataloader()
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            if self.n_datapoints >= max_datapoints:
                break

            # ! CHECK CAREFULLY
            inputs = self.forward(batch["audio"], normalize=False)
            inputs = rearrange(inputs, "b time n_mels -> (b time) n_mels")
            self.update(inputs)

            self.n_datapoints += batch["audio"].shape[0]
            if batch_idx % 100 == 0:
                self.save_data_stats(fp, batch_idx)

        self.save_data_stats(fp)
        self.check_statistics(fp, pl_datamodule, max_batches=20)

    def check_statistics(self, fp: str, pl_datamodule, max_batches: int):
        self.load_from_checkpoint(fp)
        train_loader = pl_datamodule.train_dataloader()
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            if batch_idx == max_batches:
                break

            # zero mean, unit variance
            inputs = self.forward(batch["audio"], normalize=True)
            print(inputs.mean(), inputs.std())
            # torch.testing.assert_close(inputs.mean(), torch.tensor(0, dtype=torch.float32), atol=1e-2, rtol=0)
            # torch.testing.assert_close(inputs.std(), torch.tensor(1, dtype=torch.float32), atol=1e-2, rtol=0)
