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
        return mel

    def calculate_statistics(self, pl_datamodule, max_datapoints: int = -1):
        fp = f"{pl_datamodule.__class__.__name__}_{self.__class__.__name__}.stats.pt"

        train_loader = pl_datamodule.train_dataloader()
        
        pbar = tqdm(train_loader, total=max_datapoints)
        for batch_idx, batch in enumerate(pbar):
            if max_datapoints > 0 and self.n_datapoints >= max_datapoints:
                break
            
            # ! CHECK CAREFULLY           
            inputs = self.forward(batch["audio"], normalize=False, mel=batch["mel"])

            batch_size = inputs.shape[0]
            inputs = rearrange(inputs, "b time n_mels -> (b time) n_mels")
            self.update(inputs)

            self.n_datapoints += batch_size
            if batch_idx % 100 == 0:
                self.save_data_stats(fp, batch_idx)

            pbar.update(batch_size)

        self.save_data_stats(fp)
        self.check_statistics(fp, pl_datamodule, max_batches=20)

    def check_statistics(self, fp: str, pl_datamodule, max_batches: int):
        self.load_from_checkpoint(fp)
        train_loader = pl_datamodule.train_dataloader()
        for batch_idx, batch in enumerate(tqdm(train_loader, total=max_batches)):
            if batch_idx == max_batches:
                break

            # zero mean, unit variance
            inputs = self.forward(batch["audio"], normalize=True, mel=batch["mel"])
            print(inputs.mean(), inputs.std())
            # torch.testing.assert_close(inputs.mean(), torch.tensor(0, dtype=torch.float32), atol=1e-2, rtol=0)
            # torch.testing.assert_close(inputs.std(), torch.tensor(1, dtype=torch.float32), atol=1e-2, rtol=0)

    def process_data(self, data_loader, directory: str, rank: int, padding_strategy: str):
        pattern = os.path.join(directory, str(rank), "%05d.tar")

        # TODO retain shard variety -> audio is larger, around ~400 tracks per 8GB
        # mel take up much less space, so I'm setting a manual maxcount instead

        # maxsize = (1 << 32) * 2  # 8GiB, maximum size of each shard
        maxcount = 1000
        writer = IndexShardWriter(pattern, maxcount=maxcount)

        logger.info(f"Writing shards to: {pattern}")

        audio_key = "target_audio"
        input_length_key = "input_length"
        keep_keys = ['lyrics', 'lyrics_tokens', 'lyrics_normalized_text', 'lyrics_tokens_length', 'target_tokens_length', 'style_text', 'conditions']
        # keep_keys = ["tag", "shard", "key", "tag_names"]
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch_keys = list(batch.keys())

            # choose what audio keys to keep
            for k in batch_keys:
                if k not in keep_keys:
                    batch_keys.remove(k)

            batch_size = batch[audio_key].shape[0]

            for idx in range(batch_size):
                # remove padding if done in bucket batcher:
                if input_length_key in batch_keys:
                    assert padding_strategy == "pad", "the padding strategy must be 'pad', not 'random_pad'!"
                    input_length = batch[input_length_key][idx]
                    original_audio = batch[audio_key][idx][..., :input_length] # assumes right-padding!!
                else:
                    original_audio = batch[audio_key][idx]
                
                if original_audio.ndim == 1:
                    # add channel dim
                    original_audio = original_audio.unsqueeze(dim=0)

                mel = self.forward(original_audio, normalize=False)

                # write to new index
                obj = {}
                index = {}
                for k in batch_keys:
                    if batch[k] is not None:
                        if type(batch[k]) == str:
                            index[k] = batch[k]
                        elif type(batch[k]) == torch.Tensor:
                            obj[f"{k}.npy"] = batch[k][idx].numpy()
                        else:
                            index[k] = batch[k][idx]
                
                obj["mel.npy"] = mel.squeeze(dim=0).numpy() # batch size = 1 :)
                # obj["audio.npy"] = original_audio.squeeze(dim=0).numpy() # NOTE: for debugging, preview audio with mel

                unique_id = str(uuid4())
                # original_key = index.get("key", "") # TODO: Not available for Parquet at the moment :(
                obj["__key__"] = unique_id
                writer.write(obj, index)
        writer.close()

    def save_fp(self, pl_datamodule, root_dir: str):
        return os.path.join(root_dir, f"converted_all_{pl_datamodule.__class__.__name__}_{self.__class__.__name__}")

    def convert_data(self, pl_datamodule, root_dir: str, rank: int):

        fp = self.save_fp(pl_datamodule, root_dir)

        pl_datamodule.padding_strategy = "pad"
        self.process_data(pl_datamodule.train_dataloader(), f"{fp}/train", rank, pl_datamodule.padding_strategy)
        # self.process_data(pl_datamodule.val_dataloader(), f"{fp}/validation", rank, pl_datamodule.padding_strategy) # TODO: check if this does all validation?
        # self.process_data(pl_datamodule.test_dataloader(), f"{fp}/test", rank, pl_datamodule.padding_strategy)


class WhisperSpeechTransform24k(SpeechTransform):
    _sample_rate = 24000
    _n_mels = 80
    _n_fft = 1024
    _win_length = int(_sample_rate * 0.025)  # 25ms
    _hop_length = int(_sample_rate * 0.01)  # 10ms
    _f_min = 0
    _f_max = _sample_rate // 2

    def __init__(self):
        super().__init__(
            sample_rate=self._sample_rate,
            n_mels=self._n_mels,
            n_fft=self._n_fft,
            win_length=self._win_length,
            hop_length=self._hop_length,
            f_min=self._f_min,
            f_max=self._f_max
        )
        