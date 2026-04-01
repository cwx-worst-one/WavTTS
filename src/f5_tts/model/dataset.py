import json
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

        self._resamplers = {}

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            if sample_rate not in self._resamplers:
                self._resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = self._resamplers[sample_rate](audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        return_wav_only: bool = False,
        wav_frame_len: int = 240,
        load_ssl_features: bool = False,
        wav_dataset_root: str = None,
        ssl_feature_dataset_root: str = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        self.return_wav_only = return_wav_only
        self.wav_frame_len = wav_frame_len
        self.load_ssl_features = load_ssl_features
        self.wav_dataset_root = wav_dataset_root
        self.ssl_feature_dataset_root = ssl_feature_dataset_root

        self._resamplers = {}

        if not preprocessed_mel and not return_wav_only:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                if source_sample_rate not in self._resamplers:
                    self._resamplers[source_sample_rate] = torchaudio.transforms.Resample(
                        source_sample_rate, self.target_sample_rate
                    )
                audio = self._resamplers[source_sample_rate](audio)

            # to mel spectrogram
            if not self.return_wav_only:
                mel_spec = self.mel_spectrogram(audio)
                mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
            else:
                mel_spec = None

            # load ssl features
            if self.load_ssl_features:
                ssl_feature_path = audio_path.replace(self.wav_dataset_root, self.ssl_feature_dataset_root).replace(".wav", ".npy")
                ssl_feature = torch.from_numpy(np.load(ssl_feature_path))
            else:
                ssl_feature = None

        return {
            "mel_spec": mel_spec,
            "wav": audio.squeeze(0) if self.return_wav_only else None,
            "text": text,
            "ssl_feature": ssl_feature,
        }


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])
    
    wavs = [item["wav"] for item in batch]
    has_wav = wavs[0] is not None
    if has_wav:
        wav_lengths = torch.LongTensor([w.shape[0] for w in wavs])
        max_wav_len = wav_lengths.max().item()

        padded_wavs = []
        for w in wavs:
            pad_len = max_wav_len - w.shape[0]
            padded_wavs.append(
                F.pad(w, (0, pad_len), value=0.0)
            )

        wavs = torch.stack(padded_wavs)   # [B, T_wav]
    else:
        wavs = None
        wav_lengths = None
    
    mel_specs = [item["mel_spec"] for item in batch]
    has_mel = mel_specs[0] is not None
    if has_mel:
        mel_lengths = torch.LongTensor([m.shape[-1] for m in mel_specs])
        max_mel_len = mel_lengths.max().item()

        padded_mels = []
        for m in mel_specs:
            pad_len = max_mel_len - m.shape[-1]
            padded_mels.append(
                F.pad(m, (0, pad_len), value=0.0)
            )

        mel_specs = torch.stack(padded_mels)  # [B, n_mel, T_mel]
    else:
        mel_specs = None
        mel_lengths = None

    ssl_features = [item["ssl_feature"] for item in batch]
    has_ssl_feature = ssl_features[0] is not None
    if has_ssl_feature:
        ssl_feature_lengths = torch.LongTensor([s.shape[0] for s in ssl_features])
        max_ssl_feature_len = ssl_feature_lengths.max().item()

        padded_ssl_features = []
        for s in ssl_features:
            pad_len = max_ssl_feature_len - s.shape[0]
            padded_ssl_features.append(
                F.pad(s, (0, 0, 0, pad_len), value=0.0)
            )

        ssl_features = torch.stack(padded_ssl_features)  # [B, T_ssl, d_ssl]
    else:
        ssl_features = None
        ssl_feature_lengths = None

    return dict(
        text=text,
        text_lengths=text_lengths,
        wav=wavs,
        wav_lengths=wav_lengths,
        mel=mel_specs,
        mel_lengths=mel_lengths,
        ssl_feature=ssl_features,
        ssl_feature_lengths=ssl_feature_lengths,
    )
