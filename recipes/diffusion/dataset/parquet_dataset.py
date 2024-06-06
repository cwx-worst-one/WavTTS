from collections import defaultdict
from typing import Iterator, Dict, Any
import numpy as np
import torch
import pytorch_lightning as pl
import webdataset as wds
import pickle
import pyloudnorm as pyln
import librosa
import io

from torch.utils.data import DataLoader
from webdataset.pipeline import DataPipeline
from webdataset import shardlists
from samantha.dataio.parquet.parquet_dataset import ParquetDataset
from samantha.dataio.webdataset.pipeline import WebPipeline

import torchaudio

class ParquetDatasetWrapper(WebPipeline):
    def __init__(
        self,
        data_id=408,
        resampled: bool = True,
        audio_duration: int = 30,
        audio_random_crop: bool = True,
        nodesplitter=wds.split_by_node,
        n_channels:int = 1,
        dataset_samplerate: int= 44100,
    ):
        dataset = ParquetDataset(
            data_id=data_id, resampled=resampled, nodesplitter=nodesplitter
        )
        self.audio_duration = audio_duration
        self.audio_random_crop = audio_random_crop
        self.meter = pyln.Meter(dataset_samplerate)
        self.dataset_samplerate = dataset_samplerate
        self.n_channels = n_channels
        pipeline = [{"compose": self.decode}]
        super().__init__(dataset, pipeline)

    def __iter__(self):
        return iter(self.data_pipeline)

    def decode(self, items) -> Iterator[Dict[str, Any]]:
        for item in items:
            # from byte to npy
            # wav_npy, sr = librosa.load(io.BytesIO(item["wav"]), sr=None, mono=(self.n_channels==1))
            wav_npy, sr = torchaudio.load(io.BytesIO(item["wav"]))
            wav_tensor = torch.as_tensor(wav_npy, dtype=torch.float32)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor[None, :]
            # drop samples with not enogh channels
            if wav_tensor.shape[0] != self.n_channels:
                continue
            audio_duration_samples = int(self.audio_duration * self.dataset_samplerate)
            # drop the sample its too short
            if wav_tensor.shape[-1] < (audio_duration_samples* 0.8):
                continue
            # pad audio to the length of duration
            elif wav_tensor.shape[-1] < (audio_duration_samples):
                pad_len = audio_duration_samples - wav_tensor.shape[-1]
                wav_tensor = torch.nn.functional.pad(
                    wav_tensor, (0, pad_len), "constant", 0
                )
            
            if self.audio_random_crop:
                # random crop
                start = torch.randint(
                    0, wav_tensor.shape[-1] - audio_duration_samples + 1, (1,)
                ).item()
                wav_tensor = wav_tensor[:, start : start + audio_duration_samples]
            else:
                # use first [audio_duration second]
                wav_tensor = wav_tensor[:, :audio_duration_samples]
            # loudness detection
            db = self.meter.integrated_loudness(wav_tensor.detach().cpu().numpy().T)
            if db < -50 or np.isneginf(db):
                continue

            yield {"audio": wav_tensor, "meta_song_id": item["uttid"]}


class ParquetDataModule(pl.LightningDataModule):
    data_sample_rate = None

    def __init__(
        self,
        train_data_id: int,
        valid_data_id: int,
        train_batch_size: int,
        valid_batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        resampled: bool = True,
        n_channels: int = 1,
        dataset_samplerate: int = 44100,
        audio_duration: int = 30,
        audio_random_crop: bool = True,
        prefetch_factor=2,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = ParquetDatasetWrapper(
            data_id=train_data_id,
            resampled=resampled,
            audio_duration=audio_duration,
            audio_random_crop=audio_random_crop,
            n_channels=n_channels,
            dataset_samplerate=dataset_samplerate,
        )
        self.validation_dataset = ParquetDatasetWrapper(
            data_id=valid_data_id,
            resampled=True,
            audio_duration=audio_duration,
            audio_random_crop=False,
            n_channels=n_channels,
            dataset_samplerate=dataset_samplerate,
        )
        self.predict_dataset = self.train_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def collate_fn(self, batch):
        keys = batch[0].keys()
        batch_dict = defaultdict(list)
        for b in batch:
            for k in keys:
                batch_dict[k].append(b[k])
        batch_dict["audio"] = torch.stack(batch_dict["audio"])
        return dict(batch_dict)

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset,
            wds.shuffle(self.shuffle_buffer_size),
            wds.batched(self.train_batch_size, collation_fn=self.collate_fn),
        )
        return DataLoader(
            train_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        validation_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.batched(self.valid_batch_size, collation_fn=self.collate_fn),
        )

        return DataLoader(
            validation_dataset_batched,
            batch_size=None,
            num_workers=1,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        predict_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.batched(self.batch_size, collation_fn=self.collate_fn),
        )
        return DataLoader(
            predict_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )


class OfflineFeatureParquetDatasetWrapper(ParquetDatasetWrapper):
    KEYS_REQUIRED=["vocoder_emb","umm_token","uttid","loudness"]
    def decode(self, items) -> Iterator[Dict[str, Any]]:
        for item in items:
            assert all([k in item for k in self.KEYS_REQUIRED])

            meta_song_id = item["uttid"]
            vocoder_embs = item["vocoder_emb"]
            if vocoder_embs is None:
                print(
                    f"[WARNING] get 'vocoder_emb=None' from dataset (uttid={meta_song_id})"
                )
                continue
            else:
                vocoder_embs = torch.from_numpy(pickle.loads(vocoder_embs))
                if vocoder_embs.ndim == 2:
                    vocoder_embs = vocoder_embs[None, :]

            umm_tokens = item["umm_token"]
            if umm_tokens is None:
                print(
                    f"[WARNING] get 'umm_token=None' from dataset (uttid={meta_song_id})"
                )
                continue
            else:
                umm_tokens = torch.from_numpy(pickle.loads(umm_tokens))
                if umm_tokens.ndim == 1:
                    umm_tokens = umm_tokens[None, :]
            
            loudness = pickle.loads(item["loudness"])
            if len(loudness)!=2:
                print(f"[WARINING] invalid 'loudness' with length {len(loudness)}(2) ")
                continue
            loudness, dur = loudness
            if not (isinstance(loudness,list) and isinstance(dur,float)):
                print(f"[WARINING] invalid 'loudness' with {type(loudness)}(list) and {type(dur)}(float)")
                continue

            try:
                assert len(umm_tokens) == len(vocoder_embs) == len(loudness)
                min_bs = len(umm_tokens)
            except:
                min_bs = min(len(umm_tokens), len(vocoder_embs), len(loudness))

            yield {
                "meta_song_id": meta_song_id,
                "condition_tokens": umm_tokens[:min_bs],
                "vocoder_embs": vocoder_embs[:min_bs],
                "loudness": loudness[:min_bs],
                "duration": dur,
            }


class OfflineFeatureParquetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_id: int,
        valid_data_id: int,
        train_batch_size: int,
        valid_batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        resampled: bool = True,
        dataset_samplerate: int = 44100,
        audio_duration: int = 30,
        audio_random_crop: bool = True,
        prefetch_factor=2,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_dataset = OfflineFeatureParquetDatasetWrapper(
            data_id=train_data_id,
            resampled=resampled,
            audio_duration=audio_duration,
            audio_random_crop=audio_random_crop,
            dataset_samplerate=dataset_samplerate,
        )
        self.validation_dataset = OfflineFeatureParquetDatasetWrapper(
            data_id=valid_data_id,
            resampled=True,
            audio_duration=audio_duration,
            audio_random_crop=False,
            dataset_samplerate=dataset_samplerate,
        )
        self.predict_dataset = self.train_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def collate_fn(self, batch):
        keys = batch[0].keys()
        batch_dict = defaultdict(list)
        for b in batch:
            for k in keys:
                batch_dict[k].append(b[k])
        batch_dict["condition_tokens"] = torch.stack(
            batch_dict["condition_tokens"], dim=0
        )  # [b,n,t]
        batch_dict["vocoder_embs"] = torch.stack(
            batch_dict["vocoder_embs"], dim=0
        )  # [b,n,dim,t]
        batch_dict["loudness"] = np.stack(batch_dict["loudness"],axis=0)
        return dict(batch_dict)

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset,
            wds.shuffle(self.shuffle_buffer_size),
            wds.batched(self.train_batch_size, collation_fn=self.collate_fn),
        )
        return DataLoader(
            train_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        validation_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.batched(self.valid_batch_size, collation_fn=self.collate_fn),
        )

        return DataLoader(
            validation_dataset_batched,
            batch_size=None,
            num_workers=1,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        predict_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.batched(self.batch_size, collation_fn=self.collate_fn),
        )
        return DataLoader(
            predict_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )


if __name__ == "__main__":
    dataset = ParquetDatasetWrapper(
            data_id=2190,
            resampled=True,
            audio_duration=30,
            audio_random_crop=True,
            n_channels=2,
            dataset_samplerate=44100,
    )
    

    # datamodule = ParquetDataModule(794, 793, 4, 4, 2, dataset_samplerate=24000)
    
    # train_loader = datamodule.val_dataloader()

    # for d in train_loader:
    #     # print(d.keys())
    #     print(d["audio"].shape)
    #     # assert 1==2

    # dataset=OfflineFeatureParquetDatasetWrapper(865)
    # dataset=OfflineFeatureParquetDatasetWrapper(856)
    # dataset=OfflineFeatureParquetDatasetWrapper(839)
    # dataset=OfflineFeatureParquetDatasetWrapper(808)
    for data in dataset:
        # print(data.keys())
        for k,v in data.items():
            try:
                print(k, v.shape)
            except:
                print(k, v)
        print()
    # OfflineFeatureParquetDataModule(856,856,16,16)
