from recipes.sacodec.datasets.datamodule import DataModule, SemanticTokenLengthTransform, default_bucket_batcher_fn
import random
import torchaudio
from samantha.utils.webdataset import return_self
from pedalboard import Pedalboard, MP3Compressor
import sys
import os

from typing import Iterator, Dict, Any
import torch
import pytorch_lightning as pl
import webdataset as wds
import io
from typing import Optional
from torchaudio.functional import resample

from torch.utils.data import DataLoader
from webdataset.pipeline import DataPipeline
from webdataset import shardlists
from samantha.dataio.parquet.parquet_dataset import ParquetDataset
from samantha.dataio.webdataset.pipeline import WebPipeline

import pyloudnorm as pyln
import numpy as np
import torchaudio

# __dataset_name__
NORM_DATASET = ["music_wyy-hq-part1_Swyy_N170k_T44k_v1_Clip", "music_wyy-hq-part2_Swyy_N87k_T44k_v1_Clip", "music_wyy-hq-part4_Swyy_N80k_T44k_v1_Clip"]


## accepts csv with uttid and wav path. e.g.: /mnt/bn/ashaw-lq/eval/valset_56wavs_120s/test_full.lst
def parse_meta_lst(meta_lst):
    metas = []
    with open(meta_lst, "r", encoding="utf8") as f:
        for line in f:
            splits = line.strip().split("|")
            uttid = splits[0]
            wav_path = splits[-1]
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(os.path.dirname(meta_lst), wav_path)
            meta = {
                "uttid": uttid,
                "wav_path": wav_path
            }
            metas.append(meta)
    return metas

def has_overlap(ranges, target):
    t_start, t_end = target
    for start, end in ranges:
        # Overlap occurs if the ranges are not completely apart
        if not (t_end <= start or t_start >= end):
            return True
    return False

class ParquetDatasetWrapper(WebPipeline):
    def __init__(
        self,
        data_id=408,
        resampled: bool = True,
        audio_duration: list = [5, 10],
        max_num_crops: int = 6,
        nodesplitter=wds.split_by_node,
        n_channels:int = 1,
        dataset_samplerate: int= 44100,
        resample_to_24k: bool = True,
        allow_mono: bool = False,
        allow_resample: bool = False,
        audio_augmentations: str = None,
    ):
        if isinstance(data_id, str) and data_id.endswith(".lst"):
            dataset = WebPipeline(parse_meta_lst(data_id), pipeline=[])
        else:
            if dataset_samplerate == 44100:
                # special case for data_id=16944
                extra_fields_in_data = ["audio_44k"]
            else:
                extra_fields_in_data = None
            dataset = ParquetDataset(
                data_id=data_id, resampled=resampled, nodesplitter=nodesplitter,
                extra_fields_in_data=extra_fields_in_data
            )
        self.audio_duration = audio_duration
        self.max_num_crops = max_num_crops
        self.meter = pyln.Meter(dataset_samplerate)
        # self.meter = LoudnessCheck(dataset_samplerate, threshold=.05, loudness_ratio_threshold=0.3)
        self.dataset_samplerate = dataset_samplerate
        self.n_channels = n_channels
        self.allow_mono = allow_mono
        self.allow_resample = allow_resample
        self.resample_to_24k = resample_to_24k
        audio_augmentations = audio_augmentations if audio_augmentations else ""
        self.audio_augmentations = audio_augmentations.split(",")
        if "mp3" in audio_augmentations:
            self.mp3_compress = Pedalboard([MP3Compressor()])
        pipeline = [{"compose": self.decode}]


        ## Logging module
        self.count = 0
        self.skipped = 0
        self.messages = {}
        self.log_interval = 200

        super().__init__(dataset, pipeline)

    def __iter__(self):
        return iter(self.data_pipeline)

    def decode(self, items) -> Iterator[Dict[str, Any]]:
        for item in items:
            # from byte to npy
            # wav_npy, sr = librosa.load(io.BytesIO(item["wav"]), sr=None, mono=(self.n_channels==1))
            try:
                if "wav_path" in item: # support meta.lst from local path
                    wav_npy, sr = torchaudio.load(item["wav_path"])
                elif "audio_44k" in item: # special case for data_id=16944
                    wav_npy, sr = torchaudio.load(io.BytesIO(item["audio_44k"]))
                else:
                    wav_npy, sr = torchaudio.load(io.BytesIO(item.get("audio_44k", item["wav"])))
                if sr != self.dataset_samplerate:
                    if self.allow_resample:
                        wav_npy = resample(wav_npy, sr, self.dataset_samplerate)
                    else:
                        self._update_stats(f"Invalid sample rate {sr}")
                        continue
            except Exception as e:
                self._update_stats("Unable to load audio")
                continue
            
            wav_tensor = torch.as_tensor(wav_npy, dtype=torch.float32)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor[None, :]

            if wav_tensor.shape[0] == 2 and wav_tensor.shape[0] != self.n_channels: # convert to mono
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
            # drop samples with not enogh channels
            if wav_tensor.shape[0] != self.n_channels:
                if self.allow_mono and len(wav_tensor.squeeze().shape)==1: 
                    wav_tensor = torch.stack([wav_tensor.squeeze(), wav_tensor.squeeze()], dim=0)
                else:
                    self._update_stats(f"Invalid channels {wav_tensor.shape[0]}")
                    continue
                
            audio_duration_samples = int(max(self.audio_duration) * self.dataset_samplerate)
            # drop the sample its too short
            if wav_tensor.shape[-1] < (audio_duration_samples* 0.8):
                self._update_stats("Audio too short")
                continue
            # pad audio to the length of duration
            elif wav_tensor.shape[-1] < (audio_duration_samples):
                pad_len = audio_duration_samples - wav_tensor.shape[-1]
                wav_tensor = torch.nn.functional.pad(
                    wav_tensor, (0, pad_len), "constant", 0
                )
            
            max_num_crops = int(wav_tensor.shape[-1] / self.dataset_samplerate // max(self.audio_duration) // 1.5)
            max_num_crops = max(max_num_crops, 1) # sample at least 1
            max_num_crops = min(max_num_crops, self.max_num_crops)
            ranges = []
            for i in range(max_num_crops):
                target_audio_duration = random.choice(self.audio_duration)
                target_audio_duration_samples = target_audio_duration * self.dataset_samplerate
                
                audio_duration_samples = wav_tensor.shape[-1]
                
                if target_audio_duration_samples > audio_duration_samples:
                    self._update_stats("Audio crop too short")
                    continue
                
                # random crop
                if self.max_num_crops == 1:
                    start = 0
                    end = target_audio_duration_samples
                else:
                    start = torch.randint(
                        0, wav_tensor.shape[-1] - target_audio_duration_samples + 1, (1,)
                    ).item()
                    end = start + target_audio_duration_samples

                if has_overlap(ranges, (start, end)):
                    continue
                ranges.append((start, end))

                wav_crop = wav_tensor[:, start : end]


                # loudness detection
                db = self.meter.integrated_loudness(wav_crop.detach().cpu().numpy().T)
                if db < -40 or np.isneginf(db):
                    self._update_stats("Audio not loud enough")
                    continue
                # if not self.meter(wav_crop):
                #     self._update_stats("Audio not loud enough")
                #     continue
                dataset_name = item.get("__dataset_name__", "")
                if "quiet" in self.audio_augmentations and db > -14 and random.random() < 0.75 and dataset_name in NORM_DATASET:
                    gain_db = random.randint(-6, -1)
                    wav_crop = torchaudio.functional.gain(wav_crop, gain_db=gain_db)

            
                res = {"audio": wav_crop, "meta_song_id": item["uttid"]}
            
                if self.resample_to_24k:
                    audio_24k_mono = torchaudio.functional.resample(
                        wav_crop.mean(dim=0), orig_freq=self.dataset_samplerate, new_freq=24000
                    )
                    res['audio_24k'] = audio_24k_mono
                    
                if "mp3" in self.audio_augmentations and random.random() > 0.5:
                    compress_rate = random.randint(0, 7)
                    self.mp3_compress[0].vbr_quality = compress_rate
                    audio_mp3_compress = self.mp3_compress(wav_crop.numpy(), sample_rate=self.dataset_samplerate)
                    audio_mp3_compress = torch.from_numpy(audio_mp3_compress)
                    res['audio_augmented'] = audio_mp3_compress

                if "volume" in self.audio_augmentations: 
                    ## gain audio on both. add clipping to input.
                    ## independently... reduce target audio by 0.9.

                    audio_target = res['audio']
                    # audio_24k_mono = res['audio_24k']
                    audio_augment = res['audio_augmented'] if 'audio_augmented' in res else audio_target

                    # randomly gain audio to both. randomly hard clip input.
                    if random.random() < 0.25:
                        gain_db = random.randint(-3, 3)
                        # audio_24k_mono = torchaudio.functional.gain(audio_24k_mono, gain_db=gain_db)
                        audio_target = torchaudio.functional.gain(audio_target, gain_db=gain_db)
                        audio_augment = torchaudio.functional.gain(audio_augment, gain_db=gain_db)
                        random_clamp = random.uniform(0.97, 1)
                        audio_augment = torch.clip(audio_augment, min=-1*random_clamp, max=1*random_clamp)

                    # always reduce target by 1 db
                    audio_target = torchaudio.functional.gain(audio_target, gain_db=-1)

                    res['audio'] = audio_target
                    res['audio_augmented'] = audio_augment



                yield res
            self._update_stats(None, skipped=False)

    def _update_stats(
        self,
        message: Optional[str] = None,
        skipped: bool = True
    ):
        self.count += 1
        if skipped:
            self.skipped += 1
        if message is not None:
            if message not in self.messages:
                self.messages[message] = 0
            self.messages[message] += 1
        # Print
        if self.count > 0 and self.count % self.log_interval == 0:
            worker_id = torch.utils.data.get_worker_info()
            if worker_id is not None:
                worker_id = worker_id.id
            else:
                worker_id = "Undefined"
            print(
                f"[{worker_id}] "
                f"Skipped {self.skipped}/{self.count} items, "
                f"Messages: {self.messages}",
                file=sys.stderr,
                flush=True,
            )


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
        valid_audio_duration: int = None,
        max_num_crops: int = 6,
        resample_to_24k: bool = True,
        allow_resample: bool = False,
        allow_mono: bool = False,
        audio_augmentations: str = "",
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
            max_num_crops=max_num_crops,
            n_channels=n_channels,
            dataset_samplerate=dataset_samplerate,
            resample_to_24k=resample_to_24k,
            allow_resample=allow_resample,
            allow_mono=allow_mono,
            audio_augmentations=audio_augmentations,
        )

        ### sacodec needs all songs to be the same length to calculate latent mean std. hardcoding this for now
        if valid_audio_duration is None:
            valid_audio_duration = [max(audio_duration)]
        self.validation_dataset = ParquetDatasetWrapper(
            data_id=valid_data_id,
            resampled=False,
            nodesplitter=return_self,
            audio_duration=valid_audio_duration, # 
            max_num_crops=1,
            n_channels=n_channels,
            dataset_samplerate=dataset_samplerate,
            resample_to_24k=resample_to_24k,
            # audio_augmentations=audio_augmentations, # Disable augmentations in validation set
        )
        self.dataset_samplerate = dataset_samplerate
        self.audio_duration = audio_duration
        self.valid_audio_duration = valid_audio_duration
        self.predict_dataset = self.train_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def train_dataloader(self):
        train_dataset_batched = DataPipeline(
            self.train_dataset,
            wds.map(SemanticTokenLengthTransform(sample_rate=self.dataset_samplerate, audio_key="audio")),
            wds.shuffle(self.shuffle_buffer_size),
            # default_bucket_batcher_fn already handles collate
            default_bucket_batcher_fn(
                self.dataset_samplerate, self.audio_duration, self.train_batch_size, lyrics_frame_rate=0, max_duration=max(self.audio_duration)
            ),
#             wds.batched(None, collation_fn=self.collate_fn),
        )
        return DataLoader(
            train_dataset_batched,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
#             collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        validation_dataset_batched = DataPipeline(
            self.validation_dataset,
            wds.map(SemanticTokenLengthTransform(sample_rate=self.dataset_samplerate, audio_key="audio")),
            default_bucket_batcher_fn(
                self.dataset_samplerate, self.valid_audio_duration, self.valid_batch_size, lyrics_frame_rate=0, max_duration=max(self.valid_audio_duration)
            ),
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
    pl_datamodule = ParquetDataModule(
    #     train_data_id=1421, #vocal
        train_data_id=442, # instrumental
        valid_data_id=444,
        train_batch_size=4,
        valid_batch_size=2,
        shuffle_buffer_size=0,
        num_workers=0,
        n_channels=2,
        dataset_samplerate=44100,
        audio_duration=[5, 10],
        max_num_crops=4,
        prefetch_factor=None,
        resample_to_24k=True
    )
    train_dl = pl_datamodule.train_dataloader()
    for data in iter(train_dl):
        # print(data.keys())
        for k,v in data.items():
            try:
                print(k, v.shape)
            except:
                print(k, v)
        print()
