import io
import json
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio_augmentations import Compose
from tqdm import tqdm
from webdataset import WebDataset

from recipes.datasets.base import BaseDataModule
from samantha.dataio.webdataset import ShardWriter
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
    fp32_to_int16,
)
from samantha.utils.hdfs_tools import hdfs_open
from samantha.utils.webdataset import return_self

SAMPLE_RATE = 44100


def _load_waveform(path: str, exp_sample_rate: int, read_binary: bool = True):
    if read_binary:
        with open(path, "rb") as f:
            waveform = f.read()
        sample_rate = SAMPLE_RATE
    else:
        waveform, sample_rate = torchaudio.load(path)
        if exp_sample_rate != sample_rate:
            raise ValueError(
                f"sample rate should be {exp_sample_rate}, but got {sample_rate}"
            )
    return waveform, sample_rate



def load_billboard_hot200_item(audio_fp: str):
    waveform, _ = _load_waveform(audio_fp, SAMPLE_RATE)
    return waveform




import pandas as pd


class BillboardHot200Dataset(Dataset):
    """BillboardHot200 dataset.
    Args:
    root (str or Path): Path to the directory where the dataset is found or downloaded.
    split (str, optional): The split to use (small, medium, large)
    """

    _ext_audio = ".flac"
    _ext_metadata = ".json"
    _columns = [
        "id", "name",  "disc_number", "track_number", "duration_ms", "primary_artist_name",
        "album.name", "album.release_date", "album.type", "album.id", "album.total_tracks",
        "popularity", "external_ids.isrc", "meta_song_id",
    ]

    def __init__(self, metadata_fp: str, audio_dir: str, track_features_dir: str, split: str, verify_dataset: bool = True) -> None:
        self.data = pd.read_pickle(metadata_fp)
        self._split = split
        self._audio_dir = audio_dir
        self._track_features_dir = track_features_dir

        if not os.path.isdir(self._audio_dir):
            raise RuntimeError(f"Audio dataset not found at {self._audio_dir}.")

        if not os.path.isdir(self._track_features_dir):
            raise RuntimeError(f"Audio features not found at {self._track_features_dir}.")

        if verify_dataset:
            self.verify_dataset()

        self._walker = self.data.index.tolist()
        self.total = len(self._walker)

    def verify_dataset(self):
        new_data = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Verifying audio and track features files"):
            try:
                if os.path.isfile(self.get_audio_path(row["meta_song_id"])):
                    features_fp = self.get_track_features_path(row["album.id"], row["id"])
                    self.load_track_features(features_fp)
                    new_data.append(row)
            except Exception as e:
                print(e)

        self.data = pd.DataFrame(new_data)

    def random_shuffle(self, seed: int = 42):
        random.seed(seed)
        random.shuffle(self._walker)

    def __len__(self):
        return self.total

    def get_audio_path(self, meta_song_id: str):
        return os.path.join(self._audio_dir, meta_song_id + self._ext_audio)
    
    def get_track_features_path(self, album_id: str, track_id: str):
        return os.path.join(self._track_features_dir, album_id, track_id + ".json")

    def load_track_features(self, fp: str) -> dict:
        with open(fp) as f:
            return json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        index = self._walker[idx]
        row = self.data.loc[index]
        
        audio_fp = self.get_audio_path(row["meta_song_id"])
        features_fp = self.get_track_features_path(row["album.id"], row["id"])
        track_features = self.load_track_features(features_fp)
        metadata = row[self._columns].to_dict()
        audio = load_billboard_hot200_item(audio_fp)
        return {
            "audio": audio,
            "metadata": metadata,
            "track_features": track_features,
        }


def write_index(hdfs_fp: str, index: List[str]):
    with hdfs_open(hdfs_fp, "w") as f:
        f.write("\n".join(index))

def billboard_collate_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_length = max([x["audio"].shape[-1] for x in batch])
    random_pad = RandomPad(n_samples=max_length)

    audio = []
    metadata = []
    track_features = []
    shard = []
    for idx in range(len(batch)):
        audio.append(random_pad(batch[idx]["audio"]))
        metadata.append(batch[idx]["metadata"])
        track_features.append(batch[idx]["track_features"])
        shard.append(batch[idx]["shard"])

    return {
        "audio": torch.stack(audio),
        "metadata": metadata,
        "track_features": track_features,
        "shard": shard,
    }


class BillboardHot200WebDataModule(BaseDataModule):
    data_sample_rate = SAMPLE_RATE

    def __init__(
        self,
        sample_rate: int,
        duration: float,
        split: str,
        batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        resampled: bool = True,
        shardshuffle: bool = True,
        collate_fn: Optional[Callable] = billboard_collate_fn,
    ):
        self.sample_rate = sample_rate
        self.split = split
        self.duration = duration
        self._data_stats = None

        train_shards, valid_shards = self.get_hdfs_shard_uri(split)
        train_dataset = WebDataset(
            urls=train_shards, resampled=resampled, shardshuffle=shardshuffle
        )
        validation_dataset = WebDataset(urls=valid_shards, nodesplitter=return_self)

        pipeline = []
        pipeline.append("decode")
        pipeline.append({"map": [self.wds_transform]})

        train_dataset = WebPipeline(train_dataset, pipeline)
        predict_dataset = train_dataset # TODO
        validation_dataset = WebPipeline(validation_dataset, pipeline)

        super().__init__(
            sample_rate=sample_rate,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            predict_dataset=predict_dataset,
            collate_fn=collate_fn,
        )
        self.base_transform = Compose(
            [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        )
        self.random_pad = RandomPad(self.n_audio_samples)
        self.random_crop = RandomResizedCrop(self.n_audio_samples)


    @staticmethod
    def get_hdfs_shard_uri(split: str):
        if split == "full":
            train_shards = "pipe: hdfs dfs -cat hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200/{00000..00013}.tar"
            valid_shards = "pipe: hdfs dfs -cat hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200/00014.tar"
        else:
            raise NotImplementedError("Choose between `full`")
        return train_shards, valid_shards

    @staticmethod
    def create_webdataset(dataset: BillboardHot200Dataset, pattern: str, maxsize: int, start_shard_idx: int):
        writer = ShardWriter(pattern=pattern, maxsize=maxsize, start_shard=start_shard_idx)
        index = []
        current_shard = writer.shard
        for idx, item in enumerate(tqdm(dataset)):
            if current_shard != writer.shard:
                index_fp = os.path.join(os.path.dirname(writer.fname), f"{current_shard-1:05d}.tar.index")
                print(f"Writing index: {index_fp}")
                write_index(index_fp, index)
                current_shard = writer.shard
                index = []

            id = f"{idx}-{item['metadata']['meta_song_id']}" 
            # item["audio"] = fp32_to_int16(item["audio"])
            obj = {
                "__key__": id,
                "audio.flac": item["audio"],
                "metadata.json": item["metadata"],
                "track_features.json": item["track_features"]
            }
            writer.write(obj)
            index.append(id)

        print(f"Writing index: {index_fp}")
        index_fp = os.path.join(os.path.dirname(writer.fname), f"{current_shard-1:05d}.tar.index")
        write_index(index_fp, index)
        writer.close()

    @property
    def n_audio_samples(self):
        return int(self.duration * self.sample_rate)

    def wds_transform(self, item) -> Dict[str, Any]:
        audio = item["audio.flac"]
        audio, sr = sf.read(io.BytesIO(audio))

        audio = self.base_transform(audio)
        audio = self.random_pad(audio)
        audio = self.random_crop(audio)

        shard = os.path.basename(item["__url__"])
        return {
            "audio": audio,
            "metadata": item["metadata.json"],
            "track_features": item["track_features.json"],
            "shard": shard,
        }


if __name__ == "__main__":

    split = "full" 
    audio_dir = "/mnt/bn/janne-research-xl/data/mcc/billboard_hot_200/"
    metadata_fp = "billboard_hot_200_spotify_mcc_filtered.pickle"
    track_features_dir = "/mnt/bn/janne-research-xl/data/billboard_hot200/tracks_features"
    dataset = BillboardHot200Dataset(metadata_fp=metadata_fp, audio_dir=audio_dir, track_features_dir=track_features_dir, split=split)
     
    pattern = f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200/%05d.tar"
    maxsize = (1 << 32) * 2 # 8GiB, maximum size of each shard

    dataset.random_shuffle()
    # dataset._walker = dataset._walker[13910:]
    # dataset.total = len(dataset._walker)
    start_shard_idx = 0
    BillboardHot200WebDataModule.create_webdataset(dataset, pattern, maxsize, start_shard_idx=start_shard_idx)
