import json
import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
import torch
import webdataset as wds
from lightning.pytorch import LightningDataModule
from torch.utils.data import ChainDataset, DataLoader, Dataset
from torchaudio_augmentations import Compose
from tqdm import tqdm

from recipes.datasets.mir.base import (
    LightningDataModuleBase,
    WebDataModuleBase,
    _load_waveform,
    collate_batch,
    resample,
)
from samantha.utils.webdataset import return_self
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.dataio.webdataset.writer import IndexShardWriter
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
    pad_1d,
)
from samantha.transforms.tokenizers.phoneme_tokenizer import (
    MAX_PHONE_LEN,
    Wav2VecPhonemeTokenizer,
)
from samantha.dataio.dataset import MultiIterableDataset

SAMPLE_RATE = 44100

logger = logging.getLogger(__name__)


def load_json(fp: str) -> dict:
    with open(fp) as f:
        return json.load(f)


def musixmatch_to_lrc(
    musixmatch_lyric, title: str = "", artist: str = "", album: str = ""
):
    def number_formatter(n: float) -> str:
        return f"{int(n):02d}"

    lines = []
    for d in musixmatch_lyric:
        start_time_ms = float(d["startTimeMs"])
        ss, ms = divmod(start_time_ms, 1000)
        mm, ss = divmod(ss, 60)
        lyric = d["words"]

        line = f"[{number_formatter(mm)}:{number_formatter(ss)}.{number_formatter(ms)}] {lyric}"
        lines.append(line)

    lines_str = "\n".join(lines)
    lrc = f"[ar: {artist}]\n[al: {album}]\n[ti: {title}]\n\n{lines_str}"
    return lrc


class BillboardHot200Dataset(Dataset):
    """BillboardHot200 dataset.
    Args:
    root (str or Path): Path to the directory where the dataset is found or downloaded.
    split (str, optional): The split to use (small, medium, large)
    """

    def __init__(
        self,
        metadata_fp: str,
        audio_dir: str,
        split: str,
        verify_dataset: bool = True,
        ext_audio: str = ".flac",
        lyrics_only: bool = False,
    ) -> None:
        self._split = split
        self._audio_dir = audio_dir
        self._ext_audio = ext_audio
        self._lyrics_only = lyrics_only
        data = pd.read_pickle(metadata_fp)
        data = self.process_dataframe(data)
        self.data = self.get_split(data, split)

        if not os.path.isdir(self._audio_dir):
            raise RuntimeError(f"Audio dataset not found at {self._audio_dir}.")

        if verify_dataset:
            self.verify_dataset()

        self._walker = self.data.index.tolist()
        self.total = len(self._walker)

    @property
    def audio_filepaths(self):
        return self.data["audio_fp"].tolist()

    def process_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.process_split(data)
        data = data.drop_duplicates(
            "meta_song_id"
        )  # TODO: spotify_track_id has ~250k records

        data = data.drop_duplicates("spotify_track_id")

        if self._lyrics_only:
            data = data.dropna(subset=["spotify_lyric_lines"])

            # english only
            data = data[data["spotify_lyric_language"] == "en"]

            # synced only
            data = data[data["spotify_lyric_syncType"] == "LINE_SYNCED"]
        data["audio_fp"] = data["meta_song_id"].apply(self.get_audio_path)
        return data

    def process_split(self, data: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        """This function generates a split for the Billboard Hot 200 dataset.
        We make sure to include tracks from non-overlapping artists in our train/test
        splits. The seed (default: 42) ensures that the split is reproducible.

        Args:
            data (pd.DataFrame): _description_
            seed (int, optional): _description_. Defaults to 42.

        Returns:
            pd.DataFrame: _description_
        """

        artists = data["spotify_primary_artist_name"].unique()
        n_artists = len(artists)
        n_test_artists = int(0.1 * n_artists)

        random.seed(seed)
        test_artists = random.sample(sorted(artists), k=n_test_artists)

        data["split"] = "train"
        data.loc[
            data["spotify_primary_artist_name"].isin(test_artists), "split"
        ] = "test"
        return data

    def get_split(self, data: pd.DataFrame, split: str) -> pd.DataFrame:
        if split not in ["train", "test"]:
            raise Exception("Choose either 'train' or 'test' as the data split")
        return data[data["split"] == split]

    def verify_dataset(self):
        assert (self.data["split"] == self._split).sum() == len(self.data)
        new_data = []
        for _, row in tqdm(
            self.data.iterrows(),
            total=len(self.data),
            desc="Verifying audio and track features files",
        ):
            try:
                if os.path.isfile(self.get_audio_path(row["meta_song_id"])):
                    new_data.append(row)
            except Exception as e:
                print(e)
        self.data = pd.DataFrame(new_data)

    def random_shuffle(self, seed: int = 42):
        if self._split == "train":
            random.seed(seed)
            random.shuffle(self._walker)
        else:
            logger.warning("Shuffling not done as `split != train`")

    def __len__(self):
        return self.total

    def get_audio_path(self, meta_song_id: str):
        return os.path.join(self._audio_dir, meta_song_id + self._ext_audio)

    def get_track_lyrics_path(self, album_id: str, track_id: str):
        return os.path.join(self._track_lyrics_dir, album_id, track_id + ".json")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        index = self._walker[idx]
        row = self.data.loc[index]
        return row


@dataclass
class BillboardDataResult:
    audio: torch.Tensor
    artist_gender: Union[List[str], str]
    artist_gender_label: torch.Tensor
    metadata: Union[List[dict], dict]
    shard: str
    key: str
    latent: Optional[torch.Tensor] = None


class BillboardTransform:
    _artist_gender_fp = "/mnt/bn/janne-research-xl/data/musicbrainz/artist_gender.csv"
    _artist_gender_idx = {
        "male": 0,
        "female": 1,
    }
    _num_classes = len(_artist_gender_idx)

    def __init__(
        self,
        sample_rate: int,
        duration: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.duration = duration

        self.base_transform = Compose(
            [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        )

        if self.duration is not None:
            self.random_pad = RandomPad(self.n_audio_samples)
            self.random_crop = RandomResizedCrop(self.n_audio_samples)

        df = pd.read_csv(self._artist_gender_fp)
        df.musicbrainz_artist_gender = df.musicbrainz_artist_gender.str.strip()

        self.artist_gender = dict(
            zip(df.spotify_primary_artist_name, df.musicbrainz_artist_gender)
        )

    @property
    def n_classes(self) -> int:
        return self._num_classes

    @property
    def n_audio_samples(self) -> int:
        return int(self.duration * self.sample_rate)

    @staticmethod
    def create_webdataset(
        dataset: BillboardHot200Dataset,
        sample_rate: int,
        mono: bool,
        pattern: str,
        maxsize: int,
        start_shard_idx: int,
    ):
        writer = IndexShardWriter(
            pattern=pattern, maxsize=maxsize, start_shard=start_shard_idx
        )
        for item in tqdm(dataset):
            idx = str(item.name)
            id = f"{idx}-{item['meta_song_id']}"
            # item["audio"] = fp32_to_int16(item["audio"])

            audio_fp = item["audio_fp"]
            out_fp = resample(audio_fp, sample_rate, mono)
            if not os.path.exists(out_fp):
                # print(f"{out_fp} does not exist") # TODO log this
                continue

            audio, sr = _load_waveform(out_fp, sample_rate)
            lyrics = item["spotify_lyric_lines"]
            if not type(lyrics) == list:
                lyrics = None

            del item["audio_fp"]
            del item["spotify_lyric_lines"]

            obj = {
                "__key__": id,
                "audio.npy": audio.numpy(),
                # "track_features.json": item["track_features"],
            }

            index = {"metadata": item.to_dict(), "lyrics": lyrics}
            writer.write(obj, index)
        writer.close()

    def __call__(self, items) -> Iterator[BillboardDataResult]:
        for item in items:
            audio = item["audio.npy"]
            index = item["__index_data__"]
            audio = self.base_transform(audio)

            if self.duration is not None:
                audio = self.random_pad(audio)
                audio = self.random_crop(audio)

            metadata = index["metadata"]

            primary_artist_name = metadata["spotify_primary_artist_name"]
            artist_gender = self.artist_gender[primary_artist_name]
            artist_gender_label = torch.tensor(self._artist_gender_idx[artist_gender])

            shard = item["__url__"]
            key = item["__key__"]
            yield BillboardDataResult(
                audio=audio,
                artist_gender=artist_gender,
                artist_gender_label=artist_gender_label,
                metadata=metadata,
                shard=shard,
                key=key,
            )


class BillboardMusicFMTransform:
    _artist_gender_fp = "/mnt/bn/janne-research-xl/data/musicbrainz/artist_gender.csv"
    _artist_gender_idx = {
        "male": 0,
        "female": 1,
    }
    _num_classes = len(_artist_gender_idx)

    def __init__(self):
        self.base_transform = Compose(
            [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        )
        df = pd.read_csv(self._artist_gender_fp)
        df.musicbrainz_artist_gender = df.musicbrainz_artist_gender.str.strip()

        self.artist_gender = dict(
            zip(df.spotify_primary_artist_name, df.musicbrainz_artist_gender)
        )

    def __call__(self, items) -> Iterator[BillboardDataResult]:
        for item in items:
            audio = item["audio.npy"]
            index = item["__index_data__"]
            latent = torch.tensor(item["latent.npy"])  # TODO dtype
            audio = self.base_transform(audio)

            metadata = index["metadata"]

            primary_artist_name = metadata["spotify_primary_artist_name"]

            artist_gender = self.artist_gender[primary_artist_name]
            artist_gender_label = torch.tensor(self._artist_gender_idx[artist_gender])

            shard = item["__url__"]
            key = item["__key__"]
            yield BillboardDataResult(
                audio=audio,
                latent=latent,
                artist_gender=artist_gender,
                artist_gender_label=artist_gender_label,
                metadata=metadata,
                shard=shard,
                key=key,
            )


class Billboard(LightningDataModule):
    def __init__(
        self,
        sample_rate: int,
        duration: float,
        batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # transform = BillboardMusicFMTransform()
        transform = BillboardTransform(sample_rate, duration)

        # train_dataset = IndexedWebDataset(
        #     url2index="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200_v2/24000hz/train/url2index.txt",
        #     resampled=True,
        #     shardshuffle=True,
        #     use_pipe=False,
        # )
        # test_dataset = IndexedWebDataset(
        #     url2index="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200_v2/24000hz/test/url2index.txt",
        #     resampled=False,
        #     shardshuffle=False,
        #     nodesplitter=return_self,
        #     use_pipe=False,
        # )
        # train_dataset = MultiIterableDataset(
        #     [
        #         train_dataset,
        #     ],
        # )
        # test_dataset = MultiIterableDataset(
        #     [
        #         test_dataset,
        #     ],
        # )

        male_train_dataset = IndexedWebDataset(
            url2index="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200_v2/24000hz/train/filtered/artist_gender/male/url2index.txt",
            resampled=True,
            shardshuffle=True,
            use_pipe=False,
        )

        female_train_dataset = IndexedWebDataset(
            url2index="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200_v2/24000hz/train/filtered/artist_gender/female/url2index.txt",
            resampled=True,
            shardshuffle=True,
            use_pipe=False,
        )

        male_test_dataset = IndexedWebDataset(
            url2index="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/male/url2index.txt",
            resampled=False,
            shardshuffle=False,
            nodesplitter=return_self,
            use_pipe=False,
        )

        female_test_dataset = IndexedWebDataset(
            url2index="hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/female/url2index.txt",
            resampled=False,
            shardshuffle=False,
            nodesplitter=return_self,
            use_pipe=False,
        )

        train_dataset = MultiIterableDataset(
            [
                male_train_dataset,
                female_train_dataset,
            ],
            weights=[0.5, 0.5],
        )
        test_dataset = MultiIterableDataset(
            [
                # male_test_dataset,
                female_test_dataset,
            ],
            # weights=[0.5, 0.5],
        )
        self.train_dataset = wds.DataPipeline(
            train_dataset,
            wds.decode(),
            transform,
            wds.shuffle(self.shuffle_buffer_size),
            wds.batched(self.batch_size, collation_fn=collate_batch, partial=False),
        )
        self.test_dataset = wds.DataPipeline(
            test_dataset,
            wds.decode(),
            transform,
            wds.batched(self.batch_size, collation_fn=collate_batch, partial=False),
        )
        self.validation_dataset = self.test_dataset
        self.test_dataset = self.test_dataset
        self.predict_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=None,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=None,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=None,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=None,
        )
