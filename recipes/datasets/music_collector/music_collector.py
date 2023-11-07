import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import eyed3
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchaudio_augmentations import Compose
from tqdm import tqdm

from recipes.datasets.base import BaseDataModule, _load_waveform
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
)
from samantha.dataio.webdataset.writer import ShardWriter
from samantha.utils.hdfs_tools import hdfs_open
from samantha.utils.hdfs_helper import ishdfs
from webdataset import warn_and_continue

SAMPLE_RATE = 44100

logger = logging.getLogger(__name__)


def lrc_to_timestamps(lyrics):
    lyric_timestamps = []
    for lyric in lyrics:
        m = re.match(r"\[([0-9][0-9]):([0-9][0-9]).([0-9][0-9])\]\s(.*)", lyric)
        if m is not None:
            mm = int(m[1])
            ss = int(m[2])
            ms = int(m[3])  # hundreth of seconds, so :90 = 0.9s
            lyric = m[4]
            start_time = (mm * 60) + ss + (ms / 100)

            lyric_timestamps.append({"startTimeMs": start_time * 1000, "lyrics": lyric})
    return lyric_timestamps


def resample_cmd(audio_fp: str, out_fp: str, sample_rate: int, mono: bool):
    n_channels = 1 if mono else 2
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_fp,
        "-ac",
        str(n_channels),
        "-ar",
        str(sample_rate),
        out_fp,
    ]


def resample(audio_fp: str, sample_rate: int, mono: bool):
    out_fp = audio_fp + f"-resampled_{sample_rate}hz.wav"
    if not os.path.exists(out_fp):
        p = subprocess.Popen(resample_cmd(audio_fp, out_fp, sample_rate, mono))
        p.wait()
    return out_fp


class IndexShardWriter(ShardWriter):
    def __init__(
        self,
        pattern: str,
        maxcount: int = 100000,
        maxsize: float = 3e9,
        post: Optional[Callable] = None,
        start_shard: int = 0,
        **kw,
    ):
        self.index = None
        self.url2index = []
        super().__init__(
            pattern=pattern,
            maxcount=maxcount,
            maxsize=maxsize,
            post=post,
            start_shard=start_shard,
            **kw,
        )

    @property
    def url2index_fp(self) -> str:
        return os.path.join(os.path.dirname(self.pattern), "url2index.txt")

    @staticmethod
    def open_stream(fp:str):
        if ishdfs(fp):
            return hdfs_open(fp, mode="w")
        else:
            return open(fp, "w")

    def write_url2index(self, fp: str, url2index: List[Tuple[str, str]]):
        f = self.open_stream(fp)
        for u in url2index:
            f.write(f"{u[0]}\t{u[1]}\n")
        f.close()

    def write_index(self, fp: str, index: List[Tuple[str, dict]]):
        f = self.open_stream(fp)
        for idx in index:
            f.write(f"{idx[0]}\t{json.dumps(idx[1])}\n")
        f.close()

    def finish(self):
        if self.index is not None:
            self.index_fname = self.pattern % (self.shard - 1) + ".index"
            self.write_index(self.index_fname, self.index)
            self.url2index.append((self.fname, self.index_fname))
            self.write_url2index(self.url2index_fp, self.url2index)

        self.index = []

    def write(self, obj: dict, index: dict):
        if (
            self.tarstream is None
            or self.count >= self.maxcount
            or self.size >= self.maxsize
        ):
            self.next_stream()

        key = obj["__key__"].strip()
        self.index.append((key, index))
        return super().write(obj)


class SpotDLMetadata:
    def __init__(self, artist_name, track_name, lyrics, genre, album_name):
        self.artist_name = artist_name
        self.track_name = track_name
        self.lyrics = lyrics
        self.genre = genre
        self.album_name = album_name


class MusicCollectorDataset(Dataset):
    _ext_audio = ".mp3"

    def __init__(
        self,
        root_dir: str,
        spotify_type: str,
        spotify_id: str,
        download: bool,
        n_threads: int = 4,
    ):
        super().__init__()
        assert spotify_type in ["artist", "playlist"]
        self.spotify_type = spotify_type
        self.spotify_id = spotify_id
        self.n_threads = n_threads

        self.root_dir = os.path.join(root_dir, self.spotify_type, self.spotify_id)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        if download:
            self.download()
            self.prepare_dataset()
        else:
            self.index = self.build_index()

        self.index.to_csv(self.index_fp)

    @property
    def index_fp(self):
        return os.path.join(self.root_dir, "index.csv")

    @property
    def spotify_url(self):
        if self.spotify_type == "artist":
            return f"https://open.spotify.com/artist/{self.spotify_id}"
        elif self.spotify_type == "playlist":
            return f"https://open.spotify.com/playlist/{self.spotify_id}"

    @property
    def local_audio_fp(self):
        return os.path.join(self.root_dir, "{artists}_{title}_{year}.{output-ext}")

    @property
    def download_cmd(self):
        # --only-verified-results
        # --threads
        # --fetch-albums
        # --output {isrc}
        return [
            "python3",
            "-m",
            "spotdl",
            "download",
            self.spotify_url,
            "--output",
            self.local_audio_fp,
            "--lyrics",
            "synced",
            "--generate-lrc",
            "--threads",
            str(self.n_threads),
        ]

    def download(self):
        cmd = " ".join(self.download_cmd)
        logger.info(f"Starting spotdl using the following command:\n{cmd}")
        p = subprocess.Popen(
            self.download_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # shell=True,
        )
        p.wait()
        std_out, std_err = p.communicate()

    def prepare_dataset(self):
        self.index = self.build_index()

    def prepare_lyrics(self, lyrics_fp):
        with open(lyrics_fp, "r") as f:
            lyrics = f.read().splitlines()
        return lrc_to_timestamps(lyrics)

    def build_index(self):
        index = pd.DataFrame([])
        fps = glob(os.path.join(self.root_dir, f"*{self._ext_audio}"))
        for audio_fp in fps:
            audio_fp = Path(audio_fp)
            if audio_fp.exists():
                audiofile = eyed3.load(audio_fp)

                lyrics_fp = audio_fp.with_suffix(".lrc")
                lyrics = None
                if lyrics_fp.exists():
                    lyrics = self.prepare_lyrics(lyrics_fp)

                if hasattr(audiofile.tag.genre, "name"):
                    genre = audiofile.tag.genre.name
                else:
                    genre = None

                metadata = SpotDLMetadata(
                    artist_name=audiofile.tag.artist,
                    track_name=audiofile.tag.title,
                    lyrics=lyrics,
                    genre=genre,
                    album_name=audiofile.tag.album,
                )
                metadata = vars(metadata)
                metadata.update({"audio_fp": audio_fp})

                index = pd.concat((index, pd.DataFrame([metadata])))
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        return self.index.iloc[idx].to_dict()


class MusicCollectorWebLoader(BaseDataModule):
    def __init__(
        self,
        url2index: str,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int,
        pin_memory: bool = True,
        resampled: bool = True,
        shardshuffle: bool = True,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.data_sample_rate = sample_rate  # TODO

        self.audio_transforms = Compose(
            [
                ToTensor(),
                SetAudioDimensions(),
                NormalizeAudioToFloat32(),
                RandomPad(self.duration_audio_samples),
            ]
        )
        self.rand_crop = RandomResizedCrop(self.duration_audio_samples)

        pipeline = []
        pipeline.append("decode")
        pipeline.append({"map": [self.wds_transform]})

        dataset = IndexedWebDataset(
            url2index=url2index,
            resampled=resampled,
            shardshuffle=shardshuffle,
            use_pipe=True,
            handler=warn_and_continue,
        )
        dataset = WebPipeline(dataset, pipeline)
        super().__init__(
            sample_rate=sample_rate,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=dataset,
        )

    @property
    def duration_audio_samples(self):
        return self.duration * self.sample_rate

    def wds_transform(self, item) -> Dict[str, Any]:
        audio = self.audio_transforms(item["audio.npy"])

        audio = self.rand_crop(audio)
        index = item["__index_data__"]
        return {"audio": audio, "metadata": index}

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List]:
        keys = batch[0].keys()
        batch_dict = defaultdict(list)
        for b in batch:
            for k in keys:
                batch_dict[k].append(b[k])

        batch_dict = dict(batch_dict)
        batch_dict["audio"] = torch.stack(batch_dict["audio"], dim=0)
        return batch_dict

    @staticmethod
    def create_webdataset(
        dataset: MusicCollectorDataset, sample_rate: int, mono: bool, pattern: str, maxsize: int
    ):
        # dir_fp = os.path.dirname(pattern)
        # os.makedirs(dir_fp, exist_ok=True)
        
        writer = IndexShardWriter(pattern=pattern, maxsize=maxsize)
        for idx, item in enumerate(tqdm(dataset, desc="Creating webdataset...")):
            audio_fp = item["audio_fp"]

            index = item
            del index["audio_fp"]

            out_fp = resample(str(audio_fp), sample_rate, mono)

            audio, sr = _load_waveform(out_fp, sample_rate)

            index["spotify_type"] = dataset.spotify_type

            key = f'{idx}-{item["artist_name"]} - {item["track_name"]}'
            obj = {"__key__": key, "audio.npy": audio.numpy()}

            writer.write(obj, index)
        writer.close()
