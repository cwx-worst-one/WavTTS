import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
import torch
import webdataset as wds
from torch.utils.data import Dataset
from tqdm import tqdm

from recipes.datasets.base import (
    BaseAudioTransform,
    WebDataModuleBase,
    _load_waveform,
    resample,
    audio_batcher,
    DataResult,
)
from samantha.dataio.data_bucket import data_bucket
from samantha.dataio.dataset import MultiIterableDataset
from samantha.utils.webdataset import return_self
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.writer import IndexShardWriter
from samantha.transforms.audio import (
    RandomPad,
    RandomResizedCrop,
)

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
class BillboardDataResult(DataResult):
    target_audio: torch.Tensor
    metadata: Union[List[dict], dict]
    shard: str
    key: str
    artist_gender: Optional[Union[List[str], str]] = None
    artist_gender_label: Optional[torch.Tensor] = None
    conditions: Optional[str] = None
    style_text: Optional[str] = None
    lyrics_text: Optional[str] = None
    lyrics_tokens: Optional[torch.Tensor] = None


from recipes.bigmusic.datasets.transforms.lyrics_segment import (
    crop_pad_audio_to_segment,
    lyrics_to_segments,
)
from recipes.bigmusic.datasets.transforms.lyrics import LyricsTokenTransform
from recipes.bigmusic.utils.format_utils import rewrite_metadata


class BillboardTransform:
    def __init__(
        self,
        sample_rate: int,
        duration: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.duration = duration

        self.base_transform = BaseAudioTransform()

        if self.duration is not None:
            self.random_pad = RandomPad(self.n_audio_samples)
            self.random_crop = RandomResizedCrop(self.n_audio_samples)

    @property
    def n_audio_samples(self) -> int:
        return int(self.duration * self.sample_rate)

    def __call__(self, items) -> Iterator[BillboardDataResult]:
        for item in items:
            audio = item["audio.npy"]
            index = item["__index_data__"]
            audio = self.base_transform(audio)

            if self.duration is not None:
                audio = self.random_pad(audio)
                audio = self.random_crop(audio)

            metadata = index["metadata"]

            shard = item["__url__"]
            key = item["__key__"]
            yield BillboardDataResult(
                target_audio=audio,
                metadata=metadata,
                shard=shard,
                key=key,
            )


from collections import Counter


class BillboardLyricsTransform:
    def __init__(
        self,
        sample_rate: int,
        min_duration: int,
        max_duration: int,
    ):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.base_transform = BaseAudioTransform()
        self.tokenizer = LyricsTokenTransform.init_espeak_tokenizer(
            lyrics_max_seq_len=400, enable_punctuation=True
        )

    def billboard_to_compat_style_text(self, index: dict) -> str:
        meta_dict = {}

        ## METADATA

        meta_dict["final_mood"] = None
        meta_dict["final_genre"] = index.get("genre")  # 'style'

        vocal_tags = [l for l in index["tags"]["vocal"] for l in l]
        vocal_tags = Counter(vocal_tags)

        if vocal_tags["gender_male"] > vocal_tags["gender_female"]:
            meta_dict["merge_aed"] = "Male"
        else:
            meta_dict["merge_aed"] = "Female"

        style_text = rewrite_metadata(meta_dict)
        return style_text

    def billboard_to_compat_lyrics(self, index: dict) -> List[Any]:
        utterances = index["lyrics"]["result"][0]["utterances"]
        segments = lyrics_to_segments(
            utterances,
            fixed_duration=False,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            shuffle_start=True,
            shuffle_lengths=True,
            include_intro=True,
        )
        return segments

    def __call__(self, items) -> Iterator[BillboardDataResult]:
        for item in items:
            audio = item["audio.npy"]
            index = item["__index_data__"]
            audio = self.base_transform(audio)

            metadata = index["metadata"]

            style_text = self.billboard_to_compat_style_text(index)
            segments = self.billboard_to_compat_lyrics(index)

            for segment in segments:
                clip = crop_pad_audio_to_segment(segment, audio[0], self.sample_rate)
                clip = clip.unsqueeze(dim=0)

                lyrics_dict = self.tokenizer({"lyrics": segment.text})
                if lyrics_dict is None:
                    continue

                shard = item["__url__"]
                key = item["__key__"]
                yield BillboardDataResult(
                    target_audio=clip,
                    metadata=metadata,
                    shard=shard,
                    key=key,
                    conditions="style_text,lyrics_tokens",
                    style_text=style_text,
                    lyrics_text=lyrics_dict["lyrics_normalized_text"],
                    lyrics_tokens=lyrics_dict["lyrics_tokens"],
                )


class BillboardArtistGenderTransform(BillboardTransform):
    _artist_gender_fp = "recipes/datasets/billboard/artist_gender.csv"
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
        super().__init__(sample_rate=sample_rate, duration=duration)
        df = pd.read_csv(self._artist_gender_fp)
        df.musicbrainz_artist_gender = df.musicbrainz_artist_gender.str.strip()

        self.artist_gender = dict(
            zip(df.spotify_primary_artist_name, df.musicbrainz_artist_gender)
        )

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
                target_audio=audio,
                artist_gender=artist_gender,
                artist_gender_label=artist_gender_label,
                metadata=metadata,
                shard=shard,
                key=key,
            )


class BillboardDataModule(WebDataModuleBase):
    _sample_rate: int = 24000

    def __init__(
        self,
        batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int,
        pin_memory: bool,
        duration: Optional[float] = None,
    ):
        transform = BillboardTransform(self._sample_rate, duration)

        train_dataset = IndexedWebDataset(
            url2index=data_bucket(
                "music/billboard_hot200_v2/24000hz/train/url2index.txt"
            ),
            resampled=True,
            shardshuffle=True,
            use_pipe=False,
            handler=wds.warn_and_continue
        )
        test_dataset = IndexedWebDataset(
            url2index=data_bucket(
                "music/billboard_hot200_v2/24000hz/test/url2index.txt"
            ),
            resampled=False,
            shardshuffle=False,
            nodesplitter=return_self,
            use_pipe=False,
            handler=wds.warn_and_continue
        )

        train_dataset = train_dataset.decode().compose(transform)
        test_dataset = test_dataset.decode().compose(transform)
        super().__init__(
            train_dataset=train_dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            validation_dataset=test_dataset,
            test_dataset=test_dataset,
            predict_dataset=train_dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # batcher=batcher,
        )

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


class BillboardLyricsDataModule(WebDataModuleBase):
    _sample_rate: int = 24000

    def __init__(
        self,
        batch_size: int,
        shuffle_buffer_size: int,
        buckets_sec: List[int],
        num_workers: int,
        pin_memory: bool,
    ):
        batcher = audio_batcher(self._sample_rate, batch_size, buckets_sec, length_fn=lambda x: x.target_audio.shape[-1])
        transform = BillboardLyricsTransform(
            self._sample_rate,
            min_duration=buckets_sec[0],
            max_duration=buckets_sec[-1],
        )

        train_dataset = IndexedWebDataset(
            url2index="/mnt/bn/audio-diffusion/ashaw/webdataset/billboard_v2/url2index_train.txt", # has 98/2 train/test split
            # url2index=data_bucket(
            #     "music/billboard_hot200_v2/24000hz/train/20231026_genre/url2index.txt" # 90/1 train/test split
            # ),
            resampled=True,
            shardshuffle=True,
            use_pipe=False,
            handler=wds.warn_and_continue
        )
        test_dataset = IndexedWebDataset(
            url2index="/mnt/bn/audio-diffusion/ashaw/webdataset/billboard_v2/url2index_test.txt",
            # url2index=data_bucket(
            #     "music/billboard_hot200_v2/24000hz/test/20231026_genre/url2index.txt"
            # ),
            resampled=False,
            shardshuffle=False,
            nodesplitter=return_self,
            use_pipe=False,
            handler=wds.warn_and_continue
        )

        train_dataset = train_dataset.decode().compose(transform)
        test_dataset = test_dataset.decode().compose(transform)
        super().__init__(
            train_dataset=train_dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            validation_dataset=test_dataset,
            test_dataset=test_dataset,
            predict_dataset=train_dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            batcher=batcher,
        )


class BillboardArtistGenderDataModule(WebDataModuleBase):
    _sample_rate: int = 24000
    
    def __init__(
        self,
        duration: float,
        batch_size: int,
        shuffle_buffer_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        transform = BillboardArtistGenderTransform(self._sample_rate, duration)

        train_dataset_male = IndexedWebDataset(
            url2index=data_bucket(
                "music/billboard_hot200_v2/24000hz/train/filtered/artist_gender/male/url2index.txt"
            ),
            resampled=True,
            shardshuffle=True,
            use_pipe=False,
            handler=wds.warn_and_continue
        )
        train_dataset_female = IndexedWebDataset(
            url2index=data_bucket(
                "music/billboard_hot200_v2/24000hz/train/filtered/artist_gender/female/url2index.txt"
            ),
            resampled=True,
            shardshuffle=True,
            use_pipe=False,
            handler=wds.warn_and_continue
        )

        test_dataset_male = IndexedWebDataset(
            url2index=data_bucket(
                "music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/male/url2index.txt"
            ),
            resampled=False,
            shardshuffle=False,
            nodesplitter=return_self,
            use_pipe=False,
            handler=wds.warn_and_continue
        )
        test_dataset_female = IndexedWebDataset(
            url2index=data_bucket(
                "music/billboard_hot200_v2/24000hz/test/filtered/artist_gender/female/url2index.txt"
            ),
            resampled=False,
            shardshuffle=False,
            nodesplitter=return_self,
            use_pipe=False,
            handler=wds.warn_and_continue
        )

        train_dataset = wds.DataPipeline(
            MultiIterableDataset(
                [train_dataset_male, train_dataset_female], weights=[0.3, 0.7]
            ),
            wds.decode(),
            transform,
        )
        test_dataset = wds.DataPipeline(
            MultiIterableDataset([test_dataset_male, test_dataset_female]),
            wds.decode(),
            transform,
        )

        super().__init__(
            train_dataset=train_dataset,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            validation_dataset=test_dataset,
            test_dataset=test_dataset,
            predict_dataset=test_dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
