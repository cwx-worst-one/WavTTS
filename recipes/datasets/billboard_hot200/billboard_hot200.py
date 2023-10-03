import json
import logging
import os
import random
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchaudio_augmentations import Compose
from tqdm import tqdm
# from recipes.bigmusic.datasets.tokenizers.phoneme import Wav2VecPhonemeTokenizer
from recipes.datasets.base import (
    BaseDataModule,
    _load_waveform,
    resample,
    collate_batch,
)
from samantha.dataio.webdataset import IndexShardWriter
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.dataio.webdataset.pipeline import WebPipeline
from samantha.transforms.audio import (
    NormalizeAudioToFloat32,
    RandomPad,
    RandomResizedCrop,
    SetAudioDimensions,
    ToTensor,
    pad_1d,
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


class BillboardDataModule(BaseDataModule):
    data_sample_rate = SAMPLE_RATE

    def __init__(
        self,
        url2index: str,
        sample_rate: int,
        batch_size: int,
        shuffle_buffer_size: int = 64,
        duration: Optional[float] = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        resampled: bool = True,
        shardshuffle: bool = True,
    ):
        self.url2index = url2index
        self.sample_rate = sample_rate
        self.duration = duration
        # self.phoneme_tokenizer = Wav2VecPhonemeTokenizer()

        transforms = [self.wds_transform]
        dataset = self.get_dataset(
            url2index=url2index,
            resampled=resampled,
            shardshuffle=shardshuffle,
            transforms=transforms,
        )

        super().__init__(
            sample_rate=sample_rate,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=dataset,
            validation_dataset=dataset,
            predict_dataset=dataset,
        )
        self.base_transform = Compose(
            [ToTensor(), SetAudioDimensions(), NormalizeAudioToFloat32()]
        )

        if self.duration is not None:
            self.random_pad = RandomPad(self.n_audio_samples)
            self.random_crop = RandomResizedCrop(self.n_audio_samples)

    @property
    def n_audio_samples(self) -> int:
        return int(self.duration * self.sample_rate)

    @staticmethod
    def get_dataset(
        url2index: str,
        resampled: bool,
        shardshuffle: bool,
        transforms: List[Callable] = [],
    ) -> Tuple[WebPipeline, WebPipeline, WebPipeline]:
        dataset = IndexedWebDataset(
            url2index=url2index,
            resampled=resampled,
            shardshuffle=shardshuffle,
            use_pipe=True,  # False
        )
        pipeline = []
        pipeline.append("decode")
        pipeline.append({"compose": transforms})
        return WebPipeline(dataset, pipeline)

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
        for idx, item in enumerate(tqdm(dataset)):
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
                # f"audio{dataset._ext_audio}": item["audio"],
                # "track_features.json": item["track_features"],
            }

            index = {"metadata": item.to_dict(), "lyrics": lyrics}
            writer.write(obj, index)
        writer.close()

    def collate_fn(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_dict = collate_batch(batch)
        return self.stack_batch_dict(batch_dict)

    def stack_batch_dict(self, batch_dict):
        batch_dict["audio"] = torch.stack(batch_dict["audio"], dim=0)
        return batch_dict

    def wds_transform(self, items) -> Iterator[Dict[str, Any]]:
        for item in items:
            audio = item["audio.npy"]
            index = item["__index_data__"]
            audio = self.base_transform(audio)

            if self.duration is not None:
                audio = self.random_pad(audio)
                audio = self.random_crop(audio)

            shard = os.path.basename(item["__url__"])
            yield {
                "audio": audio,
                "metadata": index["metadata"],
                "lyrics": lyrics,
                "shard": shard,
            }


class BillboardLyricsDataModule(BillboardDataModule):
    def __init__(
        self,
        url2index: str,
        sample_rate: int,
        duration: int,
        batch_size: int,
        shuffle_buffer_size: int = 64,
        num_workers: int = 32,
        pin_memory: bool = True,
        resampled: bool = True,
        shardshuffle: bool = True,
    ):
        super().__init__(
            url2index=url2index,
            sample_rate=sample_rate,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            duration=duration,
            num_workers=num_workers,
            pin_memory=pin_memory,
            resampled=resampled,
            shardshuffle=shardshuffle,
        )

    @staticmethod
    def group_lyrics(lyrics, duration: int, padding_sec: int = 5):
        # TODO: Group backwards?
        min_padding_ms = (duration - padding_sec) * 1000

        filtered_lyrics = []
        for i in range(len(lyrics) - 1):
            current_lyric = lyrics[i]
            next_lyric = lyrics[i + 1]

            current_lyric["startTimeMs"] = int(current_lyric["startTimeMs"])
            delta_time = abs(
                int(next_lyric["startTimeMs"]) - current_lyric["startTimeMs"]
            )
            current_lyric.update({"deltaTimeMs": delta_time})
            if delta_time > 0:
                filtered_lyrics.append(current_lyric)

        grouped_lyrics = []
        for i in range(len(filtered_lyrics)):
            g_lyrics = []
            acc_delta_time = 0
            for j in range(i, len(filtered_lyrics)):
                if acc_delta_time >= min_padding_ms:
                    break

                lyric = deepcopy(filtered_lyrics[j])

                acc_delta_time += lyric["deltaTimeMs"]
                lyric["totalDurationMs"] = acc_delta_time
                g_lyrics.append(lyric)

            grouped_lyrics.append(g_lyrics)

        # filter out lyrics above and below threshold
        filtered_grouped_lyrics = []
        for i in range(len(grouped_lyrics)):
            total_duration_ms = grouped_lyrics[i][-1]["totalDurationMs"]
            if total_duration_ms < min_padding_ms:
                continue

            filtered_grouped_lyrics.append(grouped_lyrics[i])
        return filtered_grouped_lyrics

    @staticmethod
    def align_audio_with_lyrics(
        audio: torch.Tensor, lyrics, sample_rate: int, max_audio_samples: int
    ) -> torch.Tensor:
        # align audio with lyrics
        start_time = lyrics[0]["startTimeMs"]
        duration_ms = lyrics[-1]["totalDurationMs"]

        start_time_samples = int((start_time / 1000) * sample_rate)
        duration_samples = int((duration_ms / 1000) * sample_rate)

        # crop, or pad, from right
        if duration_samples > max_audio_samples:
            duration_samples = max_audio_samples

        audio = audio[:, start_time_samples : start_time_samples + duration_samples]

        if audio.shape[1] < max_audio_samples:
            audio = pad_1d(audio, start_idx=0, n_samples=max_audio_samples)
        return audio

    def stack_batch_dict(self, batch_dict):
        batch_dict["audio"] = torch.stack(batch_dict["audio"], dim=0)
        # batch_dict["phoneme_tokens"] = self.phoneme_tokenizer.collate_fn(
        #     batch_dict["phoneme_tokens"]
        # )
        return batch_dict
    
    def get_phoneme_tokens(self, text: str) -> torch.Tensor:
        return self.phoneme_tokenizer([text])[0]

    def wds_transform(self, items) -> Iterator[Dict[str, Any]]:
        for item in items:
            audio = item["audio.npy"]
            index = item["__index_data__"]

            # malformed audio
            if audio.shape[1] == 0:
                continue

            audio = self.base_transform(audio)

            # lyrics
            lyrics = index["lyrics"]
            grouped_lyrics = self.group_lyrics(lyrics, duration=self.duration)
            if not len(grouped_lyrics):
                continue

            group_idx = random.randint(0, len(grouped_lyrics) - 1)
            lyrics = grouped_lyrics[group_idx]
            words = "\n".join([l["words"] for l in lyrics])

            audio = self.align_audio_with_lyrics(
                audio, lyrics, self.sample_rate, self.n_audio_samples
            )

            # phoneme_tokens = self.get_phoneme_tokens(words)

            shard = os.path.basename(item["__url__"])
            yield {
                "audio": audio,
                "metadata": index["metadata"],
                "lyrics": words,
                # "phoneme_tokens": phoneme_tokens,
                "shard": shard,
            }

