import json
import logging
import os
import random
import re
from functools import partial
from io import BytesIO
from time import time
from typing import Callable, Generator, Iterable, List, Optional, Union

import torch
import torch.nn.functional as F
import webdataset as wds
from torchaudio_augmentations import Compose
from typing_extensions import Self
from webdataset.filters import reraise_exception

from samantha.data.audio.types import AudioDataResult, AudioMeta, SegmentInfo, ShardInfo
from samantha.data.audio_utils import convert_audio
from samantha.data.av_audio import audio_info, audio_read
from samantha.data.lyrics import Lyrics
from samantha.dataio.webdataset.decoder import DecoderWithSampleKey
from samantha.dataio.webdataset.extension import IndexedWebDataset
from samantha.transforms.audio import RescaleAudio

logger = logging.getLogger(__name__)


def audio_decoder(key: str, data, sample_key: str, sample, read_fn: Callable):
    """Decode audio using the torchaudio library.

    :param key: file name extension
    :param data: data to be decoded
    """

    format = re.sub(r".*[.]", "", key)
    if format not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None

    data = BytesIO(data)
    return read_fn(data, key, format, sample_key, sample)


class AudioWebDatasetBase(IndexedWebDataset):
    def __init__(
        self,
        url2index: int,
        resampled: bool = True,
        shardshuffle: bool = True,
        detshuffle: bool = False,
        nodesplitter=wds.shardlists.split_by_node,
        shuffle_buffer_size: int = 250,  # TODO
        resolve_relative_path: bool = False,
        rng: Optional[random.Random] = None,
        resampled_split_by_nodes: bool = False,
    ):
        super().__init__(
            url2index=url2index,
            resampled=resampled,
            shardshuffle=shardshuffle,
            detshuffle=detshuffle,
            nodesplitter=nodesplitter,
            use_pipe=True,  # TODO
            handler=wds.warn_and_continue,
            resolve_relative_path=resolve_relative_path,
            rng=rng,
            resampled_split_by_nodes=resampled_split_by_nodes,
        )
        self.shuffle_buffer_size = shuffle_buffer_size

        decoders = self.decoders()
        if decoders is not None:
            self = self.decode(decoders)
        else:
            self = self.decode()

        self = self.compose(self.transform)

        # post-transform shuffling, otherwise full audio has to be
        # stored in memory
        if self.shuffle_buffer_size > 0:
            logger.info(f"Shuffle buffer size: {self.shuffle_buffer_size}")
            self = self.shuffle(self.shuffle_buffer_size)

    def decode(
        self,
        handlers: List[Callable],
        pre=None,
        post=None,
        only=None,
        partial=False,
        handler=reraise_exception,
    ) -> Self:
        decoder = DecoderWithSampleKey(
            handlers, pre=pre, post=post, only=only, partial=partial
        )
        return self.map(decoder, handler=handler)

    def decoders(self) -> List[Callable]:
        return []


class AudioWebDataset(AudioWebDatasetBase):
    def __init__(
        self,
        url2index: Union[str, List[str]],
        data_type: str,
        sample_rate: int,
        shuffle_buffer_size: int,
        channels: int = 2,
        pad: bool = True,
        segment_duration: Optional[float] = None,
        resampled: bool = True,
        shardshuffle: bool = True,
        detshuffle: bool = False,
        nodesplitter=wds.shardlists.split_by_node,
        min_segment_ratio: float = 1.0,
        max_read_retry: int = 10,
        min_audio_duration: Optional[float] = None,
        max_audio_duration: Optional[float] = None,
        lyrics_data_path: Optional[str] = None,
        crop_from_start: bool = False,
        n_segments_per_read: int = 1,
        resampled_split_by_nodes: bool = False,
        nitems: int = -1,
    ):
        assert segment_duration is None or segment_duration > 0
        assert segment_duration is None or min_segment_ratio >= 0
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration
        if self.min_audio_duration is not None and self.max_audio_duration is not None:
            assert self.min_audio_duration <= self.max_audio_duration

        self.sample_rate = sample_rate
        self.data_type = data_type
        self.channels = channels
        self.pad = pad
        self.max_read_retry = max_read_retry
        self.lyrics_data_path = lyrics_data_path
        self.crop_from_start = crop_from_start
        self.n_segments_per_read = n_segments_per_read
        self.nitems = nitems

        rng = None

        self.audio_transform = Compose(
            [
                # RandomApply([Silence()], p=0.05),
                RescaleAudio()
            ]
        )

        if lyrics_data_path is not None:
            assert self.min_audio_duration and self.max_audio_duration
            with open(
                "/mnt/bn/janne-research-xl/data/billboard_hot200/billboard_hot_200-v2_lyrics.json",
                "r",
            ) as f:
                self.lyrics_data = json.load(f)
                logger.warn(f"Lyrics data contains {len(self.lyrics_data)} entries")

        super().__init__(
            url2index=url2index,
            resampled=resampled,
            shardshuffle=shardshuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            detshuffle=detshuffle,
            nodesplitter=nodesplitter,
            resolve_relative_path=True,
            rng=rng,
            resampled_split_by_nodes=resampled_split_by_nodes,
        )

    def decoders(self) -> List[Callable]:
        return [partial(audio_decoder, read_fn=self.read_audio_segment)]

    def get_lyrics(self, key: str) -> Lyrics:
        if key in self.lyrics_data:
            lyrics_dict = self.lyrics_data[key]["lyrics"]
            return Lyrics.load_from_dict(lyrics_dict)
        return None

    def transform(self, items: Iterable) -> Generator:
        for item in items:
            # TODO: remove try/except...
            try:
                key = item["__key__"]
                segments = item["audio.mp3"]

                if not len(segments):
                    continue

                for out, segment_info in segments:
                    if self.audio_transform is not None:
                        out = self.audio_transform(out)

                    yield AudioDataResult(
                        audio=out,
                        segment_info=segment_info,
                        index=item["__index_data__"],
                        shard=item["__url__"],
                        key=key,
                        shard_info=ShardInfo(
                            url=item["__url__"], uttid=item["__key__"]
                        ),
                    )
            except Exception as e:
                logger.error(e)
                continue

    def read_audio_segment(
        self, data: BytesIO, key: str, format: str, sample_key: str, sample
    ):
        # TODO: remove try/except...
        try:
            info = audio_info(data, format=format)
            meta = AudioMeta(
                path=sample_key, duration=info.duration, sample_rate=info.sample_rate
            )

            if (
                self.min_audio_duration is not None
                and meta.duration < self.min_audio_duration
            ):
                return []

            if (
                self.max_audio_duration is not None
                and meta.duration > self.max_audio_duration
            ):
                return []

            if self.lyrics_data_path is None:
                lyrics = None
            else:
                lyrics = self.get_lyrics(sample_key)
                logger.warning("No lyrics found for %s", sample_key)
                if lyrics is None:
                    return []

            if self.segment_duration is None:
                out, sr = audio_read(data, pad=self.pad, format=format)
                out = convert_audio(out, sr, self.sample_rate, self.channels)
                n_frames = out.shape[-1]
                segment_info = SegmentInfo(
                    meta,
                    seek_time=0.0,
                    n_frames=n_frames,
                    total_frames=n_frames,
                    sample_rate=self.sample_rate,
                    channels=out.shape[0],
                    data_type=self.data_type,
                    lyrics=lyrics,
                )
                return [(out, segment_info)]
            elif self.crop_from_start:
                seek_time = 0.0
                duration = self.segment_duration

                out, sr = audio_read(
                    data, seek_time, duration, pad=False, format=format
                )
                out = convert_audio(out, sr, self.sample_rate, self.channels)
                n_frames = out.shape[-1]
                target_frames = int(duration * self.sample_rate)
                if self.pad:
                    out = F.pad(out, (0, target_frames - n_frames))

                segment_info = SegmentInfo(
                    meta,
                    seek_time,
                    n_frames=n_frames,
                    total_frames=target_frames,
                    sample_rate=self.sample_rate,
                    channels=out.shape[0],
                    data_type=self.data_type,
                    lyrics=lyrics,
                )
                return [(out, segment_info)]
            else:
                ret = []
                for _ in range(self.n_segments_per_read):
                    if lyrics is not None:
                        # Option #1: get start/end from lyrics timestamp:
                        segments = lyrics.get_segments(
                            min_duration=self.min_audio_duration,
                            max_duration=self.max_audio_duration,
                        )
                        if not len(segments):
                            logger.warning("No valid segments found for %s", meta.path)
                            return []

                        lyrics: Lyrics = random.choice(segments)
                        seek_time = lyrics.start_time
                        duration = lyrics.end_time - lyrics.start_time
                    else:
                        # Option #2: randomly sample start/end timestamp:
                        rng = torch.Generator()

                        # TODO salient excerpt
                        max_seek = max(
                            0,
                            meta.duration
                            - self.segment_duration * self.min_segment_ratio,
                        )
                        seek_time = torch.rand(1, generator=rng).item() * max_seek
                        duration = self.segment_duration

                    out, sr = audio_read(
                        data, seek_time, duration, pad=False, format=format
                    )
                    out = convert_audio(out, sr, self.sample_rate, self.channels)
                    n_frames = out.shape[-1]
                    target_frames = int(duration * self.sample_rate)
                    if self.pad:
                        out = F.pad(out, (0, target_frames - n_frames))

                    segment_info = SegmentInfo(
                        meta,
                        seek_time,
                        n_frames=n_frames,
                        total_frames=target_frames,
                        sample_rate=self.sample_rate,
                        channels=out.shape[0],
                        data_type=self.data_type,
                        lyrics=lyrics,
                    )
                    ret.append((out, segment_info))
                return ret
        except Exception as e:
            logger.error(e)
            return []


class FeatureWebDataset(AudioWebDatasetBase):
    def __init__(
        self,
        url2index: Union[str, List[str]],
        data_type: str,
        shuffle_buffer_size: int,
        channels: int = 2,
        pad: bool = True,
        segment_duration: Optional[float] = None,
        resampled: bool = True,
        shardshuffle: bool = True,
        detshuffle: bool = False,
        nodesplitter=wds.shardlists.split_by_node,
        min_segment_ratio: float = 0.5,
        max_read_retry: int = 10,
        min_audio_duration: Optional[float] = None,
        max_audio_duration: Optional[float] = None,
        lyrics_data_path: Optional[str] = None,
        crop_from_start: bool = False,
        n_segments_per_read: int = 1,
        estimate_n_samples: bool = False,
        # shuffle_seed: int = 0, # TODO: per worker?
    ):
        assert segment_duration is None or segment_duration > 0
        assert segment_duration is None or min_segment_ratio >= 0
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration
        if self.min_audio_duration is not None and self.max_audio_duration is not None:
            assert self.min_audio_duration <= self.max_audio_duration

        self.data_type = data_type
        self.channels = channels
        self.pad = pad
        self.max_read_retry = max_read_retry
        self.lyrics_data_path = lyrics_data_path
        self.crop_from_start = crop_from_start
        self.n_segments_per_read = n_segments_per_read
        # self.current_epoch: Optional[int] = None

        # self.shuffle_seed = shuffle_seed
        rng = None  # random.Random(int((os.getpid() + time()) * 1e9))

        super().__init__(
            url2index=url2index,
            resampled=resampled,
            shardshuffle=shardshuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            detshuffle=detshuffle,
            nodesplitter=nodesplitter,
            resolve_relative_path=False,
            rng=rng,
            estimate_n_samples=estimate_n_samples,
        )

    def transform(self, items: Iterable) -> Generator:
        for item in items:
            key = item["__key__"]
            original_key = item["original_key.txt"]
            # original_shard = item["original_shard.txt"]

            hidden_states = item["hidden_states.pth"]
            segment_info = item["segment_info.pickle"]

            key = (
                key if original_key is None else original_key
            )  # TODO: make sure all keys exist

            yield AudioDataResult(
                audio=hidden_states,
                segment_info=segment_info,
                index=item["__index_data__"],
                shard=item["__url__"],
                key=key,
            )
