import json
import logging
import random
import re
from abc import abstractmethod
from copy import copy
from functools import partial
from io import BytesIO
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional

import torch
import torch.nn.functional as F
import webdataset as wds
from torchaudio_augmentations import Compose
from typing_extensions import Self
from webdataset.filters import reraise_exception

from samantha.data.audio.types import AudioDataResult, AudioMeta, SegmentInfo, ShardInfo
from samantha.data.audio_utils import convert_audio
from samantha.data.av_audio import audio_info_unknown_format, audio_read
from samantha.data.lyrics import Lyrics
from samantha.dataio.batching import BucketBatcher, get_bucketed_max_length
from samantha.dataio.webdataset.decoder import DecoderWithSampleKey
from samantha.transforms.audio import Pad, RescaleAudio

logger = logging.getLogger(__name__)

from samantha.dataio.parquet import ParquetDataset


def audio_decoder(
    key: str, data, sample_key: str, sample: Dict[str, Any], read_fn: Callable
):
    """Decode audio using the torchaudio library.

    :param key: file name extension
    :param data: data to be decoded
    """

    format = re.sub(r".*[.]", "", key)
    if format not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None

    data = BytesIO(data)
    return read_fn(data, key, format, sample_key, sample)


class IndexParquetDataset(ParquetDataset):

    def __init__(
        self,
        shuffle_buffer_size: int,
        data_id: int = None,
        data_urls: List[Dict[str, str]] = None,
        resampled: bool = True,
        shardshuffle: bool = True,
        detshuffle: bool = False,
        nodesplitter=wds.shardlists.split_by_node,
        resolve_urls: bool = True
    ):
        self.shuffle_buffer_size = shuffle_buffer_size
        super().__init__(
            data_id=data_id,
            data_urls=data_urls,
            resampled=resampled,
            shardshuffle=shardshuffle,
            detshuffle=detshuffle,
            nodesplitter=nodesplitter,
            resolve_urls=resolve_urls,
            sample_config={"name": "_ParquetSampleIndex"},
        )

        self = self.rename(**{"meta.json": "meta", "text.txt": "text"})
        self = self.map(
            DecoderWithSampleKey([], only=["meta.json"]), handler=reraise_exception
        )

        self = self.compose(self.transform)

        if self.shuffle_buffer_size > 0:
            logger.info(f"Shuffle buffer size: {self.shuffle_buffer_size}")
            self = self.shuffle(self.shuffle_buffer_size)

    @abstractmethod
    def process_text(self, index: Any) -> Optional[str]:
        return index["text"]

    def transform(self, items: Iterable) -> Generator:
        for item in items:
            key = item["__key__"]

            index = item["meta.json"]
            index["text"] = item["text.txt"]
            index["text"] = self.process_text(index)

            yield AudioDataResult(
                audio=None,
                segment_info=None,
                index=index,
                shard=item["__data_url__"],
                key=key,
                shard_info=ShardInfo(url=item["__data_url__"], uttid=item["uttid"]),
            )


class AudioParquetDataset(ParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        shuffle_buffer_size: int,
        data_type: str,
        data_id: int = None,
        data_urls: List[Dict[str, str]] = None,
        channels: int = 2,
        pad: bool = False,
        segment_duration: Optional[float] = None,
        resampled: bool = True,
        shardshuffle: bool = True,
        detshuffle: bool = False,
        nodesplitter=wds.shardlists.split_by_node,
        min_segment_ratio: float = 1.0,
        min_audio_duration: Optional[float] = None,
        max_audio_duration: Optional[float] = None,
        batch_size: Optional[int] = None,
        buckets_sec: Optional[float] = None,
        use_lyrics: bool = False,
        crop_from_start: bool = False,
        n_segments_per_read: int = 1,
        resolve_urls: bool = True,
        audio_filters = None,
        nitems: int = -1,
    ):
        assert segment_duration is None or segment_duration > 0
        assert segment_duration is None or min_segment_ratio >= 0
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration

        if use_lyrics:
            assert self.segment_duration is not None
            assert self.min_audio_duration is not None
            assert self.max_audio_duration is not None

        if self.min_audio_duration is not None and self.max_audio_duration is not None:
            assert self.min_audio_duration <= self.max_audio_duration

        if buckets_sec is not None:
            assert (
                batch_size is not None
            ), "`batch_size` must be set when using BucketBatcher"
            assert (
                segment_duration is None
            ), "`segment_duration` must be set to `None` when using BucketBatcher"

            self.segment_duration = max(buckets_sec)
            self.pad = False

            logger.info(f"{self.segment_duration=}, {self.pad=}")

            self.buckets_samples = sorted(
                list(map(lambda i: i * sample_rate, buckets_sec))
            )
            self.batcher = BucketBatcher(
                buckets=self.buckets_samples,
                dynamic_batch=False,
                batch_size=batch_size,
                length_fn=lambda x: x["audio"].shape[-1],
            )
        else:
            self.batcher = None
            if segment_duration is None and batch_size is not None:
                raise Exception(
                    "`segment_duration` must be set when batch_size > 1 and not using BucketBatcher"
                )

        self.sample_rate = sample_rate
        self.data_type = data_type
        self.shuffle_buffer_size = shuffle_buffer_size
        self.channels = channels
        self.pad = pad
        self.use_lyrics = use_lyrics
        self.crop_from_start = crop_from_start
        self.n_segments_per_read = n_segments_per_read
        self.nitems = nitems
        self.audio_transform = Compose([RescaleAudio()])
        self.audio_filters = audio_filters

        super().__init__(
            data_id=data_id,
            data_urls=data_urls,
            resampled=resampled,
            shardshuffle=shardshuffle,
            detshuffle=detshuffle,
            nodesplitter=nodesplitter,
            resolve_urls=resolve_urls,
            sample_config={
                # "name": "_ParquetSample",  # default
                "name": "_ParquetSampleFast"
            },
        )

        self = self.rename(**{"meta.json": "meta", "text.txt": "text"})
        decoders = self.decoders()
        if decoders is not None:
            self = self.decode(decoders, only=["audio.wav", "audio.flac", "meta.json"])
        else:
            self = self.decode()

        self = self.compose(self.transform)

        if self.shuffle_buffer_size > 0:
            logger.info(f"Shuffle buffer size: {self.shuffle_buffer_size}")
            self = self.shuffle(self.shuffle_buffer_size)

        if self.batcher is not None:
            self = self.compose(self.bucketize)


    @abstractmethod
    def process_text(self, index: Any) -> Any:
        return index["text"]

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
        return [partial(audio_decoder, read_fn=self.read_audio_segment)]

    def bucketize(self, iterator: Iterable):
        for item in iterator:
            batch = self.batcher.collate_batch(item)
            if batch is not None:
                max_length = max([self.batcher.length_fn(item) for item in batch])

                # NOTE: this gets the max length according to the available buckets:
                # this should actually be optional, but it's useful for now because
                # some models rely on downsampling a certain factor
                max_length = get_bucketed_max_length(max_length, self.buckets_samples)

                pad = Pad(max_length, value=0.0)
                for idx in range(len(batch)):
                    data_result = batch[idx]
                    n_frames = data_result.audio.shape[-1]
                    data_result.audio = pad(data_result.audio)
                    data_result.segment_info.n_frames = n_frames
                    data_result.segment_info.total_frames = max_length
                    yield data_result

    def transform(self, items: Iterable) -> Generator:
        for item in items:
            # TODO: remove try/except...
            try:
                key = item["__key__"]

                segments = item.get("audio.wav", item.get("audio.flac"))

                if not len(segments):
                    continue

                index = item["meta.json"]
                index["text"] = item["text.txt"]
                index["text"] = self.process_text(index)

                for out, segment_info in segments:
                    if self.audio_transform is not None:
                        out = self.audio_transform(out)
                    
                    if self.audio_filters is not None:
                        filter, result = self.audio_filters(out, segment_info.sample_rate)
                        if filter:
                            logger.info(f"Skipped audio (filter): {result}")
                            continue

                    yield AudioDataResult(
                        audio=out,
                        segment_info=segment_info,
                        index=index,
                        shard=item["__data_url__"],
                        key=key,
                        shard_info=ShardInfo(
                            url=item["__data_url__"],
                            uttid=item["uttid"],
                            row_group=item["row_group"],
                            last_row_group=item["last_row_group"],
                            group_index=item["group_index"],
                            last_group=item["last_group"],
                        ),
                    )
            except Exception as e:
                logger.error(e)
                continue

    @staticmethod
    def convert_lyrics_dict(lyrics_dict: dict) -> dict:
        chunks = []
        for u in lyrics_dict["utterances"]:
            words = []
            for w in u["words"]:
                w = {
                    "timestamp": (w["start_time"] / 1000, w["end_time"] / 1000),
                    "text": w["text"],
                    "confidence": w["attribute"]["confidence"],
                    "language": None,
                }
                words.append(w)

            c = {
                "timestamp": (u["start_time"] / 1000, u["end_time"] / 1000),
                "text": u["text"],
                "confidence": u["attribute"]["confidence"],
                "language": None,
                "words": words,
            }
            chunks.append(c)

        new_lyrics_dict = {
            "text": "\n".join([c["text"] for c in chunks]),
            "chunks": chunks,
            "confidence": lyrics_dict["confidence"],
        }
        return new_lyrics_dict

    @staticmethod
    def convert_lyrics_lrc(lyrics_lrc: str) -> dict:
        lyrics = lyrics_lrc.split("\n")
        chunks = []
        for lyric in lyrics:
            m = re.match(r"\[([0-9][0-9]):([0-9][0-9]).([0-9][0-9])\]\s(.*)", lyric)
            if m is not None:
                mm = int(m[1])
                ss = int(m[2])
                ms = int(m[3])  # hundreth of seconds, so :90 = 0.9s
                lyric = m[4]
                start_time = (mm * 60) + ss + (ms / 100)

                c = {
                    "timestamp": (start_time, None),
                    "text": lyric,
                    "confidence": None,
                    "language": None,
                    "words": [],
                }
                chunks.append(c)

        new_lyrics_dict = {
            "text": "\n".join([c["text"] for c in chunks]),
            "chunks": chunks,
            "confidence": None,
        }
        return new_lyrics_dict

    def read_audio_segment(
        self,
        data: BytesIO,
        key: str,
        format: str,
        sample_key: str,
        sample: Dict[str, Any],
    ):
        # TODO: remove try/except...
        # try:
            info = audio_info_unknown_format(data, format)

            meta = AudioMeta(
                path=sample_key, duration=info.duration, sample_rate=info.sample_rate
            )

            # if meta.sample_rate < self.sample_rate:
            #     logger.error(f"Data sample rate is too low: {meta.sample_rate} < {self.sample_rate}")

            if not self.use_lyrics:
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

            if self.use_lyrics:
                meta_dict = json.loads(sample["meta.json"])

                if "lyrics" in meta_dict:
                    lyrics_dict = self.convert_lyrics_dict(meta_dict["lyrics"])
                elif "lyrics.lrc" in meta_dict and meta_dict["lyrics.lrc"] != "":
                    lyrics_dict = self.convert_lyrics_lrc(meta_dict["lyrics.lrc"])
                else:
                    return []
                
                lyrics = Lyrics.load_from_dict(lyrics_dict)
            else:
                lyrics = None

            if self.segment_duration is None:  # TODO: and self.batcher is None:
                data.seek(0)
                out, sr = audio_read(data, pad=self.pad, format=format, info=info)
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

                data.seek(0)
                out, sr = audio_read(
                    data, seek_time, duration, pad=False, format=format, info=info
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

                    data.seek(0)
                    out, sr = audio_read(
                        data, seek_time, duration, pad=False, format=format, info=info
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

                data.close()
                return ret
        # except Exception as e:
        #     logger.error(e)
        #     return []


class FeatureParquetDataset(ParquetDataset):

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        frame_rate: int,
        shuffle_buffer_size: int,
        data_type: str,
        data_id: int = None,
        data_urls: List[Dict[str, str]] = None,
        pad: bool = True,
        segment_duration: Optional[float] = None,
        resampled: bool = True,
        shardshuffle: bool = True,
        detshuffle: bool = False,
        nodesplitter=wds.shardlists.split_by_node,
        min_segment_ratio: float = 1.0,
        min_audio_duration: Optional[float] = None,
        max_audio_duration: Optional[float] = None,
        crop_from_start: bool = False,
        n_segments_per_read: int = 1,
        resolve_urls: bool = True,
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
        self.channels = channels
        self.frame_rate = frame_rate
        self.data_type = data_type
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pad = pad
        self.crop_from_start = crop_from_start
        self.n_segments_per_read = n_segments_per_read
        self.nitems = nitems

        super().__init__(
            data_id=data_id,
            data_urls=data_urls,
            resampled=resampled,
            shardshuffle=shardshuffle,
            detshuffle=detshuffle,
            nodesplitter=nodesplitter,
            resolve_urls=resolve_urls,
            sample_config={
                "name": "_ParquetSampleFast"
            },
            extra_fields_in_data=["feature"],
        )

        self = self.rename(
            **{"feature.pth": "feature", "meta.json": "meta", "text.txt": "text"}
        )

        self = self.decode([], only=["feature.pth", "meta.json"])
        self = self.compose(self.transform)

        if self.shuffle_buffer_size > 0:
            logger.info(f"Shuffle buffer size: {self.shuffle_buffer_size}")
            self = self.shuffle(self.shuffle_buffer_size)

    @abstractmethod
    def process_text(self, index: Any) -> Optional[str]:
        return None

    @abstractmethod
    def filter(self, feature: torch.Tensor) -> bool:
        return False

    def decode(
        self,
        handlers: List[Callable],
        pre=None,
        post=None,
        only=None,
        partial=False,
        handler=reraise_exception,
    ):
        decoder = DecoderWithSampleKey(
            handlers, pre=pre, post=post, only=only, partial=partial
        )
        return self.map(decoder, handler=handler)

    def transform(self, items: Iterable) -> Generator:
        for item in items:
            # TODO: remove try/except...
            try:
                key = item["__key__"]
                feature = item["feature.pth"]

                if self.filter(feature):
                    continue

                index = item["meta.json"]
                index["text"] = self.process_text(index)

                duration = feature.shape[1] / self.frame_rate
                meta = AudioMeta(
                    path=key, duration=duration, sample_rate=self.sample_rate
                )

                if self.segment_duration is None:
                    seek_time = 0
                    target_frames = feature.shape[1]
                elif self.crop_from_start:
                    seek_time = 0
                    target_frames = int(self.frame_rate * self.segment_duration)
                    feature = feature[:, :target_frames]
                else:
                    rng = torch.Generator()
                    max_seek = max(
                        0,
                        meta.duration - self.segment_duration * self.min_segment_ratio,
                    )

                    seek_time = torch.rand(1, generator=rng).item() * max_seek
                    target_frames = int(self.frame_rate * self.segment_duration)

                    start_frame = int(seek_time * self.frame_rate)
                    feature = feature[:, start_frame : start_frame + target_frames]
                    raise Exception("Not tested yet")

                if self.segment_duration is None:
                    n_frames = feature.shape[1]
                    total_frames = n_frames
                else:
                    n_frames = feature.shape[1]
                    total_frames = target_frames
                    if n_frames < target_frames and self.pad:
                        feature = F.pad(feature, (0, total_frames - n_frames))

                segment_info = SegmentInfo(
                    meta=meta,
                    seek_time=seek_time,
                    n_frames=n_frames,
                    total_frames=total_frames,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    data_type=self.data_type,
                    lyrics=None,
                )

                yield AudioDataResult(
                    audio=feature,
                    segment_info=segment_info,
                    index=index,
                    shard=item["__data_url__"],
                    key=key,
                    shard_info=ShardInfo(
                        url=item["__data_url__"],
                        uttid=item["uttid"],
                        row_group=item["row_group"],
                        last_row_group=item["last_row_group"],
                        group_index=item["group_index"],
                        last_group=item["last_group"],
                    ),
                )
            except Exception as e:
                logger.error(e)
                continue
