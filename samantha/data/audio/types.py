import json
from dataclasses import dataclass, fields
from typing import Any, Dict, List, NamedTuple, Optional, Union

import torch

from samantha.data.lyrics import Lyrics

BatchedStr = Union[str, List[str]]


@dataclass
class DataResult:
    shard: BatchedStr
    key: BatchedStr

    def get(self, name: str, default_val: Any = None):
        if not hasattr(self, name):
            return default_val
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __setitem__(self, name: str, value):
        setattr(self, name, value)

    def values(self):
        for field in fields(self):
            yield getattr(self, field.name)

    def keys(self):
        return [field.name for field in fields(self)]

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class SpeechDataResult(DataResult):
    audio: torch.Tensor
    shard: str
    key: str
    text: Optional[BatchedStr] = None
    tag: Optional[str] = None
    input_length: Optional[int] = None


class DataStats(NamedTuple):
    mean: float
    std: float
    n_datapoints: int
    total_samples: int


@dataclass(order=True)
class BaseInfo:
    @classmethod
    def _dict2fields(cls, dictionary: dict):
        return {
            field.name: dictionary[field.name]
            for field in fields(cls)
            if field.name in dictionary
        }

    @classmethod
    def from_dict(cls, dictionary: dict):
        _dictionary = cls._dict2fields(dictionary)
        return cls(**_dictionary)

    def to_dict(self):
        return {field.name: self.__getattribute__(field.name) for field in fields(self)}


@dataclass(order=True)
class AudioMeta(BaseInfo):
    path: str
    duration: float
    sample_rate: int
    amplitude: Optional[float] = None
    weight: Optional[float] = None

    @classmethod
    def from_dict(cls, dictionary: dict):
        base = cls._dict2fields(dictionary)
        return cls(**base)

    def to_dict(self):
        d = super().to_dict()
        return d


@dataclass(order=True)
class SegmentInfo(BaseInfo):
    meta: AudioMeta
    seek_time: float
    # The following values are given once the audio is processed, e.g.
    # at the target sample rate and target number of channels.
    n_frames: int  # actual number of frames without padding
    total_frames: int  # total number of frames, padding included
    sample_rate: int  # actual sample rate
    channels: int  # number of audio channels.
    data_type: str
    lyrics: Optional[Lyrics] = None
    description: Optional[List[str]] = None
    genre: Optional[str] = None


@dataclass(order=True)
class ShardInfo(BaseInfo):
    url: Optional[str] = None
    uttid: Optional[str] = None
    row_group: Optional[int] = None
    last_row_group: Optional[int] = None
    group_index: Optional[int] = None
    last_group: Optional[int] = None

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, indent=4)


@dataclass
class AudioDataResult(DataResult):
    audio: torch.Tensor
    segment_info: SegmentInfo
    index: Dict[str, Any]
    shard: str
    key: str
    shard_info: Optional[ShardInfo] = None
