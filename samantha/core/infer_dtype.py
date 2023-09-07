from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Dtype:
    ID = "id"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class Element:
    dtype: Dtype
    value: Any


@dataclass
class Container:
    input: Element
    output: Element
    meta: Dict = None
