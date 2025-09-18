import json
from pathlib import Path

from .keywords_parsing import *
from .keywords_postprocess import *
from .keywords_preprocess import *


def _build_reversed_map(merged_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    reverse_map = {}
    for k, kws in merged_dict.items():
        for kw in kws:
            if kw not in reverse_map:
                reverse_map[kw] = []
            reverse_map[kw].append(k)
    return reverse_map


def is_chinese_in_str(s: str) -> bool:
    for c in s:
        if "\u4e00" <= c <= "\u9fff":
            return True
    return False
