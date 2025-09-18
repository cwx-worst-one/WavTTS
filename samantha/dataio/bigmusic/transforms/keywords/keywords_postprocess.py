from typing import Any, Dict, List, Optional, Union

try:
    import ujson as json
except ImportError:
    import json

import ast
import copy
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict

from samantha.dataio.bigmusic.base_transform import MusicMetaRWTransform
from samantha.dataio.bigmusic.transforms.utils import *


class MetaPostprocess(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "standard_music_meta.result",
        out_key: str = "keywords",
        fixed_fields: List[str] = ["genre"],
        fixed_fields_dropout: float = 0.0,
        other_fields_dropout: float = 0.0,
        shuffle_field: bool = True,
        shuffle_keywords: bool = True,
        fuse_timbre_gender: float = 0.0,
        target_keywords_mean: int = -1,
        target_keywords_shift: int = 0,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.fixed_fields = fixed_fields
        self.fixed_fields_dropout = fixed_fields_dropout
        self.other_fields_dropout = other_fields_dropout
        self.shuffle_field = shuffle_field
        self.shuffle_keywords = shuffle_keywords
        self.fuse_timbre_gender = fuse_timbre_gender
        self.target_keywords_mean = target_keywords_mean
        self.target_keywords_shift = target_keywords_shift

        self.tempo_templates = [
            "tempo {tempo}",
            "{tempo} bpm",
            "{tempo}bpm",
            "bpm{tempo}",
            "bpm {tempo}",
        ]

        self.duration_templates = ["duration {duration}", "{duration} seconds"]

    def call(self, standard_meta: dict, **kwargs):
        standard_meta = copy.deepcopy(standard_meta)  # avoid in-place modification
        for k, v in standard_meta.items():
            if not isinstance(v, list):
                standard_meta[k] = [v]

        # tempo & duration
        duration = standard_meta.get("duration", [])
        assert len(duration) <= 1
        duration = [
            random.choice(self.duration_templates).format(duration=d) for d in duration
        ]
        standard_meta["duration"] = duration
        tempo = standard_meta.get("tempo", [])
        new_t = []
        for t in tempo:
            if is_number(t):
                new_t.append(
                    random.choice(self.tempo_templates).format(tempo=round(float(t)))
                )
            else:
                new_t.append(t)
        tempo = new_t
        standard_meta["tempo"] = tempo

        if random.random() < self.fuse_timbre_gender:
            timbre = standard_meta.get("timbre", [])
            gender = standard_meta.get("gender", [])

            timbre = [t for t in timbre if len(t.split(" ")) == 1]
            gender = [g for g in gender if len(g.split(" ")) == 1]

            if len(timbre) > 0 and len(gender) > 0:
                timbre = random.choice(timbre)
                gender = random.choice(gender)

                timbre = " ".join([timbre, gender, "voice"])
                standard_meta["timbre"] = [timbre]
                if "technique" in standard_meta:
                    del standard_meta["technique"]
                if "gender" in standard_meta:
                    del standard_meta["gender"]

        keywords = []
        other_fields = list(set(standard_meta.keys()) - set(self.fixed_fields))
        if self.shuffle_field:
            random.shuffle(other_fields)

        other_fields_dropout = self.other_fields_dropout
        if self.target_keywords_mean > 0:
            num_keywords = len(
                [
                    keyword
                    for field_keywords in standard_meta.values()
                    for keyword in field_keywords
                ]
            )
            target_keywords = random.randint(
                self.target_keywords_mean - self.target_keywords_shift,
                self.target_keywords_mean + self.target_keywords_shift,
            )
            other_fields_dropout = 1 - target_keywords / num_keywords

        field_seq = self.fixed_fields + other_fields
        for field in field_seq:
            if field in standard_meta:
                field_keys = standard_meta[field]
                dropout = (
                    self.fixed_fields_dropout
                    if field in self.fixed_fields
                    else other_fields_dropout
                )
                field_keys = [k for k in field_keys if random.random() > dropout]
                if self.shuffle_keywords:
                    random.shuffle(field_keys)
                keywords.extend(field_keys)

        return keywords


class KeywordsPostprocess(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "keywords",
        out_key: str = "prompt",
        norm: bool = True,
        filter: bool = True,
        shuffle: bool = True,
        dropout: float = 0.0,
        join_keywords: bool = True,
        blacklist_filepath: str = None,
        mapping_filepath: str = None,  # for
        **kwargs,
    ):

        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.norm = norm
        self.filter = filter
        self.shuffle = shuffle
        self.dropout = dropout
        self.join_keywords = join_keywords
        self.blacklist = get_mapping_table(blacklist_filepath)
        self.mapping = get_mapping_table(mapping_filepath)

    def call(self, keywords: str, **kwargs):
        norm_funcs = [str.lower]
        keywords = keywords_norm(keywords, norm_funcs) if self.norm else keywords

        # TODO remove this in the future
        if len(self.mapping) > 0:
            new_keywords = []
            for keyword in keywords:
                if contains_chinese(keyword):
                    if keyword in self.mapping:
                        keyword = [
                            v for _, v in self.mapping[keyword]["categories"].items()
                        ]
                        keyword = flatten_nested_list(keyword)
                        new_keywords.extend(keyword)
                    else:
                        continue
                else:
                    new_keywords.append(keyword)
            keywords = new_keywords

        keywords = keywords_dedup(keywords)
        keywords = (
            keywords_filter(
                keywords,
                match=self.blacklist.get("match", []) + FILTER_KEYWORDS,
                contain=self.blacklist.get("contain", []),
            )
            if self.filter
            else keywords
        )
        keywords = [keyword for keyword in keywords if random.random() > self.dropout]

        if self.shuffle:
            random.shuffle(keywords)

        if self.join_keywords:
            keywords = ", ".join(keywords)

        return keywords
