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
import time
from collections import defaultdict

import yaml
from cruise.utilities.hdfs_io import hcopy, hput

from samantha.dataio.bigmusic.base_transform import MusicMetaRWTransform
from samantha.dataio.bigmusic.transforms.utils import *


class WebKeywordsPreprocess(MusicMetaRWTransform):

    def __init__(
        self,
        in_key: list = ("web_keywords", "intermediate_results"),
        out_key: list = ("web_standard_meta", "intermediate_results"),
        mapping_filepath: str = None,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)
        self.mapping_table = get_mapping_table(mapping_filepath)
        self.debug = os.environ.get("TRANSFORM_DEBUG", False)

    def call(self, item, **kwargs):
        keywords, intermediate_results = item
        if isinstance(keywords, str):
            keywords = [keywords]
        intermediate_results = intermediate_results or {}
        transform = self.__class__.__name__
        intermediate_results[transform] = {}
        if self.debug:
            intermediate_results[transform]["input"] = copy.deepcopy(keywords)

        mapped_keywords = keywords_mapping(keywords, self.mapping_table)
        if self.debug:
            intermediate_results[transform]["mapping"] = copy.deepcopy(mapped_keywords)
        web_standard_meta = defaultdict(list)
        for mapped_keyword in mapped_keywords:
            for k, v in mapped_keyword.items():
                if k == "drop":
                    continue
                web_standard_meta[k].extend(v)

        if self.debug:
            intermediate_results[transform]["output"] = copy.deepcopy(web_standard_meta)

        return web_standard_meta, intermediate_results


class StandardMusicMetaKeywordsPreprocess(MusicMetaRWTransform):

    def __init__(
        self,
        in_key: list = ("meta", "standard_music_meta", "intermediate_results"),
        out_key: list = ("standard_music_meta", "intermediate_results"),
        preprocess_sources: list = (
            "human_annotation",
            "fg_tagging_model",
            "sa_tagging_model",
        ),
        mapping_filepath: str = None,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)
        self.preprocess_sources = preprocess_sources
        self.mapping_table = get_mapping_table(mapping_filepath)
        self.debug = os.environ.get("TRANSFORM_DEBUG", False)

    def call(self, item, **kwargs):
        meta, standard_music_meta, intermediate_results = item
        intermediate_results = intermediate_results or {}

        transform = self.__class__.__name__
        intermediate_results[transform] = {}
        if self.debug:
            intermediate_results[transform]["input"] = copy.deepcopy(
                standard_music_meta
            )

        for source_name in self.preprocess_sources:
            if source_name not in standard_music_meta:
                continue

            if source_name == "human_annotation":
                standard_music_meta[source_name]["genre"] = get_nested_value(
                    meta, "audio_tags.genre"
                )
                standard_music_meta[source_name]["genre_extra"] = get_nested_value(
                    meta, "audio_tags.genre_extra"
                )
                # breakpoint()

            source = standard_music_meta[source_name]
            new_standard_meta = defaultdict(list)
            for field_name in source:
                if (
                    not source[field_name]
                    or not isinstance(source[field_name], list)
                    or len(source[field_name]) == 0
                ):
                    continue
                mapped_keywords = keywords_mapping(
                    source[field_name],
                    self.mapping_table,
                    default_category_name=field_name,
                )
                for mapped_keyword in mapped_keywords:
                    for k, v in mapped_keyword.items():
                        new_standard_meta[k].extend(v)
            standard_music_meta[source_name] = new_standard_meta
        if self.debug:
            intermediate_results[transform]["output"] = copy.deepcopy(
                standard_music_meta
            )
        return standard_music_meta, intermediate_results


class GeminiKeywordsPreprocess(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: list = ["standard_music_meta.gemini", "standard_music_meta.gemini_v2"],
        out_key: list = ["standard_music_meta.gemini", "standard_music_meta.gemini_v2"],
        dropout_config_path: str = None,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)
        self.dropout_config = get_mapping_table(dropout_config_path)

    def call(self, item: dict, **kwargs):
        gemini, gemini_v2 = item
        for field_name in gemini:
            new_keywords = []
            for keyword in gemini[field_name]:
                dropout_rate = self.dropout_config.get(keyword.lower(), 0.0)
                if random.random() < dropout_rate:
                    continue
                new_keywords.append(keyword)
            gemini[field_name] = new_keywords
        for field_name in gemini_v2:
            new_keywords = []
            for keyword in gemini_v2[field_name]:
                dropout_rate = self.dropout_config.get(field_name, 0.0)
                if random.random() < dropout_rate:
                    continue
                new_keywords.append(keyword)
            gemini_v2[field_name] = new_keywords
        return gemini, gemini_v2


class MetaSelector(MusicMetaRWTransform):
    """
    hard coded for now
    """

    def __init__(
        self,
        in_key: str = ["standard_music_meta", "intermediate_results"],
        out_key: str = ["standard_music_meta.result", "intermediate_results"],
        config_path: str = "field_process.yaml",
        result_field: str = "result",
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)
        self.config = yaml.safe_load(open(config_path))
        self.result_field = result_field
        self.debug = os.environ.get("TRANSFORM_DEBUG", False)

    def call(self, item: list[dict], **kwargs):
        standard_music_meta, intermediate_results = item
        intermediate_results = intermediate_results or {}
        transform = self.__class__.__name__
        intermediate_results[transform] = {}
        if self.debug:
            intermediate_results[transform]["input"] = copy.deepcopy(
                standard_music_meta
            )

        config = self.config["fields"]
        assert "default" in config, "default field must be provided"
        default_field = config["default"]
        for field_name, field_config in config.items():
            if field_name == "default":
                continue
            for config_name in default_field.keys():
                if config_name not in field_config:
                    field_config[config_name] = default_field[config_name]

        for field_name, field_config in config.items():
            dropout = field_config["dropout"]
            default_p = dropout.get("default", 0.0)
            for source in standard_music_meta:
                if field_name not in standard_music_meta[source]:
                    continue
                if source in dropout.keys():
                    p = dropout[source]
                else:
                    p = default_p
                standard_music_meta[source][field_name] = [
                    keyword
                    for keyword in standard_music_meta[source][field_name]
                    if random.random() > p
                ]

        rev_meta = defaultdict(dict)
        for source in standard_music_meta:
            if not isinstance(standard_music_meta[source], dict):
                continue
            for field_name, v in standard_music_meta[source].items():
                if not isinstance(v, list):
                    v = [v]
                rev_meta[field_name][source] = v
        if self.debug:
            intermediate_results[transform]["rev_meta"] = copy.deepcopy(rev_meta)

        all_fields = []
        for source in standard_music_meta:
            all_fields.extend(
                standard_music_meta[source].keys()
                if isinstance(standard_music_meta[source], dict)
                else []
            )
        all_fields = set(all_fields)

        result = {}

        for field_name in rev_meta:
            aggregate_configs = config.get(field_name, config["default"])["aggregates"]
            for aggregate_config in aggregate_configs:
                aggregate_feature(rev_meta[field_name], aggregate_config)
            result[field_name] = rev_meta[field_name][self.result_field]
        if self.debug:
            intermediate_results[transform]["output"] = copy.deepcopy(result)

        return result, intermediate_results
