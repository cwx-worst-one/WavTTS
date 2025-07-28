import ast
import copy
import io
import json
import math
import operator
import pickle
import random
import re
import sys
import warnings
from collections import Counter
from functools import partial, reduce
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import zhconv
from mariana.utils.audio.audio_logger import AudioLogger

import samantha  # noqa: F401, resolve mariana python path
from samantha.dataio.bigmusic.tokenizers.sami_phoneme_tokenizer import (
    SamiPhonemeSeqPosTokenizer,
    SamiPhonemeSeqTokenizer,
    SamiPhonemeTokenizerError,
)
from samantha.dataio.bigmusic.tokenizers.style_tag_tokenizer import (
    StyleTagTokenizer as _StyleTagTokenizer,
)
from samantha.dataio.bigmusic.transforms.freeform_text import (
    format_freeform_text,
    parse_freeform_text,
)
from samantha.dataio.bigmusic.transforms.keywords import (
    expand_keyword,
    translate_zh_to_en,
)
from samantha.dataio.bigmusic.transforms.mafl import (
    MaflError,
    parse_and_format_freeform_text_legacy,
    parse_asr_lyrics,
    parse_downloaded_lyrics,
    parse_suno_lyrics,
    parse_utterances_and_structure_to_lyrics,
    preprocess_section_tags,
)
from samantha.dataio.bigmusic.transforms.song_slice import (
    Phrase,
    SongSlice,
    SongSliceError,
    drop_out_line_breaks,
    split_into_song_slices,
    validate_song_slice,
)
from samantha.dataio.bigmusic.transforms.structure import transform_raw_segments
from samantha.dataio.bigmusic.transforms.tags import (  # transform_tags,
    TagError,
    TagsProto,
    parse_style_tags,
    tokenize_tags,
    validate_tags,
)
from samantha.dataio.bigmusic.transforms.utterance import UttError, parse_utterances

logger = AudioLogger()


class MusicMetaError(Exception):
    """Raise this error when the entire item is supposed to be discarded"""

    pass


class MusicMetaMapError(Exception):
    """Raise this error when the element in the list is supposed to be filtered out"""

    pass


def err_loc() -> str:
    _, _, exc_tb = sys.exc_info()
    while (
        exc_tb.tb_next
    ):  # Traverse to the deepest traceback (where the error was raised)
        exc_tb = exc_tb.tb_next
    return f"{exc_tb.tb_frame.f_code.co_filename}:{exc_tb.tb_lineno}"


def get_nested_value(dictionary, keys):
    """
    Retrieve a value from a nested dictionary using a list of keys or a dot-separated string.

    :param dictionary: The dictionary to search in.
    :param keys: A list of keys or a dot-separated string representing the path to the value.
    :return: The value if found, otherwise None.

    Example:
        nested_dict = {
            "style_tags": {
                "gender": "male",
                "age": "30"
            }
        }

        # 使用点分隔的字符串
        value = get_nested_value(nested_dict, "style_tags.gender")
        print(value)  # 输出: male

        # 使用键列表
        value = get_nested_value(nested_dict, ["style_tags", "gender"])
        print(value)  # 输出: male

        # 键不存在的情况
        value = get_nested_value(nested_dict, "style_tags.height")
        print(value)  # 输出: None，并记录错误日志
    """
    if isinstance(keys, str):
        keys = keys.split(".")  # 如果 keys 是字符串，按 '.' 分割成列表

    current_level = dictionary
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            logger.error(f"Key '{key}' not found or current level is not a dictionary.")
            return None

    return current_level


def _auto_expand(
    in_key: Union[str, list[str], tuple[str]], pre_expansion_value
) -> Union[str, tuple[str]]:
    """Validate and auto-expand pre_expansion_key if in_key is a list/tuple
    This function is supposed to be used in item transform constructors.
    The assert errors are intended to be unrecoverable.
    """
    if isinstance(in_key, str):
        assert not isinstance(
            pre_expansion_value, (list, tuple)
        ), "merge_mode must not be a list/tuple"
        return pre_expansion_value
    else:  # multiple in-keys
        if not isinstance(pre_expansion_value, (list, tuple)):  # auto-expand
            return (pre_expansion_value,) * len(in_key)
        assert len(in_key) == len(
            pre_expansion_value
        ), "in_key and pre_expansion_key must the same number of items"
        return pre_expansion_value


class DummyItemTransform:
    def __init__(self, in_key, out_key="DDOS"):
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, item, **kwargs):
        if item is None or self.in_key not in item:
            return item

        logger.debug(f"here is a dummy case for processing {item['uttid']}")

        item[self.out_key] = item[self.in_key]
        return item


class ConvertMetaToDict:
    """
    Convert the meta to a json string
    """

    def __init__(self, out_key: str = "meta"):
        self.out_key = out_key

    def __call__(self, item, **kwargs):
        if item is None:
            return item

        if "meta" not in item:
            logger.debug(f"meta not in item check {item['uttid']}")
            return None

        temp_meta = json.loads(item["meta"])

        # Rename the wrong key word, this is should be fixed in the data directly
        if "lyrics_labled" in temp_meta:
            temp_meta["lyrics_labeled"] = temp_meta.pop("lyrics_labled")

        item[self.out_key] = temp_meta

        return item


class MusicMetaRWTransform:
    def __init__(
        self,
        in_key: Optional[Union[str, list[str], tuple[str]]],
        out_key: Optional[str],
        optional_keys: Optional[Union[list[str], tuple[str]]] = None,
        allow_empty_in: bool = False,
        remove_in_key: bool = False,
        overwrite_out_key: bool = True,
        disable_dot_ref: bool = False,
        overwrite_item: bool = False,
        maybe_return_list_output: bool = False,
        return_item_if_in_key_not_found: bool = False,
        bypass_condition: Optional[str] = None,
    ):
        """
        A simple transform that takes the value from in_key and put it into out_key.
        Args:
            in_key: key of the input item, or a list/tuple of in_keys (tuple is more recommended in case of in-place \
                modification), or None to take the entire input item.
            out_key: key of the output item, or None to not modify the item.
            optional_keys: when in_key is a list or tuple, the optional_keys are the keys that are optional. \
                If any of the optional_keys can't be find in the meta, the style tag of that category will be \
                an empty tag [""]. You might want to set allow_empty_in to True to allow empty input.
            allow_empty_in: allow empty input (in_key not found)
            overwrite_out_key: allow to overwrite the out_key if it exists
            remove_in_key: remove the in_key after the transformation
            disable_dot_ref: disable dot reference for in_key and out_key
            overwrite_item: overwrite the entire item by the returned value of the `call` method if out_key is None
            maybe_return_list_output: this flag shows the return value of the `call` method is a list, it is a \
                indicatior for following item_tranforms to do item by item
            return_item_if_in_key_not_found: return the item if in_key is not found in the item
        """
        self.in_key = in_key
        self.out_key = out_key
        self.optional_keys = optional_keys
        self.allow_empty_in = allow_empty_in
        self.remove_in_key = remove_in_key
        self.overwrite_out_key = overwrite_out_key
        self.disable_dot_ref = disable_dot_ref
        self.overwrite_item = overwrite_item
        self.maybe_return_list_output = maybe_return_list_output
        self.return_item_if_in_key_not_found = return_item_if_in_key_not_found
        self.bypass_condition = bypass_condition

        if self.overwrite_item:
            assert (
                self.out_key is None
            ), "out_key should always be None when overwrite_item is True"

        if self.bypass_condition:
            assert "==" in self.bypass_condition or "!=" in self.bypass_condition

    def __call__(self, item: dict, **kwargs):
        if item is None:
            return None

        if self.return_item_if_in_key_not_found:
            if not _validate_item_with_keys(self.in_key, item):
                return item
            if not _validate_item_with_keys("uttid", item):
                return item

        if self.bypass_condition is not None:
            sep = "!=" if "!=" in self.bypass_condition else "=="
            key, value = self.bypass_condition.split(sep)
            if sep == "!=":
                if str(item.get(key)) != value:
                    return item
            else:
                if str(item.get(key)) == value:
                    return item

        uttid = item.get("uttid")
        try:
            inner_item = self._take(item, uttid)

            if "uttid" not in kwargs:
                kwargs["uttid"] = uttid

            _validate_item_with_optional_keys(
                self.in_key, self.optional_keys, inner_item
            )
            result = self.call(inner_item, **kwargs)
            if self.overwrite_item and self.out_key is None:
                return result
            return self._put(item, result, uttid)
        except MusicMetaError as e:
            logger.debug(f"{self._get_err_log_prefix(uttid)} {str(e)} ")
            return None

    def _get_log_prefix(self, uttid):
        return f"[uttid: {uttid}][transform: {self.__class__.__name__}]"

    def _get_err_log_prefix(self, uttid):
        return self._get_log_prefix(uttid) + f"[{err_loc()}]"

    def _take_one(
        self, item: Optional[dict], in_key: str, uttid: Optional[str]
    ) -> Optional[Any]:
        if in_key is None:
            return item
        keys = [in_key] if self.disable_dot_ref else in_key.split(".")
        for idx, key in enumerate(keys):
            if key not in item:
                if not self.allow_empty_in:
                    raise MusicMetaError(f'in_key "{in_key}" is not found in item')
                else:
                    return None
            if self.remove_in_key and idx == len(keys) - 1:
                item = item.pop(key)
            else:
                item = item[key]
        return item

    def _take(self, item: Optional[dict], uttid: Optional[str]) -> Optional[Any]:
        if isinstance(self.in_key, str):
            return self._take_one(item, self.in_key, uttid)
        items = [self._take_one(item, in_key, uttid) for in_key in self.in_key]
        return items

    def _put(self, item: dict, value: Any, uttid: Optional[str]) -> dict:
        if self.out_key is None:
            return item
        inner_item = item
        keys = [self.out_key] if self.disable_dot_ref else self.out_key.split(".")
        for idx, k in enumerate(keys):
            if not isinstance(inner_item, dict):
                raise MusicMetaError(f'out_key "{self.out_key}" is not a dict')
            elif idx == len(keys) - 1:
                if not self.overwrite_out_key and k in inner_item:
                    raise MusicMetaError(
                        f'out_key "{self.out_key}" already exists in item'
                    )
                inner_item[k] = value
            elif k not in inner_item:
                inner_item[k] = {}
                inner_item = inner_item[k]
            else:
                inner_item = inner_item[k]
        return item

    def call(self, item, **kwargs):
        """
        The item is not None unless self.allow_empty_in is True. If self.in_key is a list/tuple,
        the item will be a list of values.
        """
        return item


class DropoutTransform(MusicMetaRWTransform):
    """
    Dropout the input item with the given dropout rate. Only supports str, list, dict, tuple
    """

    def __init__(self, in_key: str, dropout_rate: float, **kwargs):
        super().__init__(in_key, out_key=in_key, **kwargs)
        self.dropout_rate = dropout_rate

    def call(self, value, **kwargs):
        if value is None:
            return None
        assert isinstance(value, (str, list, dict, tuple)), value
        if random.random() < self.dropout_rate:
            return type(value)()  # creates an empty instance of the same type as value
        return value


class RegexDropoutTransform(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str,
        out_key: str = None,
        regex: str = "\\[[^\\]]*\\]\\n",
        modes: Union[tuple[str], list[str]] = ("all", "none", "positive", "negative"),
        weights: Union[tuple[float], list[float]] = (1.0, 1.0, 1.0, 1.0),
        dropout_rate: float = 1.0,
        replace_str: str = "",
        **kwargs,
    ):
        if out_key is None:
            out_key = in_key
        super().__init__(in_key=in_key, out_key=out_key, **kwargs)
        assert set(modes).issubset(["all", "none", "positive", "negative"])
        assert len(modes) == len(weights), "modes and weights must have the same length"
        self.pattern = re.compile(regex)
        self.modes = modes
        self.weights = weights
        self.dropout_rate = dropout_rate
        self.replace_str = replace_str.replace("\\n", "\n")

    def call(self, item: str, **kwargs) -> str:
        def drop_with_prob(match):
            if random.random() < self.dropout_rate:
                return self.replace_str
            return match.group(0)

        mode = random.choices(self.modes, weights=self.weights)[0]

        if mode == "positive":
            # Remove all text that matches the regex
            result = self.pattern.sub(drop_with_prob, item)
        elif mode == "negative":
            # Keep only text that matches the regex
            matches = self.pattern.findall(item)
            result = "".join(matches)
        elif mode == "none":
            return item
        else:  # mode == "all"
            result = ""

        return result


class MultiDropoutTransform(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: Union[list[str], tuple[str]],
        out_key: str,
        tasks: list[dict],
        **kwargs,
    ):
        super().__init__(in_key=in_key, out_key=out_key, **kwargs)
        self.tasks = tasks
        # [
        #     {
        #         "task": ["k1", "k2"],
        #         "weight": 0.1
        #     },
        #     {
        #         "task": ["k1", "k3"],
        #         "weight": 0.2
        #     },
        # ]

    def call(self, values, **kwargs):
        def drop_item(value):
            assert isinstance(value, (str, list, dict, tuple)), value
            return type(value)()

        items = {k: v for k, v in zip(self.in_key, values)}
        task = random.choices(self.tasks, weights=[t["weight"] for t in self.tasks])[0][
            "task"
        ]

        # drop items that are not in the task
        for k in items:
            if k in task:
                continue
            items[k] = drop_item(items[k])

        return items


class MusicMetaMapTransform(MusicMetaRWTransform):
    """
    The input item is supposed to be a tuple or list, and the `call` method will be applied to
    each element in the tuple or list. If any element should be filtered out, raise MusicMetaMapError
    in the `call` method.
    """

    def __init__(self, in_key: str, out_key: str, **kwargs):
        assert isinstance(in_key, str)
        super().__init__(in_key, out_key, **kwargs)

    def _get_item_err_log_prefix(self, uttid, idx, n):
        return self._get_err_log_prefix(uttid) + f"[item: {idx} of {n}]"

    def __call__(self, item: dict, **kwargs):
        if item is None:
            return None

        if self.return_item_if_in_key_not_found:
            if not _validate_item_with_keys(self.in_key, item):
                return item
            if not _validate_item_with_keys("uttid", item):
                return item

        uttid = item.get("uttid")
        try:
            inner_items = self._take(item, uttid)
            results = []
            err_msgs = []
            n = len(inner_items)
            for idx, inner_item in enumerate(inner_items):
                try:
                    _validate_item_with_optional_keys(
                        self.in_key, self.optional_keys, inner_item
                    )
                    result = self.call(inner_item, **kwargs)
                    results.append(result)
                except MusicMetaMapError as e:
                    # NOTE YILIN: Use debug level to avoid littering the log
                    err_msgs.append(str(e))
                    logger.debug(
                        f"{self._get_item_err_log_prefix(uttid, idx, n)} {str(e)} "
                    )
                    continue
            if len(results) < n:
                logger.debug(
                    f"{self._get_log_prefix(uttid)} {n-len(results)}/{n} items are filtered out"
                )
            if err_msgs:
                logger.debug(
                    f"{self._get_log_prefix(uttid)} reasons: {dict(Counter(err_msgs))}"
                )
            if self.overwrite_item and self.out_key is None:
                return result
            return self._put(item, results, uttid)
        except MusicMetaError as e:
            logger.debug(f"{self._get_err_log_prefix(uttid)} {str(e)} ")
            return None

    def call(self, item, **kwargs):
        return item


def _validate_item_with_keys(in_keys, item):
    """
    Check if all in_key is in the item.
    Return:
        True if all in_key is in the item, False otherwise
    """
    if isinstance(in_keys, str):
        return in_keys in item
    return all(in_key in item for in_key in in_keys)


def _validate_item_with_optional_keys(in_key, optional_keys, item):
    empty_items = []
    if isinstance(in_key, str):
        if optional_keys:
            raise ValueError("Do not set optional_keys when in_key is str")
        return
    if optional_keys is None:
        return  # skip if optional_keys is None
    for k, i in zip(in_key, item):
        if k in optional_keys:
            continue
        if i is None:
            empty_items.append(k)
    if empty_items:
        raise MusicMetaError(f"Discard because of empty {', '.join(empty_items)}")


# ================================================================


class DeserializePickle(MusicMetaRWTransform):
    def call(self, item, **kwargs):
        return pickle.loads(item)


class DeserializeJSON(MusicMetaRWTransform):
    def call(self, item, **kwargs):
        return json.loads(item)


class AstLiteralEval(MusicMetaRWTransform):
    def call(self, item, **kwargs):
        return ast.literal_eval(item)


class AstLiteralEval(MusicMetaRWTransform):
    def call(self, item, **kwargs):
        return ast.literal_eval(item)


class SerializeJSON(MusicMetaRWTransform):
    def call(self, item, **kwargs):
        return json.dumps(item)


class GroupItems(MusicMetaRWTransform):
    def __init__(self, in_key: Union[list[str], tuple[str]], out_key: str, **kwargs):
        super().__init__(in_key, out_key, **kwargs)
        assert isinstance(in_key, (tuple, list))

    def call(self, item, **kwargs):
        return item


class MaflLegacyPromptParser(MusicMetaRWTransform):
    def __init__(self, in_key: str = "meta", out_key: str = "prompt", **kwargs):
        super().__init__(in_key, out_key, **kwargs)

    def call(self, meta, **kwargs) -> str:
        try:
            return parse_and_format_freeform_text_legacy(meta)
        except MaflError as e:
            raise MusicMetaError(str(e))


class MaflFormatPrefix(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: Union[str, list[str], tuple[str]] = ("prompt", "lyrics"),
        out_key: str = "freeform_prefix",
        **kwargs,
    ):
        super().__init__(in_key, out_key, **kwargs)

    def call(self, strs, **kwargs) -> str:
        if isinstance(self.in_key, str):
            strs = [strs]
        try:
            strs = [s if s else "" for s in strs]
            joined_str = "".join(strs)
            return preprocess_section_tags(joined_str)
        except MaflError as e:
            raise MusicMetaError(str(e))


class MaflFormatPrefixHard(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: Union[str, list[str], tuple[str]] = ("prompt", "lyrics"),
        out_key: str = "freeform_prefix",
        hard_sep_start: str = "<SECTION>",
        hard_sep_end: str = "</SECTION>",
        **kwargs,
    ):
        super().__init__(in_key, out_key, **kwargs)
        self.hard_sep_start = hard_sep_start
        self.hard_sep_end = hard_sep_end

    def call(self, strs, **kwargs) -> str:
        if isinstance(self.in_key, str):
            strs = [strs]
        try:
            strs = [s if s else "" for s in strs]
            joined_str = "".join(strs)
            return self.hard_sep_start + joined_str + self.hard_sep_end
        except MaflError as e:
            raise MusicMetaError(str(e))


class TokenIdsToString(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "umm_token",
        out_key: str = "umm_string",
        first_n: int = -1,
        template: str = "<au_%s>",
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.template = template
        self.first_n = first_n

    def call(self, umm_tokens, **kwargs) -> str:
        def format(token_id: int) -> str:
            return self.template % token_id

        if self.first_n > 0:
            umm_tokens = umm_tokens[: self.first_n]

        return "".join([format(token_id) for token_id in umm_tokens])


class SimpleTokenLengthEstimation(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: Union[list[str], tuple[str]] = ("freeform_prefix", "umm_token"),
        out_key: str = "total_num_tokens",
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)

    def call(self, item, **kwargs) -> int:
        return sum([len(s) for s in item])


# ================================================================


class StandardMetaParser(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "meta.standard_music_meta",
        out_key: str = "standard_meta",
        meta_types: Union[list[str], tuple[str]] = (
            "web",
            "llm",
            "human_annotation",
            "tagging_model",
        ),
        meta_weights: Union[list[int], tuple[int]] = (4, 3, 2, 1),
        mode: str = "sample",
        norm_sep: bool = False,  # split by " / "
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.meta_types = meta_types
        self.meta_weights = meta_weights
        assert mode in [
            "sample",
            "consolidate",
        ], "mode must be either 'sample' or 'consolidate'"
        self.mode = mode
        self.norm_sep = norm_sep

    def call(self, standard_meta: dict, **kwargs) -> dict:
        if self.mode == "sample":
            sm = self._call_sample(standard_meta, **kwargs)
        else:
            sm = self._call_consolidate(standard_meta, **kwargs)
        if self.norm_sep:
            sm = {k: _norm_sep(v) for k, v in sm.items()}
        return sm

    def _call_consolidate(self, standard_meta: dict, **kwargs) -> dict:
        consolidated_standard_meta = {}
        for k in standard_meta[
            self.meta_types[0]
        ].keys():  # each meta type has the same keys
            if k == "src":
                continue  # drop src, which stands for meta source, e.g. wyy
            meta_list = []
            for meta_type in self.meta_types:
                meta = standard_meta[meta_type][k]
                if meta is None or len(meta) == 0:
                    continue
                meta_list.append(meta)
            if len(meta_list) > 0:
                meta_list_concat = _dedup_with_order(reduce(operator.add, meta_list))
                consolidated_standard_meta[k] = meta_list_concat
            else:
                continue
        return consolidated_standard_meta

    def _call_sample(self, standard_meta: dict, **kwargs) -> dict:
        weighted_standard_meta = {}
        for k in standard_meta[
            self.meta_types[0]
        ].keys():  # each meta type has the same keys
            if k == "src":
                continue  # drop src, which stands for meta source, e.g. wyy
            meta_list, weight_list = [], []
            for meta_type, weight in zip(self.meta_types, self.meta_weights):
                meta = standard_meta[meta_type][k]
                if meta is None or len(meta) == 0:
                    continue
                meta_list.append(meta)
                weight_list.append(weight)
            if len(meta_list) == 0:
                continue
            weighted_standard_meta[k] = random.choices(meta_list, weight_list)[0]
        return weighted_standard_meta


def _dedup_with_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _norm_sep(s: Union[str, list[str]], sep: str = " / ") -> list[str]:
    def split(kw: str) -> str:
        return [k.strip() for k in kw.split(sep)]

    if isinstance(s, str):
        s = [s]
    return _dedup_with_order(reduce(operator.add, [split(kw) for kw in s]))


class SplitItem(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "meta.raw.freeform_text",
        out_key: str = "tags_music.character",
        sep: str = ",",
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.sep = sep

    def call(self, item: str, **kwargs):
        return [x.strip() for x in item.split(self.sep)]


class PRDMetaParser(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "meta.raw.tags_music",
        out_key: str = "prd_meta",
        required_fields: tuple[str] = (
            "genre",
            "mood",
            "gender",
            "scene",
            "timbre",
            "instrument",
            "language",
        ),
        use_subgenre: bool = True,
        genre_repeat: int = 1,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.required_fields = required_fields
        self.genre_repeat = genre_repeat
        self.use_subgenre = use_subgenre

    def call(self, tags: dict, **kwargs) -> dict:
        meta = {}
        for field in self.required_fields:
            if field == "genre":
                genre = (
                    _dedup_with_order(
                        tags.get("genre", []) + tags.get("genre_extra", [])
                    )
                    * self.genre_repeat
                )
                if not self.use_subgenre:
                    genre = genre[:1]  # the first one is genre
                meta[field] = genre
            elif field == "mood":
                meta[field] = _dedup_with_order(tags.get("mood", []))
            elif field == "gender":
                meta[field] = _dedup_with_order(tags.get("speaker", []))
            elif field == "scene":
                meta[field] = _dedup_with_order(tags.get("scene", []))
            elif field == "timbre":
                meta[field] = _dedup_with_order(tags.get("voice", []))
            elif field == "instrument":
                meta[field] = _dedup_with_order(tags.get("instrument", []))
            elif field == "language":
                meta[field] = _dedup_with_order(tags.get("lang", []))
            elif field == "duration":
                meta[field] = tags.get("duration", [])
            elif field == "character":
                meta[field] = tags.get("character", [])
        return meta


class GenreInLyrics(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = ["prd_meta", "lyrics"],
        out_key: str = "lyrics",
        replace_prob=1.0,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.replace_prob = replace_prob

    def call(self, item, **kwargs):
        meta, lyrics = item
        if "genre" not in meta:
            return lyrics
        genre_str = f'<genre>{"|".join(x for x in meta["genre"])}</genre>'

        def replace_with_prob(match):
            if random.random() < self.replace_prob:
                return f"[{match.group(1)} {genre_str}]"
            return match.group(0)

        lyrics = re.sub(r"\[([^\]]+)\]", replace_with_prob, lyrics)
        return lyrics


class GenreInstInLyrics(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = ["prd_meta", "lyrics"],
        out_key: str = "lyrics",
        replace_prob=1.0,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.replace_prob = replace_prob

    def call(self, item, **kwargs):
        meta, lyrics = item
        insert_keys = ["genre", "instrument"]
        if not any(k in meta for k in insert_keys):
            return lyrics
        insert_str = ""
        for k in insert_keys:
            if k not in meta:
                continue
            insert_str += f"<{k}>{'|'.join(x for x in meta[k])}</{k}>"

        def replace_with_prob(match):
            if random.random() < self.replace_prob:
                return f"[{match.group(1)} {insert_str}]"
            return match.group(0)

        lyrics = re.sub(r"\[([^\]]+)\]", replace_with_prob, lyrics)
        return lyrics


class MapGenre2CN(MusicMetaRWTransform):
    def __init__(
        self, in_key: str = "prd_meta.genre", out_key: str = "prd_meta.genre", **kwargs
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.mapping = {
            "R&B/Soul": "节奏布鲁斯",
            "Rock": "摇滚",
            "Hip Hop/Rap": "嘻哈",
            "DJ": "DJ",
            "Chinese Style": "国风",
            "Jazz": "爵士",
            "Folk": "民谣",
            "Electronic": "电子",
            "Punk": "朋克",
            "Pop": "流行",
            "Reggae": "雷鬼",
        }

    def call(self, item, **kwargs):
        genres = item
        genres = [self.mapping.get(genre, genre) for genre in genres]
        return genres


class KeywordExpansion(MusicMetaRWTransform):
    def __init__(self, in_key: str, out_key: str, **kwargs):
        super().__init__(in_key, out_key, **kwargs)

    def call(self, item, **kwargs):
        def process_keyword(keyword: str) -> list[str]:
            kws = translate_zh_to_en(keyword, keep_input=False)
            if kws:  # completely replace Chinese keywords with English keywords
                return kws
            return expand_keyword(keyword, keep_input=True)

        def process_keyword_list(keywords: list[str]) -> list[str]:
            return _dedup_with_order(
                reduce(operator.add, [process_keyword(keyword) for keyword in keywords])
            )

        def process_keyword_or_keyword_list(kw: Union[str, list[str]]) -> list[str]:
            if isinstance(kw, str):
                return process_keyword(kw)
            return process_keyword_list(kw)

        if isinstance(item, dict):
            return {k: process_keyword_or_keyword_list(v) for k, v in item.items()}

        return process_keyword_or_keyword_list(item)


class TextAugmentor(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "lyrics",
        out_key: str = "lyrics",
        aug_rates: Union[float, tuple[float], list[float]] = -1,
        aug_fns: Union[str, tuple[str], list[str]] = (
            "simplified2traditional",
            "capitalize",
        ),
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)

        if isinstance(aug_fns, str):
            aug_fns = [aug_fns]
        if isinstance(aug_rates, float):
            aug_rates = [aug_rates] * len(
                aug_fns
            )  # repeat to the same length as aug_fns
        assert (
            isinstance(aug_rates, (tuple, list))
            and isinstance(aug_fns, (tuple, list))
            and len(aug_rates) == len(aug_fns)
        ), "aug_rate and aug_fns must be the same length"

        self.aug_rates = aug_rates
        self.aug_fns = aug_fns

        self.aug_fns_map = {
            "simplified2traditional": partial(zhconv.convert, locale="zh-tw"),
            "traditional2simplified": partial(zhconv.convert, locale="zh-cn"),
            "capitalize": str.capitalize,
            "upper": str.upper,
            "lower": str.lower,
        }

        assert all(
            [aug_fn in self.aug_fns_map for aug_fn in self.aug_fns]
        ), f"aug_fn must be one of {self.aug_fns_map.keys()}"

    def call(self, item, **kwargs):
        assert isinstance(
            item, (str, list, tuple)
        ), f"item must be str, list or tuple, but got {type(item), {item}}"

        orig_type = type(item)
        if orig_type == str:
            item = [item]

        aug_item = []
        for s in item:
            for aug_fn, aug_rate in zip(self.aug_fns, self.aug_rates):
                if random.random() < aug_rate:
                    s = self.aug_fns_map[aug_fn](s)
            aug_item.append(s)

        if orig_type == str:
            aug_item = aug_item[0]

        return aug_item


class PackMeta2Prompt(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "standard_meta",
        out_key: str = "prompt",
        shuffle: bool = False,
        field_dropout_rate: float = 0.0,
        dropout_rate: float = 0.0,
        system_prompt: str = "",
        keyword_mode: str = "key_value",  # or value_only
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.shuffle = shuffle
        self.field_dropout_rate = field_dropout_rate
        self.dropout_rate = dropout_rate
        self.system_prompt = system_prompt
        self.keyword_mode = keyword_mode

    def call(self, standard_meta: dict, **kwargs):
        prompt = ""
        keys = list(standard_meta.keys())
        if self.shuffle:
            random.shuffle(keys)
        for k in keys:
            v = standard_meta[k]
            if v is None or random.random() < self.field_dropout_rate:
                continue
            if self.keyword_mode == "key_value":
                prompt += f'[{k.upper()}: {"|".join(x for x in v)}]'
            elif self.keyword_mode == "value_only":
                values_lst = [x for x in v]
                if self.shuffle:
                    random.shuffle(values_lst)
                prompt += f'{",".join(values_lst)},'
        prompt = "" if random.random() < self.dropout_rate else prompt
        if self.system_prompt:
            prompt = f"[SYSTEM_PRMOPT: {self.system_prompt}]" + prompt
        return prompt


class PackMeta2PromptFull(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "standard_meta",
        out_key: str = "prompt",
        shuffle: bool = False,
        field_dropout_rate: float = 0.0,
        dropout_rate: float = 0.0,
        system_prompt: str = "",
        keyword_mode: str = "key_value",  # or value_only
        refer_path: str = "",
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.shuffle = shuffle
        self.field_dropout_rate = field_dropout_rate
        self.dropout_rate = dropout_rate
        self.system_prompt = system_prompt
        self.keyword_mode = keyword_mode
        if refer_path:
            with open(refer_path, "r") as ff:
                self.refer_prompt = json.load(ff)

    def call(self, standard_meta: dict, **kwargs):
        prompt = ""
        keys = list(standard_meta.keys())
        if self.shuffle:
            random.shuffle(keys)

        value_only_prompt_lst = []
        genre_prompt_lst = []
        for k in keys:
            v = standard_meta[k]
            if v is None or random.random() < self.field_dropout_rate:
                continue
            if not isinstance(v, (list, tuple)):
                v = [v]

            if self.keyword_mode == "key_value":
                prompt += f'<{k.lower()}>{"|".join(str(x) for x in v)}</{k.lower()}>'
            elif self.keyword_mode == "value_only":
                if k == "genre":
                    genre_prompt_lst += v
                else:
                    value_only_prompt_lst += v
            elif self.keyword_mode == "value_only_expanded":
                if k == "genre":
                    genre_prompt_lst += v
                    for x in v:
                        value_only_prompt_lst += self.refer_prompt.get(x, [])
                        # values_lst = values_lst[:random.randint(len(values_lst)//2, len(values_lst))]
                elif k == "duration":
                    value_only_prompt_lst += [f"duration:{v[0]}"]
                else:
                    value_only_prompt_lst += v

            elif self.keyword_mode == "expanded":
                if k == "genre":
                    values_lst = []
                    for x in v:
                        values_lst += self.refer_prompt.get(x, [])
                    if self.shuffle:
                        random.shuffle(values_lst)
                        values_lst = values_lst[: random.randint(1, len(values_lst))]
                    values_lst = ["|".join(str(x) for x in v)] + values_lst
                    prompt += f'<{k.lower()}>{"|".join(str(x) for x in values_lst)}</{k.lower()}>'
                else:
                    prompt += (
                        f'<{k.lower()}>{",".join(str(x) for x in v)}</{k.lower()}>'
                    )

        if (
            self.keyword_mode == "value_only"
            or self.keyword_mode == "value_only_expanded"
        ):
            if self.shuffle:
                random.shuffle(value_only_prompt_lst)
            value_only_prompt_lst = _dedup_with_order(value_only_prompt_lst)
            # there are integers in the list
            merged_lst = list(map(str, genre_prompt_lst + value_only_prompt_lst))
            merged_lst = [
                v for v in merged_lst if random.random() >= self.field_dropout_rate
            ]
            prompt = "|".join(merged_lst)

        prompt = "" if random.random() < self.dropout_rate else prompt
        if self.system_prompt:
            prompt = f"[SYSTEM_PRMOPT: {self.system_prompt}]" + prompt
        return prompt


class LyricsParser(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "meta",
        out_key: str = "lyrics",
        lyrics_type: Union[tuple[str], list[str]] = ("asr",),
        **kwargs,
    ):
        warnings.warn(
            """This transform is deprecated and will be removed in a future version. \
Please use dedicated lyrics parser transforms with SimpleSelector instead.""",
            DeprecationWarning,
        )
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        if isinstance(lyrics_type, str):
            lyrics_type = [lyrics_type]
        self.lyrics_type = lyrics_type

    def call(self, meta: dict, **kwargs) -> str:
        fns = {
            "asr": parse_asr_lyrics,  # deprecated, please use dedicated parsers instead
            "downloaded": parse_downloaded_lyrics,
            "suno": parse_suno_lyrics,
        }
        lyrics = ""
        if "data_type" in meta and meta["data_type"] == "tts":
            lyrics = meta["lyrics"]
        else:
            for t in self.lyrics_type:
                try:
                    lyrics = fns[t](meta)
                except MaflError:
                    continue
        return lyrics


class SimpleSelector(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: Union[list[str], tuple[str]],
        out_key: str,
        predicate: str = "bool",
        allow_empty_out: bool = False,
        **kwargs,
    ):
        super().__init__(in_key, out_key, **kwargs)
        assert not isinstance(self.in_key, str), "in_key must be a list or tuple"
        _selector_predicate_map = {"bool": _predicate_bool, "none": _predicate_none}
        assert (
            predicate.lower() in _selector_predicate_map
        ), f"predicate must be in {_selector_predicate_map.keys()}"
        self.predicate_fn = _selector_predicate_map[predicate.lower()]
        self.allow_empty_out = allow_empty_out

    def call(self, item: list[dict], **kwargs):
        for i in item:
            if self.predicate_fn(i):
                return i
        if self.allow_empty_out:
            return None
        raise MusicMetaError("No item satisfies the predicate")


class WeightedSelector(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: Union[list[str], tuple[str]],
        out_key: str,
        weights: Union[list[float], tuple[float]],
        **kwargs,
    ):
        super().__init__(in_key, out_key, **kwargs)
        self.weights = weights

    def call(self, item: list[dict], **kwargs):
        selected_item = random.choices(item, weights=self.weights, k=1)[0]
        return selected_item


def _predicate_bool(x):
    return bool(x)


def _predicate_none(x):
    return x is not None


class PackPromptLyrics2Tokens:
    """
    Convert the meta to a json string
    """

    def __init__(
        self,
        prompt_key: str = "prompt",
        lyrics_key: str = "lyrics",
        out_key: str = "input_strings",
        ignore_prefix=False,
    ):
        self.prompt_key = prompt_key
        self.lyrics_key = lyrics_key
        self.out_key = out_key
        self.ignore_prefix = ignore_prefix

    def __call__(self, item, **kwargs):
        if item is None:
            return item

        if self.lyrics_key not in item or self.prompt_key not in item:
            logger.debug(f"meta not in item check {item['uttid']}")
            return None

        result_item = {}
        prompt = item[self.prompt_key]
        lyrics = item[self.lyrics_key]
        result_item["prompt"] = prompt
        result_item["lyrics"] = lyrics
        prefix_text = prompt + lyrics
        prefix_text = preprocess_section_tags(prefix_text)

        prompt = preprocess_section_tags(prompt)
        lyrics = preprocess_section_tags(lyrics)

        if self.ignore_prefix:
            prefix_text = ""

        umm_tokens = pickle.loads(item["umm_token"])["umm_token"]
        token_string = "".join([f"<au_{token_id}>" for token_id in umm_tokens])
        estimated_token_length = len(prefix_text) + len(umm_tokens)
        result_item[self.out_key] = (prompt, lyrics, token_string)
        result_item["num_total_tokens"] = estimated_token_length

        # HACK
        result_item["standard_meta"] = item.get("standard_meta")

        return result_item


# ================================================================


class StructureParser(MusicMetaRWTransform):
    """Parse music structure from meta"""

    def __init__(
        self,
        in_key: Union[str, list[str], tuple[str]] = (
            "meta.music_structure_labeled.struct_result",
            "meta.musicfm_structure",
            "meta.deepchorus.raw_segments",
        ),
        out_key: str = "structure",
        confidence_threshold: Union[str, list[float], tuple[float]] = 0.0,
        merge_mode: Union[str, list[str], tuple[str]] = (
            None,  # treat human-labels as ground-truths
            None,  # skip segment merge for deepchorus v2
            "vocal",
        ),
        **kwargs,
    ):
        """
        Args:
            in_key: One or multiple sources of raw segments. \
                If it is a string, the raw segments will be obtained from the meta with the key specified by in_key. \
                If it is a list or tuple, the raw segments will be obtained by interating through each key and finding \
                the first one that is available. The raw segments will be merged using the merge_mode specified by \
                merge_mode.
            confidence_threshold: Discard the structure if the confidence is lower than this threshold. \
                If it is None, confidence filtering will be skipped. If you are not sure if the source has a \
                confidence score label, set it to None, otherwise the parser will try to access a key that might \
                not exist.
            merge_mode: The way raw segments get merged, designed for deepchorus version 1. Options: \
                - "vocal": For vocal music \
                - "inst": For instrumental music \
                - None: Do not merge

            Ways to set confidence_threshold and merge_mode: \
            - global: Set one value for all sources, no matter whether the in_key is a string or a list or tuple. \
            - per source: Set one value for each source, the length of the list/tuple should be the same as \
                the length of in_key.
        """
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)
        self.confidence_threshold = _auto_expand(self.in_key, confidence_threshold)
        self.merge_mode = _auto_expand(self.in_key, merge_mode)

    def call(self, item, **kwargs):
        if isinstance(self.in_key, str):
            in_keys = [self.in_key]
            item = [item]
            merge_mode = [self.merge_mode]
            cfds = [self.confidence_threshold]
        else:
            in_keys = self.in_key
            merge_mode = self.merge_mode
            cfds = self.confidence_threshold

        for ik, _item, mode, cfd in zip(in_keys, item, merge_mode, cfds):
            if not _item:
                continue
            results = transform_raw_segments(
                _item, mode, obtain_confidence=cfd is not None
            )
            if cfd is not None and results["confidence"] < cfd:
                raise MusicMetaError(
                    f"in_key {ik}. Discard because of low confidence {results['confidence']}"
                )
            return results["transformed"]
        raise MusicMetaError("Discard because structure is not found")


class StyleTagParser(MusicMetaRWTransform):
    """Parse styel tags from meta"""

    def __init__(
        self,
        in_key: str = "meta",
        out_key: str = "style_tags",
        vocab: str = _StyleTagTokenizer.DEFAULT_VOCAB_VER,
        dropout_rate: float = 0.0,
        standardize: bool = True,
        extend_extra: bool = True,
        source_selection: str = "fill",
        **kwargs,
    ):
        """
        Args:
            vocab: The style tag vocabulary version. \
                Check the json vocab files for available versions: \
                tokenizers/style_tag_tokenizer/vocabs/vocab.[version].json
            optional_keys: The keys that are optional. If any of the optional_keys can't be find in the meta, \
                the style tag of that category will be an empty tag [""].
            dropout_rate: The dropout rate for **individual** tag category.
            standardize: Process the incoming tag string into the standardized format. \
                Only set to True if the incoming tag string is not guaranteed to be in the standardized format, \
                while the vocab itself is already in the standardized format.
            extend_extra: Add tags to "genre_extra" and "extra" categories and potentially change the "is_sinking" \
                category. This process might introduce out-of-vocab tags, which will appear in the out-of-vocab logging.
            source_selection: "fill" or "merge" \
                - "fill": Fill the missing tags with the first available tag in the source. \
                - "merge": Merge all the tags into one list.
        """
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)
        self.vocab = _StyleTagTokenizer.get_vocab(vocab)
        self.dropout_rate = dropout_rate
        self.standardize = standardize
        self.extend_extra = extend_extra
        self.source_selection = source_selection

    def call(self, item, **kwargs) -> dict[str, list[str]]:
        uttid = kwargs.get("uttid")

        def drop_out_style_tags(style_tags: dict) -> dict:
            dropped_tags = {}
            for category, tags in style_tags.items():
                if random.random() < self.dropout_rate:
                    tags = [""]
                dropped_tags[category] = tags
            return dropped_tags

        try:
            meta = item
            # t = transform_tags(
            #     self.vocab,
            #     meta,
            #     insts,
            #     tempo,
            #     key_mode,
            #     standardize=self.standardize,
            #     extend_extra=self.extend_extra,
            #     source_selection=self.source_selection,
            # )
            t = parse_style_tags(
                self.vocab,
                meta,
                standardize=self.standardize,
                extend_extra=self.extend_extra,
            )
            tags = t["tags"]
            oov_tags = t["oov_tags"]
            if oov_tags:
                logger.debug(
                    f"{self._get_log_prefix(uttid)} {len(oov_tags)} out-of-vocab tags found: {oov_tags}"
                )
            return drop_out_style_tags(tags)
        except TagError as e:
            raise MusicMetaError(str(e))


class StyleTagFilter(MusicMetaRWTransform):
    """Filter out invalid style tags"""

    def __init__(
        self, in_key: str = "style_tags", out_key: str = "style_tags", **kwargs
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)

    def call(self, meta: dict[str, list[str]], **kwargs):
        try:
            validate_tags(meta)
            return meta
        except TagError as e:
            raise MusicMetaError(str(e))


class StyleTagTokenizer(MusicMetaRWTransform):
    """Tokenize style tags into input IDs"""

    def __init__(
        self,
        in_key: str = "style_tags",
        out_key: str = "style_input_ids",
        vocab: str = _StyleTagTokenizer.DEFAULT_VOCAB_VER,
        **kwargs,
    ):
        """
        Arg:
            vocab: The style tag vocabulary version that will be used to initialize the tokenizer. \
                Check the json vocab files for available versions: \
                tokenizers/style_tag_tokenizer/vocabs/vocab.[version].json
        """
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.tokenizer = _StyleTagTokenizer(vocab)

    def call(self, style_tags: dict[str, list[str]], **kwargs) -> dict[str, list[int]]:
        try:
            return tokenize_tags(style_tags, self.tokenizer)
        except TagError as e:
            raise MusicMetaError(str(e))


class FreeformTextParser(MusicMetaRWTransform):
    """Parse freeform text (str) from meta"""

    def __init__(
        self,
        in_key: Union[list, tuple] = ("meta", "style_tags"),
        out_key: str = "freeform_text",
        dropout_rate: float = 0.1,  # default: 0.1 chance to drop the whole freeform_text content
        long_description_rate: float = 0.0,  # default: not to use any long description
        keyword_dropout_rate: float = 0.1,  # default: 0.1 chance to drop all keywords
        style_tags_dropout_rate: float = 0.5,  # default: 0.5 chance to concat style_tags
        **kwargs,
    ):
        """
        Args:
            in_key: (meta, style_tags) in order.
            keyword_dropout_rate: The dropout rate for **individual** keyword.
            dropout_rate: The dropout rate for the **entire** freeform text.
        """
        # using invalid arguments should not be recoverable
        assert (
            not isinstance(in_key, str) and len(in_key) == 2
        ), f"in_key must be a list of length 2, got {in_key}"
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.dropout_rate = dropout_rate
        self.long_description_rate = long_description_rate
        self.keyword_dropout_rate = keyword_dropout_rate
        self.style_tags_dropout_rate = style_tags_dropout_rate

    def call(self, item, **kwargs) -> str:
        meta, style_tags = item
        if random.random() < self.dropout_rate:
            return ""
        freeform_text_dict = parse_freeform_text(meta, style_tags)
        freeform_text = format_freeform_text(
            freeform_text_dict,
            self.long_description_rate,
            self.keyword_dropout_rate,
            self.style_tags_dropout_rate,
        )
        return freeform_text


class DurationParser(MusicMetaRWTransform):
    """Parse song duration from meta"""

    def __init__(
        self,
        in_key: Union[str, list[str]] = ("meta.duration",),
        out_key: str = "duration",
        allow_empty_in: bool = True,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=allow_empty_in, **kwargs)

    def call(self, duration, **kwargs) -> float:
        if isinstance(duration, (list, tuple)):
            duration = duration[0]
        try:
            duration = int(float(duration))
        except TypeError:
            duration = -1
        return duration


class TempoParser(MusicMetaRWTransform):
    """Parse tempo (BPM) from meta"""

    DEFAULT_BPM = -1.0  # -1 means no bpm info

    def __init__(
        self,
        in_key: Union[str, list[str]] = "meta.standard_music_meta_nonbpe",
        out_key: str = "tempo",
        allow_empty_in: bool = True,  # allow data without tempo by default
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            in_key: A list of keys to try. The first valid key will be used. Available values are: \
                int, float, str of digits, e.g. "120", "120.5"
            dropout_rate: The dropout rate for the bpm value. TempoParser.DEFAULT_BPM will be used \
                if the bpm value is dropped out.
        """
        super().__init__(in_key, out_key, allow_empty_in=allow_empty_in, **kwargs)
        self.dropout_rate = dropout_rate

    def get_bpm_from_non_standard(self, bpm) -> float:
        def is_musicfm_beat(bpm) -> bool:
            return isinstance(bpm, list)

        def get_bpm_from_musicfm_beat(bpm: list[list[float]]) -> float:
            avg_beat_interval = np.mean(np.diff(np.array(bpm)[:, 0], n=1))  # spb
            return 60 / avg_beat_interval  # 60 / spb -> 60 * bps -> bpm

        if isinstance(self.in_key, str):
            bpm = [bpm]
        for bpm_value in bpm:  # pick the first valid bpm value
            if not bpm_value:  # 0 or None
                continue
            if is_musicfm_beat(bpm_value):
                bpm_value = get_bpm_from_musicfm_beat(bpm_value)
            return float(bpm_value)
        return self.DEFAULT_BPM

    def call(self, standard_music_meta_nonbpe, **kwargs) -> float:
        uttid = kwargs.get("uttid")
        if random.random() < self.dropout_rate:
            return self.DEFAULT_BPM
        if not standard_music_meta_nonbpe:
            return self.DEFAULT_BPM
        try:
            if isinstance(standard_music_meta_nonbpe, dict):
                bpm = standard_music_meta_nonbpe.get("extra_info", {}).get("bpm")
            else:
                bpm = self.get_bpm_from_non_standard(bpm=standard_music_meta_nonbpe)
            return float(bpm)
        except TypeError:
            logger.debug(f"{self._get_log_prefix(uttid)} tempo {bpm} is invalid")
        return self.DEFAULT_BPM


class KeyModeParser(MusicMetaRWTransform):
    """Parse key and mode from meta"""

    DEFAULT_KEY = ""
    DEFAULT_MODE = ""
    SEP = ":"

    def __init__(
        self,
        in_key: Union[str, list[str]] = ("meta.musicfm_plus.key",),
        out_key: str = "key_mode",
        allow_empty_in: bool = True,  # allow data without key by default
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            in_key: A dict key (str) or a list of dict keys to try. The first valid key will be used. \
                Available values: \
                1. str of format "key:mode", e.g. "C:Maj", "C:Major", "C:Min", "C:Minor" \
                2. key only, e.g. "C", "D", "E"
            dropout_rate: The dropout rate for the key value. DEFAULT_KEY+SEP+DEFAULT_MODE will be used \
                if the key value is dropped out.
        """
        super().__init__(in_key, out_key, allow_empty_in=allow_empty_in, **kwargs)
        self.dropout_rate = dropout_rate

    def call(self, keys, **kwargs) -> str:
        uttid = kwargs.get("uttid")

        if isinstance(self.in_key, str):
            keys = [keys]

        for key in keys:
            if not key:
                continue
            if random.random() < self.dropout_rate:
                return self.DEFAULT_KEY + self.SEP + self.DEFAULT_MODE
            if isinstance(key, list):  # musicfm_plus.key
                key_list = [_key[1] for _key in key]  # key name
                key_counter = Counter(key_list)
                song_key = key_counter.most_common(1)[0][0]
                if song_key != "X":
                    key_text, mode_text = song_key.split(":")
                else:
                    key_text, mode_text = "", ""
            elif isinstance(key, str) and ":" in key:
                split_text = key.split(":")
                if len(split_text) == 2:
                    key_text, mode_text = split_text
                    if not key_text or not mode_text:
                        logger.debug(
                            f"{self._get_log_prefix(uttid)} key string {key} is invalid"
                        )
                        continue
                else:
                    logger.debug(
                        f"{self._get_log_prefix(uttid)} key string {key} is invalid"
                    )
                    continue  # skip invalid key
            else:
                logger.debug(f"{self._get_log_prefix(uttid)} key {key} is invalid")
                continue
            mode_text = {"Maj": "Major", "Min": "Minor"}.get(mode_text, mode_text)
            return key_text + self.SEP + mode_text

        return self.DEFAULT_KEY + self.SEP + self.DEFAULT_MODE


class InstParser(MusicMetaRWTransform):
    """Parse instrument list (list[str]) from meta"""

    DEFAULT_INST = ""

    def __init__(
        self,
        in_key: Union[str, list[str]] = (
            "meta.instruments",
            "meta.musicfm_tagging.instrument_section.global.instruments",
        ),
        out_key: str = "instruments",
        allow_empty_in: bool = True,  # allow data without key by default
        dropout_rate: float = 0.1,
        confidence_threshold: float = 0.4,
        **kwargs,
    ):
        """
        Args:
            in_key: A list of keys to try. The first valid key will be used.
            dropout_rate: The dropout rate for the key value. DEFAULT_INST will be used \
                if the key value is dropped out.
        """
        super().__init__(in_key, out_key, allow_empty_in=allow_empty_in, **kwargs)
        self.dropout_rate = dropout_rate
        self.confidence_threshold = confidence_threshold

    def call(self, insts, **kwargs) -> list[str]:
        if random.random() < self.dropout_rate:
            return [self.DEFAULT_INST]

        if isinstance(self.in_key, str):
            insts = [insts]

        for inst in insts:
            if not inst:
                continue
            # musicfm_tagging.instrument_section.global
            if isinstance(inst, dict):
                return [
                    k
                    for k, v in inst.items()
                    if k and v >= self.confidence_threshold and k != "Vocal"
                ]
            # meta.instruments
            if inst not in ["\\N", ""]:
                return [i.strip() for i in inst.split(",")]
        return [self.DEFAULT_INST]


class UtteranceParser(MusicMetaRWTransform):
    """Parse utterances (lyrics) from meta"""

    def __init__(
        self,
        in_key: str = "meta",
        out_key: str = "utterances",
        confidence_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(in_key=in_key, out_key=out_key, allow_empty_in=False, **kwargs)
        self.confidence_threshold = confidence_threshold

    def call(self, meta, **kwargs) -> list:
        try:
            return parse_utterances(meta, self.confidence_threshold)
        except UttError as e:
            raise MusicMetaError(str(e))


class CondDropoutTransform(MusicMetaRWTransform):
    """
    Probabilistically dropout with the given dropout_rate if the condition item is True,
    forced dropout if the condition item is False.
    """

    def __init__(self, in_key: str, dropout_rate: float, **kwargs):
        """
        Args:
            in_key: (dropout_item_key, condition_item_key)
        """
        assert len(in_key) == 2, "in_key must be a tuple of length 2"
        super().__init__(in_key, out_key=in_key[0], **kwargs)
        self.dropout_rate = dropout_rate

    def call(self, value, **kwargs):
        value, cond = value
        if value is None:
            return None
        assert isinstance(value, (str, list, dict, tuple)), value
        if cond:
            if random.random() < self.dropout_rate:
                return type(
                    value
                )()  # creates an empty instance of the same type as value
            return value
        return type(value)()  # condition is False, must drop


class VoiceProportionParser(MusicMetaRWTransform):
    """Parse voice proportion value and filter out instrumental music with voice proportion higher than the threshold"""

    def __init__(
        self,
        in_key: Union[tuple[str], list[str]] = (
            "meta.vad.extra.voice_proportion",
            "utterances",
        ),
        out_key: str = "prompt_type",  # no need to use the voice proportion
        allow_empty_in: bool = True,  # allow in-keys' values to be empty
        vad_threshold: float = 0.15,
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=allow_empty_in, **kwargs)
        self.vad_threshold = vad_threshold

    def call(self, item, **kwargs) -> float:
        voice_proportion, utterances = item
        # if not voice_proportion:
        #     raise MusicMetaError(
        #         f"No vad results"
        #     )
        prompt_type = "inst"
        if voice_proportion:
            if voice_proportion > self.vad_threshold:
                prompt_type = "vocal"
        else:
            if utterances:
                prompt_type = "vocal"

        return prompt_type


class VocalOrInstFilter(MusicMetaRWTransform):
    """Filter out vocal or inst samples"""

    def __init__(
        self,
        in_key: str = "prompt_type",
        out_key=None,
        target_type: str = "inst",
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)
        self.target_type = target_type

    def call(self, prompt_type: str, **kwargs):
        if self.target_type == prompt_type:
            return

        raise MusicMetaError(
            f"Discard because of the sample is not {self.target_type} music"
        )


class PackUtteranceStructureToLyrics(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: Union[tuple[str], list[str]] = ("utterances", "structure"),
        out_key: Optional[str] = "lyrics",
        default_value: Any = "",
        **kwargs,
    ):
        """To allow empty utterances and structure, set allow_empty_in=True"""
        super().__init__(in_key, out_key, **kwargs)
        self.default_value = default_value

    def call(self, item, **kwargs) -> Any:
        utterances, structure = item
        if not utterances and not structure:
            return self.default_value
        if structure:
            return parse_utterances_and_structure_to_lyrics(utterances or [], structure)
        return "\n".join([utt["text"] for utt in utterances])


class SongSliceSplit(MusicMetaRWTransform):
    """Take the input items, and split and assign them into song slices."""

    def __init__(
        self,
        in_key: Union[list[str], tuple[str]] = (
            "structure",
            "utterances",
            "style_tags",
            "style_input_ids",
            "freeform_text",
            "vocal2midi",
            "tempo",
            "prompt_type",
        ),
        out_key: str = "song_slices",
        allow_empty_in: bool = True,
        optional_keys: Union[list[str], tuple[str]] = (
            "freeform_text",
            "vocal2midi",
            "tempo",
            "prompt_type",
        ),
        max_duration: float = 240.0,
        line_break_dropout_rate: float = 0.1,
        slice_mode: str = "section",
        **kwargs,
    ):
        """
        Args:
            in_key: (structure, utterances, style_tags, style_input_ids, freeform_text, vocal2midi, tempo) in order
            optional_keys: keys that are optional. All of these keys are allowed to not be in the item.
            max_duration: The maxium duration of a song slice
            line_break_dropout_rate: The dropout rate for the line break. If a line break is dropped out, \
                the adjacent lines will be concatenated.
            slice_mode: The mode of the song slice. "section" or "full". \
                "section" means splitting song slices whose boundaries are always section boundaries; \
                "full" means taking the full song without splitting.
        """
        # using invalid arguments should not be recoverable
        assert (
            not isinstance(in_key, str) and len(in_key) == 8
        ), f"in_key must be a list of length 8, got {in_key}"
        super().__init__(
            in_key,
            out_key,
            optional_keys=optional_keys,
            allow_empty_in=allow_empty_in,
            **kwargs,
        )
        self.max_duration = max_duration
        self.line_break_dropout_rate = line_break_dropout_rate
        self.slice_mode = slice_mode

    def call(self, item, **kwargs):
        (
            structure,
            utterances,
            style_tags,
            style_tokens,
            freeform_text,
            vocal2midi,
            tempo,
            prompt_type,
        ) = item
        tempo = TempoParser.DEFAULT_BPM if tempo is None else tempo

        try:
            return split_into_song_slices(
                self.max_duration,
                structure["tags"],
                utterances,
                style_tags,
                style_tokens,
                freeform_text,
                vocal2midi,
                tempo,
                prompt_type,
                self.slice_mode,
                self.line_break_dropout_rate,
            )
        except SongSliceError as e:
            raise MusicMetaError(str(e))


class SongSliceMapFilter(MusicMetaMapTransform):
    """Filter out invalid song slices"""

    def __init__(
        self,
        in_key: str = "song_slices",
        out_key: str = "song_slices",
        duration_range: tuple[float, float] = (20.0, 240.0),
        lyrics_confidence: float = 0.6,
        **kwargs,
    ):
        """
        Args:
            duration_range: The range of the duration of a song slice
            lyrics_confidence: The confidence of the lyrics. If there's any phrase whose lyrics confidence \
                is less than this value, the whole song slice will be filtered out.
        """
        super().__init__(in_key=in_key, out_key=out_key, allow_empty_in=False, **kwargs)
        self.duration_range = duration_range
        self.lyrics_confidence = lyrics_confidence

    def call(self, song_slice, **kwargs):
        try:
            validate_song_slice(
                song_slice,
                duration_range=self.duration_range,
                lyrics_confidence=self.lyrics_confidence,
            )
            return song_slice
        except SongSliceError as e:
            raise MusicMetaMapError(str(e))


class RandomSampleFromList(MusicMetaRWTransform):
    def __init__(
        self,
        in_key: str = "",
        out_key: str = "",
        n_items: int = 1,
        p_full: float = 0,
        **kwargs,
    ):
        """
        Args:
            n_items: The number of items to sample. Use the length of full list if the list is shorter than n_items.
            p_full: The probability of sampling the longest item.
        """
        super().__init__(in_key=in_key, out_key=out_key, allow_empty_in=False, **kwargs)
        self.n_items = n_items
        self.p_full = p_full  # 默认值为 0

    def call(self, items, **kwargs):
        if not items:
            return []

        sample_size = min(self.n_items, len(items))
        selected_items = random.sample(items, sample_size)

        # 找到 duration 最长的数据
        longest_item = max(items, key=lambda x: x["duration"])

        # 以 p_full 的概率将最长的数据添加到结果中
        if random.random() < self.p_full and longest_item not in selected_items:
            selected_items.append(longest_item)

        return selected_items


class SongSliceMapSequentialize(MusicMetaMapTransform):
    """Create a symbol sequence for the song slice, which can be used as input to the phoneme tokenizer."""

    def __init__(
        self,
        in_key: str = "song_slices",
        out_key: str = "song_slices",
        mode: str = "auto",
        note_dropout_rate: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            mode:
            - "phoneme": Encode special tags and phoneme symbols
            - "phoneme_note": Encode special tags, phoneme symbols with time stamps, \
                and notes with time stamps. Notes are appeneded to the end of the phoneme sequence.
            - "auto": use "phoneme_note" if the slice contains notes, use "phoneme" otherwise.
            All the modes have their "_cfg" suffix counterparts for AR cfg
            note_dropout_rate: The probability of dropping notes and fallback to "phoneme" mode.
        """
        super().__init__(in_key=in_key, out_key=out_key, allow_empty_in=False, **kwargs)
        self.mode = mode
        self.note_dropout_rate = note_dropout_rate

    def call(self, song_slice, **kwargs):
        try:
            song_slice = SongSlice.from_dict(song_slice)
            mode = "phoneme" if random.random() < self.note_dropout_rate else self.mode
            if song_slice.has_utterance:
                song_slice.sequentialize_inplace(mode)
            else:
                song_slice.sequentialize_inplace(mode)
            return song_slice.to_dict()
        except SongSliceError as e:
            raise MusicMetaMapError(str(e))


class SongSliceMapTokenize(MusicMetaMapTransform):
    """Tokenize each sequentialized song slice."""

    def __init__(
        self,
        in_key: str = "song_slices",
        out_key: str = "song_slices",
        tokenizer: str = "sami_phoneme",
        vocab: str = SamiPhonemeSeqTokenizer.DEFAULT_VOCAB_VER,
        section_duration_dropout_rate: float = 0.0,
        slice_duration_dropout_rate: float = 0.0,
        section_tag_dropout_rate: float = 0.0,
        singer_tag_dropout_rate: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            vocab: The phoneme vocabulary version that will be used to initialize the tokenizer. \
                Check the json vocab files for available versions: \
                tokenizers/sami_phoneme_tokenizer/vocabs/vocab.[version].json
        """
        super().__init__(in_key=in_key, out_key=out_key, allow_empty_in=False, **kwargs)
        self.tokenzier = {
            "sami_phoneme": SamiPhonemeSeqTokenizer,
            "sami_phoneme_pos": SamiPhonemeSeqPosTokenizer,
        }[tokenizer](vocab=vocab)
        self.key_dropout_config = {
            "section_duration": section_duration_dropout_rate,
            "slice_duration": slice_duration_dropout_rate,
        }
        self.symbol_dropout_config = {
            "section": section_tag_dropout_rate,
            "singer": singer_tag_dropout_rate,
        }

    def call(self, song_slice, **kwargs):
        try:
            return (
                SongSlice.from_dict(song_slice)
                .tokenize_inplace(
                    self.tokenzier,
                    key_dropout_config=self.key_dropout_config,
                    symbol_dropout_config=self.symbol_dropout_config,
                )
                .to_dict()
            )
        except SamiPhonemeTokenizerError as e:
            raise MusicMetaMapError(str(e))


class SanityCheck:
    def __init__(self, keys: list[str] = ["audio"]):
        self.keys = keys

    def __call__(self, item: dict[str, Any], **kwargs) -> dict[str, Any]:
        if item is None:
            return item

        for key in self.keys:
            if key not in item:
                # Log a warning if a key is missing
                uttid = item.get("uttid", "unknown")
                logger.error(f"Discard {uttid} because {key} is not in item")
                raise KeyError(
                    f"Required key '{key}' not found in item (uttid: {uttid})"
                )

        return item


class SongSliceAudioCut:
    """
    A transform class that cuts audio data into segments based on song slice information.
    Each song slice will get its corresponding audio segment from the full audio.
    """

    def __init__(
        self,
        in_key: str = "song_slices",
        audio_key="audio",
        duration_leeway_in_s: float = 2.0,
        no_slice: bool = False,
        **kwargs,
    ):
        """
        Args:
            duration_leeway_in_s: The song slice end time should not exceed this value plus the audio duration.
            no_slice: Use the full audio without slicing. It only works under the condition that len(song_slices) == 1.
        """
        # Key in the input dictionary that contains song slice information
        self.in_key = in_key
        # List of required keys that must be present in the input item
        self.required_keys = ["uttid", audio_key]
        self.audio_key = audio_key
        # Expected sample rate for the audio data
        self.sample_rate = 24000
        # NOTE: This parameter could introduce discrepancy between song slice duration and audio duration.
        # Please make sure that all audio duration-related values are only derived from the sliced audio array
        # after this transform.
        self.duration_leeway_in_samples = round(duration_leeway_in_s * self.sample_rate)
        self.no_slice = no_slice

    def __call__(self, item, **kwargs):
        # Skip processing if item is None or doesn't contain song slices
        if item is None or self.in_key not in item:
            return None

        # Validate that all required keys are present in the input item
        for key in self.required_keys:
            if key not in item:
                logger.error(f"Missing required key: {key} in {item['uttid']}")
                return None

        # Ensure the audio sample rate matches the expected rate
        if self.sample_rate != item["sample_rate"]:
            logger.error(
                f"Sample rate mismatch for {item['uttid']}: expected {self.sample_rate}, got {item['sample_rate']}"
            )
            return None

        song_slices = item[self.in_key]
        if self.no_slice and len(song_slices) != 1:
            logger.error(
                f"Discard {item['uttid']} because no_slice is True but len(song_slices) != 1"
            )
            return None

        # Process each song slice
        for song_slice_index, song_slice in enumerate(song_slices):
            # Create a unique ID for each slice by appending a zero-padded index
            song_slice["uttid"] = f"{item['uttid']}-{str(song_slice_index).zfill(2)}"
            try:
                # Convert time-based start/end points to sample indices
                start_sample = int(song_slice["start"] * item["sample_rate"])
                end_sample = int(song_slice["end"] * item["sample_rate"])

                # Ensure the slice boundaries are within the audio data
                if (
                    start_sample < 0
                    or end_sample
                    > item[self.audio_key].shape[1] + self.duration_leeway_in_samples
                ):
                    logger.error(
                        f"Invalid start or end sample for {song_slice['uttid']}"
                    )
                    return None

                # Extract the audio segment for this slice
                # Preserves all channels (dimension 0) while slicing the time dimension (dimension 1)
                if self.no_slice:
                    song_slice[self.audio_key] = item[self.audio_key]
                else:
                    song_slice[self.audio_key] = item[self.audio_key][
                        :, start_sample:end_sample
                    ]

            except Exception as e:
                # Log any errors during slice processing
                logger.debug(f"Discard {song_slice['uttid']} because of {e}")

        return item


class SongSliceAudioCutWithMetaData(SongSliceAudioCut):
    """
    A transform class that cuts audio data into segments based on song slice information.
    Each song slice will get its corresponding audio segment from the full audio.
    original item key will be copy into new item
    This is a intermedia version for merging to master puprose
    """

    def __init__(
        self,
        in_key: str = "song_slices",
        audio_key="audio",
        duration_leeway_in_s: float = 2.0,
        no_slice: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_key=in_key,
            audio_key=audio_key,
            duration_leeway_in_s=duration_leeway_in_s,
            no_slice=no_slice,
            **kwargs,
        )
        # excluded_keys will NOT be copied into new item
        self.excluded_keys = {"audio", "uttid", "text", "audio_44k", "wav"}

    def __call__(self, item, **kwargs):
        # Skip processing if item is None or doesn't contain song slices

        if item is None or self.in_key not in item:
            return None

        # Validate that all required keys are present in the input item
        for key in self.required_keys:
            if key not in item:
                logger.error(f"Missing required key: {key} in {item['uttid']}")
                return None

        # Ensure the audio sample rate matches the expected rate
        if self.sample_rate != item["sample_rate"]:
            logger.error(
                f"Sample rate mismatch for {item['uttid']}: expected {self.sample_rate}, got {item['sample_rate']}"
            )
            return None

        song_slices = copy.deepcopy(item[self.in_key])
        if self.no_slice and len(song_slices) != 1:
            logger.error(
                f"Discard {item['uttid']} because no_slice is True but len(song_slices) != 1"
            )
            return None
        # Process each song slice
        for song_slice_index, song_slice in enumerate(song_slices):
            # Create a unique ID for each slice by appending a zero-padded index
            song_slice["uttid"] = f"{item['uttid']}-{str(song_slice_index).zfill(2)}"
            try:
                # Convert time-based start/end points to sample indices
                start_sample = int(song_slice["start"] * item["sample_rate"])
                end_sample = int(song_slice["end"] * item["sample_rate"])

                if (
                    start_sample < 0
                    or end_sample
                    > item[self.audio_key].shape[1] + self.duration_leeway_in_samples
                ):
                    logger.error(
                        f"Invalid start or end sample for {song_slice['uttid']}"
                    )
                    return None

                # Extract the audio segment for this slice
                # Preserves all channels (dimension 0) while slicing the time dimension (dimension 1)
                if self.no_slice:
                    song_slice[self.audio_key] = item[self.audio_key]
                else:
                    song_slice[self.audio_key] = item[self.audio_key][
                        :, start_sample:end_sample
                    ]
                song_slice["audio_shape"] = song_slice[self.audio_key].shape[-1]

                # Copy other keys from the original item to the song slice

                for key, value in item.items():
                    if key in self.excluded_keys:
                        continue
                    if isinstance(value, np.ndarray) and value.size > 8192:
                        logger.info(f"{key} is larger than 8192. discard")
                        continue
                    song_slice[key] = value

            except Exception as e:
                # Log any errors during slice processing
                logger.debug(f"Discard {song_slice['uttid']} because of {e}")

        item[self.in_key] = song_slices

        return item


class CalculateToken:
    """
    A transform class that calculates the total number of tokens by combining:
    1. Audio tokens: audio samples downsampled by token_rate
    2. Phoneme tokens: length of phoneme token sequence
    """

    def __init__(
        self,
        token_rate: int = 25,
        phoneme_tokens_key: str = "phoneme_tokens",
        audio_length_out_key: str = "target_tokens_length",
        lyrics_length_out_key: str = "lyrics_tokens_length",
        total_token_length: str = "num_total_tokens",
        sample_rate: int = 24000,
        **kwargs,
    ):
        """
        Initialize the token calculator
        Args:
            token_rate: Downsampling rate for audio (samples per token)
            audio_length_out_key: Key to store the calculated audio token count
            lyrics_length_out_key: Key to store the calculated lyrics token count
            total_token_length: Key to store the calculated total token count
            sample_rate: Sample rate of the audio
        """
        self.token_rate = token_rate
        self.sample_rate = sample_rate
        self.phoneme_tokens_key = phoneme_tokens_key
        self.audio_length_out_key = audio_length_out_key
        self.lyrics_length_out_key = lyrics_length_out_key
        self.total_token_length = total_token_length

    def __call__(
        self, item: Optional[dict[str, Any]] = None, **kwargs
    ) -> Optional[dict[str, Any]]:
        """
        Calculate total tokens for the given item
        Args:
            item: Dictionary containing:
                - audio: Audio tensor [channels, samples]
                - phoneme_tokens: Dictionary with 'tokens' list
        Returns:
            item: Updated with total token count, or None if invalid
        """
        if item is None:
            return None

        if "audio" in item and isinstance(item["audio"], np.ndarray):
            item[self.audio_length_out_key] = (
                math.ceil(int(item["audio"].shape[-1] / self.sample_rate))
                * self.token_rate
            )

        else:
            item[self.audio_length_out_key] = 0
            logger.debug(f"Missing audio for {item['uttid']}")

        item[self.lyrics_length_out_key] = len(
            item[self.phoneme_tokens_key]["input_ids"]
        )

        item[self.total_token_length] = (
            item[self.audio_length_out_key] + item[self.lyrics_length_out_key]
        )

        return item


class SpeakerIdParser:
    """
    A class to parse speaker IDs based on gender information from input data.

    Attributes:
        in_key (str): The key in the input data where the gender information is stored.
        out_key (str): The key in the output data where the speaker ID will be stored.
        default_value (int): The default speaker ID to use if gender information is missing.
    """

    def __init__(
        self,
        in_key: str = "style_tags.gender",
        out_key: str = "speaker_id",
        default_value: int = None,
        **kwargs,
    ):
        """
        Initializes the SpeakerIdParser with the given keys and default value.

        Args:
            in_key (str): The key to look up gender information in the input data.
            out_key (str): The key to store the parsed speaker ID in the output data.
            default_value (int): The default speaker ID to use if gender information is missing.
            **kwargs: Additional keyword arguments (not used in this implementation).
        """
        self.in_key = in_key
        self.out_key = out_key
        self.default_value = default_value

    def __call__(self, item, **kwargs):
        """
        Processes the input item to assign a speaker ID based on gender information.

        Args:
            item (dict): The input data containing gender information.
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            dict: The input data with the speaker ID added, or None if the item is invalid.
        """
        # If the input item is None, return None
        if item is None:
            return None

        # Initialize the output speaker ID
        output_id = 0

        # Retrieve the gender information from the input item using the specified key
        gender = get_nested_value(item, self.in_key)

        # If gender information is missing, handle it appropriately
        if gender is None:
            logger.debug(f"Missing gender for {item['uttid']}")
            if self.default_value is not None:
                # Use the default value if provided
                output_id = self.default_value
            else:
                # Return None if no default value is provided
                return None
        else:
            # Assign speaker ID based on gender
            if "Neutral" in gender:
                output_id = 43  # ID for Neutral
            elif "Multiple" in gender:
                output_id = 44  # ID for Multiple
            elif "Chorus" in gender:
                output_id = 45  # ID for Chorus
            elif "Child" in gender:
                output_id = 46  # ID for Child
            elif "Male" in gender and "Female" in gender:
                output_id = 47  # ID for Duet
            elif "Male" in gender:
                output_id = 48  # ID for Male
            elif "Female" in gender:
                output_id = 49  # ID for Female
            elif "" in gender:  # ID for empty (dropout)
                output_id = 0
            else:
                # Log a warning if the gender is unknown
                logger.debug(f"Unknown gender {gender} for {item['uttid']}")

        # Add the speaker ID to the output item
        item[self.out_key] = output_id

        # Return the modified item
        return item


class RenameKey:
    """
    A class to rename keys in a dictionary.

    Attributes:
        key_map (dict): A dictionary mapping old keys to new keys.
    """

    def __init__(self, in_key_list: list, out_key_list: list, in_place: bool = False):
        """Initialize list collate function, setup list key."""

        self.in_key_list = in_key_list
        self.out_key_list = out_key_list
        self.in_place = in_place

        assert len(self.in_key_list) == len(self.out_key_list)

    def __call__(self, item, **kwargs):

        if item is None:
            return item

        for key_index, key in enumerate(self.in_key_list):

            if key not in item:
                logger.debug(f"Missing key {key} for {item['uttid']}")
                return None

            if self.in_place:
                item[self.out_key_list[key_index]] = item.pop(key)
            else:
                item[self.out_key_list[key_index]] = item[key]

        return item


class ConvertDictToList:
    def __init__(
        self,
        in_key="style_tags",
        out_key="style_text",
        list_order=[
            "genre",
            "genre_extra",
            "extra",
            "mood",
            "scene",
            "speaker",
            "voice",
            "lang",
            "sinking",
            "instrument",
            "tempo",
            "key",
            "mode",
        ],
    ):

        self.in_key = in_key
        self.out_key = out_key
        self.list_order = list_order

    def __call__(self, item, **kwargs):

        if item is None:
            return None

        temp_result = []

        for temp_key in self.list_order:
            temp_result.append(item[self.in_key][temp_key])

        item[self.out_key] = temp_result
        return item


class AddField:
    def __init__(
        self,
        key="conditions",
        value="style_category,speaker_id,tempo,freeform_text,lyrics_tokens",
    ):

        self.key = key
        self.value = value

    def __call__(self, item, **kwargs):
        if item is None:
            return item

        item[self.key] = self.value
        return item


class NormalizeAudioToFloat32:
    def __init__(self, in_key="audio", out_key="audio"):
        self.in_key = in_key
        self.out_key = out_key

    def normalize_audio_to_float32(self, array: np.ndarray) -> np.ndarray:
        # 如果输入已经是 float32 类型，直接返回
        if array.dtype == np.float32:
            return array

        # 如果是 float64 类型，转换为 float32
        if array.dtype == np.float64:
            return array.astype(np.float32)

        # 对于整数类型，进行归一化
        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            abs_max = 2 ** (info.bits - 1)
            return array.astype(np.float32) / abs_max

    def __call__(self, item, **_kwargs):
        if item is None or self.in_key not in item:
            return item

        item[self.out_key] = self.normalize_audio_to_float32(item[self.in_key])

        # sf.write("test_new.wav", np.squeeze(item[self.out_key]), 24000)
        return item


def read_wave_as_int16(wav: BytesIO):
    if int.from_bytes(wav.getbuffer()[20:22], "little") == 3:
        waveform, sample_rate = sf.read(wav, dtype="float32")
        waveform = (waveform * np.iinfo(np.int16).max).astype(np.int16)
    else:
        waveform, sample_rate = sf.read(wav, dtype="int16")
    return waveform, sample_rate


class MusicParser:
    """extract waveform from wav binary data"""

    def __init__(
        self, in_key="audio", out_key="audio", min_len=0.05, max_len=9e9, skip_prob=0.0
    ):
        """init
        Args:
            in_key: which key contain wave binary data
            min_len: the minimum value of audio length
            max_len: the maximum value of audio length
        """
        self.in_key = in_key
        self.out_key = out_key
        self.min_len = min_len
        self.max_len = max_len
        self.skip_prob = skip_prob

    def __call__(self, item, **_kwargs):
        """extract waveform and sample_rate"""
        if item is None or self.in_key not in item:
            return item
        if self.skip_prob > 0 and random.uniform(0, 1) < self.skip_prob:
            # skip parser waveform for TOG
            return item

        try:
            waveform, sample_rate = read_wave_as_int16(io.BytesIO(item[self.in_key]))
        except Exception:
            logger.info(f"Failed to parse {item['uttid']}")
            return None

        # Ensure [n_channel, n_samples]
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        elif waveform.ndim == 2:
            if waveform.shape[0] > waveform.shape[1]:
                waveform = waveform.T
        else:
            logger.info(f"Failed to parse {item['uttid']}, ndim: {waveform.ndim}")
            return None

        channel_num = waveform.shape[0]
        duration = waveform.shape[1] / float(sample_rate)

        if self.min_len < duration < self.max_len:
            item["bits"] = np.iinfo(waveform.dtype).bits  # 位深度 (16, 24, 32等)
            item["sample_rate"] = sample_rate  # 采样率 (如 44100, 48000)
            item["channel_num"] = channel_num  # 通道数
            item["duration"] = duration  # 音频时长 (秒)

            # 如果多通道，只取第一个通道 (单声道处理)
            if channel_num > 1:
                waveform = waveform[
                    0, :
                ]  # 取第一个通道 [n_channels, n_samples] -> [n_samples]
                channel_num = 1

            item[self.out_key] = waveform.reshape(channel_num, -1)
            return item
        else:
            logger.info(
                f"""f{item['uttid']} duration is {duration} which is less than \
{self.min_len} or large than {self.max_len}"""
            )
            return None


class MusicWavResample:
    """resample the waveform"""

    def __init__(
        self, sample_rate=24000, key="waveform", resample_backend="librosa", **_kwargs
    ):
        """init
        Args:
            sample_rate: target sample rate
            key: the key to resample
        """
        self.sample_rate = sample_rate
        self.key = key
        self.resample_backend = resample_backend

    def __call__(self, item, **_kwargs):
        """call to resample"""
        if (
            item is None
            or "sample_rate" not in item
            or item["sample_rate"] == self.sample_rate
        ):
            return item
        if self.key not in item:
            logger.info(f"{self.key} is not in data {item['uttid']}")
            return None
        if self.resample_backend == "sox":
            raise NotImplementedError("sox is not implemented")
        elif self.resample_backend == "librosa":
            data = item[self.key]
            # data_type = data.dtype
            p_data = librosa.core.resample(
                data[0, :],
                orig_sr=item["sample_rate"],
                target_sr=self.sample_rate,
                res_type="soxr_hq",
            )
            item["sample_rate"] = self.sample_rate
            # PATCH: Do not cast to int16
            # item[self.key] = p_data.astype(np.int16).astype(data_type).reshape(1, -1)
            resampled_waveform = p_data.reshape(1, -1)
            item[self.key] = resampled_waveform
        else:
            raise ValueError(f"Unknown resample backend: {self.resample_backend}")
        return item


class CaptureWorkerInfo:
    def __init__(self, out_key="worker_info"):
        self.out_key = out_key

    def __call__(self, item, **kwargs):
        if item is None:
            return item
        if not isinstance(item, list):
            item = [item]
        for it in item:
            it[self.out_key] = kwargs
        if len(item) == 1:
            return item[0]
        return item


class GetItem:
    def __init__(
        self,
        in_key,
        out_key,
        convert_str_to_dict=False,
        return_item_if_in_key_not_found=False,
    ):
        self.in_key = in_key
        self.out_key = out_key
        self.convert_str_to_dict = convert_str_to_dict
        self.return_item_if_in_key_not_found = return_item_if_in_key_not_found

    def __call__(self, item, **kwargs):
        if item is None:
            return None

        data = get_nested_value(item, self.in_key)

        if self.convert_str_to_dict and isinstance(data, str):
            data = ast.literal_eval(data)

        if data is None:
            if self.return_item_if_in_key_not_found:
                return item
            else:
                logger.error(f"{self.in_key} is not right for data {item['uttid']}")
                return None
        item[self.out_key] = data

        return item


class DeleteItem:
    def __init__(self, in_key):
        self.in_key = in_key

    def __call__(self, item, **kwargs):
        if item is None:
            return None
        item_path, item_key = self.in_key.rsplit(".", 1)
        data = get_nested_value(item, item_path)
        if item_key in data:
            del data[item_key]
        return item


class DataAdaptor:
    def __init__(self):
        pass

    def __call__(self, item, **kwargs):
        # breakpoint()
        if item is None:
            return None

        if "text" in item.keys() and item["text"] != "":
            item["meta"]["lyrics"] = item["text"]
            item["meta"]["data_type"] = "tts"
            item["data_type"] = "tts"
            item["meta"]["standard_music_meta"] = {
                "web": {"genre": ["spoken"]},
                "human_annotation": {"genre": ["spoken"]},
                "tagging_model": {"genre": ["spoken"]},
            }

        else:
            item["meta"]["data_type"] = "l2s"
            item["data_type"] = "l2s"

        return item


class InferDataAdaptor:
    def __init__(self, out_key="input_strings", prompt_type="keyword"):
        self.out_key = out_key
        self.prompt_type = prompt_type

    def __call__(self, item, **kwargs):

        if item is None:
            return None

        index = item["index"]
        digits_only = "".join([char for char in index if char.isdigit()])
        index = int(digits_only)

        if index % 2 == 0:
            data_type = "tts"
            if self.prompt_type == "keyword":
                prompt = "spoken"
            elif self.prompt_type == "special_token":
                prompt = "<genre>spoken</genre>"
            lyrics = item["lyrics"]

            pattern = re.compile("\\[[^\\]]*\\]\\n?")
            lyrics = pattern.sub("", lyrics)

        else:
            data_type = "l2s"
            prompt = item["prompt"]
            lyrics = item["lyrics"]
            section_tags = re.findall(r"\[.*?\]", lyrics)
            lyrics = "\n".join(section_tags) + "\n"

        item["prompt"] = prompt
        item["lyrics"] = lyrics
        item["data_type"] = data_type

        item[self.out_key] = [prompt, lyrics]

        return item


class GetFrontendResults:
    def __init__(self, out_key="frontend_results"):
        self.out_key = out_key
        self.extra_field = "meta.raw.extra"

    def __call__(self, item, **kwargs):
        if item is None:
            return item

        temp = get_nested_value(item, self.extra_field)
        if temp is None:
            logger.debug(f"{self.extra_field} is not right for data {item['uttid']}")
            return None

        try:
            temp = ast.literal_eval(temp)
        except (ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse extra field for {item['uttid']}: {str(e)}")
            return None

        if "frontend_results" not in temp:
            return None

        item[self.out_key] = temp["frontend_results"]
        return item


class PromptToSongSliceTransformParquet:
    """Transform the Chinese Vocal V4/V5 prompts to song slices with tokenized phonemes and style tags"""

    def __init__(
        self,
        phoneme_vocab: Optional[str] = SamiPhonemeSeqTokenizer.DEFAULT_VOCAB_VER,
        style_tag_vocab: Optional[str] = _StyleTagTokenizer.DEFAULT_VOCAB_VER,
        phoneme_tokenizer: Optional[str] = "sami_phoneme",
        use_cfg: bool = True,
        cfg_items: Union[list[str], tuple[str]] = (),
        standardize: bool = False,
        **kwargs,
    ):
        """
        Args:
            phoneme_vocab: The phoneme vocabulary version that will be used to initialize the tokenizer. \
                Check the json vocab files for available versions: \
                tokenizers/sami_phoneme_tokenizer/vocabs/vocab.[version].json
            style_tag_vocab: The style tag vocabulary version that will be used to initialize the tokenizer. \
                Check the json vocab files for available versions: \
                tokenizers/style_tag_tokenizer/vocabs/vocab.[version].json
        """
        self.phoneme_tokenizer = {
            "sami_phoneme": SamiPhonemeSeqTokenizer,
            "sami_phoneme_pos": SamiPhonemeSeqPosTokenizer,
        }[phoneme_tokenizer](vocab=phoneme_vocab)
        self.style_tag_tokenizer = (
            _StyleTagTokenizer(vocab=style_tag_vocab) if style_tag_vocab else None
        )
        self.use_cfg = use_cfg
        self.cfg_items = [
            self._remap_style_tag_key(item)
            for item in cfg_items
            if item != "speaker_id"
        ]  # force map key value
        self.standardize = standardize

    def _generate_data(
        self,
        phrases: List[Phrase],
        style_tags: Dict[str, Any],
        freeform_text: str,
        tempo_xval: float,
    ) -> SongSlice:
        style_input_ids = self._tokenize_style_tags(style_tags)
        song_slice = SongSlice(
            phrases=phrases,
            style_tags=style_tags,
            style_input_ids=style_input_ids,
            freeform_text=freeform_text,
            tempo=tempo_xval,
        )
        if self.phoneme_tokenizer:
            song_slice.sequentialize_inplace("phoneme")
            song_slice.tokenize_inplace(self.phoneme_tokenizer)
        return song_slice

    def _generate_cfg_data(
        self,
        phrases: List[Phrase],
        style_tags: Dict[str, Any],
        freeform_text: str,
        tempo_xval: float,
    ) -> SongSlice:
        """Generate configuration data for the song slice"""
        if "lyrics" in self.cfg_items:
            key_dropout_config = {"section_duration": 1.0, "slice_duration": 1.0}
            symbol_dropout_config = {"section": 1.0, "singer": 1.0}
        else:
            key_dropout_config = None
            symbol_dropout_config = None
        if "line_break" in self.cfg_items:
            if "lyrics" in self.cfg_items:
                phrases = [p for p in phrases if p.has_utterance]
            phrases = drop_out_line_breaks(phrases, 1.0)
        if "freeform_text" in self.cfg_items:
            freeform_text = ""
        style_tags_cfg = {
            k: ([""] if k in self.cfg_items else v) for k, v in style_tags.items()
        }
        style_input_ids_cfg = self._tokenize_style_tags(style_tags_cfg)
        song_slice_cfg = SongSlice(
            phrases=phrases,
            style_tags=style_tags_cfg,
            style_input_ids=style_input_ids_cfg,
            freeform_text=freeform_text,
            tempo=tempo_xval,
        )
        if self.phoneme_tokenizer:
            song_slice_cfg.sequentialize_inplace("phoneme")
            song_slice_cfg.tokenize_inplace(
                self.phoneme_tokenizer,
                key_dropout_config=key_dropout_config,
                symbol_dropout_config=symbol_dropout_config,
            )
        return song_slice_cfg

    def _remap_style_tag_key(self, key: str) -> str:
        """Map old style tag keys to new ones"""
        return {
            "speaker": "vocal_gender",
            "voice": "vocal_timbre",
            "lang": "language",
            "sinking": "is_sinking",
            # Add other mappings as needed
        }.get(key, key)

    def _adjust_phrase_time_span(self, phrases: List[Phrase], total_duration: float):
        """Adjust phrase timings to evenly distribute across total duration"""
        phrase_duration = total_duration / len(phrases)
        for i, phrase in enumerate(phrases):
            start_time = i * phrase_duration
            end_time = (i + 1) * phrase_duration
            phrase.time_span = (start_time, end_time)

    def _tokenize_style_tags(
        self, style_tags: dict[str, list[str]]
    ) -> Optional[dict[str, list[int]]]:
        """Tokenize style tags using the style tag tokenizer"""
        if not self.style_tag_tokenizer:
            return None
        if self.standardize:
            tags = TagsProto.from_dict(style_tags)
            tags.standardize_inplace(self.style_tag_tokenizer.vocab)
            tags, oov = tags.remove_oov_tags(self.style_tag_tokenizer.vocab)
            if oov:
                logger.debug(f"OOV tags: {oov}")
            style_tags = tags.fill_empty_inplace().to_dict()
        return tokenize_tags(style_tags, self.style_tag_tokenizer)

    def _merge_slices(
        self, song_dict: Dict[str, Any], cfg_dict: Dict[str, Any], suffix: str = "_cfg"
    ) -> Dict[str, Any]:
        """merge two dict, if key in both dict, and value is different, add suffix to key in cfg_dict

        Args:
            song_dict: song_slice_dict
            cfg_dict: cfg_version
            suffix: suffix

        Returns:
            merged_dict: merged dict
        """
        merged_dict = {}
        all_keys = set(song_dict.keys()) | set(cfg_dict.keys())

        for key in all_keys:
            song_val = song_dict.get(key)
            cfg_val = cfg_dict.get(key)
            if key in song_dict and key in cfg_dict:
                # NOTE: Getting rid of this if statement ensures the existence of
                # new keys does not depend their values. Otherwise, the result
                # might not pass SanityCheck all the time.
                # Another solution is to remove the checking of _cfg fields in
                # SanityCheck. The downside is that it won't be able to catch
                # the error if _cfg fields are not correctly set.
                # if song_val == cfg_val:
                #     merged_dict[key] = song_val
                # else:
                merged_dict[key] = song_val
                merged_dict[f"{key}{suffix}"] = cfg_val
            elif key in song_dict:
                merged_dict[key] = song_val
            else:
                logger.error(f"{key} not in cfg_dict and not in song_dict")
                raise ValueError(f"{key} not in cfg_dict and not in song_dict")

        return merged_dict

    def __call__(self, item, **kwargs):

        # Validate required fields
        required_fields = ["frontend_results", "duration", "tags_music"]
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Input item must contain '{field}' key")

        # convert to phrases
        frontend_results = item["frontend_results"]
        phrases = [
            Phrase.parse(text=r["text"], phonemes=r["phonemes"])
            for r in frontend_results
        ]

        duration = float(item["duration"])
        # HACK: To make SongSlice.duration reflect the total duration, assign fake time stamps
        # for each phrase.
        # TODO: Support cover song
        if not phrases:
            phrases = [Phrase(time_span=(0, duration))]
        else:
            self._adjust_phrase_time_span(phrases, duration)

        # Process style tags
        style_tags = {
            self._remap_style_tag_key(k): v
            for k, v in item["tags_music"].items()
            if k != "speaker_id"
        }

        freeform_text = item["freeform_text"]
        tempo_xval = item.get(
            "tempo", TempoParser.DEFAULT_BPM
        )  # NOTE: set default here to make it compatible with the previous code

        # Create song slice
        try:
            song_slice = self._generate_data(
                phrases, style_tags, freeform_text, tempo_xval
            )
        except TagError as e:
            logger.error(f"Failed to tokenize style tags: {str(e)}")
            return None
        song_slice_dict = song_slice.to_dict()

        if self.use_cfg:
            try:
                song_slice_cfg = self._generate_cfg_data(
                    phrases, style_tags, freeform_text, tempo_xval
                )
            except TagError as e:
                logger.error(f"Failed to tokenize style tags: {str(e)}")
                return None
            song_slice_cfg_dict = song_slice_cfg.to_dict()
            merged_dict = self._merge_slices(song_slice_dict, song_slice_cfg_dict)
            item.update(merged_dict)

            # Extremely unprofession code
            assert "speaker_id" in item
            item["speaker_id_cfg"] = 0

        else:
            item.update(song_slice_dict)

        return item


class ExtractLyrics:
    def __init__(
        self,
        in_key: str = "phrase",
        out_key: str = "raw_lyrics",
        with_section_tag: bool = True,
        only_lyrics: bool = False,
    ):
        self.in_key = in_key
        self.out_key = out_key
        self.with_section_tag = with_section_tag
        self.only_lyrics = only_lyrics

    def __call__(self, item, **kwargs):

        if item is None:
            return None

        if self.in_key not in item:
            logger.info(f"{self.in_key} not in item:{item['uttid']}")
            return None

        results = []

        for phrase in item[self.in_key]:
            if phrase["section_tag"] is None:
                results.append(phrase["text"])
            elif phrase["section_tag"] is not None and self.with_section_tag:
                results.append(f"[{phrase['section_tag']}]")
        lyrics = "\n".join(results)
        if self.only_lyrics:
            item[self.out_key] = lyrics
            return item
        instruct = "Generate music from the given musical style tags and segmented lyrics, processing each segment incrementally."  # noqa:E501
        item[self.out_key] = "user\n{}\ntags:\n{}\nlyrics:\n{}\nassistant\n".format(
            instruct, item["freeform_text"], lyrics
        )
        # logger.info(f"{self.out_key}={item[self.out_key]}")

        return item


class TokenParser:
    def __init__(
        self,
        in_key="umm_token",
        out_key="umm_token",
        duration_key="duration",
        min_len=0.05,
        max_len=9e9,
        skip_prob=0.0,
        token_frame_rate=25,
        duration_leeway_in_s: float = 2.0,
    ):
        """init
        Args:
            in_key: which key contain wave binary data
            min_len: the minimum value of audio length
            max_len: the maximum value of audio length
        """
        self.in_key = in_key
        self.out_key = out_key
        self.min_len = math.ceil(min_len * token_frame_rate)
        self.max_len = math.ceil(max_len * token_frame_rate)
        self.skip_prob = skip_prob
        self.duration_key = duration_key
        self.duration_leeway_in_s = duration_leeway_in_s

    def __call__(self, item, **_kwargs):
        """extract waveform and sample_rate"""
        if item is None or self.in_key not in item:
            return item
        if self.skip_prob > 0 and random.uniform(0, 1) < self.skip_prob:
            # skip parser waveform for TOG
            return item
        if item[self.in_key] is None:
            return None

        try:
            umm_token_feature = pickle.loads(item[self.in_key])
        except Exception as e:
            logger.error(f"Parser umm_token failed for {item['uttid']} with error {e}")
            umm_token_feature = None
        if umm_token_feature is None:
            logger.error(
                f"Invalid umm_token for {item['uttid']} (umm_token_feature is None)"
            )
            return None
        assert isinstance(umm_token_feature, dict)
        umm_token = umm_token_feature.get("umm_token", None)
        time_span = umm_token_feature.get("time_span", None)
        if umm_token is None:
            logger.error(f"Invalid umm_token for {item['uttid']} (umm_token is None)")
            return None
        else:
            if isinstance(umm_token, np.ndarray):
                if self.min_len < umm_token.shape[-1] < self.max_len:
                    item[self.out_key] = umm_token
                    return item
                else:
                    # fmt: off
                    logger.error(f"skip umm_token for {item['uttid']} due to {umm_token.shape[-1]=} not in [{self.min_len}, {self.max_len}]")  # noqa:E501
                    # fmt: on
                    return None
            elif isinstance(umm_token, list):
                valid_umm_token = []
                valid_time_span = []
                valid_duration = []

                full_song_duration = item.get(self.duration_key, None)
                for seg_idx in range(len(umm_token)):
                    if (
                        full_song_duration is not None
                        and time_span[seg_idx][-1]
                        > float(full_song_duration) + self.duration_leeway_in_s
                    ):
                        # fmt: off
                        logger.error(f"skip umm_token for {item['uttid']} due to {time_span[seg_idx][-1]} > {full_song_duration + self.duration_leeway_in_s}")  # noqa:E501
                        # fmt: on
                        return None
                    if self.min_len <= umm_token[seg_idx].shape[-1] <= self.max_len:
                        valid_umm_token.append(umm_token[seg_idx])
                        valid_time_span.append(umm_token_feature["time_span"][seg_idx])
                        valid_duration.append(umm_token_feature["duration"][seg_idx])
                if len(valid_umm_token) > 0:
                    item[self.out_key] = {
                        "umm_token": valid_umm_token,
                        "time_span": valid_time_span,
                        "duration": valid_duration,
                    }
                    return item
                else:
                    logger.error(
                        f"skip umm_token for {item['uttid']} due to empty invalid umm_token"
                    )
                    return None
            else:
                logger.error(
                    f"Invalid umm_token type '{type(umm_token)}' for {item['uttid']}"
                )
                return None


class SongSliceTokenCut:
    """
    A transform class that cuts audio data into segments based on song slice information.
    Each song slice will get its corresponding audio segment from the full audio.
    """

    def __init__(
        self,
        in_key="song_slices",
        token_key="umm_token",
        duration_leeway_in_s: float = 2.0,
        **kwargs,
    ) -> None:
        self.in_key = in_key
        self.required_keys = ["uttid", token_key]
        self.token_key = token_key

        self.token_frame_rate = kwargs.get("token_frame_rate", 25)
        # self.sample_rate = kwargs.get("sample_rate", 24000)
        self.duration_leeway_in_tokens = round(
            duration_leeway_in_s * self.token_frame_rate
        )

    def __call__(self, item, **kwargs) -> Any:
        if item is None or self.in_key not in item:
            return None
        for key in self.required_keys:
            if key not in item:
                logger.error(f"Missing required key: {key} in {item['uttid']}")
                return None

        umm_token = pickle.loads(item[self.token_key])
        if umm_token is None:
            logger.error(f"Invalid umm_token for {item['uttid']}")
            return None

        if isinstance(umm_token, dict):
            umm_token = umm_token["umm_token"]

        song_slices = item[self.in_key]
        # Process each song slice
        for song_slice_index, song_slice in enumerate(song_slices):
            song_slice["uttid"] = f"{item['uttid']}-{str(song_slice_index).zfill(2)}"
            try:
                # Convert time-based start/end points to sample indices
                start_token = int(song_slice["start"] * self.token_frame_rate)
                end_token = int(song_slice["end"] * self.token_frame_rate)
                # Ensure the slice boundaries are within the audio data
                if (
                    start_token < 0
                    or end_token > umm_token.shape[-1] + self.duration_leeway_in_tokens
                ):
                    # fmt: off
                    logger.error(f"Invalid start or end token for {song_slice['uttid']} ({start_token=}, {end_token=}, {umm_token.shape[-1]=})")  # noqa:E501
                    # fmt: on
                    return None
                # Extract the audio segment for this slice
                song_slice[self.token_key] = item[self.token_key][start_token:end_token]

                # Extract the audio segment for this slice
                # Preserves all channels (dimension 0) while slicing the time dimension (dimension 1)
                song_slice[self.token_key] = umm_token[start_token:end_token]

            except Exception as e:
                # Log any errors during slice processing
                logger.debug(f"Discard {song_slice['uttid']} because of {e}")

        return item


class SongSliceTokenMatch:
    def __init__(
        self,
        in_key="song_slices",
        token_key="umm_token",
        duration_leeway_in_s: float = 2.0,
        n_items=10,
        **kwargs,
    ) -> None:
        self.in_key = in_key
        self.required_keys = ["uttid", token_key]
        self.token_key = token_key

        self.token_frame_rate = kwargs.get("token_frame_rate", 25)
        # self.sample_rate = kwargs.get("sample_rate", 24000)
        self.duration_leeway_in_tokens = round(
            duration_leeway_in_s * self.token_frame_rate
        )
        self.n_items = n_items

    def __call__(self, item, **kwargs) -> Any:
        if item is None or self.in_key not in item:
            return None
        for key in self.required_keys:
            if key not in item:
                logger.error(f"Missing required key: {key} in {item['uttid']}")
                return None

        umm_token = item[self.token_key]
        if umm_token is None:
            logger.error(f"Invalid umm_token for {item['uttid']}")
            return None

        if isinstance(umm_token, dict):
            umm_token = umm_token["umm_token"]

        song_slices = item.pop(self.in_key)
        umm_token_items = item.pop(self.token_key)
        umm_token_map = {
            umm_token_items["time_span"][i]: umm_token_items["umm_token"][i]
            for i in range(len(umm_token_items["umm_token"]))
        }
        # print(f"[before:0]{type(song_slices)}")
        # Process each song slice
        picked_song_slices = []
        for song_slice_index, song_slice in enumerate(song_slices):
            # print(f"[before:1]{type(song_slice)}")
            song_slice["uttid"] = f"{item['uttid']}-{str(song_slice_index).zfill(2)}"
            time_span = song_slice["time_span"]
            if time_span in umm_token_map:
                song_slice[self.token_key] = umm_token_map[time_span]
                picked_song_slices.append(song_slice)
        picked_song_slices = random.sample(
            picked_song_slices, min(len(picked_song_slices), self.n_items)
        )
        item[self.in_key] = picked_song_slices
        return item


class CalculateTokenV2:
    def __init__(
        self,
        audio_length_out_key: str = "target_tokens_length",
        lyrics_length_out_key: str = "lyrics_tokens_length",
        total_token_length: str = "num_total_tokens",
        **kwargs,
    ) -> None:
        self.audio_length_out_key = audio_length_out_key
        self.lyrics_length_out_key = lyrics_length_out_key
        self.total_token_length = total_token_length

    def __call__(
        self, item: Optional[dict[str, Any]] = None, **kwargs
    ) -> Optional[dict[str, Any]]:
        """
        Calculate total tokens for the given item
        Args:
            item: Dictionary containing:
                - songslice.umm_token: Audio tensor [channels, samples]
                - phoneme_tokens: Dictionary with 'tokens' list
        Returns:
            item: Updated with total token count, or None if invalid
        """
        if item is None:
            return None

        if "target_token_ids" not in item:
            item[self.audio_length_out_key] = 0
            logger.debug(f"Missing audio for {item['uttid']}")
        else:
            item[self.audio_length_out_key] = item["target_token_ids"].shape[-1]

        item[self.lyrics_length_out_key] = len(item["phoneme_tokens"]["input_ids"])

        item[self.total_token_length] = (
            item[self.audio_length_out_key] + item[self.lyrics_length_out_key]
        )

        return item


class CheckDataType:

    def __init__(self, out_key: str = "data_type"):

        self.out_key = out_key

    def __call__(self, item, **__kwargs):
        if item is None:
            return None

        if "text" in item and len(item["text"]) > 0:
            item[self.out_key] = "speech"
            return item

        if "data_type" in item["meta"] and item["meta"]["data_type"] == "instrument":
            item[self.out_key] = "instrument"
            return item

        item[self.out_key] = "vocal"
        return item


class TextNormalizer:
    def __init__(self, in_key="text", text_length_key="text_length"):

        from string import punctuation

        from zhon.hanzi import punctuation as punctuation_zh

        self.in_key = in_key
        self.text_length_key = text_length_key

        self.punctuation_zh = punctuation_zh
        self.punctuation_en = punctuation.replace("'", "")

    def __call__(self, item, **kwargs):

        if item is None or self.in_key not in item:
            return None

        text = item[self.in_key]

        text = self._normalize_text(text)
        text = self._remove_punc(text)
        text = self._remove_cn_space(text)
        item[self.in_key] = text

        item[self.text_length_key] = len(text)
        return item

    def _normalize_text(self, text):
        text = text.replace("&", " and ").replace("/", " ")
        return text.translate(str.maketrans("", "", self.punctuation_en)).strip()

    def _remove_punc(self, text):
        # Remove all Chinese + English punctuation (but preserve apostrophe)
        all_punc = self.punctuation_zh + self.punctuation_en
        return re.sub(f"[{re.escape(all_punc)}]", "", text)

    def _remove_cn_space(self, text):
        # Remove spaces between consecutive Chinese characters
        return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


class UtteranceSegmentSampler:
    def __init__(
        self,
        out_key: str = "audio_slice",
        data_type_key: str = "data_type",
        instrument_text: str = "<PURE_INSTRUMENT>",
        num_segments: int = 3,
        full_song: bool = True,
        min_duration: float = 20.0,
        max_duration: float = 60.0,
    ):

        self.out_key = out_key
        self.data_type_key = data_type_key
        self.instrument_text = instrument_text  # Not used for now
        self.num_segments = num_segments
        self.full_song = full_song
        self.min_duration = min_duration  # in seconds
        self.max_duration = max_duration  # in seconds

    def _get_utterances(self, item):
        # Get basic identifiers for error messages
        uttid = item.get("uttid", "unknown")
        crs_filename = item.get("crs_filename", "None")

        # Safely get lyrics_data with proper type checking
        lyrics_data = item.get("meta", {}).get("lyrics")

        if lyrics_data is None:
            logger.debug(f"lyrics_data is None: {uttid}, {crs_filename}")
            return None

        if not isinstance(lyrics_data, dict):
            logger.debug(f"lyrics_data is not a dict: {uttid}, {crs_filename}")
            return None

        # Try to get utterances directly first
        utterances = lyrics_data.get("utterances")
        if utterances:
            return utterances

        # Fallback to checking result list if utterances not found directly
        result_list = lyrics_data.get("result")
        if isinstance(result_list, list) and result_list:
            first_result = result_list[0]
            if isinstance(first_result, dict):
                utterances = first_result.get("utterances")
                if utterances:
                    return utterances

        # If we get here, no utterances were found
        logger.error(
            f"Utterances not found for uttid: {uttid}, crs_filename: {crs_filename}"
        )
        return None

    def _handle_speech_data(
        self, item: Dict[str, Any], duration_sec: float
    ) -> Dict[str, Any]:
        """Handles speech data type."""

        if duration_sec < self.min_duration or duration_sec > self.max_duration:
            logger.debug(
                f"Duration {duration_sec} is outside the allowed range for uttid: {item.get('uttid', 'unknown_uttid')},"
                f" crs_filename: {item.get('crs_filename', 'None')}"
            )
            return None

        if "text" not in item:
            logger.debug(
                f"Text not found for uttid: {item.get('uttid', 'unknown_uttid')}, "
                f"crs_filename: {item.get('crs_filename', 'None')}"
            )
            return None

        results = [{"start": 0.0, "end": duration_sec, "text": item["text"]}]
        item[self.out_key] = results
        return item

    def _handle_instrument_data(
        self, item: Dict[str, Any], duration_sec: float
    ) -> Dict[str, Any]:
        """Handles instrument data type."""
        results: List[Dict[str, Any]] = []

        dur = random.uniform(self.min_duration, min(self.max_duration, duration_sec))
        dur = round(dur, 3)
        start_time = round(random.uniform(0, max(0, duration_sec - dur)), 3)
        end_time = min(start_time + dur, duration_sec)
        results.append({"start": start_time, "end": end_time, "text": ""})
        item[self.out_key] = results
        return item

    def _handle_vocal_data(
        self, item: Dict[str, Any], duration_sec: float
    ) -> Optional[Dict[str, Any]]:
        """Handles vocal data type with target duration distribution."""

        # Extract and preprocess utterances
        utterances = self._prepare_utterances(item, duration_sec)
        if not utterances:
            return None

        # Generate segments based on utterance count
        if len(utterances) < self.num_segments:
            # Simple case: create one segment per utterance
            segments = self._create_simple_segments(utterances, duration_sec)
        else:
            # Complex case: generate multiple segments by combining utterances
            segments = self._create_combined_segments(utterances, duration_sec)

        # Return result or None if no valid segments
        if not segments:
            logger.debug(
                f"No valid segments created for uttid: {item.get('uttid', 'unknown_uttid')}"
            )
            return None

        item[self.out_key] = segments
        return item

    def _prepare_utterances(
        self, item: Dict[str, Any], duration_sec: float
    ) -> List[Dict[str, Any]]:
        """Extract utterances and convert time units."""
        utterances_orig = self._get_utterances(item)

        if not utterances_orig:
            logger.debug(
                f"Could not extract utterances for uttid: {item.get('uttid', 'unknown_uttid')}, "
                f"crs_filename: {item.get('crs_filename', 'None')}"
            )
            return []

        # Create deep copies and convert milliseconds to seconds
        utterances = [dict(u) for u in utterances_orig]
        MS_TO_SECONDS = 1000.0

        for u in utterances:
            u["start_time"] = u.get("start_time", 0) / MS_TO_SECONDS
            u["end_time"] = u.get("end_time", 0) / MS_TO_SECONDS

        # Handle full song mode: extend first and last utterances
        if self.full_song and utterances:
            utterances[0]["start_time"] = 0.0
            utterances[-1]["end_time"] = float(duration_sec)

        # Sort by start time for proper chronological ordering
        utterances.sort(key=lambda x: x["start_time"])
        return utterances

    def _create_simple_segments(
        self, utterances: List[Dict], duration_sec: float
    ) -> List[Dict[str, Any]]:
        """Create segments from individual utterances when count is low."""
        segments = []

        for u in utterances:
            duration = u["end_time"] - u["start_time"]
            if (
                self.min_duration <= duration <= self.max_duration
                and u["end_time"] <= duration_sec
            ):
                segments.append(
                    {
                        "start": round(u["start_time"], 3),
                        "end": round(u["end_time"], 3),
                        "text": u["text"].strip(),
                    }
                )

        return segments

    def _create_combined_segments(
        self, utterances: List[Dict], duration_sec: float
    ) -> List[Dict[str, Any]]:
        """Generate segments by combining multiple utterances."""
        segments = []
        n = len(utterances)

        for _ in range(self.num_segments):
            # Generate random target duration
            target_duration = random.uniform(
                self.min_duration, min(self.max_duration, duration_sec)
            )
            target_duration = round(target_duration, 3)

            # Find valid starting utterances
            valid_starts = []
            for i, u in enumerate(utterances):
                potential_end_time = min(
                    u["start_time"] + self.max_duration, duration_sec
                )
                if potential_end_time - u["start_time"] >= self.min_duration:
                    valid_starts.append((i, u))

            if not valid_starts:
                continue

            # Select starting utterance and build segment
            selected_index, selected_u = random.choice(valid_starts)
            start_time = selected_u["start_time"]
            text_segments = [selected_u["text"]]
            end_time = selected_u["end_time"]

            # Extend segment by adding consecutive utterances
            for i in range(selected_index + 1, n):
                u = utterances[i]
                potential_end = u["end_time"]
                potential_duration = potential_end - start_time

                # Stop if adding this utterance would exceed maximum duration
                if potential_duration > self.max_duration:
                    break

                # Stop if we've reached target and current segment is valid
                current_duration = end_time - start_time
                if (
                    current_duration >= target_duration
                    and current_duration >= self.min_duration
                ):
                    break

                # Add this utterance to the segment
                text_segments.append(u["text"])
                end_time = potential_end

            # Validate and add segment
            actual_duration = end_time - start_time
            if (
                self.min_duration <= actual_duration <= self.max_duration
                and end_time <= duration_sec
            ):
                segments.append(
                    {
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "text": " ".join(text_segments).strip(),
                    }
                )

        return segments

    def __call__(self, item: Dict[str, Any], **kargs) -> Optional[Dict[str, Any]]:
        if not item or "meta" not in item:
            return None

        duration_sec = item.get("duration", None)
        if not duration_sec:
            logger.debug(
                f"There is no duration in {item.get('uttid', 'unknown_uttid')}"
            )
            return None

        data_type = item.get(self.data_type_key, None)
        if not data_type:
            logger.error(
                f"There is no data_type in {item.get('uttid', 'unknown_uttid')}"
            )
            return None

        if data_type == "speech":
            return self._handle_speech_data(item, duration_sec)

        if data_type == "instrument":
            return self._handle_instrument_data(item, duration_sec)

        if data_type == "vocal":
            return self._handle_vocal_data(item, duration_sec)

        logger.warning(
            f"Unknown data_type '{data_type}' for uttid: {item.get('uttid', 'unknown_uttid')}"
        )
        return None


class UtteranceSegmentSamplerRandom:
    def __init__(
        self,
        out_key: str = "audio_slice",
        num_segments: int = 3,
        min_duration: float = 20.0,
        max_duration: float = 180.0,
    ):

        self.out_key = out_key
        self.num_segments = num_segments
        self.min_duration = min_duration  # in seconds
        self.max_duration = max_duration  # in seconds

    def __call__(self, item: Dict[str, Any], **kargs) -> Optional[Dict[str, Any]]:

        # Get duration in seconds, return None if invalid
        duration_sec = item.get("duration", None)
        if not duration_sec:
            logger.debug(f"There is no duration in {item['uttid']}")
            return None
        results: List[Dict[str, Any]] = []
        for _ in range(self.num_segments):
            dur = random.uniform(
                self.min_duration, min(self.max_duration, duration_sec)
            )
            start_time = random.uniform(0, max(0, duration_sec - dur))
            end_time = min(start_time + dur, duration_sec)
            results.append({"start": start_time, "end": end_time, "text": ""})
        item[self.out_key] = results

        return item
