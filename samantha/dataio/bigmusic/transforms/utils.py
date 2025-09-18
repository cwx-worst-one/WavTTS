import ast
import copy
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from samantha.dataio.bigmusic.base_exception import MusicMetaError

KEYWORD_MARKED_DELETE = "DELETE"
KEYWORD_MARKED_DUPLICATE = "DUPLICATE"
FILTER_KEYWORDS = [KEYWORD_MARKED_DELETE, KEYWORD_MARKED_DUPLICATE]

NUM_LONG_WORDS = 6


def get_mapping_table(mapping_filepath: str, lower_key: bool = False):
    if mapping_filepath is not None and os.path.exists(mapping_filepath):
        mapping_table = json.load(open(mapping_filepath))
    else:
        mapping_table = {}
    if lower_key:
        keys = list(mapping_table.keys())
        for k in keys:
            if k.lower() == k:
                continue
            mapping_table[k.lower()] = mapping_table[k]
    return mapping_table


def err_loc() -> str:
    _, _, exc_tb = sys.exc_info()
    while (
        exc_tb.tb_next
    ):  # Traverse to the deepest traceback (where the error was raised)
        exc_tb = exc_tb.tb_next
    return f"{exc_tb.tb_frame.f_code.co_filename}:{exc_tb.tb_lineno}"


def contains_chinese(text):
    """判断字符串是否包含中文字符"""
    return re.search("[\u4e00-\u9fa5]", text)


def word_count(keyword: str):
    # chinese
    if chr(0x4E00) <= keyword[0] <= chr(0x9FFF):
        return len(keyword), "chinese"
    else:
        return len(keyword.split()), "other"


def keywords_marking(keywords: List[str], cond_func, mark=KEYWORD_MARKED_DELETE):
    keywords = [mark if cond_func(k) else k for k in keywords]
    return keywords


def keywords_dedup(keywords: List[str], mark=KEYWORD_MARKED_DUPLICATE):
    keywords_set = set()
    new_keywords = []
    for k in keywords:
        if k in keywords_set:
            new_keywords.append(mark)
        else:
            new_keywords.append(k)
            keywords_set.add(k)
    return new_keywords


def keywords_filter(keywords: List[str], match=[], contain=[]):
    keywords = [k for k in keywords if k not in match]
    new_keywords = []
    for keyword in keywords:
        if any(bad_keyword in keyword for bad_keyword in contain):
            continue
        new_keywords.append(keyword)
    keywords = new_keywords
    return keywords


def keywords_mapping(
    keywords: List[str],
    mapping_table: Dict[str, str],
    default_category_name: str = "additional",
    category_num=1,
):
    if not isinstance(keywords, list):
        keywords = [keywords]
    new_keywords = []

    for keyword in keywords:
        if keyword in mapping_table:
            mapped_keyword = copy.deepcopy(
                mapping_table[keyword]
            )  # prevent inplace change of mapping_table
            if isinstance(mapped_keyword, str):
                mapped_keyword = [mapped_keyword]
            if not isinstance(mapped_keyword, dict):  # close vocab mapping
                mapped_keyword = {"categories": {default_category_name: mapped_keyword}}
            else:
                if "drop" in mapped_keyword and mapped_keyword["drop"]:
                    mapped_keyword = {"categories": {"drop": [keyword]}}
        else:
            if contains_chinese(keyword) and len(keyword) > NUM_LONG_WORDS:
                mapped_keyword = {"categories": {"drop": [keyword]}}
            elif (
                len(keyword.split(" ")) > NUM_LONG_WORDS
                and ">" not in keyword
                and " - " not in keyword
                and " & " not in keyword
            ) or len(keyword.split(",")) >= 10:
                mapped_keyword = {"categories": {"description": [keyword]}}
            else:
                mapped_keyword = {"categories": {default_category_name: [keyword]}}

        for category in mapped_keyword["categories"].keys():
            # TODO DROP
            mapped_keyword["categories"][category] = random.choices(
                mapped_keyword["categories"][category], k=category_num
            )
        new_keywords.append(mapped_keyword["categories"])
    return new_keywords


def keywords_norm(keywords: List[str], norm_funcs):
    for norm_func in norm_funcs:
        keywords = [norm_func(k) for k in keywords]
    return keywords


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
            return None

    return current_level


def flatten_nested_list(nested_list):
    if not isinstance(nested_list, list):
        return nested_list
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list


def remove_none_in_list(obj):
    for tag, value in obj.items():
        current_value = value
        # Ensure the value is treated as a list
        if not isinstance(current_value, list):
            current_value = [current_value]

        # Use a list comprehension to safely and efficiently remove all empty strings.
        obj[tag] = [item for item in current_value if item != ""]
    return obj


def remove_empty(object):
    if isinstance(object, dict):
        new_dict = {k: remove_empty(v) for k, v in object.items()}
        return {
            k: v for k, v in new_dict.items() if not isinstance(v, (list, dict)) or v
        }
    elif isinstance(object, list):
        return [remove_empty(elem) for elem in object]
    return object


def dict_rename_key(obj, mapping, keep_unmapped=False):
    renamed_obj = {}
    for k, v in obj.items():
        if k in mapping:
            renamed_obj[mapping[k]] = v
        elif keep_unmapped:
            renamed_obj[k] = v
    return renamed_obj


def dedup_with_order(string_list, ignore_case=False):
    """
    Deduplicates a list of strings, ignoring case, but preserving the case of the first occurrence.
    """
    seen_lower = set()
    result = []
    for item in string_list:
        if isinstance(item, str) and ignore_case:
            lower_item = item.lower()
        else:
            lower_item = item
        if lower_item not in seen_lower:
            seen_lower.add(lower_item)
            result.append(item)
    return result


def is_number(s):
    pattern = r"^-?\d+(\.\d+)?$"  # 匹配整数和小数
    return bool(re.match(pattern, s))


def has_chinese(text):
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def is_chinese(char):
    """Check if a character is a Chinese character."""
    return "\u4e00" <= char <= "\u9fff"


def validate_item_with_keys(in_keys, item):
    """
    Check if all in_key is in the item.
    Return:
        True if all in_key is in the item, False otherwise
    """
    if isinstance(in_keys, str):
        return in_keys in item
    return all(in_key in item for in_key in in_keys)


def validate_item_with_optional_keys(in_key, optional_keys, item):
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


# 每行首字母大写
def capitalize_first_letter(text, aug_rate=1.0):
    if not text:
        return text
    text = "\n".join([line.capitalize() for line in text.split("\n")])


def aggregate_feature(meta: dict, config: dict):
    inputs = config["inputs"]
    output = config["output"]
    meta[output] = []
    if config["mode"] == "merge":
        for field in inputs:
            if field not in meta:
                continue
            meta[output].extend(meta[field])
    elif config["mode"] == "priority_sample":
        for field in inputs:
            value = meta.get(field, None)
            if not value or len(value) == 0:
                continue
            meta[output] = value
            break
    elif config["mode"] == "random_sample":
        sample_pool = [value for field, value in meta.items() if field in inputs]
        meta[output] = random.choice(sample_pool)
    elif config["mode"] == "weighted_sample":
        weights = config.get("weights", None)
        assert weights, "weights must be provided for weighted_sample"
        assert len(inputs) == len(
            weights
        ), "inputs and weights must have the same length"
        sample_pool = []
        sample_weights = []
        for field, weight in zip(inputs, weights):
            value = meta.get(field, None)
            if not value or len(value) == 0:
                continue
            sample_pool.append(value)
            sample_weights.append(weight)
        if len(sample_pool) > 0 and len(sample_weights) > 0:
            meta[output] = random.choices(sample_pool, weights=sample_weights)[0]
    elif config["mode"] == "drop":
        pass
    elif config["mode"] == "concat":
        sep = config.get("sep", " ")
        return_empty_if_missing_field = config.get(
            "return_empty_if_missing_field", True
        )
        feat_seq = []
        for field in inputs:
            if return_empty_if_missing_field and (
                field not in meta or meta[field] is None or len(meta[field]) == 0
            ):
                meta[output] = ""
                return
            else:
                feat_seq.append(meta[field])
        feat_seq = flatten_nested_list(feat_seq)
        if config.get("shuffle", False):
            random.shuffle(feat_seq)
        meta[output] = sep.join(feat_seq)
    elif config["mode"] == "random_concat":
        sep = config.get("sep", " ")
        feat_seq = flatten_nested_list([meta.get(field, []) for field in inputs])
        target_len = config.get("target_len", 1)
        if target_len == "full":
            target_len = len(feat_seq)
        target_len = min(target_len, len(feat_seq))
        if config.get("shuffle", False):
            random.shuffle(feat_seq)
        feat_seq = random.sample(feat_seq, target_len)
        meta[output] = sep.join(feat_seq)
    else:
        raise ValueError(f"Unknown mode {config['mode']}")
