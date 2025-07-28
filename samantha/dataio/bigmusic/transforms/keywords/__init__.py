import json
from pathlib import Path


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


KEYWORD_MAPPING_PATH = Path(__file__).parent / "mapping.json"
KEYWORD_MAPPING = json.loads(KEYWORD_MAPPING_PATH.read_text())
KEYWORD_MAPPING_LOWERK = {
    k.lower(): v for k, v in KEYWORD_MAPPING.items()
}  # for case-insensitive mapping
REVERSE_KEYWORD_MAPPING = _build_reversed_map(KEYWORD_MAPPING)


def expand_keyword(keyword: str, keep_input: bool = False) -> list[str]:
    if keyword.lower() not in KEYWORD_MAPPING_LOWERK:
        return []
    keywords = KEYWORD_MAPPING_LOWERK[keyword.lower()]
    if keep_input:
        keywords = [keyword] + keywords
    return keywords


def translate_zh_to_en(keyword: str, keep_input: bool = False) -> list[str]:
    if not is_chinese_in_str(keyword):
        if keep_input:
            return [keyword]
        return []
    return expand_keyword(keyword, keep_input)
