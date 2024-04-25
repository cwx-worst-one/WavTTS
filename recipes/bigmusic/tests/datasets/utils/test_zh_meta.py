from itertools import cycle

import pytest
import mock

from recipes.datasets.mcc.sami_tokenizer import Phrase
from recipes.bigmusic.datasets.utils.zh_meta import (
    SongSlice,
    drop_out_line_breaks,
    drop_out_section_tags,
    move_out_section_tags,
)


PHONE_MOCK_INPUT = [
    "sil\t0\tS\t0\tO\t\tS",
    "C0uo\t3\tS\t0\tO\t我\tS",
    "C0c\t2\tS\t1\tO\t曾\tB",
    "C0eng\t2\tS\t1\tO\t曾\tE",
    "C0j\t1\tS\t0\tO\t将\tB",
    "C0iang\t1\tS\t0\tO\t将\tE",
    "C0n\t3\tS\t1\tO\t你\tB",
    "C0i\t3\tS\t1\tO\t你\tE",
    "C0ch\t1\tB\t0\tO\t春天\tB",
    "C0uen\t1\tB\t0\tO\t春天\tE",
    "C0t\t1\tE\t0\tO\t\tB",
    "C0ian\t1\tE\t0\tO\t\tE",
    "C0d\t5\tS\t1\tO\t的\tB",
    "C0e\t5\tS\t1\tO\t的\tE",
    "C0n\t4\tB\t0\tO\t诺言\tB",
    "C0uo\t4\tB\t0\tO\t诺言\tE",
    "C0ian\t2\tE\t4\tO\t\tS",
    "。\t0\tS\t4\tO\t\tS",
]


REFORMAT_CASES = [
    {
        "in": [
            "[intro]",
            "[verse] 你好",
            "[verse] 我好",
            "[inst]",
            "[chorus] 大家好",
        ],
        "out": [
            "[intro]",
            "[verse]",
            "你好",
            "我好",
            "[inst]",
            "[chorus]",
            "大家好",
        ],
        "desc": "regular case",
    },
    {
        "in": [
            "没有",
            "[verse] 你好",
            "[verse] 我好",
            "很好",
            "[chorus] 真好"
        ],
        "out": [
            "没有",
            "[verse]",
            "你好",
            "我好",
            "很好",
            "[chorus]",
            "真好"
        ],
        "desc": "first and middle phrases have no section tag",
    },
    {
        "in": [
            "没有",
            "无",
            "空",
        ],
        "out": [
            "没有",
            "无",
            "空",
        ],
        "desc": "no section tag at all",
    },
    {
        "in": [
            "[verse] 没有",
            "无",
            "空",
        ],
        "out": [
            "[verse]",
            "没有",
            "无",
            "空",
        ],
        "desc": "only first phrase has section tag",
    },
    {
        "in": [
            "[intro]",
            "[verse]",
            "你好",
            "我好",
            "[inst]",
            "[chorus]",
            "大家好",
        ],
        "out": [
            "[intro]",
            "[verse]",
            "你好",
            "我好",
            "[inst]",
            "[chorus]",
            "大家好",
        ],
        "desc": "no change",
    },
]

@pytest.mark.parametrize(
    "test_data",
    REFORMAT_CASES,
    ids=[d["desc"] for d in REFORMAT_CASES],
)
def test_move_out_section_tags(test_data):
    in_phrases = [Phrase.parse(text=line) for line in test_data["in"]]
    out_phrases = [Phrase.parse(text=line) for line in test_data["out"]]
    assert move_out_section_tags(in_phrases) == out_phrases


PHRASE_CONCAT_CASES = [
    {
        "phrase_a": Phrase(
            phonemes = "\n".join(PHONE_MOCK_INPUT),
            text="嘻嘻嘻",
            singer_tag="男",
            section_tag="verse",
            time_span=(0, 5),
        ),
        "phrase_b": Phrase(
            phonemes = "\n".join(PHONE_MOCK_INPUT),
            text="哈哈哈",
            singer_tag="男",
            section_tag="verse",
            time_span=(6, 10),
        ),
        "out": Phrase(
            phonemes = "\n".join(PHONE_MOCK_INPUT[:-1]+PHONE_MOCK_INPUT),
            text="嘻嘻嘻哈哈哈",
            singer_tag="男",
            section_tag="verse",
            time_span=(0, 10),
        ),
        "desc": "regular case"
    },
    {
        "phrase_a": Phrase(
            phonemes = "\n".join(PHONE_MOCK_INPUT),
            text="嘻嘻嘻",
            singer_tag="男",
            section_tag="verse",
            time_span=(0, 5),
        ),
        "phrase_b": Phrase(
            phonemes = "\n".join(PHONE_MOCK_INPUT),
            text="hahaha",
            singer_tag="男",
            section_tag="verse",
            time_span=(6, 10),
        ),
        "out": Phrase(
            phonemes = "\n".join(PHONE_MOCK_INPUT[:-1]+PHONE_MOCK_INPUT),
            text="嘻嘻嘻 hahaha",
            singer_tag="男",
            section_tag="verse",
            time_span=(0, 10),
        ),
        "desc": "mix lang case"
    }
]


PHRASE_CONCAT_INVALID_CASES = [
    {
        "phrase_a": Phrase(
            phonemes = "\n".join(PHONE_MOCK_INPUT),
            text="嘻嘻嘻",
            singer_tag="男",
        ),
        "phrase_b": Phrase(
            phonemes = "\n".join(PHONE_MOCK_INPUT),
            text="哈哈哈",
            section_tag="verse",
        ),
        "desc": "prefix_tags do not match"
    },
]


@pytest.mark.parametrize(
    "test_data",
    PHRASE_CONCAT_CASES,
    ids=[d["desc"] for d in PHRASE_CONCAT_CASES],
)
def test_phrase_concat(test_data):
    assert Phrase.concat(test_data["phrase_a"], test_data["phrase_b"]) == test_data["out"]


@pytest.mark.parametrize(
    "test_data",
    PHRASE_CONCAT_INVALID_CASES,
    ids=[d["desc"] for d in PHRASE_CONCAT_INVALID_CASES],
)
def test_phrase_concat_invalid(test_data):
    with pytest.raises(ValueError):
        Phrase.concat(test_data["phrase_a"], test_data["phrase_b"])


class MockRandom:
    def __init__(self, seed):
        self.iter = cycle([0.4, 0.6])

    def random(self):
        return next(self.iter)


LINE_BREAK_DROPOUT_INPUT = [
    Phrase(text="hello", section_tag="verse"),
    Phrase(text="world", section_tag="verse"),
    Phrase(text="foo", section_tag="chorus"),
    Phrase(text="bar", section_tag="chorus"),
]


LINE_BREAK_DROPOUT_CASES = [
    {
        "rate": 0,
        "out": LINE_BREAK_DROPOUT_INPUT,
        "desc": "no dropout",
    },
    {
        "rate": 1,
        "out": [
            Phrase(text="hello world", section_tag="verse"),
            Phrase(text="foo bar", section_tag="chorus"),
        ],
        "desc": "all dropout",
    },
    {
        "rate": 0.5,
        "out": [
            Phrase(text="hello", section_tag="verse"),
            Phrase(text="world", section_tag="verse"),
            Phrase(text="foo bar", section_tag="chorus"),
        ],
        "desc": "all dropout",
    }
]


@pytest.mark.parametrize(
    "test_data",
    LINE_BREAK_DROPOUT_CASES,
    ids=[d["desc"] for d in LINE_BREAK_DROPOUT_CASES],
)
def test_drop_out_line_breaks(test_data):
    with mock.patch("recipes.bigmusic.datasets.utils.zh_meta.Random", MockRandom):
        assert drop_out_line_breaks(LINE_BREAK_DROPOUT_INPUT, test_data["rate"]) == test_data["out"]


SECTION_TAG_DROPOUT_INPUT = [
    Phrase(section_tag="verse#1"),
    Phrase(text="hello"),
    Phrase(section_tag="verse#2"),
    Phrase(text="hello"),
    Phrase(section_tag="chorus#3"),
    Phrase(text="world"),
]


SECTION_TAG_DROPOUT_OUTPUT = [
    Phrase(section_tag="verse#1"),
    Phrase(text="hello"),
    Phrase(section_tag="verse#2"),
    Phrase(text="hello"),
    Phrase(section_tag="chorus#3"),
    Phrase(text="world"),
]


SECTION_TAG_DROPOUT_CASES = [
    # it only has 2 cases
    {
        "rate": 0,
        "out": SECTION_TAG_DROPOUT_OUTPUT,
        "desc": "no dropout",
    },
    {
        "rate": 1,
        "out": [
            Phrase(text="hello"),
            Phrase(text="hello"),
            Phrase(text="world"),
        ],
        "desc": "all dropout",
    },
]


@pytest.mark.parametrize(
    "test_data",
    SECTION_TAG_DROPOUT_CASES,
    ids=[d["desc"] for d in SECTION_TAG_DROPOUT_CASES],
)
def test_drop_out_section_tags(test_data):
    with mock.patch("recipes.bigmusic.datasets.utils.zh_meta.Random", MockRandom):
        assert drop_out_section_tags(SECTION_TAG_DROPOUT_INPUT, test_data["rate"]) == test_data["out"]


def test_reformat_and_dropout():
    assert SongSlice.reformat_and_dropout(0, 0, SECTION_TAG_DROPOUT_INPUT) == [
        Phrase(section_tag="verse"),
        Phrase(text="hello"),
        Phrase(section_tag="verse"),
        Phrase(text="hello"),
        Phrase(section_tag="chorus"),
        Phrase(text="world"),
    ]