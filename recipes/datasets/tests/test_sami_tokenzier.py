from itertools import cycle, product

import pytest
import mock
import numpy as np

from recipes.datasets.mcc.sami_tokenizer import (
    Phrase,
    SamiOfflineTokenizer,
    SamiTokenizerError,
    drop_out_line_breaks,
    drop_out_section_tags,
    move_out_section_tags,
    singer_tags,
    section_tags,
    phone_to_int,
    tone_to_int,
    convert_labels_to_text_id,
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

PHONE_MOCK_OUTPUT = (
    np.array(
        [
            [
                2,
                250,
                265,
                263,
                166,
                200,
                265,
                263,
                172,
                207,
                265,
                263,
                176,
                204,
                265,
                263,
                167,
                245,
                265,
                182,
                206,
                265,
                263,
                168,
                196,
                265,
                263,
                176,
                250,
                265,
                206,
                265,
                263,
                85,
            ],
            [
                2,
                5,
                19,
                17,
                4,
                4,
                19,
                17,
                3,
                3,
                19,
                17,
                5,
                5,
                19,
                17,
                3,
                3,
                19,
                3,
                3,
                19,
                17,
                7,
                7,
                19,
                17,
                6,
                6,
                19,
                4,
                19,
                17,
                2,
            ],
        ]
    ),
    [
        "sil",
        "C0uo",
        "syl_sep",
        "zh_word_sep",
        "C0c",
        "C0eng",
        "syl_sep",
        "zh_word_sep",
        "C0j",
        "C0iang",
        "syl_sep",
        "zh_word_sep",
        "C0n",
        "C0i",
        "syl_sep",
        "zh_word_sep",
        "C0ch",
        "C0uen",
        "syl_sep",
        "C0t",
        "C0ian",
        "syl_sep",
        "zh_word_sep",
        "C0d",
        "C0e",
        "syl_sep",
        "zh_word_sep",
        "C0n",
        "C0uo",
        "syl_sep",
        "C0ian",
        "syl_sep",
        "zh_word_sep",
        "。",
    ],
    [
        "0",
        "3",
        "syl_sep",
        "zh_word_sep",
        "2",
        "2",
        "syl_sep",
        "zh_word_sep",
        "1",
        "1",
        "syl_sep",
        "zh_word_sep",
        "3",
        "3",
        "syl_sep",
        "zh_word_sep",
        "1",
        "1",
        "syl_sep",
        "1",
        "1",
        "syl_sep",
        "zh_word_sep",
        "5",
        "5",
        "syl_sep",
        "zh_word_sep",
        "4",
        "4",
        "syl_sep",
        "2",
        "syl_sep",
        "zh_word_sep",
        "0",
    ],
)


def test_sami_tokenizer():
    tokens, phonemes, tones = convert_labels_to_text_id(PHONE_MOCK_INPUT)
    gt_tokens, gt_phonemes, gt_tones = PHONE_MOCK_OUTPUT
    assert np.array_equal(tokens, gt_tokens)
    assert phonemes == gt_phonemes
    assert tones == gt_tones


def _get_mock_phrase(section_tag, singer_tag):
    # prepend the singer tag to the beginning of the first line
    if section_tag and singer_tag:
        mock_input = [f"[{section_tag}] {singer_tag}:{PHONE_MOCK_INPUT[0]}"] + PHONE_MOCK_INPUT[1:]
    elif section_tag:
        mock_input = [f"[{section_tag}] {PHONE_MOCK_INPUT[0]}"] + PHONE_MOCK_INPUT[1:]
    elif singer_tag:
        mock_input = [f"{singer_tag}:{PHONE_MOCK_INPUT[0]}"] + PHONE_MOCK_INPUT[1:]
    else:
        mock_input = PHONE_MOCK_INPUT
    return Phrase.parse(phonemes="\n".join(mock_input))


@pytest.mark.parametrize("prefix_tags", product([None] + section_tags, [None] + singer_tags))
def test_sami_tokenizer_with_prefix_tags(prefix_tags):
    section_tag, singer_tag = prefix_tags
    phrase = _get_mock_phrase(section_tag, singer_tag)
    tokens, phonemes, tones = convert_labels_to_text_id(phrase.phonemes.split("\n"), phrase.prefix_tags)
    gt_tokens, gt_phonemes, gt_tones = PHONE_MOCK_OUTPUT

    section_phone_id = phone_to_int.get(section_tag)
    section_tone_id = tone_to_int.get(section_tag)
    singer_phone_id = phone_to_int.get(singer_tag)
    singer_tone_id = tone_to_int.get(singer_tag)

    prepend_tokens = list(filter(None, [section_tag, singer_tag]))
    prepend_phone_ids = list(filter(None, [section_phone_id, singer_phone_id]))
    prepend_tone_ids = list(filter(None, [section_tone_id, singer_tone_id]))

    assert np.array_equal(tokens, np.hstack([[prepend_phone_ids, prepend_tone_ids], gt_tokens]))
    assert phonemes == prepend_tokens + gt_phonemes
    assert tones == prepend_tokens + gt_tones


def test_tokenize_phrase():
    section_tag = "verse"
    singer_tag = "合"
    phrase = _get_mock_phrase(section_tag, singer_tag)

    section_phone_id = phone_to_int.get(section_tag)
    singer_phone_id = phone_to_int.get(singer_tag)

    prepend_phone_ids = [section_phone_id, singer_phone_id]

    gt_tokens, _, _ = PHONE_MOCK_OUTPUT

    tokenizer = SamiOfflineTokenizer()
    assert np.array_equal(tokenizer.tokenize_phrase(phrase), np.concatenate([prepend_phone_ids, gt_tokens[0]]))


def test_tokenize_phrase_invalid_inputs():
    tokenizer = SamiOfflineTokenizer()

    with pytest.raises(SamiTokenizerError):
        tokenizer.tokenize_phrase(Phrase.parse(""))  # empty phrase

    with pytest.raises(SamiTokenizerError):
        tokenizer.tokenize_phrase(Phrase.parse("[verse] abc"))  # invalid phoneme label


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
    with pytest.raises(SamiTokenizerError):
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
    with mock.patch("recipes.datasets.mcc.sami_tokenizer.Random", MockRandom):
        assert drop_out_line_breaks(LINE_BREAK_DROPOUT_INPUT, test_data["rate"]) == test_data["out"]


SECTION_TAG_DROPOUT_INPUT = [
    Phrase(section_tag="verse"),
    Phrase(text="hello"),
    Phrase(section_tag="chorus"),
    Phrase(text="world"),
]


SECTION_TAG_DROPOUT_CASES = [
    # it only has 2 cases
    {
        "rate": 0,
        "out": SECTION_TAG_DROPOUT_INPUT,
        "desc": "no dropout",
    },
    {
        "rate": 1,
        "out": [
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
    with mock.patch("recipes.datasets.mcc.sami_tokenizer.Random", MockRandom):
        assert drop_out_section_tags(SECTION_TAG_DROPOUT_INPUT, test_data["rate"]) == test_data["out"]


# TODO (Yilin) Test SamiTokenizer