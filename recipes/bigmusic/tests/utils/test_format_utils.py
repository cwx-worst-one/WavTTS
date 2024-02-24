from cgitb import enable
import pytest

from recipes.bigmusic.utils.format_utils import normalize_text, normalize_text_sami_tokenizer, reformat_zh_text_input

def test_normalize_text():
    assert normalize_text("HELLO! WORLD?", enable_punctuation=True, lowercase=False) == "HELLO <n> WORLD <n>"
    assert normalize_text("HELLO! WORLD?", enable_punctuation=False, lowercase=True) == "hello world"
    assert normalize_text("R&B hip-hop/rap") == "R and B hip hop rap", "Should split - and /"
    assert normalize_text("Don't stop! Believing", enable_punctuation=True, lowercase=True) == "don't stop <n> believing", "Should keep contractions"
    assert normalize_text("Don't stop! Believing", enable_punctuation=False, lowercase=True) == "don't stop believing", "Should keep contractions"
    assert normalize_text("This. Is. A. Test.", enable_punctuation=True, lowercase=True) == "this <n> is <n> a <n> test <n>", "Should convert periods to newlines"
    assert normalize_text("Hello.... Yo??", enable_punctuation=True, lowercase=False) == "Hello <n> Yo <n>", "Should remove double new lines"


def test_normalize_text_sami_tokenizer():
    # This function is supposed to normalize the lyrics content without touching special tags
    lines = [
        "[intro]",
        "[verse] 一句话包含 English 单词",
        "",
        "[chorus] 合:另一句歌词"
    ]
    lines_no_empty_line = [l for l in lines if l]
    assert normalize_text_sami_tokenizer("\n".join(lines), enable_punctuation=True, lowercase=False) == " <n> ".join(lines_no_empty_line)


REFORMAT_ZH_TEXT_CASES = [
    {
        "in": [
            "[verse]",
            "男:今天天气真好",
            "女:今天心情真好",
            "[chorus]",
            "合:你好我也好",
            "大家一起好",
        ],
        "out": [
            "[verse] 男:今天天气真好",
            "[verse] 女:今天心情真好",
            "[chorus] 合:你好我也好",
            "[chorus] 大家一起好",
        ],
        "desc": "regular case"
    },
    {
        "in": [
            "[intro]",
            "[verse]",
            "男:今天天气真好",
            "女:今天心情真好",
            "[chorus]",
            "合:你好我也好",
            "大家一起好",
        ],
        "out": [
            "[intro]",
            "[verse] 男:今天天气真好",
            "[verse] 女:今天心情真好",
            "[chorus] 合:你好我也好",
            "[chorus] 大家一起好",
        ],
        "desc": "leading inst tag"
    },
    {
        "in": [
            "[intro]",
            "[outro]",
        ],
        "out": [
            "[intro]",
            "[outro]",
        ],
        "desc": "inst tags only"
    }
]

REFORMAT_ZH_TEXT_INVALID_CASES = [
    {
        "in": [
            "男:今天天气真好",
            "女:今天心情真好",
            "[chorus]",
            "合:你好我也好",
            "大家一起好",
        ],
        "desc": "no leading section tag"
    },
    {
        "in": [
            "男:今天天气真好",
            "女:今天心情真好",
            "合:你好我也好",
            "大家一起好",
        ],
        "desc": "no section tag"
    },
    {
        "in": [
        ],
        "desc": "empty input"
    }
]

@pytest.mark.parametrize(
    "test_data",
    REFORMAT_ZH_TEXT_CASES,
    ids=[d["desc"] for d in REFORMAT_ZH_TEXT_CASES],
)
def test_reformat_zh_text_input(test_data):
    assert reformat_zh_text_input("\n".join(test_data["in"])) == "\n".join(test_data["out"])


@pytest.mark.parametrize(
    "test_data",
    REFORMAT_ZH_TEXT_INVALID_CASES,
    ids=[d["desc"] for d in REFORMAT_ZH_TEXT_INVALID_CASES],
)
def test_reformat_zh_text_input_invalid(test_data):
    with pytest.raises(ValueError):
        reformat_zh_text_input("\n".join(test_data["in"]))
