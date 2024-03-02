from cgitb import enable
import pytest

from recipes.bigmusic.utils.format_utils import normalize_text, normalize_text_sami_tokenizer

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
