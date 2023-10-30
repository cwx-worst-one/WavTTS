from recipes.bigmusic.utils.format_utils import normalize_text

def test_normalize_text():
    assert normalize_text("HELLO! WORLD?", enable_punctuation=True, lowercase=False) == "HELLO <n> WORLD <n>"
    assert normalize_text("HELLO! WORLD?", enable_punctuation=False, lowercase=True) == "hello world"
    assert normalize_text("R&B hip-hop/rap") == "R and B hip hop rap", "Should split - and /"
    assert normalize_text("Don't stop! Believing", enable_punctuation=True, lowercase=True) == "don't stop <n> believing", "Should keep contractions"
    assert normalize_text("Don't stop! Believing", enable_punctuation=False, lowercase=True) == "don't stop believing", "Should keep contractions"
    assert normalize_text("This. Is. A. Test.", enable_punctuation=True, lowercase=True) == "this <n> is <n> a <n> test <n>", "Should convert periods to newlines"
    assert normalize_text("Hello.... Yo??", enable_punctuation=True, lowercase=False) == "Hello <n> Yo <n>", "Should remove double new lines"