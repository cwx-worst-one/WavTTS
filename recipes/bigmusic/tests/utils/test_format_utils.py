from recipes.bigmusic.utils.format_utils import normalize_text

def test_normalize_text():
    assert normalize_text("HELLO! WORLD?", enable_punctuation=True) == "hello <n> world"
    assert normalize_text("HELLO! WORLD?", enable_punctuation=False) == "hello world"
    assert normalize_text("R&B hip-hop/rap") == "r and b hip hop rap", "Should split - and /"
    assert normalize_text("Don't stop! Believing", enable_punctuation=True) == "don't stop <n> believing", "Should keep contractions"
    assert normalize_text("Don't stop! Believing", enable_punctuation=False) == "don't stop believing", "Should keep contractions"
    assert normalize_text("This. Is. A. Test.", enable_punctuation=True) == "this <n> is <n> a <n> test <n>", "Should convert periods to newlines"