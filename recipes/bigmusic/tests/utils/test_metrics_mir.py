from recipes.bigmusic.utils.metrics_mir import is_key_relative_key_error, is_key_fifth_error


def test_key_min_maj_error():
    assert is_key_relative_key_error("C:Maj", "A:Min")
    assert is_key_relative_key_error("A:Maj", "F#:Min")

    assert not is_key_relative_key_error("A:Maj", "B:Min")
    assert not is_key_relative_key_error("A:Maj", "A:Min")

    assert not is_key_relative_key_error("A:Maj", "N")
    assert not is_key_relative_key_error("N", "A:Min")


def test_key_fifth_error():
    assert is_key_fifth_error("C:Maj", "G:Maj")
    assert is_key_fifth_error("G:Min", "C:Min")
    assert is_key_fifth_error("G:Maj", "D:Maj")

    assert not is_key_fifth_error("C:Maj", "G:Min")
    assert not is_key_fifth_error("A:Maj", "G:Maj")
    assert not is_key_fifth_error("A:Maj", "A:Min")

    assert not is_key_fifth_error("N", "G:Maj")
    assert not is_key_fifth_error("C:Maj", "N")