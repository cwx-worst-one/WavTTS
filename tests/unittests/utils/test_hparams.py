from samantha.utils import hparams as hp


def test_hparams_dotdict():
    dotdict = hp.DotDict({"key1": "val1", "key2": {"key3": "val3"}}, key4="val4")

    assert dotdict.key1 == "val1"
    assert dotdict["key1"] == "val1"

    assert dotdict.key2.key3 == "val3"
    assert dotdict["key2"]["key3"] == "val3"

    dotdict.key1 = "new_val"
    dotdict.key2.key3 = "new_val"
    assert dotdict.key1 == "new_val"
    assert dotdict.key2.key3 == "new_val"

    assert dotdict["key4"] == "val4"

    empty = hp.DotDict()
    empty.key1 = "val"
    assert empty.key1 == "val"
