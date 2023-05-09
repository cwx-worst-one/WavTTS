from samantha.utils.datastructures import select_keys


def test_select_keys():
    d = {"key1": 1, "key2": 2}
    out_dict = select_keys(d, keys=["key1"])
    assert out_dict == {"key1": 1}
