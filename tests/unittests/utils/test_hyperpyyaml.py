from hyperpyyaml import load_hyperpyyaml


def test_float_conversion():
    yaml_string = """
    a: 0.00001
    b: 1e-5
    c: 1.e-5
    d: !ref <c> * 1
    """

    loaded = load_hyperpyyaml(yaml_string)
    assert loaded["a"] == 1e-5
    assert loaded["b"] == 1e-5
    assert loaded["c"] == 1e-5
    assert loaded["d"] == 1e-5
