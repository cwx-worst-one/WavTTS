import pytest
from hyperpyyaml import load_hyperpyyaml


@pytest.fixture
def hparams():
    hconf = "tests/unittests/apps/bigtts/umm/ar/test_data_utils.yaml"
    return load_hyperpyyaml(open(hconf))


def test_gen_duration_buckets(hparams):
    expect_buckets = [
        50,
        75,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        275,
        325,
        375,
        425,
        475,
        525,
        600,
        675,
        750,
        825,
        925,
        1025,
        1150,
        1275,
        1425,
        1500,
    ]
    assert hparams["buckets"] == expect_buckets


def test_init_tokenizer(hparams):
    assert hparams["tokenizer"] == "sami"


def test_normalize_text(hparams):
    text = "samantha&mariana/are good friends."
    expect = "samantha and mariana are good friends"

    normalize_text = hparams["normalize_text"]
    assert normalize_text(text) == expect
