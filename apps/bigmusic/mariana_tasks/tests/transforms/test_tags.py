import pytest

from samantha.dataio.bigmusic.tokenizers.style_tag_tokenizer import StyleTagVocab
from samantha.dataio.bigmusic.tokenizers.style_tag_tokenizer.vocab import _VOCAB_DIR
from samantha.dataio.bigmusic.transforms.tags import _standardize_tag


def get_style_tag_vocabs() -> list[StyleTagVocab]:
    return [
        fp
        for fp in _VOCAB_DIR.glob("vocab.*.json")
        if fp.stem not in ["vocab.v0", "vocab.v1"]
    ]


@pytest.fixture(params=get_style_tag_vocabs())
def style_tag_vocab(request):
    return StyleTagVocab.from_json(request.param)


def test_standardization(style_tag_vocab: StyleTagVocab):
    for cat, tags in style_tag_vocab.token_to_id.items():
        if cat == "instrument":
            continue
        for tag in tags:
            assert _standardize_tag(tag) == tag
