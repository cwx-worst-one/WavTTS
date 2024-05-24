from recipes.bigmusic.datasets.mir_data_util import get_categorical_vocab


def test_vocab_compatibility():
    sa_vocab, n_cat_sa = get_categorical_vocab("SA")
    sa_uni_vocab, n_cat_uni = get_categorical_vocab("SA_unified")
    assert n_cat_sa == 5
    assert n_cat_uni == 5
    for tag_name, id in sa_vocab.items():
        assert tag_name in sa_uni_vocab
        assert id == sa_uni_vocab[tag_name]

