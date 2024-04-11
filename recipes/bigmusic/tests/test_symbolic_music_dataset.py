import os

from samantha.dataio.parquet import ParquetDataset
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import (
    BMDfsDictBuilder,
)


def test_utterance_phoneme_match_with_sami_tokenizer():
    from recipes.bigmusic.inference.utils import get_sami_tokenizer
    from recipes.datasets.mcc.sami_tokenizer import (
        convert_labels_to_text_id
    )

    dataset = ParquetDataset(data_id=1800)
    data_iter = dataset.__iter__()
    sample = next(data_iter)

    def preproc_each_sample(sample):
        dfs_dict = BMDfsDictBuilder(sample)\
            .pre_load_meta()\
            .add_df_lyrics()\
            .create_output()
        return dfs_dict

    dfs_dict = preproc_each_sample(sample)

    df_lyrics = dfs_dict["df_lyrics"]

    phoneme_0 = convert_labels_to_text_id(df_lyrics.iloc[0].phoneme.split("\n"))[0][0]
    phoneme_1 = convert_labels_to_text_id(df_lyrics.iloc[1].phoneme.split("\n"))[0][0]

    tokenizer = get_sami_tokenizer()
    tokenizer(df_lyrics.iloc[0].text + "\n" + df_lyrics.iloc[1].text)

    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
