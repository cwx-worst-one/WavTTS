from recipes.bigmusic.datasets.symbolic_music.base import IndexerConfig
from recipes.bigmusic.datasets.symbolic_music.trans5stems_codec import Trans5StemsCodec
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.dataset import MultiIterableDataset


def test_trans5stems_codec():
    dataset = MultiIterableDataset(datasets=[
        ParquetDataset(data_id=1837),
        ParquetDataset(data_id=1881),
        ParquetDataset(data_id=1885),
    ])
    data_iter = dataset.__iter__()
    sample = next(data_iter)
    config = IndexerConfig(
        include_drum_events=True,
        include_prompt=True,
        include_section_indicator_each_bar=True,
        include_bpm_levels=True,
    )
    codec = Trans5StemsCodec(config)

    def preproc_each_sample(sample):
        dfs_dict = BMDfsDictBuilder(sample)\
            .pre_load_meta()\
            .add_df_note(subsets=["vocal", "piano", "guitar", "bass", "drums"])\
            .add_df_beat()\
            .add_df_section()\
            .add_df_chord()\
            .quantize_chord_to_beat()\
            .quantize_section_to_downbeat()\
            .create_output()
        return codec.encode(dfs_dict)

    res = preproc_each_sample(sample)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip