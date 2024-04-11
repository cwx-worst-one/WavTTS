from recipes.bigmusic.datasets.symbolic_music.decorators import (
    split_section,
    until,
)
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.dataset import MultiIterableDataset


def test_split_section():
    dataset = MultiIterableDataset(datasets=[
        ParquetDataset(data_id=1837),
        ParquetDataset(data_id=1881),
        ParquetDataset(data_id=1885),
    ], sample_chunk_size=3)
    data_iter = dataset.__iter__()

    @split_section(["verse", "chorus"])
    @until(1)
    def dfs_dict_iter():
        for sample in data_iter:
            yield BMDfsDictBuilder(sample)\
                .pre_load_meta()\
                .add_key()\
                .add_df_note(subsets=["vocal", "piano", "guitar", "bass", "drums"])\
                .add_df_beat()\
                .add_df_section()\
                .add_df_chord()\
                .quantize_chord_to_beat()\
                .quantize_section_to_downbeat()\
                .create_output()
    
    for dfs_dict in dfs_dict_iter():
        print(dfs_dict["key"])
