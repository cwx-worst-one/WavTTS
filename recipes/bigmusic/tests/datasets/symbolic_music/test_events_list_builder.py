import os
from samantha.dataio.parquet import ParquetDataset
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import (
    BMDfsDictBuilder
)
from recipes.bigmusic.datasets.symbolic_music.events_list_builder import (
    EventsListBuilder
)
import pickle

current_dir = os.path.dirname(__file__)


def get_sample_dfs_dict():
    dataset = ParquetDataset(data_id=1800)
    data_iter = dataset.__iter__()
    sample = next(data_iter)
    note_subsets = ["vocal", "piano", "guitar", "bass", "drums"]
    return (
        BMDfsDictBuilder(sample)
        .pre_load_meta()
        .add_key()
        .add_df_note(subsets=note_subsets)
        .add_df_beat()
        .add_df_section()
        .add_df_chord()
        .quantize_chord_to_beat()
        .quantize_section_to_downbeat()
        .create_output()
    )


def test_events_list_builder():
    # dfs_dict = get_sample_dfs_dict()

    ## If you have a quick local cache:
    dfs_dict_path = os.path.join(current_dir, "../../data/20240326.dfs_dict.pkl")
    with open(dfs_dict_path, "rb") as f:
        dfs_dict = pickle.load(f)

    event_str_seq = (
        EventsListBuilder(dfs_dict)
        .add_chord_events()
        .add_note_events_from_trans5stem(stem_list=["vocal", "drum"])
        .add_bar_events()
        .add_section_events_each_bar()
        .add_bpm_level_events()
        .create_event_str_seq()
    )

    # from recipes.bigmusic.datasets.symbolic_music.base import IndexerConfig
    # from recipes.bigmusic.datasets.symbolic_music.trans5stems_codec import Trans5StemsCodec
    # config = IndexerConfig(
    #     inclue_phoneme=False,
    #     include_utterance_phoneme_tokens=False,
    #     include_bar_event = True,
    #     include_stem_indicator = True,
    #     include_drum_events = True,
    #     include_section_indicator_each_bar = True,
    #     include_bpm_levels = True,
    #     include_prompt = True,
    # )
    # codec = Trans5StemsCodec(config)
    # events_list_2 = codec.encode(dfs_dict)
    # for a, b in zip(events_list[:10], events_list_2[:10]):
    #     print(a, b)
    #     print()

    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
