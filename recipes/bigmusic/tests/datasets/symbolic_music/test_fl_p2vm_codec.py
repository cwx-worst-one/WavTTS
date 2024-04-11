import os

from samantha.dataio.parquet import ParquetDataset
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import (
    BMDfsDictBuilder,
)
from recipes.bigmusic.datasets.symbolic_music.fl_p2vm_codec import (
    FLP2VMCodec,
)


def test_fl_p2vm_codec_encode():
    """Verify tokenization works
    """
    dataset = ParquetDataset(data_id=1839)
    data_iter = dataset.__iter__()
    sample = next(data_iter)
    config = FLP2VMCodec.Config(
        lyrics_seq_len=4000,
        leadsheet_seq_len=8000,
    )
    codec = FLP2VMCodec(config=config)

    def preproc_each_sample(sample):
        dfs_dict = BMDfsDictBuilder(sample)\
            .pre_load_meta()\
            .add_df_lyrics()\
            .add_df_vocal2midi()\
            .add_df_beat()\
            .add_df_chord()\
            .add_key()\
            .quantize_chord_to_beat()\
            .create_output()
        return codec.encode(dfs_dict)

    res = preproc_each_sample(sample)
    assert len(res) == 12000
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


def test_fl_p2vm_codec_get_datamodule():
    config = FLP2VMCodec.Config(
        lyrics_seq_len=400,
        leadsheet_seq_len=800,
    )

    dm = FLP2VMCodec.get_datamodule(
        config=config,
        train_id=1838,
        val_id=1839,
        batch_size=20,
        num_workers=10,
        shuffle_buffer=0,
    )

    num_samples = 0
    for i, batch in enumerate(dm.val_dataloader()):
        num_samples += batch["target_ids"].size()[0]
        print(i, num_samples)
    print("total number of val samples:", num_samples)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
