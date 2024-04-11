import os
from recipes.bigmusic.datasets.symbolic_music.base import IndexerConfig
from recipes.bigmusic.datasets.symbolic_music.fixed_length_phoneme_and_vocal2midi_no_silence_codec import (
    FixedLengthPhonemeAndVocal2MidiNoSilenceCodec,
    FixedLengthPhonemeAndVocal2MidiCodec,
)
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import BMDfsDictBuilder
from samantha.dataio.parquet import ParquetDataset
from samantha.dataio.dataset import MultiIterableDataset


def test_fixed_length_phoneme_and_vocal2midi_no_silence_codec():
    dataset = MultiIterableDataset(datasets=[
        ParquetDataset(data_id=1838),
        ParquetDataset(data_id=1839),
    ])
    data_iter = dataset.__iter__()
    sample = next(data_iter)
    config = FixedLengthPhonemeAndVocal2MidiCodec.Config(
        lyrics_seq_len=500,
        leadsheet_seq_len=2000,
    )
    codec = FixedLengthPhonemeAndVocal2MidiNoSilenceCodec(config)

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
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


def test_fixed_length_phoneme_and_vocal2midi_no_silence_codec_datamodule():
    config = FixedLengthPhonemeAndVocal2MidiCodec.Config(
        lyrics_seq_len=400,
        leadsheet_seq_len=8000,
    )

    dm = FixedLengthPhonemeAndVocal2MidiNoSilenceCodec.get_datamodule(
        config=config,
        train_id=1838,
        val_id=1839,
        batch_size=20,
        num_workers=10,
        shuffle_buffer=100,
    )

    num_samples = 0
    for i, batch in enumerate(dm.val_dataloader()):
        num_samples += len(batch["__key__"])
        print(i, num_samples)
    ## val #samples: 5142
    print("total number of val samples:", num_samples)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
