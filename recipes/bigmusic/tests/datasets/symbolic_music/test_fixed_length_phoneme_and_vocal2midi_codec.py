import os

from samantha.dataio.parquet import ParquetDataset
from recipes.bigmusic.datasets.symbolic_music.bm_dfs_dict_builder import (
    BMDfsDictBuilder,
)
from recipes.bigmusic.datasets.symbolic_music.fixed_length_phoneme_and_vocal2midi_codec import (
    FixedLengthPhonemeAndVocal2MidiCodec,
)


def test_fixed_length_phoneme_and_vocal2midi_codec():
    """Verify tokenization works
    """
    dataset = ParquetDataset(data_id=1800)
    data_iter = dataset.__iter__()
    sample = next(data_iter)
    config = FixedLengthPhonemeAndVocal2MidiCodec.Config(
        lyrics_seq_len=4000,
        leadsheet_seq_len=8000,
    )
    codec = FixedLengthPhonemeAndVocal2MidiCodec(config=config)

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


def test_fixed_length_phoneme_and_vocal2midi_codec_datamodule():
    config = FixedLengthPhonemeAndVocal2MidiCodec.Config(
        lyrics_seq_len=400,
        leadsheet_seq_len=800,
    )

    dm = FixedLengthPhonemeAndVocal2MidiCodec.get_datamodule(
        config=config,
        train_id=1838,
        val_id=1839,
        batch_size=20,
        num_workers=10,
        shuffle_buffer=100,
    )

    num_samples = 0
    for batch in dm.val_dataloader():
        num_samples += batch.size()[0]
        print(num_samples, batch.size())
    print("total number of val samples:", num_samples)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


def test_fixed_length_phoneme_and_vocal2midi_codec_decode():
    from recipes.bigmusic.datasets.utils.symbolic_music import pretty_midi_obj_to_midi_bytes

    # arr = [472, 472, 456, 483, 467, 495, 472, 456, 483, 459, 495, 463, 483, 472, 456, 495, 464, 476, 468, 495, 472, 456, 495, 463, 476, 472, 456, 495, 472, 456, 495, 464, 476, 467, 483, 472, 456, 495, 472, 456, 495, 472, 456, 495, 546, 459, 431, 138, 149, 134, 141, 157, 159, 467, 426, 157, 135, 470, 425, 126, 149, 547, 463, 425, 359, 465, 427, 361, 469, 425, 363, 471, 424, 366, 472, 456, 495, 460, 476, 468, 483, 546, 456, 427, 143, 156, 547, 456, 432, 366, 465, 430, 359, 472, 456, 483, 464, 495, 546, 456, 428, 126, 149, 461, 428, 127, 137, 153, 125, 156, 466, 427, 135, 153, 140, 470, 436, 133, 153, 140, 138, 547, 460, 425, 359, 462, 429, 361, 468, 424, 363, 469, 426, 361, 472, 456, 495, 547, 456, 432, 366, 465, 430, 361, 472, 456, 476, 459, 483, 546, 461, 427, 127, 153, 135, 125, 467, 427, 126, 154, 137, 471, 426, 143, 156, 547, 462, 430, 369, 469, 426, 368, 472, 456, 478, 546, 458, 426, 140, 163, 461, 426, 144, 162, 137, 464, 428, 143, 157, 135, 125, 160, 547, 456, 427, 368, 460, 425, 366, 462, 426, 368, 465, 429, 365, 471, 430, 361, 472, 456, 476, 546, 470, 429, 132, 158, 136, 138, 472, 456, 495, 459, 483, 546, 460, 424, 134, 158, 461, 428, 130, 148, 159, 157, 159, 466, 426, 147, 135, 547, 456, 426, 358, 459, 426, 361, 462, 425, 363, 464, 425, 366, 466, 434, 366, 472, 456, 483, 468, 473, 546, 464, 426, 140, 163, 467, 427, 154, 142, 137, 158, 471, 427, 143, 155, 125, 547, 466, 425, 358, 468, 425, 359, 470, 425, 361, 472, 456, 480, 464, 545, 467, 478, 546, 459, 427, 144, 163, 463, 427, 138, 156, 547, 456, 429, 363, 462, 425, 365, 464, 432, 366, 472, 456, 478, 463, 545, 468, 483, 546, 469, 427, 153, 547, 471, 425, 358, 472, 456, 483, 460, 495, 464, 483, 468, 495, 546, 457, 428, 156, 135, 140, 462, 432, 128, 160, 157, 159, 471, 439, 135, 160, 130, 143, 154, 137, 547, 457, 425, 360, 459, 427, 361, 463, 427, 363, 467, 429, 361, 472, 456, 483, 463, 482, 467, 483, 547, 459, 427, 363, 463, 455, 366, 472, 456, 483, 472, 456, 483, 472, 456, 483, 456, 483]
    arr = [472, 460, 474, 463, 486, 468, 474, 472, 456, 486, 463, 479, 467, 474, 472, 456, 476, 472, 456, 474, 468, 476, 546, 462, 426, 138, 149, 134, 141, 157, 159, 465, 425, 157, 135, 467, 424, 126, 149, 468, 424, 143, 156, 469, 424, 126, 149, 470, 425, 127, 137, 153, 125, 156, 472, 456, 476, 464, 488, 468, 476, 546, 456, 425, 135, 153, 140, 458, 432, 133, 153, 140, 138, 472, 456, 474, 464, 476, 472, 456, 474, 467, 476, 472, 456, 476, 459, 474, 468, 476, 472, 456, 476, 464, 474, 468, 476, 472, 456, 476, 463, 479, 467, 480, 472, 456, 476, 468, 481, 472, 456, 474, 460, 473, 463, 474, 468, 476, 546, 468, 426, 127, 153, 135, 125, 471, 426, 126, 154, 137, 472, 456, 476, 464, 479, 546, 458, 426, 143, 156, 461, 424, 140, 163, 462, 425, 144, 162, 137, 464, 427, 143, 157, 135, 125, 160, 472, 456, 474, 464, 481, 468, 476, 546, 468, 426, 132, 158, 136, 138, 471, 425, 134, 158, 472, 456, 476, 459, 480, 463, 479, 467, 480, 546, 457, 427, 130, 148, 159, 157, 159, 461, 425, 147, 135, 463, 426, 140, 163, 466, 425, 154, 142, 137, 158, 468, 426, 143, 155, 125, 471, 426, 144, 163, 472, 456, 474, 463, 477, 546, 458, 427, 138, 156, 547, 465, 425, 360, 472, 456, 479, 464, 484, 467, 477, 546, 460, 428, 153, 465, 428, 156, 135, 140, 470, 426, 128, 160, 157, 159, 547, 463, 427, 355, 467, 427, 357, 471, 426, 357, 472, 456, 477, 546, 457, 440, 135, 160, 130, 143, 154, 137, 547, 458, 428, 357, 463, 425, 355, 465, 426, 357, 468, 440, 355, 472, 456, 477, 460, 476, 464, 481, 468, 477, 472, 456, 477, 464, 474, 468, 473]
    lyrics_tokens = [  2, 138, 149, 134, 141, 157, 159, 264, 157, 135, 264, 126, 149,
       264, 143, 156, 264, 126, 149, 264, 127, 137, 153, 125, 156, 264,
       135, 153, 140, 264, 133, 153, 140, 138, 264,  85, 264,   2, 127,
       153, 135, 125, 264, 126, 154, 137, 264, 143, 156, 264, 140, 163,
       264, 144, 162, 137, 264, 143, 157, 135, 125, 160, 264,  85, 264,
         2, 132, 158, 136, 138, 264, 134, 158, 264, 130, 148, 159, 157,
       159, 264, 147, 135, 264, 140, 163, 264, 154, 142, 137, 158, 264,
       143, 155, 125, 264, 144, 163, 264, 138, 156, 264,  85, 264,   2,
       204, 265, 263, 156, 135, 140, 264, 128, 160, 157, 159, 264, 135,
       160, 130, 143, 154, 137, 264,  85]

    config = FixedLengthPhonemeAndVocal2MidiCodec.Config(
        lyrics_seq_len=2000,
        leadsheet_seq_len=5000,
    )

    output_dir = os.path.join(
        os.environ["DUMP_DIR"], "ai_music/tmp/"
    )

    codec = FixedLengthPhonemeAndVocal2MidiCodec(config)
    midi_obj = codec.decode(arr)
    midi_bytes = pretty_midi_obj_to_midi_bytes(midi_obj)
    midi_output_path = os.path.join(output_dir, f"20240314.20240309.l2l.base.0.mid")
    with open(midi_output_path, "wb") as f:
        f.write(midi_bytes)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip