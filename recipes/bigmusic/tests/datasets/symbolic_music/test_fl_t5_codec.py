import os

from recipes.bigmusic.datasets.symbolic_music.fl_t5_codec import FLT5Codec


def test_fl_t5_codec_decode():
    from recipes.bigmusic.datasets.utils.symbolic_music import pretty_midi_obj_to_midi_bytes

    config = FLT5Codec.Config(
        leadsheet_seq_len=5000,
    )
    dm = FLT5Codec.get_datamodule(
        config=config,
        train_id=1838,
        val_id=1839,
        batch_size=1,
        num_workers=0,
        shuffle_buffer=0,
    )
    codec = FLT5Codec(config)

    for i, batch in enumerate(dm.val_dataloader()):
        from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
        target_length = batch["target_lengths"][0] - 1
        leadsheet_tokens = batch["target_ids"][0][:target_length].numpy()
        midi_obj = codec.decode(leadsheet_tokens)
