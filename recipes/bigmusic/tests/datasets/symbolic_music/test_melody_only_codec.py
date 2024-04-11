import os
from recipes.bigmusic.datasets.symbolic_music.melody_only_codec import (
    MelodyOnlyCodec,
)


def test_melody_only_codec_datamodule():
    config = MelodyOnlyCodec.Config(
        leadsheet_seq_len=8000,
    )
    dm = MelodyOnlyCodec.get_datamodule(
        config=config,
        train_id=1838,
        val_id=1839,
        batch_size=20,
        num_workers=0,
        shuffle_buffer=100,
    )

    num_samples = 0
    for batch in dm.val_dataloader():
        from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
        num_samples += batch.size()[0]
        print(num_samples, batch.size())
