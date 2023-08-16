import json

import pytest
import torchaudio
from tqdm import tqdm

from recipes.datasets.billboard_hot200.billboard_hot200 import (
    SAMPLE_RATE,
    BillboardHot200WebDataModule,
)


@pytest.mark.skip()
def test_billboard200():
    pl_datamodule = BillboardHot200WebDataModule(
        sample_rate=SAMPLE_RATE, batch_size=8, shuffle_buffer_size=100
    )
    test_loader = pl_datamodule.val_dataloader()
    batch = next(iter(test_loader))

    assert "audio" in batch
    assert "metadata" in batch
    assert "track_features" in batch
    assert "lyrics" in batch

    fp = f"{batch['metadata'][0]['name']} - {batch['metadata'][0]['primary_artist_name']}"
    torchaudio.save(fp + ".flac", batch["audio"][0], SAMPLE_RATE)

    with open(f"{fp}_track_features.json", "w") as f:
        json.dump(batch["track_features"][0], f)

    with open(f"{fp}_lyrics.json", "w") as f:
        json.dump(batch["lyrics"][0], f)

    with open(f"{fp}_metadata.json", "w") as f:
        json.dump(batch["metadata"][0], f)


def test_billboard200_throughput():
    batch_size = 64
    shuffle_buffer_size = 200
    num_workers = 8
    duration = 10.0
    shardshuffle = False
    pl_datamodule = BillboardHot200WebDataModule(
        sample_rate=SAMPLE_RATE,
        duration=duration,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        shardshuffle=shardshuffle,
    )
    test_loader = pl_datamodule.train_dataloader()
    for batch in tqdm(test_loader):
        pass
