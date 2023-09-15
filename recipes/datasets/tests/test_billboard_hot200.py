import json

import pytest
import torchaudio
from tqdm import tqdm

from recipes.datasets.billboard_hot200.billboard_hot200 import (
    BillboardHot200PreprocessedWebDataModule,
    BillboardHot200WebDataModule,
)
from recipes.soundstorm2.lightning.dac import DACModel
from tests.helpers.testing_utils import torch_device


def test_billboard200():
    sample_rate = 24000
    pl_datamodule = BillboardHot200PreprocessedWebDataModule(
        sample_rate=sample_rate, batch_size=8, shuffle_buffer_size=8, num_workers=8
    )
    test_loader = pl_datamodule.val_dataloader()
    batch = next(iter(test_loader))

    assert "audio" in batch
    assert "metadata" in batch
    assert "track_features" in batch
    assert "lyrics" in batch

    fp = f"{batch['metadata'][0]['name']} - {batch['metadata'][0]['primary_artist_name']}"
    torchaudio.save(fp + ".flac", batch["audio"][0], sample_rate)

    with open(f"{fp}_track_features.json", "w") as f:
        json.dump(batch["track_features"][0], f)

    with open(f"{fp}_lyrics.json", "w") as f:
        json.dump(batch["lyrics"][0], f)

    with open(f"{fp}_metadata.json", "w") as f:
        json.dump(batch["metadata"][0], f)

def test_billboard200_throughput():
    sample_rate = 24000
    batch_size = 16
    shuffle_buffer_size = 8
    num_workers = 8
    duration = 30.0
    shardshuffle = False
    num_batches = 10
    # dac = DACModel(src_sample_rate=SAMPLE_RATE, target_sample_rate=SAMPLE_RATE).to(torch_device)

    # pl_datamodule = BillboardHot200WebDataModule(
    #     sample_rate=SAMPLE_RATE,
    #     batch_size=batch_size,
    #     shuffle_buffer_size=shuffle_buffer_size,
    #     duration=duration,
    #     num_workers=num_workers,
    #     shardshuffle=shardshuffle,
    # )
    # test_loader = pl_datamodule.train_dataloader()

    # for idx, batch in tqdm(enumerate(test_loader)):
    #     if idx == num_batches:
    #         break

    pl_datamodule_preproc = BillboardHot200PreprocessedWebDataModule(
        sample_rate=sample_rate,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        duration=duration,
        num_workers=num_workers,
        shardshuffle=shardshuffle,
    )
    test_loader = pl_datamodule_preproc.train_dataloader()

    for idx, batch in tqdm(enumerate(test_loader)):
        if idx == num_batches:
            break