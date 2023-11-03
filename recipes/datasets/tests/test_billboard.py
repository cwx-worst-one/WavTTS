import json

import pytest

pytestmark = pytest.mark.data

import torchaudio
from tqdm import tqdm

from recipes.datasets.billboard.billboard import (
    BillboardDataModule,
    BillboardLyricsDataModule,
    BillboardArtistGenderDataModule,
    BillboardDataResult,
    musixmatch_to_lrc,
)

sample_rate = 24000
batch_size = 8
buckets_sec = [10, 15, 20, 25, 30]
shuffle_buffer_size = 8
num_workers = 0


def get_fp(metadata):
    track_name = metadata["spotify_track_name"]
    artist_name = metadata["spotify_primary_artist_name"]
    album_name = metadata["spotify_album_name"]
    fp = f"{track_name} - {artist_name}".replace("/", "-")
    return fp


@pytest.fixture()
def billboard_datamodule():
    return BillboardDataModule(
        duration=30,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        pin_memory=True,
    )


@pytest.fixture()
def billboard_lyrics_datamodule():
    return BillboardLyricsDataModule(
        buckets_sec=buckets_sec,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        pin_memory=True,
    )


@pytest.fixture()
def billboard_artist_datamodule():
    return BillboardArtistGenderDataModule(
        duration=30,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        pin_memory=True,
    )


@pytest.fixture()
def batch(billboard_datamodule):
    train_loader = billboard_datamodule.train_dataloader()
    return next(iter(train_loader))


@pytest.fixture()
def batch_lyrics(billboard_lyrics_datamodule):
    train_loader = billboard_lyrics_datamodule.train_dataloader()
    return next(iter(train_loader))


@pytest.fixture()
def batch_artist(billboard_artist_datamodule):
    train_loader = billboard_artist_datamodule.train_dataloader()
    return next(iter(train_loader))


@pytest.mark.skip()
def test_billboard200(batch):
    assert type(batch) == BillboardDataResult

    for batch_idx in range(batch_size):
        fp = get_fp(batch.metadata[batch_idx])
        torchaudio.save(fp + ".flac", batch.target_audio[batch_idx], sample_rate)


def test_billboard200_lyrics(batch_lyrics):
    assert type(batch_lyrics) == BillboardDataResult

    for batch_idx in range(batch_size):
        fp = get_fp(batch_lyrics.metadata[batch_idx])
        torchaudio.save(fp + ".flac", batch_lyrics.target_audio[batch_idx], sample_rate)

        with open(fp + "_lyrics.txt", "w") as f:
            f.write(batch_lyrics.lyrics_text[batch_idx])


@pytest.mark.skip()
def test_billboard200_artist(batch_artist):
    assert type(batch_artist) == BillboardDataResult

    for batch_idx in range(batch_size):
        fp = get_fp(batch_artist.metadata[batch_idx])

        fp = f"{fp} - {batch_artist.artist_gender[batch_idx]}"
        torchaudio.save(fp + ".flac", batch_artist.target_audio[batch_idx], sample_rate)


def test_billboard200_throughput(billboard_datamodule):
    num_batches = 100
    train_loader = billboard_datamodule.train_dataloader()

    for idx, batch in enumerate(tqdm(train_loader)):
        if idx == num_batches:
            break
