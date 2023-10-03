import json

import pytest
import torchaudio
from tqdm import tqdm

from recipes.datasets.billboard_hot200.billboard_hot200 import (
    BillboardDataModule,
    BillboardLyricsDataModule,
    musixmatch_to_lrc,
)
from recipes.soundstorm2.lightning.dac import DACModel
from tests.helpers.testing_utils import torch_device
# from recipes.bigmusic.datasets.mix import Wav2VecPhonemeTokenizer
from recipes.bigmusic.utils.metrics_asr import wav2lyrics


sample_rate = 24000
batch_size = 8
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
        url2index=f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200_v2/{sample_rate}hz/train/url2index.txt",
        duration=30,
        sample_rate=sample_rate,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
    )


@pytest.fixture()
def billboard_lyrics_datamodule():
    return BillboardLyricsDataModule(
        url2index=f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/billboard_hot200_v2_lyrics/{sample_rate}hz/train/url2index.txt",
        duration=30,
        sample_rate=sample_rate,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
    )


@pytest.fixture()
def batch(billboard_datamodule):
    train_loader = billboard_datamodule.train_dataloader()
    return next(iter(train_loader))


@pytest.fixture()
def lyrics_batch(billboard_lyrics_datamodule):
    train_loader = billboard_lyrics_datamodule.train_dataloader()
    return next(iter(train_loader))


@pytest.mark.skip()
def test_billboard200(batch):
    assert "audio" in batch
    assert "metadata" in batch
    assert "lyrics" in batch
    assert "shard" in batch

    track_name = batch["metadata"][0]["spotify_track_name"]
    artist_name = batch["metadata"][0]["spotify_primary_artist_name"]
    album_name = batch["metadata"][0]["spotify_album_name"]

    fp = f"{track_name} - {artist_name}"
    torchaudio.save(fp + ".flac", batch["audio"][0], sample_rate)

    # with open(f"{fp}_track_features.json", "w") as f:
    #     json.dump(batch["track_features"][0], f)

    with open(f"{fp}_lyrics.json", "w") as f:
        json.dump(batch["lyrics"][0], f)

    with open(f"{fp}_metadata.json", "w") as f:
        json.dump(batch["metadata"][0], f)

    if batch["lyrics"][0] is not None:
        lrc = musixmatch_to_lrc(
            batch["lyrics"][0], title=track_name, artist=artist_name, album=album_name
        )
        with open(f"{fp}.lrc", "w") as f:
            f.write(lrc)


@pytest.mark.skip()
def test_billboard_lyrics(lyrics_batch):
    assert "audio" in lyrics_batch
    assert "metadata" in lyrics_batch
    assert "lyrics" in lyrics_batch
    assert "shard" in lyrics_batch

    fp = get_fp(lyrics_batch["metadata"][0])
    torchaudio.save(fp + ".flac", lyrics_batch["audio"][0], sample_rate)

    with open(f"{fp}_lyrics.txt", "w") as f:
        f.write(lyrics_batch["lyrics"][0])


def test_billboard_lyrics_asr_transcription(billboard_lyrics_datamodule):
    # phoneme_tokenizer = Wav2VecPhonemeTokenizer()
    dataloader = billboard_lyrics_datamodule.train_dataloader()

    for lyrics_batch in tqdm(dataloader):
        audio = lyrics_batch["audio"]
        gt_lyrics = lyrics_batch["lyrics"]
        # gt_phoneme_tokens = lyrics_batch["phoneme_tokens"]

        lyrics, wavs = wav2lyrics(audio, sr=sample_rate, do_itn=True)

        # phoneme_tokens = []
        # for l in lyrics:
        #     phoneme_tokens.append(phoneme_tokenizer([l])[0])

        # phoneme_tokens = phoneme_tokenizer.collate_fn(phoneme_tokens)

        for idx in range(len(audio)):
            fp = get_fp(lyrics_batch["metadata"][idx])
            torchaudio.save(f"{fp}_{idx}.mp3", audio[idx], sample_rate)

            with open(f"{fp}_{idx}_lyrics.txt", "w") as f:
                f.write(gt_lyrics[idx])

            with open(f"{fp}_{idx}_asr_lyrics.txt", "w") as f:
                f.write("\n".join(lyrics[idx].split(". ")))
        break


@pytest.mark.skip()
def test_billboard200_throughput(billboard_datamodule):
    num_batches = 10
    train_loader = billboard_datamodule.train_dataloader()

    for idx, batch in tqdm(enumerate(train_loader)):
        if idx == num_batches:
            break
