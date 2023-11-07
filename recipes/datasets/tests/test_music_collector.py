import json

import torchaudio

from recipes.datasets.music_collector import MusicCollectorWebLoader


def test_music_collector():
    sample_rate = 24000
    batch_size = 8 
    duration = 10
    url2index = "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/music/youtube_sft/playlist/1CIaSnLlv5FgE0V1ZFkWAU/url2index.txt"
    pl_datamodule = MusicCollectorWebLoader(
        url2index=url2index,
        sample_rate=sample_rate,
        duration=duration,
        batch_size=batch_size,
        shuffle_buffer_size=batch_size,
        num_workers=0,
        pin_memory=True,
        resampled=True,
        shardshuffle=True,
    )

    train_loader = pl_datamodule.train_dataloader()
    batch = next(iter(train_loader))

    for idx in range(batch_size):
        audio = batch["audio"][idx]
        artist_name = batch["metadata"][idx]["artist_name"]
        track_name = batch["metadata"][idx]["track_name"]
        genre = batch["metadata"][idx]["genre"]
        lyrics = batch["metadata"][idx]["lyrics"]

        fp = f"{track_name} - {artist_name} ({genre})".replace("/", "-")
        torchaudio.save(f"{fp}.wav", audio, sample_rate=sample_rate)

        with open(f"{fp}_lyrics.txt", "w") as f:
            json.dump(lyrics, f, indent=True, ensure_ascii=False)
        
