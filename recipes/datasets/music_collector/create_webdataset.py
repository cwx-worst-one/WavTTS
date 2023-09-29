import json
import os
from argparse import ArgumentParser
from glob import glob
import logging
import torchaudio

from recipes.datasets.music_collector import (
    MusicCollectorDataset,
    MusicCollectorWebLoader,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--root_dir", default="./music_collector", type=str)
    parser.add_argument("--webdataset_root_dir", default="./webdataset", type=str)
    parser.add_argument(
        "--spotify_type", type=str, default="artist", choices=["artist", "playlist"]
    )
    parser.add_argument("--spotify_id", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--mono", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()

    dataset = MusicCollectorDataset(
        root_dir=args.root_dir,
        spotify_type=args.spotify_type,
        spotify_id=args.spotify_id,
        download=args.download,
        n_threads=args.n_threads,
    )

    pattern = os.path.join(
        args.webdataset_root_dir, args.spotify_type, args.spotify_id, "%05d.tar"
    )

    maxsize = 4 << 30  # 32  # 4GiB, maximum size of each shard
    MusicCollectorWebLoader.create_webdataset(
        dataset=dataset,
        sample_rate=args.sample_rate,
        mono=args.mono,
        pattern=pattern,
        maxsize=maxsize,
    )

    url2index = os.path.join(os.path.dirname(pattern), "url2index.txt")
    pl_datamodule = MusicCollectorWebLoader(
        url2index=url2index,
        sample_rate=args.sample_rate,
        duration=args.duration,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        resampled=True,
        shardshuffle=True,
    )

    logger.info(f"Finished creating webdataset at {url2index}")

    logger.info("Rendering a single batch...")
    train_loader = pl_datamodule.train_dataloader()
    batch = next(iter(train_loader))

    out_fp = os.path.join("test_collector", args.spotify_id)
    os.makedirs(out_fp, exist_ok=True)
    for idx in range(args.batch_size):
        audio = batch["audio"][idx]
        artist_name = batch["metadata"][idx]["artist_name"]
        track_name = batch["metadata"][idx]["track_name"]
        genre = batch["metadata"][idx]["genre"]
        lyrics = batch["metadata"][idx]["lyrics"]

        filename = f"{track_name} - {artist_name} ({genre})".replace("/", "-")
        fp = os.path.join(out_fp, filename)
        torchaudio.save(f"{fp}.wav", audio, sample_rate=args.sample_rate)

        with open(f"{fp}_lyrics.txt", "w") as f:
            json.dump(lyrics, f, indent=True, ensure_ascii=False)
