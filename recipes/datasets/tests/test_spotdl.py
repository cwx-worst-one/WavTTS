from recipes.datasets.music_collector import MusicCollectorDataset, MusicCollectorWebLoader


def test_spotdl_dataset():
    # dataset = SpotDLDataset(root_dir="./data", artist_id=artist_id)
    # dataset.download()
    # dataset.prepare_dataset()

    spotify_type = "artist"
    spotify_id = "74ASZWbe4lXaubB36ztrGX"
    sample_rate = 24000
    mono = True
    download = False
    dataloader = MusicCollectorWebLoader(
        root_dir="./spotdl_data",
        spotify_type=spotify_type,
        spotify_id=spotify_id,
        download=download,
        sample_rate=sample_rate,
        mono=mono,
        n_threads=8,
    )

    pattern = f"./webdataset/{spotify_type}/{spotify_id}/%05d.tar"
    maxsize = 1 << 32  # 4GiB, maximum size of each shard
    dataloader.create_webdataset(pattern=pattern, maxsize=maxsize)
