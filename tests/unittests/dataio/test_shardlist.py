from samantha.dataio.parquet.shardlists import ResampledShards


def test_resampled_shards():
    urls = [f"fake_url_{i}" for i in range(10)]

    shard = ResampledShards(urls=urls, nshards=100)

    itered_urls = []
    for i, url in enumerate(shard):
        itered_urls.append(url)
        if (i + 1) % len(urls) == 0:
            assert set(itered_urls) == set(urls)
