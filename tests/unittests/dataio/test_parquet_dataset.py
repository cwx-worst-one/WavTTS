import torch
import torch.utils.data

from samantha.dataio.parquet import ParquetDataset


def test_parquet_dataset():
    urls = [
        {
            "index": "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tests/dataset/index_1/*.parquet",  # noqa
            "data": "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tests/dataset/data/*.parquet",  # noqa
            "wvae_1.0": "hdfs://haruna/home/byte_speech_sv/user/wangxin.colin/tests/dataset/wvae_1.0/*.parquet",  # noqa
        }
    ]
    ds = ParquetDataset(data_urls=urls, resampled=False)
    dl = torch.utils.data.DataLoader(ds, num_workers=2, batch_size=None)
    cnt = 0
    for item in dl:
        assert "__index_url__" in item
        assert "__data_url__" in item
        assert "__wvae_1.0_url__" in item
        assert item["__dataset_name__"] == "dataset"
        cnt += 1
        break
    assert cnt == 1
