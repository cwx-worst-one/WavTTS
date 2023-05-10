'''
test data fetcher
'''
from core.dataset import get_paths, DataFetcher, TargetDataFetcher

# pylint: disable='line-too-long'
def test_data_fetcher(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/youyongbin/data/lm/bytebot/hdfs_data/',
    file_prefix='shard',
    shard_num=12,
):
    '''test data fetcher'''
    paths = get_paths(data_root=data_root, file_prefix=file_prefix, shard_num=shard_num)
    data_fetcher = DataFetcher(paths)
    for _ in range(10):
        data = data_fetcher.get_data()
        print(len(data))


def test_target_datas_fetcher(
    data_root='hdfs://haruna/home/byte_arnold_hl_speech_asr/user/youyongbin/data/lm/bytebot/hdfs_data/',
    file_prefix='shard',
    shard_num=12,
):
    '''test data fetcher'''
    paths = get_paths(data_root=data_root, file_prefix=file_prefix, shard_num=shard_num)
    data_fetcher = TargetDataFetcher(paths)
    for i in range(10):
        data = data_fetcher.get_data(target_shard=i)
        print(len(data))
