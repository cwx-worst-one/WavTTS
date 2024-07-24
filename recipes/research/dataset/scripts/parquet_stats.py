from tqdm import tqdm
from webdataset.shardlists import split_by_node
from webdataset.utils import pytorch_worker_info

from samantha.dataio.parquet.extension import _ParquetReader
from samantha.utils.hdfs_helper import fast_glob_files

if __name__ == "__main__":

    rank, world_size, worker, num_workers = pytorch_worker_info()
    urls = fast_glob_files(
        "hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data_store/BigMusic/playlist_v5_2M/index_1/*/*.parquet"
    )

    urls = list(split_by_node(urls))

    total_rows = 0
    for url in tqdm(urls, desc=f"{rank}/{world_size}: reading..."):
        index_reader = _ParquetReader(url)
        # total_rows += index_reader.total_rows
        total_rows += index_reader.count_rows()
        index_reader.close()

    print(f"{rank}/{world_size}: {total_rows}")
