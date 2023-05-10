"""
Data Fetcherator
"""
import os
import random
from collections import deque
from dataloader import FalconReader
from core.utils import logging


def get_paths(data_paths=None, data_root=None, file_prefix=None, shard_list=None, shard_num=None):
    """
    get data paths
    Args:
        data_paths(str or list): means all data-paths
        data_root(str): data root
        file_prefix(str): file prefix
        shard_list(list of str): means all shard names
        shard_num(int): means all shard num
    """
    # First priority
    if data_paths is not None:
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        return data_paths
    if data_root is None:
        logging.error("DataFetcher Error: No file path entered")
        return None
    if file_prefix is None:
        path_prefix = data_root
    else:
        path_prefix = os.path.join(data_root, file_prefix)
    if shard_list:
        data_paths = [f"{path_prefix}{shard}" for shard in shard_list]
    elif shard_num > 0:
        data_paths = [f"{path_prefix}{shard}" for shard in range(shard_num)]
    else:
        data_paths = [path_prefix]
    return data_paths


class DataFetcher:
    """Data Fetcher
    random get datas from a list of files
    can get data in native mode or global mode
    """

    def __init__(
        self,
        data_paths=None,
        chunk_size=8,
        parrallel_chunk_num=4,
        io_thread_num=3,
        fd_cache_size=40,
        cached_name='data_fetcher',
        shuffle=False,
        mem_shared=False,
        sampler_mode='native',
    ):
        """init."""
        self.data_paths = data_paths
        self.shard_num = len(self.data_paths)
        self.reader = None
        self.chunk_size = chunk_size
        self.parrallel_chunk_num = parrallel_chunk_num
        self.io_thread_num = io_thread_num
        self.fd_cache_size = fd_cache_size
        self.cached_name = cached_name
        self.entry_nums = 0
        self.sampler_mode = sampler_mode
        self.shuffle = shuffle
        self.mem_shared = mem_shared
        self.chunks_buf = deque()
        self.data_buf = deque()

    def get_reader(self, worker_num=1, pid=0):
        """
        get FalconReader
        """
        if self.reader is not None:
            return
        self.reader = FalconReader(
            self.data_paths,
            self.fd_cache_size,  # fd_cache_size
            self.io_thread_num,  # io_thread_num
            5,  # io_retry
            self.cached_name,  # unused
            worker_num,  # GPU num in one worker
            pid,  # GPU idx in one worker
            self.chunk_size,  # chunk_size
        )
        if self.sampler_mode == 'global':
            self.shard_entry_nums = self.reader.get_entry_num(
                list(range(self.shard_num)), self.mem_shared, True
            )
            self.entry_nums = sum(self.shard_entry_nums)

    def get_chunks(self):
        """get chunks"""
        if self.reader is None:
            self.get_reader()
        if self.sampler_mode == 'global':
            chunks_idxs = [i * self.chunk_size for i in range(self.entry_nums // self.chunk_size)]
        else:
            shard = random.randint(0, self.shard_num - 1)
            entry_nums = self.reader.get_entry_num([shard], False)
            chunks_idxs = [i * self.chunk_size for i in range(entry_nums // self.chunk_size)]
        if self.shuffle:
            random.shuffle(chunks_idxs)
        self.chunks_buf.extend(
            [
                chunks_idxs[i : i + self.parrallel_chunk_num]
                for i in range(0, len(chunks_idxs), self.parrallel_chunk_num)
            ]
        )

    def update_buf(self):
        '''update data buf'''
        if not self.chunks_buf:
            self.get_chunks()
        chunks = self.chunks_buf.popleft()
        try:
            datas = sum(self.reader.read_many(chunks), [])
            self.data_buf.extend(datas)
        except Exception:
            logging.warning("Data Fetcher may occur Error")

    def get_data(self):
        """get data"""
        while not self.data_buf:
            self.update_buf()
        item = self.data_buf.popleft()
        return item


class TargetDataFetcher(DataFetcher):
    """
    get Target file_idx datas
    """

    def __init__(
        self,
        data_paths=None,
        chunk_size=8,
        parrallel_chunk_num=4,
        io_thread_num=2,
        fd_cache_size=256,
        cached_name='data_fetcher',
        shuffle=False,
        mem_shared=False,
    ):
        """init"""
        super().__init__(
            data_paths,
            chunk_size,
            parrallel_chunk_num,
            io_thread_num,
            fd_cache_size,
            cached_name,
            shuffle,
            mem_shared,
            sampler_mode='global',
        )
        self.file_shards = list(range(self.shard_num))
        self.mapped_chunks = dict()
        self.mapped_datas = dict()
        for shard in self.file_shards:
            self.mapped_datas[shard] = deque()
            self.mapped_chunks[shard] = None

    def pre_get_file_chunks(self):
        """pre get file chunks"""
        if self.reader is None:
            self.get_reader()
        all_chunks = [i * self.chunk_size for i in range(self.entry_nums // self.chunk_size)]
        chunk_offset = 0
        for shard, shard_entry_num in zip(self.file_shards, self.shard_entry_nums):
            shard_chunk_num = shard_entry_num // self.chunk_size
            self.mapped_chunks[shard] = all_chunks[chunk_offset : chunk_offset + shard_chunk_num]
            chunk_offset += shard_chunk_num
        del all_chunks

    def update_buf(self, target_shard):
        """
        update data buf
        """
        if self.mapped_chunks[target_shard] is None:
            self.pre_get_file_chunks()
        chunks = random.sample(self.mapped_chunks[target_shard], self.parrallel_chunk_num)
        try:
            datas = sum(self.reader.read_many(chunks), [])
            if self.shuffle:
                random.shuffle(datas)
            self.mapped_datas[target_shard].extend(datas)
            del datas
        except Exception:
            logging.warning("Data Fetcher may occur Error")

    def get_data(self, target_shard=None):
        """get target data"""
        if target_shard is None:
            target_shard = random.choice(self.file_shards)
        if not self.mapped_datas[target_shard]:
            self.update_buf(target_shard)

        item = self.mapped_datas[target_shard].popleft()
        return item
