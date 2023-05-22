"""
IterableDataset
ParquetDataset support reading Parquet
"""

import random
import numpy as np
import torch
from torch.utils.data import IterableDataset

from core.extensions import mpu
from core.dataset.sampler import setup_sampler_cfg
from core.dataset.utils import get_parquet_file_handle
from core.utils.dist_util import get_local_rank, get_local_size
from core.utils import logging


class ParquetDataset(IterableDataset):
    """
    Read Parquet IterableDataset
    """

    def __init__(
        self,
        path_list,
        cfg,
        item_transform=None,
        shuffle=True,
        split_path_list_by_rank=True,
        rank=0,
        world_size=1,
    ):
        self.path_list = path_list
        self.read_batch_size = cfg.get('read_batch_size', 16)
        self.pipeline = not mpu.is_unitialized()
        self.rank = rank if not self.pipeline else mpu.get_data_parallel_rank()
        self.world_size = world_size if not self.pipeline else mpu.get_data_parallel_world_size()
        self.split_path_list_by_rank = split_path_list_by_rank
        # preprocess each item, set on demand
        self.item_transform = item_transform
        self.epoch = 0
        self.resume = 0
        cfg.sampler = 'ParquetSampler'
        self.sampler_cfg = setup_sampler_cfg(cfg)
        self.shuffle = shuffle
        self.sampler = None
        self.columns = cfg.get('columns', None)
        self.use_threads = cfg.get('use_threads', True)
        self.file_handle = None
        self.base_seed = cfg.get('seed', 123356)
        self.io_reuse = cfg.get('io_reuse', 1)
        self.local_world_size = get_local_size()
        self.local_rank = get_local_rank()

    def set_seed(self):
        '''set seed'''
        seed = self.base_seed + self.epoch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_sampler(self):
        '''init sampler'''
        self.local_world_size = get_local_size()
        self.local_rank = get_local_rank()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.prefetch_num = 1
            self.pid = 0
        else:
            self.prefetch_num = worker_info.num_workers
            self.pid = worker_info.id
        self.local_process_num = (
            self.local_world_size * self.prefetch_num
        )  # the process num in one worker
        self.local_pid = (
            self.local_rank * self.prefetch_num + self.pid
        )  # the process pid in one worker
        sampler_cls = self.sampler_cfg[0]
        self.sampler = sampler_cls(
            self.pid,
            self.prefetch_num,
            self.local_rank,
            self.local_world_size,
            self.rank,
            self.world_size,
            self.shuffle,
            self.split_path_list_by_rank,
            self.path_list,
            self.sampler_cfg[1],
        )
        logging.get_logger(log_level="WARNING")

    def __iter__(self):
        '''iter'''
        self.set_seed()
        if self.sampler is None:
            self.init_sampler()
        self.sampler.reset(self.base_seed, skip_num=self.resume)
        self.resume = 0
        for row_groups, path_idx in self.sampler:
            kwargs = {
                'local_rank': self.local_rank,
                'pid': self.pid,
                'local_process_num': self.local_process_num,
                'local_pid': self.local_pid,
            }
            kwargs['path_idx'] = path_idx
            parquet_file, _, self.file_handle = get_parquet_file_handle(self.path_list[path_idx])
            for row_group in row_groups:
                parquet_iter = parquet_file.iter_batches(
                    self.read_batch_size,
                    row_groups=[row_group],
                    columns=self.columns,
                    use_threads=self.use_threads,
                )
                for group_data in parquet_iter:
                    group_datas = group_data.to_pandas()
                    for row in group_datas.iterrows():
                        item = row[1].to_dict()
                        if self.item_transform is not None:
                            try:
                                item = self.item_transform(item, **kwargs)
                            except Exception:
                                logging.warning(
                                    "rank %d: prefetch %d item_transform failed!",
                                    self.rank,
                                    self.pid,
                                    exc_info=True,
                                )
                        yield item

    def reset(self, epoch_count=0, skip_item_num=0):
        '''
        dataset reset
        '''
        self.epoch = epoch_count
        self.skip_item_num = skip_item_num
