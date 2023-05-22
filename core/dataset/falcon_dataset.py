"""
IterableDataset
falcon dataset support tfrecord and tensorbundle use FalconReader
"""

import random
import numpy as np
import torch
from core.extensions import mpu
from core.dataset.sampler import setup_sampler_cfg
from core.utils.dist_util import get_local_rank, get_local_size
from core.utils import logging
from torch.utils.data import IterableDataset
from dataloader import FalconReader


class FalconDataset(IterableDataset):
    """
    Use FalconReader's IterableDataset
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
        self.chunk_size = cfg.get('chunk_size', 16)
        self.io_thread_num = cfg.get('io_thread_num', 12)
        self.fd_cache_size = cfg.get('fd_cache_size', 2048)
        self.io_retry = cfg.get('io_retry', 5)
        self.cache_name = cfg.get('cache_name', 'falcon_dataset')
        self.prefetch_chunk_num = cfg.get('prefetch_chunk_num', 30)
        self.pipeline = not mpu.is_unitialized()
        self.rank = rank if not self.pipeline else mpu.get_data_parallel_rank()
        self.world_size = world_size if not self.pipeline else mpu.get_data_parallel_world_size()
        self.split_path_list_by_rank = split_path_list_by_rank
        self.reader = None
        self.item_transform = item_transform
        self.epoch = 0
        self.resume = 0
        self.sampler_cfg = setup_sampler_cfg(cfg)
        self.shuffle = shuffle
        self.sampler = None
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

    def init_falconreader(self):
        '''init falconreader
        the dataset will be build in main process
        but the FalconReader must be build in sub process
        '''
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
        self.reader = FalconReader(
            self.path_list,
            self.fd_cache_size,
            self.io_thread_num,
            self.io_retry,
            self.cache_name,
            self.local_process_num,
            self.local_pid,
            self.chunk_size,
        )

    def init_sampler(self):
        '''init sampler'''
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
            self.chunk_size,
            self.prefetch_chunk_num,
            len(self.path_list),
            self.reader,
            self.sampler_cfg[1],
        )
        logging.get_logger(log_level="WARNING")

    def __iter__(self):
        '''iter'''
        self.set_seed()
        if self.reader is None:
            self.init_falconreader()
        if self.sampler is None:
            self.init_sampler()
        self.sampler.reset(self.base_seed, skip_num=self.resume)
        self.resume = 0
        for chunks, path_idxs in self.sampler:
            kwargs = {
                'local_rank': self.local_rank,
                'pid': self.pid,
                'local_process_num': self.local_process_num,
                'local_pid': self.local_pid,
            }
            try:
                raw_datas = self.reader.read_many(chunks, True)
            except Exception:
                logging.warning(
                    "rank %d: prefetch %d FalconReader read chunks failed!\n"
                    "Current path is %s, io_thread_num is %d",
                    self.rank,
                    self.pid,
                    self.path_list[path_idxs[0]],
                    self.io_thread_num,
                    exc_info=True,
                )
                continue
            for path_idx, raw_data in zip(path_idxs, raw_datas):
                kwargs['path_idx'] = path_idx
                if self.io_reuse > 1:
                    raw_data = [bytes(item) for item in raw_data]
                    raw_data *= self.io_reuse
                for item in raw_data:
                    try:
                        item = self.item_transform(item, **kwargs)
                    except Exception:
                        logging.warning(
                            "rank %d: prefetch %d item_transform failed!",
                            self.rank,
                            self.pid,
                            exc_info=True,
                        )
                        continue
                    if item is not None:
                        yield item

    def reset(self, epoch_count=0, skip_item_num=0):
        '''
        dataset reset
        '''
        self.epoch = epoch_count
        self.skip_item_num = skip_item_num
