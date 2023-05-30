"""
IterableDataset
ParquetDataset support reading Parquet
"""

import random
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset

from core.extensions import mpu
from core.utils import logging
from core.utils.dist_util import get_local_rank, get_local_size


class WdsDataset(IterableDataset):
    """
    Read webdataset IterableDataset
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
        self.skip_item_num = cfg.get('skip_item_num', 0)
        self.shuffle = shuffle
        self.base_seed = cfg.get('seed', 123356)
        self.kwargs = cfg.get('wds', dict())
        self.wds_dataset = None
        self.local_world_size = get_local_size()
        self.local_rank = get_local_rank()
        self.dataset_kind = self.__class__.__name__

    def set_seed(self):
        '''set seed'''
        seed = self.base_seed + self.epoch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_wdsdataset(self):
        '''init wdsdataset'''
        urls = [f"pipe: hdfs dfs -cat {url}" for url in self.path_list]
        self.wds_dataset = wds.WebDataset(
            urls=urls, nodesplitter=wds.split_by_node, **self.kwargs
        ).decode()
        logging.get_logger(log_level="WARNING")
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

    def __iter__(self):
        '''iter'''
        self.set_seed()
        if self.wds_dataset is None:
            self.init_wdsdataset()
        self.skip_item_num = 0
        data_count = 0
        for item in self.wds_dataset:
            kwargs = dict()
            if self.item_transform is not None:
                try:
                    item = self.item_transform(item, **kwargs)
                except Exception:
                    logging.warning(
                        "rank %d: %s prefetch %d item_transform failed!",
                        self.rank,
                        self.__class__.__name__,
                        self.pid,
                        exc_info=True,
                    )
                    data_count += 1
                    continue
            data_count += 1
            if item is not None:
                # can't get path_idx
                item['dataset_state'] = (self.dataset_kind, data_count, -1)
                yield item
                data_count = 0
        self.epoch += 1

    def reset(self, epoch_count=0, skip_item_num=0):
        '''
        dataset reset
        '''
        self.epoch = epoch_count
        self.skip_item_num += skip_item_num
