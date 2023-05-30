'''
Raw Dataset Module.
'''
import random
import numpy as np
import torch
from dataloader import FalconReader  # pylint:disable=import-error
from core.utils import get_dist_info, logging
from core.utils.dist_util import get_local_rank, get_local_size
from core.dataset.sampler import setup_sampler_cfg
from .base import BaseDataset


class DebugHDFSDataset(BaseDataset):
    """DebugHDFSDataset based on BaseDataset for debug

    Exmaple::

        root_path = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user' +
                    '/jiangbo/arnold_hdfs_data/ag20kh_bpe'
        train_path_list = [f'{root_path}/shard{i}' for i in range(1024)]
        train_batch_dataset = DebugHDFSDataset(
                                           train_path_list,
                                           dataset_cfg,
                                           self.parse_fn,
                                           self.draw_batch_fn,
                                           split_path_list_by_rank=True,
                                           shuffle=True,
                                           )

    """

    def __init__(
        self,
        path_list,
        bucket_schedule,
        cfg,
        item_transform,
        batch_transforms,
        device_transforms=None,
        split_path_list_by_rank=True,
        prefetch_block_num=200,
        epoch_count=0,
        shuffle=False,
        is_cuda_available=True,
    ):
        super().__init__(
            path_list,
            bucket_schedule,
            cfg,
            epoch_count,
            *get_dist_info(),
            shuffle,
            split_path_list_by_rank,
        )
        self.prefetch_worker_num = cfg.get('prefetch_worker_num', 4)
        self.deterministic = cfg.get('deterministic', False)
        self.cache_name = cfg.get('cache_name', 'hdfs_train')
        self.sampler_cfg = setup_sampler_cfg(cfg)
        self.item_transform = item_transform
        self.batch_transforms = batch_transforms
        self.device_transforms = device_transforms
        self.prefetch_block_num = prefetch_block_num
        self.is_cuda_available = is_cuda_available
        self.io_thread_num = cfg.get('io_thread_num', 12)
        self.io_retry = cfg.get('io_retry', 5)
        self.fd_cache_size = cfg.get('fd_cache_size', 8000)
        self.chunk_size = cfg.get('chunk_size', 40)
        self.prefetch_chunk_num = cfg.get("prefetch_chunk_num", 40)

        cuda_cache_size = cfg.get('cuda_cache_size', 2)
        if self.is_cuda_available:
            self.init_cuda_event(cuda_cache_size)
        self.local_world_size = get_local_size()
        self.local_rank = get_local_rank()
        self.init_falconreader(
            self.rank,
            self.world_size,
            self.local_world_size,  #
            self.local_rank,  #
            self.origin_path_list,
            self.fd_cache_size,
            self.io_thread_num,
            self.io_retry,
            self.cache_name,
            self.chunk_size,
            self.prefetch_chunk_num,
            self.sampler_cfg,
            self.shuffle,
            self.split_path_list_by_rank,
        )

    def init_falconreader(
        self,
        rank,
        world_size,
        local_world_size,
        local_rank,
        path_list,
        fd_cache_size,
        io_thread_num,
        io_retry,
        cache_name,
        chunk_size,
        prefetch_chunk_num,
        sampler_cfg,
        shuffle,
        split_path_list_by_rank,
    ):
        '''init falconreader'''
        self.local_process_num = local_world_size  # the process num in one worker
        self.local_pid = local_rank  # the process pid in one worker
        pid = 0
        prefetch_num = 1
        self.reader = FalconReader(
            path_list,
            fd_cache_size,
            io_thread_num,
            io_retry,
            cache_name,
            self.local_process_num,
            self.local_pid,
            chunk_size,
        )
        sampler_cls = sampler_cfg[0]
        self.chunk_sampler = sampler_cls(
            pid,
            prefetch_num,
            local_rank,
            local_world_size,
            rank,
            world_size,
            shuffle,
            split_path_list_by_rank,
            chunk_size,
            prefetch_chunk_num,
            len(path_list),
            self.reader,
            sampler_cfg[1],
        )

    def reset(self):
        '''reset this dataset, and data will be shuffled,
        and can be read again.
        '''
        self.epoch_count += 1
        if self.stream is None and self.is_cuda_available:
            self.stream = torch.cuda.current_stream()
        self.batch_iter = self._prefetch()

    def next(self):
        '''
        next.
        '''
        retry = self.next_retry
        batch_data, event = None, None
        while retry > 0:
            try:
                batch_data = next(self.batch_iter)
                if self.is_cuda_available:
                    batch_data, is_success, event = self.prefetch_to_cuda(
                        batch_data, retry=self.prefetch_retry
                    )
                    if not is_success:
                        continue
                    batch_data = self.do_cuda_transform(batch_data)
                break
            except Exception:
                pass
            retry -= 1
        if event:
            event.wait()
        return batch_data

    def _prefetch(self):
        '''get every batch for the dataset.'''
        random.seed(self.base_seed + self.epoch_count)
        np.random.seed(self.base_seed + self.epoch_count)
        torch.manual_seed(self.base_seed + self.epoch_count)
        self.chunk_sampler.reset(self.base_seed + self.epoch_count)
        for chunks, path_idxs in self.chunk_sampler:
            kwargs = {
                'local_rank': self.local_rank,
                'local_process_num': self.local_process_num,
                'local_pid': self.local_pid,
            }
            try:
                raw_datas = self.reader.read_many(chunks, True)
            except Exception:
                logging.warning(
                    "rank %d: FalconReader read chunks failed!\n"
                    "Current path is %s, io_thread_num is %d",
                    self.rank,
                    self.origin_path_list[path_idxs[0]],
                    self.io_thread_num,
                    exc_info=True,
                )
                continue
            for path_idx, raw_data in zip(path_idxs, raw_datas):
                kwargs['path_idx'] = path_idx
                for item in raw_data:
                    try:
                        item = self.item_transform(item, **kwargs)
                    except Exception:
                        logging.warning("rank %d: item_transform failed!", self.rank, exc_info=True)
                        continue
                    if item is None:
                        continue
                    batch_data = self.batch_strategy.collate_batch(item, self.max_batch_size)
                    if batch_data is None:
                        # data is discarded or batch is not full
                        continue
                    try:
                        batch_data = self.batch_transforms(batch_data)
                    except Exception:
                        logging.warning("rank %d: item_transform failed!", self.rank, exc_info=True)
                        continue
                    yield batch_data
                    del batch_data
                del raw_data
            del raw_datas

        for data in self.batch_strategy.collect_last_batch():
            if self.drop_last or len(data) == 0:
                continue
            batch_data = self.batch_transforms(data)
            yield batch_data
            del batch_data
