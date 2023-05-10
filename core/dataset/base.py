'''
Base Dataset API.
'''

from abc import abstractmethod
import os
import random
import math
import time
import torch
from dataloader import KVReader
from core.utils.split import split_list
from core.utils import logging
from core.extensions import mpu, AmpEnable
from .batching import get_batch_strategy
from .cuda import to_cuda, pin_memory


class BaseDataset:
    '''Abstract class Base Dataset.

    Dataset can process multi dataset path(list of path).
    '''

    # avoid memory increase due to Arnold dataloader
    os.environ['LIBHDFS_OPTS'] = os.getenv('LIBHDFS_OPTS', '') + ' -Xms1g -Xmx3g'
    os.environ['KRB5CCNAME'] = '/tmp/krb5cc'

    def __init__(
        self,
        path_list,
        bucket_schedule,
        cfg,
        epoch=0,
        rank=0,
        world_size=1,
        shuffle=True,
        split_path_list_by_rank=True,
        max_batch_size=None,
        bucket_schedule_key=None,
    ):
        '''do init.'''
        assert isinstance(path_list, list)
        #  Remove all whitespace characters
        self.origin_path_list = ["".join(path.split()) for path in path_list]
        self.path_list = None
        self.epoch_count = epoch
        self.pipeline = not mpu.is_unitialized()
        self.rank = rank if not self.pipeline else mpu.get_data_parallel_rank()
        self.world_size = world_size if not self.pipeline else mpu.get_data_parallel_world_size()
        self.shuffle = shuffle
        self.split_path_list_by_rank = split_path_list_by_rank
        self.chunk_size = cfg.get('chunk_size', 20)
        self.batch_means_tokens = cfg.get('batch_means_tokens', True)
        self.bucket_schedule_key = cfg.get('bucket_schedule_key', '')
        self.drop_last = cfg.get('drop_last', False)
        self.next_timeout = cfg.get("next_timeout", 1)
        self.next_retry = cfg.get('next_retry', 1200)
        self.use_old_bucket = cfg.get('use_old_bucket', False)
        self.base_seed = cfg.get('seed', 123356)
        self.io_reuse = cfg.get('io_reuse', 1)
        self.item_reuse = cfg.get('item_reuse', 1)
        self.batch_reuse = cfg.get('batch_reuse', 1)
        self.pin_memory = cfg.get('pin_memory', False)
        self.skip_item_num = 0  # start read data idx

        if self.split_path_list_by_rank and len(self.origin_path_list) < self.world_size:
            logging.warning(
                "rank %d HDFSDataset try split path by rank,"
                "but path list less than world size %d,"
                "auto set split index !!!",
                self.rank,
                self.world_size,
            )
            self.split_path_list_by_rank = False

        if max_batch_size is None:
            max_batch_size = cfg.get('max_batch_size', 1)
        if bucket_schedule_key is not None:
            self.bucket_schedule_key = bucket_schedule_key

        if len(bucket_schedule) == 0:
            self.bucket_schedule = None
            self.max_batch_size = max_batch_size
        else:
            self.bucket_schedule = [int(item) for item in bucket_schedule.split(',')]
            if isinstance(max_batch_size, str) and ',' in cfg.max_batch_size:
                # max_batch_size for every bucket is set by user.
                self.max_batch_size = [int(item) for item in max_batch_size.strip().split(',')]
                assert len(self.max_batch_size) == len(self.bucket_schedule)
            else:
                # max_batch_size for every bucket is calculated by max_batch_scale.
                max_batch_scale = cfg.get('max_batch_scale', 0)
                if max_batch_scale != 0:
                    assert self.batch_means_tokens and max_batch_scale > 0
                max_bucket = min(self.bucket_schedule[-1], 2000)
                self.max_batch_size = [
                    max_batch_size + max_batch_scale * max(max_bucket - item, 0)
                    for item in self.bucket_schedule
                ]
        logging.info("rank %d bucket schedule key %r", rank, self.bucket_schedule_key)
        logging.info("rank %d bucket schedule %r", rank, self.bucket_schedule)
        logging.info("rank %d max batch size for every bucket %r", rank, self.max_batch_size)

        logging.info(
            'rank %d env LIBHDFS_OPTS %s, KRB5CCNAME %s',
            rank,
            os.environ['LIBHDFS_OPTS'],
            os.environ['KRB5CCNAME'],
        )
        self.stream = None
        self.events = None
        self.event_idx = 0
        self.device_transforms = None
        self.prefetch_retry = cfg.get('prefetch_retry', 3)
        self.batch_strategy = get_batch_strategy(
            cfg, self.bucket_schedule, bucket_schedule_key=self.bucket_schedule_key
        )

    def split_paths(self):
        '''split path(child dataset) to different rank.'''
        self.path_list = self.origin_path_list.copy()
        random.seed(123356 + self.epoch_count)
        if self.shuffle:
            random.shuffle(self.path_list)
        if self.split_path_list_by_rank:
            # split path list
            self.path_list = split_list(self.path_list, self.world_size)[self.rank]

    def reset_epoch_count(self, epoch_cout, skip_item_num=0):
        '''reset epoch count

        It's for resume.

        Args:
            epoch_cout(int): epoch count
            skip_item_num(int): reset item idx in the epoch
        '''
        self.epoch_count = epoch_cout
        self.skip_item_num = skip_item_num
        logging.all_rank_info(
            "rank %d: %s reset_epoch_count, epoch count %d, skip_item_num %d",
            self.rank,
            self.__class__.__name__,
            self.epoch_count,
            self.skip_item_num,
        )

    def create_sampler(self, lens):
        '''create sampler list.

        Args:
            lens(int): length for a single dataset file.
        Return:
            list of int: sample list of current rank.
        '''
        return self.static_create_sampler(
            lens,
            self.chunk_size,
            self.epoch_count,
            self.shuffle,
            self.rank,
            self.world_size,
            self.split_path_list_by_rank,
        )

    @staticmethod
    def static_create_sampler(
        lens, chunk_size, epoch_count, shuffle, rank, world_size, split_path_list_by_rank
    ):
        '''create sampler list.

        Args:
            lens(int): length for a single dataset file.
            chunk_size(int): chunk size.
            epoch_count(int): current epoch count, for random seed.
            shuffle(bool): whether do shuffle.
            rank(int): rank of communication world.
            world_size(int): world size of communication world.
            split_path_list_by_rank(bool): whether do split path by rank.
                                           Do sample list split if not split_path_list_by_rank.
        Return:
            list of int: sample list of current rank.
        '''
        start_key_num = (lens + chunk_size - 1) // chunk_size
        start_key_list = [i * chunk_size for i in range(start_key_num)]
        random.seed(12345678 + epoch_count)
        if not split_path_list_by_rank:
            start_key_list = split_list(start_key_list, world_size)[rank]
        if shuffle:
            random.shuffle(start_key_list)
        sample_list = []
        for start_key in start_key_list:
            end_key = min(start_key + chunk_size, lens)
            sample_list += range(start_key, end_key)
        return sample_list

    @abstractmethod
    def reset(self):
        '''reset method.

        it means this dataset has been go through.
        do anothor time.
        '''
        raise NotImplementedError

    @abstractmethod
    def next(self):
        '''
        Dataset must be iterable.
        '''
        raise NotImplementedError

    def terminate(self):
        '''
        terminate this dataset.
        '''

    def set_max_batch_size(self, bucket_idx, max_batch_size):
        '''set max batch size'''
        if bucket_idx < 0:
            return
        self.max_batch_size[bucket_idx] = max_batch_size

    def get_max_batch_size(self, bucket_idx):
        '''get max batch size'''
        if bucket_idx < 0:
            return None
        return self.max_batch_size[bucket_idx]

    def get_max_batch_size_list(self):
        '''get max batch size list'''
        return self.max_batch_size

    def round_up_max_batch_size(self):
        '''round up for max batch size list'''
        for idx, item in enumerate(self.max_batch_size):
            self.max_batch_size[idx] = math.ceil(item)

    def get_batch_means_tokens(self):
        '''get batch mean tokens'''
        return self.batch_means_tokens

    @staticmethod
    def get_src_shape(batch_data):
        '''get src shape'''
        if 'src' in batch_data:
            src_shape = batch_data['src'].shape
            return src_shape
        return None

    def set_oom_info(self):
        '''set oom info'''

    @staticmethod
    def path_list_check(path_list, check_path=False, retry=10):
        '''path list check'''
        if not check_path:
            return
        for path in path_list:
            check_retry = retry
            while check_retry > 0:
                try:
                    KVReader(path)
                    break
                except Exception:
                    check_retry -= 1
            if check_retry <= 0:
                raise ValueError('Dataset {} failed'.format(path))

    def prefetch_to_cuda(self, batch_data, retry=3):
        '''to cuda'''
        event = self.get_cuda_event()
        if batch_data is None:
            return batch_data, True, event
        need_retry = True
        while retry > 0 and need_retry:
            try:
                retry -= 1
                new_batch_data = batch_data
                with torch.cuda.stream(self.stream):
                    if self.pin_memory:
                        new_batch_data = pin_memory(new_batch_data)
                    new_batch_data = to_cuda(new_batch_data)
                    event.record(self.stream)
                need_retry = False
            except Exception:
                if retry <= 0:
                    logging.error(
                        'rank %d: %s prefetch to_cuda thread Exception',
                        self.rank,
                        self.__class__.__name__,
                        exc_info=True,
                    )
                else:
                    time.sleep(0.01)  # 10ms.
        return new_batch_data, not need_retry, event

    def do_cuda_transform(self, batch_data):
        '''do cuda transform'''
        if not self.device_transforms:
            return batch_data
        if batch_data is None:
            return batch_data
        # The retry logic is removed because
        # it is executed in the main thread without thread conflict
        try:
            with AmpEnable(enabled=False):
                batch_data = self.device_transforms(batch_data)
        except Exception:
            logging.error(
                'rank %d: %s do cuda transform Exception allocated mem %.2f GB batch %r',
                self.rank,
                self.__class__.__name__,
                torch.cuda.memory_allocated() / (2**30),
                {k: v.shape for k, v in batch_data.items() if hasattr(v, 'shape')},
                exc_info=True,
            )
            return None
        return batch_data

    def init_cuda_event(self, cuda_cache_size):
        '''init cuda events, we will resue them. Add 3 for safety.'''
        self.events = [torch.cuda.Event() for _ in range(cuda_cache_size + 3)]

    def get_cuda_event(self):
        '''get a cuda event to use.'''
        self.event_idx += 1
        if self.event_idx >= len(self.events):
            self.event_idx -= len(self.events)
        return self.events[self.event_idx]

    # pylint: disable=no-self-use
    def state_dict(self):
        '''get dataloader state dict.'''
        return dict()
