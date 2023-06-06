"""
An Mixed Dataloader
"""
import torch
import time
from torch.utils.data import DataLoader

from core.extensions import mpu, AmpEnable
from core.utils import get_dist_info, logging
from core.dataset.batching import get_batch_strategy
from core.dataset.falcon_dataset import FalconDataset
from core.dataset.parquet_dataset import ParquetDataset
from core.dataset.wds_dataset import WdsDataset
from core.dataset.cuda import to_cuda, pin_memory
from samantha.dataio.dataset import MultiIterableDataset, SplitIterableDataset


class MixedDataLoader:
    '''MixedDataset'''

    def __init__(
        self,
        dataset,
        bucket_schedule,
        cfg,
        batch_transforms,
        device_transforms=None,
        rank=0,
        split_each_dataset=False,
    ):
        '''init'''
        self.dataset = dataset
        self.prefetch_worker_num = cfg.get('prefetch_worker_num', 4)
        self.batch_means_tokens = cfg.get('batch_means_tokens', True)
        self.bucket_schedule_key = cfg.get('bucket_schedule_key', '')
        self.drop_last = cfg.get('drop_last', False)
        self.use_old_bucket = cfg.get('use_old_bucket', False)
        max_batch_size = cfg.get('max_batch_size', 1)
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
        self.batch_strategy = get_batch_strategy(cfg, self.bucket_schedule)
        logging.info("bucket schedule %r", self.bucket_schedule)
        logging.info("max batch size for every bucket %r", self.max_batch_size)
        self.dataloader = None
        self.batch_transforms = batch_transforms
        self.pipeline = not mpu.is_unitialized()
        self.rank = rank if not self.pipeline else mpu.get_data_parallel_rank()
        self.pin_memory = cfg.get('pin_memory', False)
        self.stream = None
        self.events = None
        self.event_idx = 0
        self.device_transforms = device_transforms
        self.cuda_cache_size = cfg.get('cuda_cache_size', 2)
        self.init_cuda_event(self.cuda_cache_size)
        self.prefetch_retry = cfg.get('prefetch_retry', 3)
        self.persistent_workers = cfg.get('persistent_workers', True)
        self.split_each_dataset = split_each_dataset
        self.dataloader_state_dict = dict()
        for dataset in self.dataset._datasets:
            dataset_kind = dataset.__class__.__name__
            self.dataloader_state_dict[dataset_kind] = 0

    def update_state_dict(self, data):
        '''flush dataloader state dict'''
        ### update data
        data_state = data.pop('dataset_state', None)
        if data_state:
            # dataset class name
            # this data and skip None data count
            cur_dataset_kind, cur_data_cnt, _path_idx = data_state
            self.dataloader_state_dict[cur_dataset_kind] += cur_data_cnt

    def __iter__(self):
        '''iter'''
        if self.dataloader is None:
            persistent_workers = self.prefetch_worker_num > 0
            self.dataloader = DataLoader(
                self.dataset,
                num_workers=self.prefetch_worker_num,
                batch_size=None,
                collate_fn=MixedDataLoader._collate_fn,
                persistent_workers=self.persistent_workers,
            )

        # flash dataloader state dict
        for dataset_kind in self.dataloader_state_dict:
            self.dataloader_state_dict[dataset_kind] = 0

        for data in self.dataloader:
            # flush dataloader state
            # TODO: add there or add after draw batch
            self.update_state_dict(data)
            if data is None and self.split_each_dataset:
                for data in self.batch_strategy.collect_last_batch():
                    if not self.drop_last and len(data) > 0:
                        try:
                            batch_data = self.batch_transforms(data)
                        except Exception:
                            logging.warning(
                                "rank %d: batch transforms failed", self.rank, exc_info=True
                            )
                            continue
                        yield self._prefetch_batch(batch_data)
                yield
                continue

            batch_data = self.batch_strategy.collate_batch(data, self.max_batch_size)
            if batch_data is None:
                # data is discarded or batch is not full
                continue
            try:
                batch_data = self.batch_transforms(batch_data)
            except Exception:
                logging.warning("rank %d: batch transforms failed.", self.rank, exc_info=True)
                # skip this bucket batch
                continue
            yield self._prefetch_batch(batch_data)

        for data in self.batch_strategy.collect_last_batch():
            if not self.drop_last and len(data) > 0:
                try:
                    batch_data = self.batch_transforms(data)
                except Exception:
                    logging.warning("rank %d: batch transforms failed", self.rank, exc_info=True)
                    continue
                yield self._prefetch_batch(batch_data)

    @staticmethod
    def _collate_fn(item):
        return item

    def init_cuda_event(self, cuda_cache_size):
        '''init cuda events, we will resue them. Add 3 for safety.'''
        self.events = [torch.cuda.Event() for _ in range(cuda_cache_size + 3)]

    def get_cuda_event(self):
        '''get a cuda event to use.'''
        self.event_idx += 1
        if self.event_idx >= len(self.events):
            self.event_idx -= len(self.events)
        return self.events[self.event_idx]

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

    def _prefetch_batch(self, batch_data):
        '''
        do thread prefetch.
        '''

        batch_data, _is_success, event = self.prefetch_to_cuda(
            batch_data, retry=self.prefetch_retry
        )
        batch_data = self.do_cuda_transform(batch_data)
        event.wait()
        return batch_data

    def __del__(self):
        '''del
        we need del dataloader because we set persistent_workers=True'''
        if self.dataloader is not None:
            del self.dataloader

class MixedHDFSDataset:
    '''
    MixedHDFSDatset which has same api with HDFSDataset
    '''

    def __init__(
        self,
        path_list,
        bucket_schedule,
        cfg,
        item_transform,
        batch_transforms,
        device_transforms=None,
        split_path_list_by_rank=True,
        epoch_count=0,
        shuffle=False,
    ):
        '''init func'''
        self.rank, self.world_size = get_dist_info()
        self.cfg = cfg
        self.origin_path_list = path_list  # mixed
        # Distinguish file formats by file name
        self.init_path_list(path_list)
        self.bucket_schedule = bucket_schedule
        self.item_transform = item_transform
        self.batch_transforms = batch_transforms
        self.device_transforms = device_transforms
        self.num_samples = cfg.get('num_samples', -1)
        self.weight = cfg.get('weight', None)
        self.split_path_list_by_rank = split_path_list_by_rank
        self.shuffle = shuffle
        self.epoch_count = epoch_count
        self.skip_item_num = 0
        self.weight = self.cfg.get('weight', dict())
        self.dataloader = None
        self._state = dict()

    def init_path_list(self, path_list):
        '''filter parquet file list'''
        self.kv_path_list = []
        self.parquet_path_list = []
        self.wds_path_list = []
        for path in path_list:
            if '.parquet' in path:
                self.parquet_path_list.append(path)
            elif '.tar' in path:
                self.wds_path_list.append(path)
            else:
                self.kv_path_list.append(path)

    def build_dataset(self):
        '''build dataset'''
        datasets = []
        weights = []
        if self.kv_path_list:
            # build falcon dataset
            falcon_dataset = FalconDataset(
                path_list=self.kv_path_list,
                cfg=self.cfg,
                item_transform=self.item_transform,
                rank=self.rank,
                world_size=self.world_size,
                shuffle=self.shuffle,
                split_path_list_by_rank=self.split_path_list_by_rank,
            )
            datasets.append(falcon_dataset)
            weights.append(self.weight.get('kv', None))

        # build parquet dataset
        if self.parquet_path_list:
            parquet_dataset = ParquetDataset(
                path_list=self.parquet_path_list,
                cfg=self.cfg,
                item_transform=self.item_transform,
                rank=self.rank,
                world_size=self.world_size,
                shuffle=self.shuffle,
                split_path_list_by_rank=self.split_path_list_by_rank,
            )
            datasets.append(parquet_dataset)
            weights.append(self.weight.get('parquet', None))

        # build web dataset
        if self.wds_path_list:
            web_dataset = WdsDataset(
                path_list=self.wds_path_list,
                cfg=self.cfg,
                item_transform=self.item_transform,
                rank=self.rank,
                world_size=self.world_size,
                shuffle=self.shuffle,
                split_path_list_by_rank=self.split_path_list_by_rank,
            )

            datasets.append(web_dataset)
            weights.append(self.weight.get('wds', None))

        if sum([w is None for w in weights]) > 0:
            weights = None
        return datasets, weights

    def build_dataloader(self):
        '''build dataloader'''
        datasets, weights = self.build_dataset()

        # dataset reset
        for dataset in datasets:
            dataset_kind = dataset.__class__.__name__
            dataset.reset(self.epoch_count, self._state.get(dataset_kind, 0))
            self._state[dataset_kind] = 0

        multi_iterable_dataset = MultiIterableDataset(
            datasets, num_samples=self.num_samples, weights=weights
        )

        self.dataloader = MixedDataLoader(
            dataset=multi_iterable_dataset,
            bucket_schedule=self.bucket_schedule,
            cfg=self.cfg,
            batch_transforms=self.batch_transforms,
            device_transforms=self.device_transforms,
            rank=self.rank,
        )

    def reset(self):
        '''reset'''
        logging.all_rank_info(
            "rank %d: %s reset, epoch count %d, resume_state_dict %s",
            self.rank,
            self.__class__.__name__,
            self.epoch_count,
            self._state,
        )
        if self.dataloader is None:
            self.build_dataloader()
        self.data_generator = iter(self.dataloader)
        self.epoch_count += 1

    def reset_epoch_count(self, epoch_cout, state_dict=0):
        '''reset epoch count

        It's for resume.

        Args:
            epoch_cout(int): epoch count
            skip_item_num(int): reset item idx in the epoch
        '''
        self.epoch_count = epoch_cout
        self._state = state_dict
        logging.all_rank_info(
            "rank %d: %s reset_epoch_count, epoch count %d, resume dataloader state %s",
            self.rank,
            self.__class__.__name__,
            self.epoch_count,
            self._state,
        )

    def terminate(self):
        '''terminate'''
        logging.all_rank_info("rank %d: MixedHDFSDataset terminate", self.rank)
        del self.dataloader

    def state_dict(self):
        '''state dict'''
        return self.dataloader.dataloader_state_dict

    def set_oom_info(self):
        '''see oom info'''
        return

    def next(self):
        '''next'''
        try:
            data = next(self.data_generator)
            return data
        except StopIteration:
            return None


class MixedValidHDFSDataset(MixedHDFSDataset):
    '''ValidHDFSDataset support reading multi datasets'''

    def __init__(
        self,
        path_list,
        bucket_schedule,
        cfg,
        item_transform,
        batch_transforms,
        device_transforms=None,
        split_path_list_by_rank=True,
        epoch_count=0,
        shuffle=False,
        split_each_dataset=False,
    ):
        cfg = cfg.copy()
        self.split_each_dataset = split_each_dataset
        cfg.prefetch_worker_num = 1
        cfg.persistent_workers = False
        cfg.global_shuffle = False
        cfg.cache_name = cfg.get('valid_cache_name', 'falcon_dataset_valid')
        super().__init__(
            path_list,
            bucket_schedule,
            cfg,
            item_transform,
            batch_transforms,
            device_transforms,
            split_path_list_by_rank,
            epoch_count,
            shuffle,
        )

    def build_dataset(self):
        '''build dataset'''
        if not self.split_each_dataset:
            return super().build_dataset()
        datasets = []
        weights = []
        for kv_path in self.kv_path_list:
            falcon_dataset = FalconDataset(
                path_list=[kv_path],
                cfg=self.cfg,
                item_transform=self.item_transform,
                rank=self.rank,
                world_size=self.world_size,
                shuffle=self.shuffle,
                split_path_list_by_rank=False,
            )
            datasets.append(falcon_dataset)

        # build parquet dataset
        for parquet_path in self.parquet_path_list:
            parquet_dataset = ParquetDataset(
                path_list=[parquet_path],
                cfg=self.cfg,
                item_transform=self.item_transform,
                rank=self.rank,
                world_size=self.world_size,
                shuffle=self.shuffle,
                split_path_list_by_rank=False,
            )
            datasets.append(parquet_dataset)

        return datasets, weights

    def build_dataloader(self):
        if not self.split_each_dataset:
            return super().build_dataloader()

        datasets, _weights = self.build_dataset()

        split_iterable_dataset = SplitIterableDataset(datasets)

        self.dataloader = MixedDataLoader(
            dataset=split_iterable_dataset,
            bucket_schedule=self.bucket_schedule,
            cfg=self.cfg,
            batch_transforms=self.batch_transforms,
            device_transforms=self.device_transforms,
            rank=self.rank,
            split_each_dataset=self.split_each_dataset,
        )
