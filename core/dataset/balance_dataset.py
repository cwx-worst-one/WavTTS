'''
Balanced HDFS Dataset. Used to SID mainly.
'''

import queue as tq
import random
import os
import multiprocessing as mp
from threading import Thread
import torch
from dataloader import FalconReader
from core.utils import get_dist_info
from core.dataset.sampler import BalancedSampler
from core.utils.dist_util import get_local_rank, get_local_size
from core.utils import logging
from .base import BaseDataset
from .queue_utils import safe_get, safe_put


class BalancedHDFSDataset(BaseDataset):
    '''
    balanced dataset for identity task.
    '''

    def __init__(
        self,
        path_list,
        cfg,
        item_transform,
        batch_transforms,
        device_transforms=None,
        epoch_count=0,
        none_after_epoch=False,
    ):
        super().__init__(
            path_list, '', cfg, epoch_count, split_path_list_by_rank=False, *get_dist_info()
        )
        self.batch_size_per_class = cfg.batch_size_per_class
        self.batch_class_num = cfg.batch_class_num

        self.prefetch_worker_num = cfg.get('prefetch_worker_num', 3)
        self.preprocess_worker_num = cfg.get('preprocess_worker_num', 1)
        # for FalconReader
        self.io_thread_num = cfg.get('io_thread_num', 12)
        self.io_retry = cfg.get('io_retry', 5)
        self.fd_cache_size = cfg.get('fd_cache_size', 2048)
        self.cache_name = cfg.get('balance_cache_name', 'balance_train')
        # Due to the particularity of the BalanceDataset, chunk_size must be set to 1.
        # In order to avoid setting chunk_size in config, it is temporarily fixed to 1.
        self.chunk_size = 1
        self.cuda_cache_size = cfg.get('cuda_cache_size', 2)
        self.proc_cache_size = cfg.get('proc_cache_size', 16)
        self.rand_seed = cfg.get('rand_seed', 13411)
        self.process_mutex = cfg.get('process_mutex', False)
        self.item_transform = item_transform
        self.batch_transforms = batch_transforms
        self.device_transforms = device_transforms

        self._launched = False
        self.mp_manager = None
        self.stop_queue = None
        self.raw_data_queues = []
        self.batch_queue = None
        self.thread_queue = None
        self.all_procs = []
        self.init_cuda_event(self.cuda_cache_size)

        # If the runner don't have iters_per_epoch arg,
        # we need to put a None after the end of sampler's iter
        # to help the dataset recognize the end of epoch.
        self.none_after_epoch = none_after_epoch
        # To working with none_after_epoch, the sum of sampler workers'
        # iteration should be a full dataset.
        self.balance_epoch = cfg.get('balance_epoch', False)

    def reset(self):
        '''reset this dataset, and data will be shuffled,
        and can be read again.
        do nothing, but logging. do split path in prefetch process.
        '''
        if not self._launched:
            self._launched = True
            self._launch_process()

        if self.stream is None and torch.cuda.is_available():
            self.stream = torch.cuda.current_stream()
        logging.all_rank_info(
            "rank %d: Balanced HDFS dataset reset, epoch count %d", self.rank, self.epoch_count
        )
        self.epoch_count += 1

    def next(self):
        '''iter the dataset.'''
        retry = 60
        while retry > 0:
            try:
                retry -= 1
                data = self.thread_queue.get(block=True, timeout=30)
            except tq.Empty:
                if not self.stop_queue.empty():
                    break
                continue
            data, is_success, event = self.prefetch_to_cuda(data, retry=self.prefetch_retry)
            data = self.do_cuda_transform(data)
            if event:
                event.wait()

            if not is_success:
                continue

            return data
        logging.error("rank: %d: Data loading failed.", self.rank)
        return None

    def _launch_process(self):
        '''launch batching process.'''
        self.mp_manager = mp.Manager()

        # for reset
        self.stop_queue = self.mp_manager.Queue()

        # IO
        # Each raw_data_queue contains a list of data
        self.raw_data_queues = [
            self.mp_manager.Queue(self.proc_cache_size) for _ in range(self.prefetch_worker_num)
        ]
        data_prefetch_procs = [
            mp.Process(
                target=self._prefetch,
                args=(
                    self.raw_data_queues[pid],
                    self.stop_queue,
                    pid,
                    self.origin_path_list,
                    # for FalconReader
                    self.fd_cache_size,
                    self.io_thread_num,
                    self.io_retry,
                    self.chunk_size,
                    self.cache_name,
                    get_local_rank(),
                    get_local_size(),
                    self.batch_size_per_class,
                    self.batch_class_num,
                    self.rand_seed,
                    self.prefetch_worker_num,
                    self.rank,
                    self.world_size,
                    self.item_transform,
                    self.epoch_count,
                    self.process_mutex,
                    self.io_reuse,
                    self.item_reuse,
                    self.none_after_epoch,
                    self.balance_epoch,
                ),
            )
            for pid in range(self.prefetch_worker_num)
        ]
        for proc in data_prefetch_procs:
            proc.daemon = True
            proc.start()

        # batch
        self.batch_queue = torch.multiprocessing.Queue(
            self.prefetch_worker_num * self.proc_cache_size
        )
        data_preprocess_procs = [
            mp.Process(
                target=self._preprocess,
                args=(
                    self.stop_queue,
                    pid,
                    self.raw_data_queues,
                    self.batch_queue,
                    self.batch_size_per_class,
                    self.batch_class_num,
                    self.rand_seed,
                    self.preprocess_worker_num,
                    self.rank,
                    self.batch_transforms,
                    self.batch_reuse,
                    self.prefetch_worker_num,
                ),
            )
            for pid in range(self.preprocess_worker_num)
        ]
        for proc in data_preprocess_procs:
            proc.daemon = True
            proc.start()

        self.thread_queue = tq.Queue(self.cuda_cache_size)
        prefetch_thread = Thread(target=self._cuda_prefetch)
        prefetch_thread.daemon = True
        prefetch_thread.start()

        # launch all child process
        self.all_procs = data_prefetch_procs + data_preprocess_procs

    @staticmethod
    def _prefetch(
        data_queue,
        stop_queue,
        pid,
        path_list,
        # for FalconReader
        fd_cache_size,
        io_thread_num,
        io_retry,
        chunk_size,
        cache_name,
        local_rank,
        local_world_size,
        batch_size_per_class,
        batch_class_num,
        rand_seed,
        prefetch_worker_num,
        rank,
        world_size,
        item_transform,
        epoch_count,
        process_mutex,
        io_reuse,
        batch_reuse,
        none_after_epoch,
        balance_epoch,
    ):
        '''
        do prefetch in child process.
        Args:
            data_queue: The queue that saves the training data.
                        The data consists of (class, feats*n), where n is the
                        number of samples per speaker in one batch.
            stop_queue: An indicator that stop the reading process.
            pid(int): prefetch process index.
            batch_size_per_class: The num of egs per class.
            rand_seed: The random seed.
            prefetch_worker_num: The num of workers.
            batch_class_num: The num of class in one batch
            rank: The global rank id.
            world_size: The total num of ranks
            item_transform: The item transform called when loading the data
            process_mutex: different processes fetch different class
        '''
        # pylint:disable=too-many-locals
        seed = rand_seed
        random.seed(seed + rank * prefetch_worker_num + pid)
        logging.get_logger(log_level="WARNING")
        local_process_num = local_world_size * prefetch_worker_num  # the process num in one worker
        local_pid = local_rank * prefetch_worker_num + pid  # the process pid in one worker
        reader = FalconReader(
            path_list,
            fd_cache_size,
            io_thread_num,
            io_retry,
            cache_name,
            local_process_num,
            local_pid,
            chunk_size,
        )

        balanced_sampler = BalancedSampler(
            pid,
            prefetch_worker_num,
            local_rank,
            local_world_size,
            rank,
            world_size,
            reader,
            batch_size_per_class,
            batch_class_num,
            cache_name,
            list(range(len(path_list))),
            process_mutex,
            balance_epoch,
        )
        # Make sure when training is resumed, the epoch_count won't be the same as the one
        # before resuming
        epoch_count *= 100
        while stop_queue.empty():
            balanced_sampler.reset(seed + epoch_count)
            for clss_buf, keys_buf in balanced_sampler:
                BalancedHDFSDataset._real_prefetch(
                    reader,
                    keys_buf,
                    clss_buf,
                    batch_size_per_class,
                    stop_queue,
                    data_queue,
                    item_transform,
                    rank,
                    pid,
                    io_reuse,
                    batch_reuse,
                )
            epoch_count += 1
            if none_after_epoch:
                safe_put(stop_queue, data_queue, None)

    @staticmethod
    def _real_prefetch(
        reader,
        keys_buf,
        cls_buf,
        batch_size_per_class,
        stop_queue,
        data_queue,
        item_transform,
        rank,
        pid,
        io_reuse,
        item_reuse,
    ):
        '''
        prefetch for hdfs data.
        Args:
            reader: FalconReader
            keys_buf: The utt-id list
            cls_buf: The speaker name list
            batch_size_per_class:
            stop_queue:
            data_queue:
            item_transform:
        '''
        try:
            # the return value is list of list
            # so there contains a trans
            vals = sum(reader.read_many(keys_buf, True), [])
            if io_reuse > 1:
                vals = [bytes(v) for v in vals]
            vals = [
                vals[i * batch_size_per_class : (i + 1) * batch_size_per_class]
                for i in range(len(cls_buf))
            ]
        except Exception:
            logging.warning(
                "rank %d: prefetch %d, read from KVPeader error", rank, pid, exc_info=True
            )
            del keys_buf, cls_buf
            return
        buf = []
        if io_reuse > 1:
            cls_buf *= io_reuse
            vals *= io_reuse
        for cs, vs in zip(cls_buf, vals):
            datas = []
            for v in vs:
                try:
                    v = item_transform(v)
                    if v is None:
                        continue
                    v.pop('file_type', None)
                except Exception:
                    logging.warning(
                        "rank %d: prefetch %d, item transform error", rank, pid, exc_info=True
                    )
                    break
                datas.append(v)
                del v
            if len(datas) == batch_size_per_class:
                buf.append((cs, datas))
            del datas, vs
        if len(buf) > 0:
            for _ in range(item_reuse):
                safe_put(stop_queue, data_queue, buf)
        del vals, buf
        del keys_buf, cls_buf

    @staticmethod
    def _preprocess(
        stop_queue,
        pid,
        data_queues,
        batch_queue,
        _batch_size_per_class,
        _batch_class_num,
        rand_seed,
        preprocess_worker_num,
        rank,
        batch_transforms,
        batch_reuse,
        prefetch_worker_num,
    ):
        '''
        preprocess the data and form them into batches.

        Args:
            stop_queue:
            pid:
            data_queues: Multiple data queues with data loaded from prefetch workers
            batch_queue: Feed the queue with the output batched data
            batch_class_num: The num of classes in a batch
            rand_seed:
            preprocess_worker_num:
            rank:
            batch_transform: The transform that performs during the batch packing.
        '''
        seed = rand_seed + rank * preprocess_worker_num + pid
        random.seed(seed)
        logging.get_logger(log_level="WARNING")
        torch.set_num_threads(1)
        # the number of None received from workers
        end_num = 0

        # worker_index = 0
        while stop_queue.empty():
            # Sample the data_queues iteratively (do not want to sample the same
            # queue continuously.) But this may not be hold anytime, since one
            # worker could be blocked and there is no data in the corresponding
            # queues.
            dq = random.choice(data_queues)
            # dq = data_queues[worker_index]
            # worker_index = (worker_index + 1) % len(data_queues)
            data_buf = safe_get(stop_queue, dq, timeout=0.001, retry=1)
            # receive an end flag of a worker
            if data_buf is None:
                end_num += 1
                # Because the process of all worker are the same,
                # There is no need to considering the pid
                if end_num == prefetch_worker_num:
                    end_num = 0
                    safe_put(stop_queue, batch_queue, None)
                continue
            # failed to get the data
            if not data_buf:
                continue
            batch_data = []
            clss_set = []  # unused
            for clss, datas in data_buf:
                batch_data += datas
                clss_set.append(clss)
            del clss_set, data_buf

            # got a batch
            # batch check guaranteed in sampler class
            batch_data = batch_transforms(batch_data)
            for _ in range(batch_reuse):
                safe_put(stop_queue, batch_queue, batch_data)
            del batch_data

    def _cuda_prefetch(self):
        '''
        do cuda prefetch.
        '''
        while self.stop_queue.empty():
            batch_data = safe_get(self.stop_queue, self.batch_queue, retry=8000)
            if isinstance(batch_data, tuple) and len(batch_data) == 0:
                logging.error(
                    "rank %d: BalancedHDFSDataset prefetch_batch timeout, retry...", self.rank
                )
                continue

            safe_put(self.stop_queue, self.thread_queue, batch_data)

    def terminate(self):
        '''terminate'''
        if self._launched:
            self.stop_queue.put(1)
        for proc in self.all_procs:
            proc.join(2)
        self._launched = False
        os.system('rm -rf /tmp/falconreader/{}'.format(self.cache_name))
        os.system('rm -rf /dev/shm/falconreader_*')
