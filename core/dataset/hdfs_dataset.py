"""
HDFS dataset.
"""

import os
import multiprocessing as mp
from threading import Thread
import queue as tq
import random
import numpy as np
import torch

from dataloader import FalconReader
from core.utils import logging
from core.dataset.sampler import setup_sampler_cfg
from core.utils.dist_util import get_local_rank, get_local_size
from core.utils import get_dist_info
from core.utils.split import split_list
from .base import BaseDataset
from .adaptive_batch import ProcessedAdaptiveBatch
from .queue_utils import safe_get, safe_put
from .cuda import to_cuda


class ProcessedDataset(BaseDataset):
    """
    base Dataset with subprocess.
    """

    PREFETCH_GET_RETRY = 60 * 60 * 5  # 5 hours

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
        max_batch_size=None,
        bucket_schedule_key=None,
    ):
        super().__init__(
            path_list,
            bucket_schedule,
            cfg,
            epoch_count,
            *get_dist_info(),
            shuffle,
            split_path_list_by_rank,
            max_batch_size=max_batch_size,
            bucket_schedule_key=bucket_schedule_key,
        )
        self.item_transform = item_transform
        self.batch_transforms = batch_transforms
        self.device_transforms = device_transforms
        self.proc_cache_size = cfg.get("proc_cache_size", 4)
        self.cuda_cache_size = cfg.get("cuda_cache_size", 2)
        self.io_thread_num = cfg.get("io_thread_num", 12)
        self.io_retry = cfg.get("io_retry", 5)
        self.fd_cache_size = cfg.get("fd_cache_size", 8000)
        self.chunk_size = cfg.get("chunk_size", 40)
        self.prefetch_chunk_num = cfg.get("prefetch_chunk_num", 40)
        self.prefetch_torch_thread = cfg.get("prefetch_torch_thread", 4)

        self._launched = False
        self.stop_queue = None
        self.batch_queue = None
        self.procs = []
        self.thread_queue = None
        self.prefetch_thread = None
        self.path_list_check(path_list, cfg.get("check_path", False))
        self.init_cuda_event(self.cuda_cache_size)

    def _launch_process(self):
        """launch sub process."""
        raise NotImplementedError

    def reset(self):
        """
        reset this dataset.
        """
        if self.stream is None:
            self.stream = torch.cuda.current_stream()

        if not self._launched:
            self._launched = True
            self._launch_process()

        self.epoch_count += 1

    def next(self):
        """iter the dataset."""
        retry = self.next_retry
        while retry > 0:
            try:
                retry -= 1
                data, event = self.thread_queue.get(block=True, timeout=self.next_timeout)
                data = self.do_cuda_transform(data)
            except tq.Empty:
                continue
            event.wait()
            return data
        return None

    def _prefetch_batch(self):
        """
        do thread prefetch.
        """
        while self.stop_queue.empty():
            batch_data = safe_get(self.stop_queue, self.batch_queue, retry=self.PREFETCH_GET_RETRY)
            if isinstance(batch_data, tuple) and len(batch_data) == 0:
                logging.error(
                    "rank %d: %s prefetch_batch timeout, retry again",
                    self.rank,
                    self.__class__.__name__,
                )
                continue

            batch_data, is_success, event = self.prefetch_to_cuda(
                batch_data, retry=self.prefetch_retry
            )
            if not is_success:
                continue

            safe_put(self.stop_queue, self.thread_queue, (batch_data, event))

    def terminate(self):
        """terminate"""
        if self._launched:
            self.stop_queue.put(1)
        for proc in self.procs:
            proc.join(10)
        self._launched = False


class HDFSDataset(ProcessedDataset):
    """HDFSDataset based on ProcessedDataset

    Example::

        root_path = 'hdfs://haruna/home/byte_arnold_hl_speech_asr/user' +
                    '/jiangbo/arnold_hdfs_data/ag20kh_bpe'
        train_path_list = [f'{root_path}/shard{i}' for i in range(1024)]
        train_batch_dataset = HDFSDataset(
                                         train_path_list,
                                         dataset_cfg,
                                         self.parse_fn,
                                         self.draw_batch_fn,
                                         split_path_list_by_rank=True,
                                         shuffle=True,
                                        )

        when the train is done, please call:
        `
        self.train_batch_dataset.terminate()
        `

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
        epoch_count=0,
        shuffle=False,
        max_batch_size=None,
        bucket_schedule_key=None,
        cache_name=None,
    ):
        super().__init__(
            path_list,
            bucket_schedule,
            cfg,
            item_transform,
            batch_transforms,
            device_transforms,
            split_path_list_by_rank=split_path_list_by_rank,
            epoch_count=epoch_count,
            shuffle=shuffle,
            max_batch_size=max_batch_size,
            bucket_schedule_key=bucket_schedule_key,
        )
        self.prefetch_worker_num = cfg.get('prefetch_worker_num', 4)
        self.deterministic = cfg.get('deterministic', False)
        if cache_name is None:
            self.cache_name = cfg.get('cache_name', 'hdfs_train')
        else:
            self.cache_name = cache_name
        self.sampler_cfg = setup_sampler_cfg(cfg)
        logging.info(
            "rank %d: HDFSDataset prefetch_worker_num %d  proc_cache_size %d "
            "cuda_cache_size %d prefetch_chunk_num %d deterministic %r",
            self.rank,
            self.prefetch_worker_num,
            self.proc_cache_size,
            self.cuda_cache_size,
            self.prefetch_chunk_num,
            self.deterministic,
        )
        self.raw_data_queues = []
        self.batch_size_queues = []
        self.adaptive_batch_enble = cfg.get("adaptive_batch_size", False)
        if self.adaptive_batch_enble:
            self.adaptive_batch = ProcessedAdaptiveBatch(
                cfg, len(bucket_schedule), self.adaptive_batch_enble
            )
        # data_count is a process shared variable that records the number of data
        # read by all sub processes on each card.
        # When resuming, it will be divided evenly by the sampler
        self.data_count = mp.Manager().dict()
        self.dataset_weights = cfg.get("dataset_weights", None)
        self.dataset_length = cfg.get("dataset_length", None)
        if self.dataset_weights:
            assert self.dataset_length is not None
            assert len(self.dataset_weights) == len(self.dataset_length)
            assert sum(self.dataset_length) == len(self.origin_path_list)
            logging.info(
                "rank %d: HDFSDataset dataset weights: %r",
                self.rank,
                self.dataset_weights,
            )
            logging.info(
                "rank %d: HDFSDataset dataset lengths: %r",
                self.rank,
                self.dataset_length,
            )

    def reset(self):
        """
        reset this dataset.
        """
        logging.all_rank_info(
            "rank %d: %s reset, epoch count %d, skip_item_num %d",
            self.rank,
            self.__class__.__name__,
            self.epoch_count,
            self.skip_item_num,
        )
        self.data_count["get"] = self.skip_item_num
        self.data_count["save"] = self.skip_item_num
        self.skip_item_num = 0
        super().reset()

    def terminate(self):
        super().terminate()
        os.system("rm -rf /tmp/falconreader/{}".format(self.cache_name))
        os.system("rm -rf /dev/shm/falconreader_*")

    def state_dict(self):
        # use process0
        inner_data_count = self.data_count["save"]
        state_dict = {"inner_data_count": inner_data_count}
        return state_dict

    def next(self):
        """iter the dataset."""
        retry = self.next_retry
        if self.adaptive_batch_enble:
            self.adaptive_batch.before_iter(self)
        while retry > 0:
            try:
                retry -= 1
                data, event = self.thread_queue.get(block=True, timeout=self.next_timeout)
                data = self.do_cuda_transform(data)
                if self.adaptive_batch_enble and self.adaptive_batch.is_run:
                    src_shape = self.get_src_shape(data)
                    if src_shape is not None:
                        self.adaptive_batch.set_batch_info("batch_num", src_shape[0])
                        self.adaptive_batch.set_batch_info("batch_size", src_shape[1])
            except tq.Empty:
                continue
            event.wait()
            return data
        return None

    def _launch_process(self):
        """launch child process."""
        preprocess_worker_num = 1
        proc_cache_size = self.proc_cache_size

        # for reset
        self.stop_queue = mp.Queue()

        # FalconReader
        self.raw_data_queues = [mp.Queue(proc_cache_size) for _ in range(self.prefetch_worker_num)]
        data_prefetch_procs = [
            mp.Process(
                target=self._prefetch,
                args=(
                    self.raw_data_queues[pid],
                    self.stop_queue,
                    self.origin_path_list,
                    self.dataset_weights,
                    self.dataset_length,
                    # world info.
                    pid,
                    self.prefetch_worker_num,
                    get_local_rank(),
                    get_local_size(),
                    self.rank,
                    self.world_size,
                    # for reader
                    self.chunk_size,
                    self.cache_name,
                    self.io_thread_num,
                    self.fd_cache_size,
                    self.io_retry,
                    self.prefetch_chunk_num,
                    # sampler
                    self.sampler_cfg,
                    self.shuffle,
                    self.split_path_list_by_rank,
                    self.epoch_count,
                    self.base_seed,
                    # item trans
                    self.prefetch_torch_thread,
                    self.item_transform,
                    self.io_reuse,
                    self.item_reuse,
                    # for resume
                    self.data_count,  # for dataloader resume
                ),
            )
            for pid in range(self.prefetch_worker_num)
        ]
        for proc in data_prefetch_procs:
            proc.daemon = True
            proc.start()

        # transform, bucket, batch
        self.batch_queue = torch.multiprocessing.Queue(self.prefetch_worker_num * proc_cache_size)
        if self.adaptive_batch_enble:
            batch_adjust_num = self.adaptive_batch.get_batch_adjust_num()
            self.batch_size_queues = [
                mp.Queue(batch_adjust_num + 5) for _ in range(preprocess_worker_num)
            ]
        else:
            self.batch_size_queues = [[] for _ in range(preprocess_worker_num)]
        data_preprocess_procs = [
            mp.Process(
                target=self._preprocess,
                args=(
                    self.stop_queue,
                    self.raw_data_queues,
                    self.batch_queue,
                    self.prefetch_worker_num,
                    self.max_batch_size,
                    self.drop_last,
                    self.batch_transforms,
                    self.batch_size_queues[pid],
                    self.adaptive_batch_enble,
                    self.deterministic,
                    self.rank,
                    self.batch_reuse,
                    self.batch_strategy,
                ),
            )
            for pid in range(preprocess_worker_num)
        ]
        for proc in data_preprocess_procs:
            proc.daemon = True
            proc.start()

        # launch all child process
        self.procs = data_prefetch_procs + data_preprocess_procs

        self.thread_queue = tq.Queue(self.cuda_cache_size)
        self.prefetch_thread = Thread(target=self._prefetch_batch)
        # set thread daemon, No need to join this thread when terminate.
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    @staticmethod
    def _prefetch(
        data_queue,
        stop_queue,
        path_list,
        dataset_weights,
        dataset_length,
        pid,
        prefetch_num,
        local_rank,
        local_world_size,
        rank,
        world_size,
        chunk_size,
        cache_name,
        io_thread_num,
        fd_cache_size,
        io_retry,
        prefetch_chunk_num,
        sampler_cfg,
        shuffle,
        split_path_list_by_rank,
        epoch_count,
        base_seed,
        prefetch_torch_thread,
        item_transform,
        io_reuse,
        item_reuse,
        data_count,
    ):
        """
        do prefetch in child process.
        Args:
            ds_id(int): prefetch process index.
        """
        # pylint:disable=too-many-branches
        # pylint:disable=too-many-locals
        logging.get_logger(log_level="WARNING")
        torch.set_num_threads(prefetch_torch_thread)

        local_process_num = local_world_size * prefetch_num  # the process num in one worker
        local_pid = local_rank * prefetch_num + pid  # the process pid in one worker
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
        sampler_cls = sampler_cfg[0]
        chunk_sampler = sampler_cls(
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
            reader,
            sampler_cfg[1],
            dataset_weights,
            dataset_length,
        )
        skip_item_num = data_count["get"]
        while stop_queue.empty():
            # setup seed
            seed = base_seed + epoch_count
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # get my path
            chunk_sampler.reset(seed, skip_num=skip_item_num)
            for chunks, path_idxs in chunk_sampler:
                if not stop_queue.empty():
                    break
                kwargs = {
                    "local_rank": local_rank,
                    "pid": pid,
                    "local_process_num": local_process_num,
                    "local_pid": local_pid,
                }
                try:
                    raw_datas = reader.read_many(chunks, True)
                except Exception:
                    logging.warning(
                        "rank %d: prefetch %d FalconReader read chunks failed!\n"
                        "Current path is %s, io_thread_num is %d",
                        rank,
                        pid,
                        path_list[path_idxs[0]],
                        io_thread_num,
                        exc_info=True,
                    )
                    continue
                datas = []
                for path_idx, raw_data in zip(path_idxs, raw_datas):
                    kwargs["path_idx"] = path_idx
                    # only rank0 will save checkpoints
                    if local_rank == 0:
                        data_count["save"] += len(raw_data)
                    if io_reuse > 1:
                        raw_data = [bytes(item) for item in raw_data]
                        raw_data *= io_reuse
                    for item in raw_data:
                        try:
                            item = item_transform(item, **kwargs)
                        except Exception:
                            logging.warning(
                                "rank %d: prefetch %d item_transform failed!",
                                rank,
                                pid,
                                exc_info=True,
                            )
                            continue
                        if item is not None:
                            datas.append(item)
                        del item
                    del raw_data
                    if datas:
                        for _ in range(item_reuse):
                            safe_put(stop_queue, data_queue, datas)
                        datas = []
                del raw_datas
                if datas:
                    for _ in range(item_reuse):
                        safe_put(stop_queue, data_queue, datas)
                del datas

            epoch_count += 1
            skip_item_num = 0
            safe_put(stop_queue, data_queue, None)

    @staticmethod
    def init_epoch_start_state(deterministic, prefetch_worker_num):
        """init state for queue choice."""
        pids = list(range(prefetch_worker_num))
        if deterministic:
            return (0, pids)
        return (set(), pids)

    @staticmethod
    def preprocess_get_data(deterministic, state, stop_queue, data_queues):
        """init state for queue choice."""
        pids = state[1]
        if deterministic:
            cur_idx = state[0]
            cur_pid = pids[cur_idx]
            datas = safe_get(stop_queue, data_queues[cur_pid], retry=8000000)
            return cur_pid, datas
        pid = random.choice(pids)
        datas = safe_get(stop_queue, data_queues[pid], timeout=0.001, retry=1)
        return pid, datas

    @staticmethod
    def end_of_a_process(deterministic, state, cur_pid, prefetch_worker_num):
        """init state for queue choice."""
        pids = state[1]
        if deterministic:
            pids.remove(cur_pid)
            if len(pids) == 0:
                return True, (0, list(range(prefetch_worker_num)))
            cur_idx = state[0]
            if cur_idx >= len(pids):
                cur_idx -= len(pids)
            return False, (cur_idx, pids)
        end_pids = state[0]
        end_pids.add(cur_pid)
        if len(end_pids) == prefetch_worker_num:
            return True, (set(), list(range(prefetch_worker_num)))
        return False, state  # state[0] has changed

    @staticmethod
    def _preprocess(
        stop_queue,
        data_queues,
        batch_queue,
        prefetch_worker_num,
        max_batch_size,
        drop_last,
        batch_transforms,
        batch_size_queue,
        adaptive_batch_flag,
        deterministic,
        rank,
        batch_reuse,
        batch_strategy,
    ):
        """
        do preprocess(item transform) in child process.
        Args:
            wid(int): prefetch process index.
        """
        # pylint:disable=too-many-branches
        logging.get_logger(log_level="WARNING")
        torch.set_num_threads(1)

        pids_state = HDFSDataset.init_epoch_start_state(deterministic, prefetch_worker_num)
        while stop_queue.empty():
            if adaptive_batch_flag:
                batch_size = safe_get(stop_queue, batch_size_queue, timeout=0.001, retry=1)
                if batch_size is not None:
                    if len(batch_size) == 0:
                        adaptive_batch_flag = False
                    else:
                        max_batch_size = batch_size

            cur_pid, datas = HDFSDataset.preprocess_get_data(
                deterministic, pids_state, stop_queue, data_queues
            )

            if datas is None:
                # end of a propress
                is_end, pids_state = HDFSDataset.end_of_a_process(
                    deterministic, pids_state, cur_pid, prefetch_worker_num
                )
                if not is_end:
                    continue
                for data in batch_strategy.collect_last_batch():
                    if not drop_last and len(data) > 0:
                        try:
                            batch_data = batch_transforms(data)
                        except Exception:
                            logging.warning("rank %d: batch transforms failed", rank, exc_info=True)
                            continue
                        # put last bucket
                        safe_put(stop_queue, batch_queue, batch_data)
                        del batch_data
                safe_put(stop_queue, batch_queue, None)
                continue
            if deterministic:
                cur_idx, pids = pids_state
                cur_idx += 1
                if cur_idx >= len(pids):
                    cur_idx -= len(pids)
                pids_state = (cur_idx, pids)
            for data in datas:
                batch_data = batch_strategy.collate_batch(data, max_batch_size)
                del data
                if batch_data is None:
                    # data is discarded or batch is not full
                    continue
                try:
                    batch_data = batch_transforms(batch_data)
                except Exception:
                    logging.warning(
                        "rank %d: batch transforms failed.",
                        rank,
                        exc_info=True,
                    )
                    # skip this bucket batch
                    continue
                for _ in range(batch_reuse):
                    safe_put(stop_queue, batch_queue, batch_data)
                del batch_data
            del datas

    def get_adaptive_batch(self):
        """get adaptive batch"""
        return self.adaptive_batch

    def set_oom_info(self):
        if not self.adaptive_batch_enble:
            return
        self.adaptive_batch.set_batch_info("out_of_memory", True)


class ValidHDFSDataset(ProcessedDataset):
    """
    only have 1 sub process.
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
        epoch_count=0,
        shuffle=False,
        split_each_dataset=False,
    ):
        super().__init__(
            path_list,
            bucket_schedule,
            cfg,
            item_transform,
            batch_transforms,
            device_transforms,
            split_path_list_by_rank=split_path_list_by_rank,
            epoch_count=epoch_count,
            shuffle=shuffle,
        )
        self.cache_name = cfg.get("valid_cache_name", "hdfs_valid")
        self.split_each_dataset = split_each_dataset
        logging.info(
            "rank %d: ValidHDFSDataset proc_cache_size %d "
            "cuda_cache_size %d prefetch_chunk_num %d",
            self.rank,
            self.proc_cache_size,
            self.cuda_cache_size,
            self.prefetch_chunk_num,
        )
        wav_list = cfg.get("wav_list", "").strip()
        self.is_wav_test = bool(wav_list)
        self.wav_iter = iter(self.read_wavs(wav_list)) if self.is_wav_test else None

    def reset(self):
        """
        reset this dataset.
        """
        if self.is_wav_test:
            return
        super().reset()

    def next(self):
        """iter the dataset."""
        if self.is_wav_test:
            if self.wav_iter is not None:
                try:
                    return next(self.wav_iter)
                except StopIteration:
                    self.wav_iter = None
            return None
        return super().next()

    def terminate(self):
        if self.is_wav_test:
            return
        super().terminate()

    def read_wavs(self, wav_list):
        """read wavs file list for test."""
        if wav_list.endswith(".scp"):
            with open(wav_list, "r") as f:
                wav_list = [tuple(line.strip().split()) for line in f.read().strip().split("\n")]
        else:
            wav_list = [(fn, fn) for fn in wav_list.split(",")]

        for utt, wav_path in wav_list:
            with open(wav_path.strip(), "rb") as f:
                wav_bytes = f.read()
            item = {"uttid": utt, "label": ["fake", "label"], "wav": wav_bytes}
            item = self.item_transform(item)
            batch_data = self.batch_transforms([item])
            batch_data = to_cuda(batch_data)
            yield batch_data

    def _launch_process(self):
        """launch child process."""
        self.stop_queue = torch.multiprocessing.Queue()
        self.batch_queue = torch.multiprocessing.Queue(self.proc_cache_size)
        proc = mp.Process(
            target=self._process_worker,
            args=(
                self.stop_queue,
                self.batch_queue,
                self.origin_path_list,
                self.chunk_size,
                self.io_thread_num,
                self.epoch_count,
                self.shuffle,
                self.rank,
                self.world_size,
                get_local_rank(),
                get_local_size(),
                self.split_path_list_by_rank,
                self.fd_cache_size,
                self.io_retry,
                self.prefetch_chunk_num,
                self.cache_name,
                self.item_transform,
                self.batch_transforms,
                self.max_batch_size,
                self.drop_last,
                self.base_seed,
                self.split_each_dataset,
                self.batch_strategy,
            ),
        )
        proc.daemon = True
        proc.start()
        self.procs = [proc]

        self.thread_queue = tq.Queue(self.cuda_cache_size)
        self.prefetch_thread = Thread(target=self._prefetch_batch)
        # set thread daemon, No need to join this thread when terminate.
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    @staticmethod
    def split_path(
        path_list,
        shuffle,
        split_path_list_by_rank,
        rank,
        world_size,
        prefetch_id=0,
        prefetch_num=1,
    ):
        """split paths."""
        my_paths = path_list.copy()
        if shuffle:
            random.shuffle(my_paths)
        if split_path_list_by_rank:
            my_paths = split_list(my_paths, world_size)[rank]
        my_paths = split_list(my_paths, prefetch_num)[prefetch_id]
        return my_paths

    @staticmethod
    def safe_do_read(reader, chunks):
        """safe do read."""
        try:
            raw_datas = reader.read_many(chunks, True)
            return raw_datas
        except Exception:
            pass
        return []

    @staticmethod
    def safe_do_item_trans(item_transform, item, kwargs):
        """safe do item trans."""
        try:
            item = item_transform(item, **kwargs)
            del item["file_type"]
        except Exception:
            return None
        return item

    @staticmethod
    def clear_bucket_list(stop_queue, queue, batch_transforms, batch_strategy, rank, drop_last):
        """clear the bucket list"""
        for data in batch_strategy.collect_last_batch():
            if not drop_last and len(data) > 0:
                try:
                    batch_data = batch_transforms(data)
                except Exception:
                    logging.warning("rank %d: batch transforms failed", rank, exc_info=True)
                    continue
                # put last bucket
                safe_put(stop_queue, queue, batch_data)
                del batch_data
        safe_put(stop_queue, queue, None)

    @staticmethod
    def _process_worker(
        stop_queue,
        batch_queue,
        path_list,
        chunk_size,
        io_thread_num,
        epoch_count,
        shuffle,
        rank,
        world_size,
        local_rank,
        local_world_size,
        split_path_list_by_rank,
        fd_cache_size,
        io_retry,
        prefetch_chunk_num,
        cache_name,
        item_transform,
        batch_transforms,
        max_batch_size,
        drop_last,
        base_seed,
        split_each_dataset,
        batch_strategy,
    ):
        """
        do all in one process.
        """
        # pylint:disable=too-many-branches,too-many-locals,too-many-nested-blocks
        logging.get_logger(log_level="WARNING")
        torch.set_num_threads(1)
        reader = FalconReader(
            path_list,
            fd_cache_size,
            io_thread_num,
            io_retry,
            cache_name,
            local_world_size,
            local_rank,
            chunk_size,
        )
        path_list_idxs = list(range(len(path_list)))
        while stop_queue.empty():
            random.seed(base_seed + epoch_count)
            np.random.seed(base_seed + epoch_count)
            torch.manual_seed(base_seed + epoch_count)

            my_paths = ValidHDFSDataset.split_path(
                path_list_idxs,
                shuffle,
                split_path_list_by_rank,
                rank,
                world_size,
                prefetch_id=0,
                prefetch_num=1,
            )
            for path_idx in my_paths:
                keys = reader.list_keys([path_idx], False)
                kwargs = {
                    "local_rank": local_rank,
                    "pid": 0,
                    "path_idx": path_idx,
                    "local_process_num": local_world_size,
                    "local_pid": local_rank,
                }
                entry_nums = len(keys)
                chunk_idxs = [i * chunk_size for i in range(entry_nums // chunk_size)]
                random.seed(12345678 + epoch_count)
                if not split_path_list_by_rank:
                    chunk_idxs = split_list(chunk_idxs, world_size)[rank]
                if shuffle:
                    random.shuffle(chunk_idxs)
                for i in range(0, len(chunk_idxs), prefetch_chunk_num):
                    chunks = chunk_idxs[i : i + prefetch_chunk_num]
                    raw_datas = ValidHDFSDataset.safe_do_read(reader, chunks)
                    if raw_datas is None:
                        continue
                    for chunk_idx, raw_data in enumerate(raw_datas):
                        for idx, item in enumerate(raw_data):
                            kwargs["keys"] = keys[chunks[chunk_idx] + idx]
                            item = ValidHDFSDataset.safe_do_item_trans(item_transform, item, kwargs)
                            if item is None:
                                continue
                            batch_data = batch_strategy.collate_batch(item, max_batch_size)
                            if batch_data is None:
                                continue
                            try:
                                batch_data = batch_transforms(batch_data)
                                safe_put(stop_queue, batch_queue, batch_data)
                            except Exception:
                                logging.warning(
                                    "rank %d: batch transforms failed.",
                                    rank,
                                    exc_info=True,
                                )
                    del raw_datas
                if split_each_dataset:
                    ValidHDFSDataset.clear_bucket_list(
                        stop_queue,
                        batch_queue,
                        batch_transforms,
                        batch_strategy,
                        rank,
                        drop_last,
                    )
            if not split_each_dataset:
                ValidHDFSDataset.clear_bucket_list(
                    stop_queue,
                    batch_queue,
                    batch_transforms,
                    batch_strategy,
                    rank,
                    drop_last,
                )
            epoch_count += 1
