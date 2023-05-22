'''
sampler module.
'''
import random
import pickle
import os
import time
import bisect
from itertools import accumulate
from core.utils import logging
from core.utils import Registry
from core.utils.split import split_list
from core.dataset.utils import get_parquet_file_info

SAMPLER = Registry("sampler")


def setup_sampler_cfg(cfg):
    '''
    get sampler according to data-config

    Sampler method:
    NativeSampler (default, raw sampler method)
    GlobalSampler (global shuffle)
    '''
    sampler_args = dict()
    sampler_args['NativeSampler'] = cfg.get('native_parallel_file_num', 1)
    sampler_args['GlobalSampler'] = cfg.get('global_shuffle_speed_up', False)
    sampler_args['ParquetSampler'] = cfg.get('parquet_rows_group_shuffle', False)
    global_shuffle = cfg.get('global_shuffle', False)
    sampler_method = cfg.get('sampler', 'GlobalSampler' if global_shuffle else 'NativeSampler')
    sampler_cfg = (SAMPLER.get(sampler_method), sampler_args.get(sampler_method))
    return sampler_cfg


def norm_sample_weight(global_weights, sizes):
    '''norm sample weights'''
    # sum of weights
    sum_weight = sum(global_weights)
    # need_dataset_nums = weight/sum_weights
    # real_dataset_nums = dataset_sizes
    # sample_weight = need_dataset_nums / real_dataset_nums
    sample_weights = [
        global_weight / sum_weight / size for global_weight, size in zip(global_weights, sizes)
    ]
    # set max_sample_weights to 1, that others will lower than 1
    max_sample_weights = max(sample_weights)
    sample_weights = [w / max_sample_weights for w in sample_weights]
    return sample_weights


class BaseSampler:
    """BaseSampler class
    It is an abstract class without implementation.
    You can see the specific implementation below.
    usage example:
    `
    sampler = BaseSampler(args) # instantiation
    sampler.reset(args) # must do
    sampler.reset
    for idx in sampler: # get data
        do_something()
    `
    """

    def __init__(self, pid, prefetch_worker_num, local_rank, local_world_size, rank, world_size):
        '''
        init.
        args:
        pid: process_idx in one GPU
        prefetch_worker_num: process_num in one GPU
        local_rank: GPU idx in one worker
        local_world_size: GPU num in one worker
        rank: GPU idx in all worker
        world_size: GPU num in all worker
        '''
        self.pid = pid
        self.prefetch_worker_num = prefetch_worker_num
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.rank = rank
        self.world_size = world_size

    def reset(self):
        '''reset random seed, must do before use.'''
        raise NotImplementedError

    def __iter__(self):
        '''yield the ret data'''
        raise NotImplementedError


@SAMPLER.register_module()
class NativeSampler(BaseSampler):
    """Sampler class for HDFSDataset
    shuffle_method: Native shuffle strategy

    usage example:
    `
    sampler = ChunkSampler(pid, prefetch_worker_num, local_rank, local_world_size,
                           rank, world_size, shuffle, split_path_by_rank,
                           chunk_size, prefetch_chunk_num, files_num, reader)
    sampler.reset(random_seed) # must do
    for chunk_idxs, file_idxs in sampler:
        do()
    `
    Args:
        pid: process_idx in one GPU
        prefetch_worker_num: process_num in one GPU
        local_rank: GPU idx in one worker
        local_world_size: GPU num in one worker
        rank: GPU idx in all worker
        world_size: GPU num in all worker
        split_path_list_by_rank: Whether read data by rank and pid
        chunk_info: A list of chunks num in each file
        shuffle: Whether shuffle
        prefetch_chunk_num: prefetch_chunk_num of chunks will return
        parallel_file_num: read files num
    '''
    """

    def __init__(
        self,
        pid,
        prefetch_worker_num,
        local_rank,
        local_world_size,
        rank,
        world_size,
        shuffle,
        split_path_by_rank,
        chunk_size,
        prefetch_chunk_num,
        files_num,
        reader,
        parallel_file_num=1,
        dataset_weights=None,
        dataset_length=None,
    ):
        '''
        init.
        '''
        super().__init__(pid, prefetch_worker_num, local_rank, local_world_size, rank, world_size)
        self.shuffle = shuffle
        self.split_path_by_rank = split_path_by_rank
        self.chunk_size = chunk_size
        self.prefetch_chunk_num = prefetch_chunk_num
        self.files_num = files_num
        self.reader = reader
        self.seed = 12345
        self.parallel_file_num = max(1, parallel_file_num)
        self.multiples = 1
        self.check_file_nums()
        self.dataset_weights = None
        if dataset_weights:
            self.get_dataset_weights(dataset_weights, dataset_length)

    def check_file_nums(self):
        '''
        check whether need expand file nums
        '''
        multiples = 4 * (
            self.world_size * self.prefetch_worker_num
            if self.split_path_by_rank
            else self.prefetch_worker_num
        )
        if self.files_num < multiples:
            logging.warning(
                "You are using a non-global_shuffle sampler policy, "
                "but the number of your files(%d) is too small to divide into subprocesses, "
                "(world_size: %d , process_worker_num: %d)"
                " and now the files are automatically copied for you to make them separable",
                self.files_num,
                self.world_size,
                self.prefetch_worker_num,
            )
            multiples = (multiples + self.files_num - 1) // self.files_num
            self.multiples = multiples

    def reset(self, seed, skip_num=0):
        '''
        Update the random seed, refresh the sampler list
        '''
        self.seed = seed
        self.skip_num = (skip_num + self.prefetch_worker_num - 1) // self.prefetch_worker_num
        # random seed has been set in dataset, no need to set again for shuffle and split_list
        path_idxs = list(range(self.files_num))
        path_idxs *= self.multiples
        if self.shuffle:
            random.shuffle(path_idxs)
        if self.split_path_by_rank:
            path_idxs = split_list(path_idxs, self.world_size)[self.rank]
        self.path_idxs = split_list(path_idxs, self.prefetch_worker_num)[self.pid]

    def __iter__(self):
        '''
        for yeild datas
        '''
        for st in range(0, len(self.path_idxs), self.parallel_file_num):
            file_entry_nums = self.reader.get_entry_num(
                self.path_idxs[st : st + self.parallel_file_num], False, True
            )
            breakpoints = list(accumulate(file_entry_nums))
            chunk_idxs = []
            pre_sum = 0
            random.seed(self.seed)
            for offset, entry_num in enumerate(file_entry_nums):
                this_chunk_indexs = [
                    pre_sum + i * self.chunk_size for i in range(entry_num // self.chunk_size)
                ]
                pre_sum += entry_num
                path_idx = self.path_idxs[st + offset]
                if self.dataset_weights:
                    path_weight = self.dataset_weights[path_idx]
                    sample_num = int(len(this_chunk_indexs) * path_weight)
                    if self.shuffle:
                        this_chunk_indexs = random.sample(this_chunk_indexs, sample_num)
                    else:
                        this_chunk_indexs = this_chunk_indexs[:sample_num]
                chunk_idxs.extend(this_chunk_indexs)
            if self.shuffle:
                random.shuffle(chunk_idxs)
            if not self.split_path_by_rank:
                chunk_idxs = split_list(chunk_idxs, self.world_size)[self.rank]
            if self.skip_num >= len(chunk_idxs) * self.chunk_size:
                self.skip_num -= len(chunk_idxs) * self.chunk_size
                continue
            start_chunk = self.skip_num // self.chunk_size
            for i in range(start_chunk, len(chunk_idxs), self.prefetch_chunk_num):
                chunks = chunk_idxs[i : i + self.prefetch_chunk_num]
                chunk_path_idxs = [bisect.bisect(breakpoints, chunk_idx) for chunk_idx in chunks]
                chunk_path_idxs = [self.path_idxs[st + i] for i in chunk_path_idxs]
                yield chunks, chunk_path_idxs

    def get_dataset_weights(self, dataset_weights, dataset_length):
        '''get dataset weights'''
        # get entry num of each file
        self.entry_nums = self.reader.get_entry_num(list(range(self.files_num)), True, True)
        # get dataset sizes
        dataset_sizes = []
        file_offset = 0
        for file_num in dataset_length:
            dataset_sizes.append(sum(self.entry_nums[file_offset : file_offset + file_num]))
            file_offset += file_num
        logging.warning("rank %d: HDFSDataset dataset sizes: %r", self.rank, dataset_sizes)
        # get sample weights
        sample_weights = norm_sample_weight(dataset_weights, dataset_sizes)
        self.dataset_weights = []
        for dataset_weight, file_num in zip(sample_weights, dataset_length):
            self.dataset_weights += [dataset_weight] * file_num
        logging.warning(
            "rank %d: HDFSDataset dataset sample weighted: %r", self.rank, sample_weights
        )


@SAMPLER.register_module()
class GlobalSampler(BaseSampler):
    """Sampler class for HDFSDataset
    shuffle_method: global_shuffle

    usage example:
    `
    sampler = GlobalChunkSampler(pid, prefetch_worker_num, local_rank, local_world_size,
                                 rank, world_size, shuffle, split_path_by_rank,
                                 chunk_size, prefetch_chunk_num, files_num, reader)
    sampler.reset(random_seed) # must do
    for chunk_idxs, file_idxs in sampler:
        do()
    `
    Args:
        pid: process_idx in one GPU
        prefetch_worker_num: process_num in one GPU
        local_rank: GPU idx in one worker
        local_world_size: GPU num in one worker
        rank: GPU idx in all worker
        world_size: GPU num in all worker
        shuffle: Whether shuffle
        prefetch_chunk_num: prefetch_chunk_num of chunks will return
        chunks_num: Total chunks num
    '''
    """

    def __init__(
        self,
        pid,
        prefetch_worker_num,
        local_rank,
        local_world_size,
        rank,
        world_size,
        shuffle,
        split_path_by_rank,
        chunk_size,
        prefetch_chunk_num,
        files_num,
        reader,
        proceess_split=False,
        dataset_weights=None,
        dataset_length=None,
    ):
        '''
        init.
        '''
        super().__init__(pid, prefetch_worker_num, local_rank, local_world_size, rank, world_size)
        self.shuffle = shuffle
        self.split_path_by_rank = split_path_by_rank
        self.chunk_size = chunk_size
        self.prefetch_chunk_num = prefetch_chunk_num
        self.files_num = files_num
        self.reader = reader
        self.proceess_split = proceess_split
        self.dataset_weights = None
        if dataset_weights:
            self.get_dataset_weights(dataset_weights, dataset_length)
        else:
            # global shuffle sampler use cache to reduce memory.
            self.entry_nums = self.reader.get_entry_num(list(range(self.files_num)), True, True)
        self.breakpoints = list(accumulate(self.entry_nums))

    def reset(self, _seed, skip_num=0):
        '''
        reset the sampler list, must do
        '''
        self.skip_num = (skip_num + self.prefetch_worker_num - 1) // self.prefetch_worker_num
        # random seed has been set in dataset, no need to set again for shuffle and split_list
        chunk_idxs = []
        pre_sum = 0
        for path_idx, entry_num in enumerate(self.entry_nums):
            this_chunk_indexs = [
                pre_sum + i * self.chunk_size for i in range(entry_num // self.chunk_size)
            ]
            pre_sum += entry_num
            if self.dataset_weights:
                path_weight = self.dataset_weights[path_idx]
                sample_num = int(len(this_chunk_indexs) * path_weight)
                if self.shuffle:
                    this_chunk_indexs = random.sample(this_chunk_indexs, sample_num)
                else:
                    this_chunk_indexs = this_chunk_indexs[:sample_num]
            chunk_idxs.extend(this_chunk_indexs)
        if self.proceess_split:
            all_rank_chunks = split_list(chunk_idxs, self.prefetch_worker_num)
            chunk_idxs = all_rank_chunks[self.pid]
            if self.shuffle:
                random.shuffle(all_rank_chunks)
                chunk_idxs = all_rank_chunks[self.pid]
                random.shuffle(chunk_idxs)
            chunk_idxs = split_list(chunk_idxs, self.world_size)[self.rank]
        else:
            if self.shuffle:
                random.shuffle(chunk_idxs)
            chunk_idxs = split_list(chunk_idxs, self.world_size)[self.rank]  # default
            chunk_idxs = split_list(chunk_idxs, self.prefetch_worker_num)[self.pid]
        self.chunk_idxs = chunk_idxs

    def __iter__(self):
        '''
        yeild the datas
        return chunk indexs
        '''
        start_chunk = self.skip_num // self.chunk_size
        for i in range(start_chunk, len(self.chunk_idxs), self.prefetch_chunk_num):
            chunks = self.chunk_idxs[i : i + self.prefetch_chunk_num]
            chunk_path_idxs = [bisect.bisect(self.breakpoints, chunk_idx) for chunk_idx in chunks]
            yield chunks, chunk_path_idxs
        del self.chunk_idxs

    def get_dataset_weights(self, dataset_weights, dataset_length):
        '''get dataset weights'''
        # get entry num of each file
        self.entry_nums = self.reader.get_entry_num(list(range(self.files_num)), True, True)
        # get dataset sizes
        dataset_sizes = []
        file_offset = 0
        for file_num in dataset_length:
            dataset_sizes.append(sum(self.entry_nums[file_offset : file_offset + file_num]))
            file_offset += file_num
        logging.warning("rank %d: HDFSDataset dataset sizes: %r", self.rank, dataset_sizes)
        # get sample weights
        sample_weights = norm_sample_weight(dataset_weights, dataset_sizes)
        self.dataset_weights = []
        for dataset_weight, file_num in zip(sample_weights, dataset_length):
            self.dataset_weights += [dataset_weight] * file_num
        logging.warning(
            "rank %d: HDFSDataset dataset sample weighted: %r", self.rank, sample_weights
        )


@SAMPLER.register_module()
class BalancedSampler(BaseSampler):
    """Sampler class for SID task
    Usage example:
    `
    sampler = BalancedSampler(pid,
                              prefetch_worker_num,
                              local_rank,
                              local_world_size,
                              rank,
                              world_size,
                              reader,
                              batch_size_per_class,
                              batch_class_num,
                              cached_name,
                              path_idxs,
                              process_mutex,)

    sampler.reset(seed)
    for clss_buf, clss_keys_buf in sampler:
        reader.read_many(clss_keys_buf)
    `
    """

    def __init__(
        self,
        pid,
        prefetch_worker_num,
        local_rank,
        local_world_size,
        rank,
        world_size,
        reader,
        batch_size_per_class,
        batch_class_num,
        cached_name,
        path_idxs,
        process_mutex,
        balance_epoch,
    ):
        '''
        Args:
        pid: process_idx in one GPU
        prefetch_worker_num: process_num in one GPU
        local_rank: GPU idx in one worker
        local_world_size: GPU num in one worker
        rank: GPU idx in all worker
        world_size: GPU num in all worker
        reader: FalconReader
        batch_size_per_class: num of items in one class
        batch_class_num: class num in one batch
        cached_name: saved file name for share datas,
                     shared data will be saved in /tmp/falconreader/{cahced_name}
        path_idxs: list of path_idx, eg: [0, 1, 2 ..]
        process_mutex: different processes fetch different class
        '''
        super().__init__(pid, prefetch_worker_num, local_rank, local_world_size, rank, world_size)
        self.batch_size_per_class = batch_size_per_class
        self.batch_class_num = batch_class_num
        self.reader = reader
        saved_dir = '/tmp/falconreader/' + cached_name
        if not os.path.exists(saved_dir) and local_rank == 0 and pid == 0:
            os.makedirs(saved_dir)
        self.data_saved_name = os.path.join(saved_dir, 'class.data')
        self.flag_save_name = os.path.join(saved_dir, 'class.flag')
        self.path_idxs = path_idxs
        # local_pid : process idx in one worker
        self.local_pid = self.local_rank * self.prefetch_worker_num + self.pid
        # local_pid : process idx in one worker
        self.local_process_num = self.local_world_size * self.prefetch_worker_num
        # process idx in all workers
        self.global_pid = self.rank * self.prefetch_worker_num + self.pid
        # all process num in all workers
        self.global_process_num = self.world_size * self.prefetch_worker_num
        self.cls2idxs, self.items_num = BalancedSampler.shared_cls2idxs(
            reader,
            self.local_process_num,
            self.local_pid,
            self.path_idxs,
            self.data_saved_name,
            self.flag_save_name,
        )
        self.cls_id2key = list(self.cls2idxs.keys())
        self.process_mutex = process_mutex
        # Apply the virtual_items_num:  max{len(class_data)} * class_num.
        # Besides, considering the multithreading,
        # we should divide the prefetch_worker_num to keep the epoch batch num as expected
        if balance_epoch:
            self.items_num = max(len(i) for _, i in self.cls2idxs.items()) * len(
                self.cls2idxs.keys()
            )
            self.items_num //= self.prefetch_worker_num

    def reset(self, random_seed):
        if not self.process_mutex:
            random_seed += self.rank * self.prefetch_worker_num + self.pid
        random.seed(random_seed)

    def __iter__(self):
        '''
        will yield all data in the file in one epoch,
        will yield batch_class_num class once
        '''
        # clss_idxs: all classes idxs len(class_idxs) == all_class_nums
        clss_idxs = list(range(len(self.cls2idxs)))
        random.shuffle(clss_idxs)
        for _ in range(
            0,
            self.items_num,
            self.batch_class_num * self.batch_size_per_class,
        ):
            cls_picked = self.pick_class(clss_idxs)
            clss_buf = []  # str
            entries_buf = []
            for clss_idx in cls_picked:
                clss = self.cls_id2key[clss_idx]
                clss_buf.append(clss)
                clss_data_num = len(self.cls2idxs[clss])
                if clss_data_num < self.batch_size_per_class:
                    logging.warning(
                        "while sampler class(%s), the class's data num(%d)"
                        " is less than wanted data num(%d),"
                        " and now will oversample this class",
                        clss,
                        clss_data_num,
                        self.batch_size_per_class,
                    )
                    self.cls2idxs[clss] *= (
                        self.batch_size_per_class + clss_data_num - 1
                    ) // clss_data_num
                keys_buf = random.sample(self.cls2idxs[clss], self.batch_size_per_class)
                entries_buf.extend(keys_buf)
                del keys_buf
            yield clss_buf, entries_buf
            del cls_picked, clss_buf, entries_buf

    @staticmethod
    def get_cls2idxs(reader, path_idxs):
        """
        get cls-to-idx dict
        """
        keys = reader.list_keys(path_idxs, False)
        entry_num = len(keys)
        cls2idxs = dict()
        key2idx = dict()
        meta_idxs = []
        for idx, key in enumerate(keys):
            if key == 'meta':
                meta_idxs.append(idx)
            else:
                key2idx[key] = idx
        del keys
        raw_datas = reader.read_many(meta_idxs, True)
        del meta_idxs
        raw_datas = sum(raw_datas, [])
        for item in raw_datas:
            cls2keys = pickle.loads(item)
            for clss, clss_keys in cls2keys.items():
                for cls_key in clss_keys:
                    try:
                        cls2idxs.setdefault(clss, []).append(key2idx[cls_key])
                    except Exception:
                        logging.error(
                            "BalancedSampler occur error while creating cls2idxs",
                        )
            del cls2keys
        del raw_datas
        return cls2idxs, entry_num

    @staticmethod
    def shared_cls2idxs(
        reader, local_process_num, local_pid, path_idxs, data_saved_name, flag_saved_name
    ):
        """
        get cls-to-idxs dict and entry nums
        it will be write to file for single in-machine sharing
        """
        if local_process_num == 1:
            cls2idxs, entry_num = BalancedSampler.get_cls2idxs(reader, path_idxs)
            return cls2idxs, entry_num
        # more than one process in one worker
        cls2idxs, entry_num = None, None
        if local_pid == 1:
            cls2idxs, entry_num = BalancedSampler.get_cls2idxs(reader, path_idxs)
            if cls2idxs is None:
                logging.error(
                    "BalancedSampler get cls2idxs error: cls2idxs is None, ",
                    "local_pid is %d local_process_num is %d",
                    local_pid,
                    local_process_num,
                )
            try:
                with open(data_saved_name, 'wb') as f:
                    pickle.dump(cls2idxs, f)
                # pylint:disable=consider-using-with
                open(flag_saved_name, 'a', encoding='utf-8').close()
            except Exception as e:
                logging.error(
                    "BalancedSampler write cls2idxs error: %s, "
                    "while writing cls2idxs to shared memory, there contains an error "
                    "local_pid is %d local_process_num is %d",
                    repr(e),
                    local_pid,
                    local_process_num,
                )
        else:
            entry_num = reader.get_entry_num(path_idxs, True)
            time.sleep(1)
            while not os.path.exists(flag_saved_name):
                time.sleep(1)
            try:
                with open(data_saved_name, 'rb') as f:
                    cls2idxs = pickle.load(f)
            except Exception as e:
                logging.error(
                    "BalancedSampler unpickle cls2idxs error: %s , "
                    "while unpickling cls2idxs from shared memory, there contains an error, ",
                    "the shared memory file is %s, ",
                    "the file size is %d",
                    "local_pid is %d local_process_num is %d",
                    repr(e),
                    data_saved_name,
                    os.path.getsize(data_saved_name),
                    local_pid,
                    local_process_num,
                )
        return cls2idxs, entry_num

    def pick_class(self, clss_idxs):
        '''
        picked classes in one iter on different sampler method
        '''
        class_num = len(clss_idxs)
        picked_cls_num = self.batch_class_num
        if self.process_mutex:
            picked_cls_num *= self.global_process_num

        if class_num < picked_cls_num:
            logging.warning(
                "Sampler Warning: the class num(%d) is less than picked class num(%d)",
                class_num,
                picked_cls_num,
            )
            clss_idxs *= (picked_cls_num + class_num - 1) // class_num

        cls_picked = random.sample(clss_idxs, picked_cls_num)
        if self.process_mutex:
            cls_picked = split_list(cls_picked, self.global_process_num)[self.global_pid]
        return cls_picked


@SAMPLER.register_module()
class ParquetSampler(BaseSampler):
    """ParquetSampler class
    usage example:
    `
    sampler = ParquetSampler(args) # instantiation
    sampler.reset(args) # must do
    sampler.reset
    for idx in sampler: # get data
        do_something()
    `
    """

    def __init__(
        self,
        pid,
        prefetch_worker_num,
        local_rank,
        local_world_size,
        rank,
        world_size,
        shuffle,
        split_path_by_rank,
        file_list,
        rows_group_shuffle,
    ):
        '''
        init.
        args:
        pid: process_idx in one GPU
        prefetch_worker_num: process_num in one GPU
        local_rank: GPU idx in one worker
        local_world_size: GPU num in one worker
        rank: GPU idx in all worker
        world_size: GPU num in all worker
        shuffle: whether use data shuffle,
        split_path_by_rank: split datas to different rank,
        file_num: parquet file num
        '''
        super().__init__(pid, prefetch_worker_num, local_rank, local_world_size, rank, world_size)
        self.shuffle = shuffle
        self.split_path_by_rank = split_path_by_rank
        self.file_list = file_list
        self.rows_group_shuffle = rows_group_shuffle
        self.parquet_info = [None] * len(self.file_list)

    def reset(self, seed, skip_num=0):
        '''reset random seed, must do before use.'''
        self.seed = seed
        self.skip_num = skip_num
        path_idxs = list(range(len(self.file_list)))
        if self.shuffle:
            random.shuffle(path_idxs)
        if self.split_path_by_rank:
            path_idxs = split_list(path_idxs, self.world_size)[self.rank]
        self.path_idxs = split_list(path_idxs, self.prefetch_worker_num)[self.pid]

    def __iter__(self):
        '''yield the ret data'''
        for path_idx in self.path_idxs:
            if self.parquet_info[path_idx] is None:
                self.parquet_info[path_idx] = get_parquet_file_info(self.file_list[path_idx])
            num_row_groups = self.parquet_info[path_idx]['num_row_groups']
            num_rows = self.parquet_info[path_idx]['num_rows']
            # implement a temporary resume solution.
            # TODO: providing a more fine-grained resume implementation
            if self.skip_num >= num_rows:
                self.skip_num -= num_rows
                continue
            num_row_groups = list(range(num_row_groups))
            if self.rows_group_shuffle:
                random.shuffle(num_row_groups)
            yield num_row_groups, path_idx