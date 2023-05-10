'''adaptive batch'''
import math
import torch
from core.utils import logging, get_local_rank
from .batching import find_bucket_idx
from .queue_utils import safe_put


class BaseAdaptiveBatch:
    '''Adaptive batch size
    the solution on https://bytedance.feishu.cn/docs/doccnKDWnkonKYxUAw3wuyJfRXb#
    '''

    def __init__(self, cfg, bucket_schedule_length, is_run):
        '''init.'''
        self.batch_adjust_num = cfg.get('batch_adjust_num', 500)  # the solution of times
        # if there is too much out of memory results, we can reduce the parameter
        self.expected_memory_usage = cfg.get('expected_memory_usage', 0.625)
        local_rank = get_local_rank()
        self.current_device_memory = torch.cuda.get_device_properties(local_rank).total_memory
        # record the last batch size
        self.last_batch_size = [0] * bucket_schedule_length
        # record minimal batch size which caused out of memory
        self.oom_batch_size = [math.inf] * bucket_schedule_length
        self.is_run = is_run
        self.is_iter_oom = False
        self.batch_info = {}
        self.iter = 0

    def before_iter(self, data_loader):
        '''before iter'''
        if not self.is_run or self.iter > self.batch_adjust_num:
            return
        if self.iter == 0:
            self.start(data_loader)
            self.iter += 1
        else:
            self.after_iter(data_loader)
            self.batch_info.clear()
            self.iter += 1
            if self.iter == self.batch_adjust_num:
                self.stop_adjust_batch(data_loader)
        torch.cuda.reset_max_memory_allocated()

    def after_iter(self, data_loader):
        '''after iter'''
        if not self.is_run or self.iter > self.batch_adjust_num:
            return
        batch_size = self.batch_info.get('batch_size', 0)
        batch_num = self.batch_info.get('batch_num', 0)
        out_of_memory = self.batch_info.get('out_of_memory', False)
        if not data_loader.get_batch_means_tokens():
            batch_size = 1
        bucket_idx = find_bucket_idx(data_loader.bucket_schedule, batch_size)
        if bucket_idx < 0:
            return
        max_batch_size = data_loader.get_max_batch_size(bucket_idx)
        if out_of_memory:
            oom_batch_size = self.oom_batch_size[bucket_idx]
            self.oom_batch_size[bucket_idx] = min(oom_batch_size, batch_size * batch_num)
        # we improved max batch size but the batch composed by last max batch size
        if batch_size * batch_num < max_batch_size:
            return
        # we reduced max batch size bu the batch composed by last max batch size
        if self.last_batch_size[bucket_idx] > max_batch_size and (
            batch_size * batch_num >= self.last_batch_size[bucket_idx]
        ):
            return
        expected_memory_usage = self.expected_memory_usage
        memory_usage = torch.cuda.max_memory_allocated() / self.current_device_memory
        self.improve_batch(
            data_loader, bucket_idx, memory_usage, expected_memory_usage, out_of_memory
        )

    def improve_batch(
        self, data_loader, bucket_idx, memory_usage, expected_memory_usage, out_of_memory=False
    ):
        '''improve batch size'''
        if out_of_memory:
            self.reduce_batch(data_loader, bucket_idx, memory_usage, expected_memory_usage, True)
        elif memory_usage > expected_memory_usage:
            self.reduce_batch(data_loader, bucket_idx, memory_usage, expected_memory_usage)
        else:
            self.increase_batch(data_loader, bucket_idx, memory_usage, expected_memory_usage)

    def stop_adjust_batch(self, data_loader):
        '''stop adjust batch size'''
        self.is_run = False
        data_loader.round_up_max_batch_size()
        local_rank = get_local_rank()
        logging.all_rank_info(
            'rank %d last max batch size for every bucket %r',
            local_rank,
            data_loader.get_max_batch_size_list(),
        )

    def increase_batch(self, data_loader, bucket_idx, memory_usage, expected_memory_usage):
        '''increase batch size'''
        max_batch_size = data_loader.get_max_batch_size(bucket_idx)
        if memory_usage * 1.5 <= expected_memory_usage - 0.1:
            max_batch_size = max_batch_size * 1.5
        else:
            max_batch_size = max_batch_size * (
                1
                + (expected_memory_usage / memory_usage)
                * (0.01 * (math.log(100 * (expected_memory_usage - memory_usage)) + 1))
            )
        # if max_batch_size more than minimal batch size which caused out of memmory
        # we can't improve the batch size
        if max_batch_size >= self.oom_batch_size[bucket_idx]:
            return
        self.last_batch_size[bucket_idx] = data_loader.get_max_batch_size(bucket_idx)
        data_loader.set_max_batch_size(bucket_idx, max_batch_size)

    def reduce_batch(
        self, data_loader, bucket_idx, memory_usage, expected_memory_usage, out_of_memory=False
    ):
        '''reduce batch size'''
        max_batch_size = data_loader.get_max_batch_size(bucket_idx)
        if out_of_memory:
            max_batch_size = max_batch_size / 2
        else:
            max_batch_size = max_batch_size * (
                1
                - (expected_memory_usage / memory_usage)
                * (0.01 * (math.log(100 * (memory_usage + expected_memory_usage)) + 1))
            )
        self.last_batch_size[bucket_idx] = data_loader.get_max_batch_size(bucket_idx)
        data_loader.set_max_batch_size(bucket_idx, max_batch_size)

    def get_batch_adjust_num(self):
        '''get batch adjust num'''
        return self.batch_adjust_num

    def start(self, _data_loader):
        '''start'''

    def clear_iter_oom(self):
        '''clear iter oom'''
        self.is_iter_oom = False

    def set_iter_oom(self):
        '''set iter oom'''
        self.is_iter_oom = True

    def clear_batch_info(self):
        '''clear batch data'''
        self.batch_info.clear()

    def set_batch_info(self, key, value):
        '''set batch data'''
        if not self.is_run:
            return
        self.batch_info[key] = value


class ProcessedAdaptiveBatch(BaseAdaptiveBatch):
    '''Adaptive batch size for MultiProcess'''

    def after_iter(self, data_loader):
        '''after iter'''
        super().after_iter(data_loader)
        if self.iter > self.batch_adjust_num:
            return
        # to do communication
        for bsz in data_loader.batch_size_queues:
            safe_put(
                data_loader.stop_queue,
                bsz,
                data_loader.get_max_batch_size_list(),
            )

    def start(self, data_loader):
        '''start'''
        # if batch size isn't runnng, we need put empty list
        if self.is_run:
            max_batch_size = data_loader.get_max_batch_size_list()
            for bsz in data_loader.batch_size_queues:
                safe_put(data_loader.stop_queue, bsz, max_batch_size)

    def stop_adjust_batch(self, data_loader):
        '''stop ddjust batch'''
        super().stop_adjust_batch(data_loader)
        for bsz in data_loader.batch_size_queues:
            safe_put(data_loader.stop_queue, bsz, [])
