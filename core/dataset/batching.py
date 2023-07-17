'''
bucket process module.
'''

import random
from core.utils import Registry, logging

BATCH_STRATEGY = Registry("batch_strategy")


def find_bucket_idx(bucket_schedule, size):
    '''
    find bucket idx for a size.
    Args:
        bucket_schedule(list of int): bucket schedule.
        size(int): normaly should be fbank.shape[0].
    Return:
        int: the minimum bucket idx for this size, which match
             `size <= bucket_schedule[idx]`.
             -1 means no bucket match.
    '''
    if bucket_schedule == '' or bucket_schedule is None:
        return 0
    bucket_length = len(bucket_schedule)
    if size > bucket_schedule[-1]:
        return -1
    low = -1
    high = bucket_length - 1
    while low + 1 < high:
        mid = (high + low) >> 1
        if bucket_schedule[mid] < size:
            low = mid
        else:
            high = mid
    return high


def get_batch_strategy(cfg, bucket_schedule=None, batch_strategy=None, bucket_schedule_key=None):
    '''Get batch collate strategy.'''
    if batch_strategy is None:
        if bucket_schedule is None:
            # for sid vaild dataset
            batch_strategy = cfg.get('batch_strategy', 'NoPaddingBucketBatching')
        else:
            # for asr tasks
            batch_strategy = cfg.get('batch_strategy', 'BucketBatching')
    strategy_cls = BATCH_STRATEGY.get(batch_strategy)
    if bucket_schedule_key is None:
        bucket_schedule_key = cfg.get('bucket_schedule_key', '')
    return strategy_cls(
        bucket_schedule=bucket_schedule,
        bucket_schedule_key=bucket_schedule_key,
        bucket_schedule_shape=cfg.get('bucket_schedule_shape', 0),
        use_old_bucket=cfg.get('use_old_bucket', False),
        batch_means_tokens=cfg.get('batch_means_tokens', True),
        bucket_skip_warning_num=cfg.get('bucket_skip_warning_num', 10000),
        bucket_size=cfg.get('bucket_size', 20),
        domain_num=cfg.get('domain_num', 1),
        domain_key=cfg.get('domain_key', 'domain'),
    )


def setup_bucket(cfg, batch_strategy='BucketBatching'):
    ''' bucket. '''
    batch_means_tokens = cfg.get('batch_means_tokens', True)
    bucket_schedule_key = cfg.get('bucket_schedule_key', '')
    max_batch_size = cfg.get('max_batch_size', 1)
    bucket_schedule = cfg.get('bucket_schedule', None)
    if bucket_schedule is not None:
        bucket_schedule = [int(item) for item in bucket_schedule.split(',')]
    batch_strategy = globals()[batch_strategy]

    if isinstance(max_batch_size, str) and ',' in cfg.max_batch_size:
        # max_batch_size for every bucket is set by user.
        max_batch_size = [int(item) for item in max_batch_size.strip().split(',')]
        if bucket_schedule is not None:
            assert len(max_batch_size) == len(bucket_schedule)
    elif bucket_schedule is not None:
        # max_batch_size for every bucket is calculated by max_batch_scale.
        max_batch_scale = cfg.get('max_batch_scale', 0)
        if max_batch_scale != 0:
            assert batch_means_tokens and max_batch_scale > 0
        max_bucket = min(bucket_schedule[-1], 2000)
        max_batch_size = [
            max_batch_size + max_batch_scale * max(max_bucket - item, 0)
            for item in bucket_schedule
        ]

    return batch_strategy(
        bucket_schedule=bucket_schedule,
        bucket_schedule_key=cfg.get('bucket_schedule_key', ''),
        bucket_schedule_shape=cfg.get('bucket_schedule_shape', 0),
        use_old_bucket=cfg.get('use_old_bucket', False),
        batch_means_tokens=cfg.get('batch_means_tokens', True),
        bucket_skip_warning_num=cfg.get('bucket_skip_warning_num', 10000),
        bucket_size=cfg.get('bucket_size', 20),
        domain_num=cfg.get('domain_num', 1),
        domain_key=cfg.get('domain_key', 'domain'),
        domain_id_key=cfg.get('dimain_id_key', 'domain_id'),
    )


class BaseBatching:
    '''base batch collater.'''

    def __init__(self):
        '''init.'''
        self.throw_num = 0

    @staticmethod
    def get_item_size(_data_item):
        '''get item size.'''
        return NotImplementedError

    # pylint: disable=no-self-use
    def collate_batch(self):
        '''
        draw data_item for collate batch.
        Args:
            data_item(any): data item.
            max_batch_size(int): max batch size.
        Return:
            batch_data(any): collated batch data if batch is full else None.
        '''
        return NotImplementedError

    def collect_last_batch(self):
        '''collect batch data(s) that has not been get.'''
        return NotImplementedError

    def clear(self):
        '''clear data buffer(s)'''
        return NotImplementedError


@BATCH_STRATEGY.register_module()
class BucketBatching(BaseBatching):
    '''
    batch token bucket schedule.
    collate batch data depending on bsz means token.
    '''

    def __init__(
        self,
        bucket_schedule,
        bucket_schedule_key,
        bucket_schedule_shape,
        use_old_bucket=True,
        batch_means_tokens=True,
        bucket_skip_warning_num=10000,
        **_kwargs,
    ):
        super().__init__()
        self.bucket_schedule = bucket_schedule
        self.bucket_schedule_key = bucket_schedule_key
        self.bucket_schedule_shape = bucket_schedule_shape
        self.bucket_num = 1 if bucket_schedule is None else len(bucket_schedule)
        self.use_old_bucket = use_old_bucket
        self.batch_means_tokens = batch_means_tokens
        self.bucket_list = [[] for _ in range(self.bucket_num)]
        self.bucket_batch_size = [0 for _ in range(self.bucket_num)]
        self.bucket_max_size = [0 for _ in range(self.bucket_num)]
        self.bucket_skip_warning_num = bucket_skip_warning_num

    def find_and_push_bucket(self, data_item):
        '''find a suitable bucket and push to bucket.'''
        size = self.get_item_size(data_item)
        if size == -1:
            return -1, -1
        bucket_idx = find_bucket_idx(self.bucket_schedule, size)
        if bucket_idx < 0:
            return -1, -1
        self.bucket_list[bucket_idx].append(data_item)
        self.bucket_batch_size[bucket_idx] += size
        self.bucket_max_size[bucket_idx] = max(self.bucket_max_size[bucket_idx], size)
        return size, bucket_idx

    def get_item_size(self, data_item):
        '''get item size.'''
        try:
            if self.bucket_schedule is not None:
                if hasattr(data_item[self.bucket_schedule_key], 'shape'):
                    size = data_item[self.bucket_schedule_key].shape[self.bucket_schedule_shape]
                elif isinstance(data_item[self.bucket_schedule_key], int):
                    size = data_item[self.bucket_schedule_key]
                else:
                    size = -1
            else:
                size = 1
        except Exception:
            size = -1
        return size

    def collate_batch(self, data_item, max_batch_size):
        '''
        push data_item to bucket_list for collate batch.
        Args:
            data_item(any): data item.
            max_batch_size(int): max batch size.
        Return:
            batch_data(any): collated batch data if batch is full else None.
        '''
        size, bucket_idx = self.find_and_push_bucket(data_item)
        if size == -1:
            self.throw_num += 1
            if self.throw_num % self.bucket_skip_warning_num == 100:
                logging.warning(
                    "cannot find suitable bucket. You has already skipped %d data_item",
                    self.throw_num,
                )
            return None
        if isinstance(max_batch_size, (list, tuple)):
            max_batch_size = max_batch_size[bucket_idx]
        bsz = len(self.bucket_list[bucket_idx])

        if self.batch_means_tokens:
            if self.use_old_bucket and bsz * self.bucket_schedule[bucket_idx] > max_batch_size:
                batch_data = self.bucket_list[bucket_idx]
                self.clear(bucket_idx)
                return batch_data
            if (not self.use_old_bucket) and bsz * self.bucket_max_size[
                bucket_idx
            ] > max_batch_size:
                batch_data = self.bucket_list[bucket_idx]
                self.clear(bucket_idx)
                return batch_data
        if not self.batch_means_tokens and bsz > max_batch_size:
            batch_data = self.bucket_list[bucket_idx]
            self.clear(bucket_idx)
            return batch_data
        return None

    def collect_last_batch(self):
        '''collect batch data(s) that has not been get.'''
        last_batch = self.bucket_list
        self.clear(-1)
        return last_batch

    def clear(self, bucket_idx=-1):
        '''clear data buffer'''
        if bucket_idx >= 0:
            self.bucket_list[bucket_idx] = []
            self.bucket_batch_size[bucket_idx] = 0
            self.bucket_max_size[bucket_idx] = 0
        elif bucket_idx < 0:
            self.bucket_list = [[] for _ in range(self.bucket_num)]
            self.bucket_batch_size = [0 for _ in range(self.bucket_num)]
            self.bucket_max_size = [0 for _ in range(self.bucket_num)]


@BATCH_STRATEGY.register_module()
class SimpleBatching(BaseBatching):
    '''
    batch num bucket schedule.
    collate batch data depending on data item num.
    '''

    def __init__(self, **_kwargs):
        super().__init__()
        self.data_buffer = []

    def collate_batch(self, data_item, max_batch_size):
        '''
        draw data_item to buffer for collate batch.
        Args:
            data_item(any): data item.
            max_batch_size(int): max batch size.
        Return:
            batch_data(any): collated batch data if batch is full else None.
        '''
        self.data_buffer.append(data_item)
        bsz = len(self.data_buffer)
        if bsz >= max_batch_size:
            batch_data = self.data_buffer
            self.clear()
            return batch_data
        return None

    def collect_last_batch(self):
        '''collect batch data(s) that has not been get.'''
        last_batch = [self.data_buffer]
        self.clear()
        return last_batch

    def clear(self, _bucket_idx=-1):
        '''clear data buffer'''
        self.data_buffer = []


@BATCH_STRATEGY.register_module()
class RandomSimpleBatching(BaseBatching):
    '''
    batch num bucket schedule.
    collate batch data depending on data item num.
    '''

    def __init__(self, bucket_size, **_kwargs):
        super().__init__()
        self.bucket_size = bucket_size
        self.bucket_list = [[] for _ in range(bucket_size)]

    def find_and_push_bucket(self, data_item):
        '''find a suitable bucket and push to bucket.'''
        bucket_idx = random.randint(0, self.bucket_size - 1)
        self.bucket_list[bucket_idx].append(data_item)
        return bucket_idx

    def collate_batch(self, data_item, max_batch_size):
        '''
        draw data_item to buffer for collate batch.
        Args:
            data_item(any): data item.
            max_batch_size(int): max batch size.
        Return:
            batch_data(any): collated batch data if batch is full else None.
        '''
        bucket_idx = self.find_and_push_bucket(data_item)
        bsz = len(self.bucket_list[bucket_idx])
        if bsz >= max_batch_size:
            batch_data = self.bucket_list[bucket_idx]
            self.clear(bucket_idx)
            return batch_data
        return None

    def collect_last_batch(self):
        '''collect batch data(s) that has not been get.'''
        last_batch = self.bucket_list
        self.clear(-1)
        return last_batch

    def clear(self, bucket_idx=-1):
        '''clear data buffer'''
        if bucket_idx >= 0:
            self.bucket_list[bucket_idx] = []
        elif bucket_idx < 0:
            self.bucket_list = [[] for _ in range(self.bucket_size)]


@BATCH_STRATEGY.register_module()
class NoPaddingBucketBatching(BucketBatching):
    '''
    total size bucket schedule.
    collate batch data depending on total item size.
    '''

    def collate_batch(self, data_item, max_batch_size):
        '''
        push data_item to bucket_list for collate batch.
        Args:
            data_item(any): data item.
            max_batch_size(int): max batch size.
        Return:
            batch_data(any): collated batch data.
            batch_state(int):
                >=0, means this data_item has already been put into a bucket with
                bucket_idx=batch_state and the current bucket is already full.
                -1, otherwise.
        '''
        size, bucket_idx = self.find_and_push_bucket(data_item)
        if size == -1:
            self.throw_num += 1
            if self.throw_num % self.bucket_skip_warning_num == 100:
                logging.warning(
                    "cannot find suitable bucket. You has already skipped %d data_item",
                    self.throw_num,
                )
            return None
        if isinstance(max_batch_size, (list, tuple)):
            max_batch_size = max_batch_size[bucket_idx]
        if self.bucket_batch_size[bucket_idx] < max_batch_size:
            return None
        batch_data = self.bucket_list[bucket_idx]
        self.clear(bucket_idx)
        return batch_data


@BATCH_STRATEGY.register_module()
class FixedBucketBatching(BucketBatching):
    '''
    Fixed batch token bucket schedule.
    (bsz + 1) * bucket_frame > max_batch_size should draw batch.
    '''

    def collate_batch(self, data_item, max_batch_size):
        '''
        push data_item to bucket_list for collate batch.
        Args:
            data_item(any): data item.
            max_batch_size(int): max batch size.
        Return:
            batch_data(any): collated batch data if batch is full else None.
        '''
        size, bucket_idx = self.find_and_push_bucket(data_item)
        if size == -1:
            self.throw_num += 1
            if self.throw_num % self.bucket_skip_warning_num == 100:
                logging.warning(
                    "cannot find suitable bucket. You has already skipped %d data_item",
                    self.throw_num,
                )
            return None
        if isinstance(max_batch_size, (list, tuple)):
            max_batch_size = max_batch_size[bucket_idx]
        bsz = len(self.bucket_list[bucket_idx])
        if self.batch_means_tokens:
            next_bsz = (bsz + 1) * self.bucket_schedule[bucket_idx]
        else:
            next_bsz = bsz + 1

        if next_bsz > max_batch_size:
            batch_data = self.bucket_list[bucket_idx]
            self.clear(bucket_idx)
            return batch_data
        return None


@BATCH_STRATEGY.register_module()
class DomainBucketBatching(BaseBatching):
    '''
    batch token bucket schedule support different domain.
    collate batch data depending on bsz means token.
    '''

    def __init__(
        self,
        bucket_schedule,
        bucket_schedule_key,
        bucket_schedule_shape,
        use_old_bucket=True,
        batch_means_tokens=True,
        bucket_skip_warning_num=10000,
        domain_num=1,
        domain_key='',
        **_kwargs,
    ):
        super().__init__()
        self.domain_num = domain_num
        self.domain_key = domain_key
        self.bucket_schedule = bucket_schedule
        self.bucket_schedule_key = bucket_schedule_key
        self.bucket_schedule_shape = bucket_schedule_shape
        self.bucket_num = 1 if bucket_schedule is None else len(bucket_schedule)
        self.use_old_bucket = use_old_bucket
        self.batch_means_tokens = batch_means_tokens
        self.bucket_skip_warning_num = bucket_skip_warning_num
        self.bucket_list = [
            [[] for _ in range(self.bucket_num)] for _domian_id in range(self.domain_num)
        ]
        self.bucket_batch_size = [
            [0 for _ in range(self.bucket_num)] for _domian_id in range(self.domain_num)
        ]
        self.bucket_max_size = [
            [0 for _ in range(self.bucket_num)] for _domian_id in range(self.domain_num)
        ]

    def find_and_push_bucket(self, data_item):
        '''find a suitable bucket and push to bucket.'''
        size = self.get_item_size(data_item)
        if size == -1:
            return -1, -1, -1
        bucket_idx = find_bucket_idx(self.bucket_schedule, size)
        if bucket_idx < 0:
            return -1, -1, -1
        domain_id = data_item[self.domain_key]
        self.bucket_list[domain_id][bucket_idx].append(data_item)
        self.bucket_batch_size[domain_id][bucket_idx] += size
        self.bucket_max_size[domain_id][bucket_idx] = max(
            self.bucket_max_size[domain_id][bucket_idx], size
        )
        return size, bucket_idx, domain_id

    def get_item_size(self, data_item):
        '''get item size.'''
        try:
            if self.bucket_schedule is not None:
                if hasattr(data_item[self.bucket_schedule_key], 'shape'):
                    size = data_item[self.bucket_schedule_key].shape[self.bucket_schedule_shape]
                elif isinstance(data_item[self.bucket_schedule_key], int):
                    size = data_item[self.bucket_schedule_key]
                else:
                    size = -1
            else:
                size = 1
        except Exception:
            size = -1
        return size

    def collate_batch(self, data_item, max_batch_size):
        '''
        push data_item to bucket_list for collate batch.
        Args:
            data_item(any): data item.
            max_batch_size(int): max batch size.
        Return:
            batch_data(any): collated batch data if batch is full else None.
        '''
        size, bucket_idx, domain_id = self.find_and_push_bucket(data_item)
        if size == -1:
            self.throw_num += 1
            if self.throw_num % self.bucket_skip_warning_num == 100:
                logging.warning(
                    "cannot find suitable bucket. You has already skipped %d data_item",
                    self.throw_num,
                )
            return None
        if isinstance(max_batch_size, (list, tuple)):
            max_batch_size = max_batch_size[bucket_idx]
        bsz = len(self.bucket_list[domain_id][bucket_idx])

        if self.batch_means_tokens:
            if self.use_old_bucket and bsz * self.bucket_schedule[bucket_idx] > max_batch_size:
                batch_data = self.bucket_list[domain_id][bucket_idx]
                self.clear(bucket_idx, domain_id)
                return batch_data
            if (not self.use_old_bucket) and bsz * self.bucket_max_size[domain_id][
                bucket_idx
            ] > max_batch_size:
                batch_data = self.bucket_list[domain_id][bucket_idx]
                self.clear(bucket_idx, domain_id)
                return batch_data
        if not self.batch_means_tokens and bsz > max_batch_size:
            batch_data = self.bucket_list[domain_id][bucket_idx]
            self.clear(bucket_idx, domain_id)
            return batch_data
        return None

    def collect_last_batch(self):
        '''collect batch data(s) that has not been get.'''
        last_batch = sum(self.bucket_list, [])  # [[[]]] -> [[]]
        self.clear(-1)
        return last_batch

    def clear(self, bucket_idx=-1, domain_id=-1):
        '''clear data buffer'''
        if bucket_idx >= 0:
            self.bucket_list[domain_id][bucket_idx] = []
            self.bucket_batch_size[domain_id][bucket_idx] = 0
            self.bucket_max_size[domain_id][bucket_idx] = 0
        elif bucket_idx < 0:
            self.bucket_list = [
                [[] for _ in range(self.bucket_num)] for _domian_id in range(self.domain_num)
            ]
            self.bucket_batch_size = [
                [0 for _ in range(self.bucket_num)] for _domian_id in range(self.domain_num)
            ]
            self.bucket_max_size = [
                [0 for _ in range(self.bucket_num)] for _domian_id in range(self.domain_num)
            ]
