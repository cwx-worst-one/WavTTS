"""bucket process module"""
import logging
from typing import Callable, List

logger = logging.getLogger(__name__)


class BucketBatcher:

    r"""Separate samples into different buckets according its size calculated by
    ``length_fn``, and collate batches from each bucket once their size satisfied
    the ``maximum_bucket_size`` when ``dynamic_batch`` is on, ``batch_size`` when off.

    Args:
        buckets (List[int]): Each number indicates the longest sample the bucket
            accepts. For example, if the buckets set as [100, 200, 300, 400, 500], then
            first bucket only accepts if sample length in (0, 100], second bucket is
            (100, 200], etc.
        dynamic_batch (bool): Dynamic batch or fixed batch. If True, each batch collated
            has non-deterministic batch size, but the total length of the batch will
            strictly less than ``maximum_bucket_size``. If False, each batch has fixed
            batch size set by ``batch_size``.
        maximum_bucket_size (int): Maximum length of total length of data in each batch,
            only valid when ``dynamic_batch`` is True.
        batch_size (int): Batch size of each batch, only valid when ``dynamic_batch``
            is False.
        length_fn (Callable): Function to be applied to each element to get lengths.
            len(data) is used by default.
        bucket_skip_warning_num (int): False tolerant number.
    """

    def __init__(
        self,
        buckets: List[int] = None,
        dynamic_batch: bool = True,
        maximum_bucket_size: int = None,
        batch_size: int = None,
        length_fn: Callable = len,
        bucket_skip_warning_num: int = 10000,
    ):

        if buckets is None:
            buckets = [2**31]

        if dynamic_batch and maximum_bucket_size is None:
            raise ValueError(
                "Expecting maximum_bucket_size be provided when dynamic_batch is True."
            )

        if not dynamic_batch and batch_size is None:
            raise ValueError(
                "Expecting batch_size be provided when dynamic_batch is False."
            )

        self.buckets = buckets
        self.dynamic_batch = dynamic_batch
        self.maximum_bucket_size = maximum_bucket_size
        self.batch_size = batch_size
        self.length_fn = length_fn
        self.bucket_skip_warning_num = bucket_skip_warning_num
        self.bucket_num = 1 if self.buckets is None else len(self.buckets)
        self.bucket_list = [[] for _ in range(self.bucket_num)]
        self.bucket_size = [0 for _ in range(self.bucket_num)]
        self.bucket_max_size = [0 for _ in range(self.bucket_num)]
        self.throw_num = 0

    def find_bucket(self, data_item):
        """find a suitable bucket and push to bucket."""
        size = self.get_item_size(data_item)
        if size is None:
            return None, None
        bucket_idx = self.find_bucket_idx(size)
        if bucket_idx is None:
            return None, None
        return size, bucket_idx

    def find_bucket_idx(self, size):
        r"""find bucket idx for a size.

        Args:
            size(int): size of a data item
        Returns:
            int: the minimum bucket idx for this size, which match
                 `size <= bucket_schedule[idx]`.
                 -1 means no bucket match.
        """

        if size > self.buckets[-1]:
            logger.warning(
                f"{size=} exceeding the maximum bucket size {self.buckets[-1]}."
            )
            return None

        bucket_length = len(self.buckets)
        low = -1
        high = bucket_length - 1
        while low + 1 < high:
            mid = (high + low) >> 1
            if self.buckets[mid] < size:
                low = mid
            else:
                high = mid
        return high

    def push_bucket(self, data_item, size, bucket_idx):
        self.bucket_list[bucket_idx].append(data_item)
        self.bucket_size[bucket_idx] += size
        self.bucket_max_size[bucket_idx] = max(self.bucket_max_size[bucket_idx], size)

    def get_item_size(self, data_item):
        """get item size."""
        try:
            size = self.length_fn(data_item)
        except Exception as e:
            logger.warning(f"Failed to calculate data length with error message {e}")
            size = None
        return size

    def collate_batch(self, data_item):
        """
        push data_item to bucket_list for collate batch.
        Args:
            data_item(any): data item.
        Returns:
            batch_data(any): collated batch data if batch is full else None.
        """
        size, bucket_idx = self.find_bucket(data_item)
        if size is None:
            self.throw_num += 1
            if self.throw_num % self.bucket_skip_warning_num == 100:
                logger.warning(
                    f"Cannot find suitable bucket. You have already "
                    f"skipped {self.throw_num} data_item"
                )
            return None

        max_batch_size = max(self.bucket_max_size[bucket_idx], size)
        bsz = len(self.bucket_list[bucket_idx]) + 1

        if self.dynamic_batch:
            total_size = bsz * max_batch_size

            if total_size == self.maximum_bucket_size:
                self.push_bucket(data_item, size, bucket_idx)
                batch_data = self.bucket_list[bucket_idx]
                self.clear(bucket_idx)
                return batch_data
            elif total_size > self.maximum_bucket_size:
                batch_data = self.bucket_list[bucket_idx]
                self.clear(bucket_idx)
                self.push_bucket(data_item, size, bucket_idx)
                return batch_data
        else:
            if bsz == self.batch_size:
                self.push_bucket(data_item, size, bucket_idx)
                batch_data = self.bucket_list[bucket_idx]
                self.clear(bucket_idx)
                return batch_data

        self.push_bucket(data_item, size, bucket_idx)

        return None

    def collect_last_batch(self):
        """collect batch data(s) that has not been get."""
        last_batch = self.bucket_list
        self.clear()
        return last_batch

    def clear(self, bucket_idx=None):
        """clear data buffer"""
        if bucket_idx is None:
            self.bucket_list = [[] for _ in range(self.bucket_num)]
            self.bucket_size = [0 for _ in range(self.bucket_num)]
            self.bucket_max_size = [0 for _ in range(self.bucket_num)]
        else:
            assert bucket_idx >= 0
            self.bucket_list[bucket_idx] = []
            self.bucket_size[bucket_idx] = 0
            self.bucket_max_size[bucket_idx] = 0
