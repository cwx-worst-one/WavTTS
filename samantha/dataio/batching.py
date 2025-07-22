import copy
import operator
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from cruise.utilities.logger import get_cruise_logger

logger = get_cruise_logger()


def setup_batcher_fn(cfg):
    """Factory function to create and configure a batcher instance.

    This function dynamically creates a batcher object based on the specified type
    in the configuration. It extracts common parameters with default values and
    passes them to the appropriate batcher class constructor.

    The function supports different batcher types:
    - SimpleBatcher: Basic batching with fixed batch size
    - BucketBatcher: Groups samples into buckets based on their size
    - TaggedBucketBatcher: Groups samples by tag and size
    - TokenBucketBatcher: Specialized for token-based batching

    Args:
        cfg (dict): Configuration dictionary containing:
            - type: Batcher class name (default: 'SimpleBatcher')
            - maximum_bucket_size: Maximum size for dynamic batching (default: 1)
            - dynamic_batch: Whether to use dynamic batching (default: True)
            - batch_size: Fixed batch size for non-dynamic batching (default: 1)
            - bucket_skip_warning_num: Warning threshold for skipped items (default: 10000)
            - buckets: List of bucket sizes (default: None)
            - bucket_schedule_key: Key to extract item size (default: 'num_total_tokens')
            - Additional parameters specific to each batcher type

    Returns:
        object: Configured batcher instance of the specified type
    """
    # Create a deep copy to avoid modifying the original config
    cfg = copy.deepcopy(cfg)

    # Extract common parameters with default values
    batch_strategy = cfg.pop("type", "SimpleBatcher")
    maximum_bucket_size = cfg.pop("maximum_bucket_size", 1)
    dynamic_batch = cfg.pop("dynamic_batch", True)
    batch_size = cfg.pop("batch_size", 1)
    bucket_skip_warning_num = cfg.pop("bucket_skip_warning_num", 10000)
    buckets = cfg.pop("buckets", None)
    bucket_schedule_key = cfg.pop("bucket_schedule_key", "num_total_tokens")

    # Get the batcher class from the global namespace
    batch_strategy = globals()[batch_strategy]

    # Create and return the batcher instance with extracted parameters
    # Any remaining config items are passed as additional kwargs
    batcher = batch_strategy(
        buckets=buckets,
        dynamic_batch=dynamic_batch,
        maximum_bucket_size=maximum_bucket_size,
        batch_size=batch_size,
        bucket_skip_warning_num=bucket_skip_warning_num,
        bucket_schedule_key=bucket_schedule_key,
        **cfg,
    )

    return batcher


class SimpleBatcher:
    """
    batch num bucket schedule.
    collate batch data depending on data item num.
    """

    def __init__(self, batch_size=1, **kwargs):
        self.data_buffer = []
        self.batch_size = batch_size

    def collate_batch(self, data_item):
        """
        draw data_item to buffer for collate batch.
        Args:
            data_item(any): data item.
            max_batch_size(int): max batch size.
        Return:
            batch_data(any): collated batch data if batch is full else None.
        """
        self.data_buffer.append(data_item)
        bsz = len(self.data_buffer)
        if bsz >= self.batch_size:
            batch_data = self.data_buffer
            self.clear()
            return batch_data
        return None

    def collect_last_batch(self):
        """collect batch data(s) that has not been get."""
        last_batch = [self.data_buffer]
        self.clear()
        return last_batch

    def clear(self, _bucket_idx=-1):
        """clear data buffer"""
        self.data_buffer = []


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
        length_fn: Callable = None,
        bucket_skip_warning_num: int = 10000,
        bsz_evaluator: Optional[Callable] = None,
        bucket_size_fn: Optional[Callable] = None,
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
        self.length_fn = length_fn or len
        self.bucket_skip_warning_num = bucket_skip_warning_num
        self.bucket_num = 1 if self.buckets is None else len(self.buckets)
        self.bucket_list = [[] for _ in range(self.bucket_num)]
        self.bucket_size = [0 for _ in range(self.bucket_num)]
        self.bucket_max_size = [0 for _ in range(self.bucket_num)]
        self.throw_num = 0
        self.bsz_evaluator = bsz_evaluator or operator.mul
        self.bucket_size_fn = bucket_size_fn

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

    def _max_batch_size(self, bucket_idx, current_size):
        if self.bucket_size_fn is not None:
            return self.bucket_size_fn(self.bucket_list[bucket_idx])
        return max(self.bucket_max_size[bucket_idx], current_size)

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

        max_batch_size = self._max_batch_size(bucket_idx, size)
        bsz = len(self.bucket_list[bucket_idx]) + 1

        if self.dynamic_batch:
            total_size = self.bsz_evaluator(bsz, max_batch_size)

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


class TaggedBucketBatcher:
    def __init__(
        self,
        buckets: Dict[str, List[int]],
        batch_size: int,
        tag_fn: Callable = lambda x: x["bucket_tag"],
        length_fn: Callable = len,
        bucket_skip_warning_num: int = 10000,
    ):
        self.buckets = buckets
        self.batch_size = batch_size
        self.tag_fn = tag_fn
        self.length_fn = length_fn
        self.bucket_skip_warning_num = bucket_skip_warning_num
        self.bucket_num = {k: len(v) for k, v in self.buckets.items()}
        self.bucket_list = {
            k: [[] for _ in range(v)] for k, v in self.bucket_num.items()
        }
        self.throw_num = 0

    def find_bucket(self, data_item):
        """find a suitable bucket and push to bucket."""
        tag, size = self.get_item_size(data_item)
        if tag is None or size is None:
            return None, None, None
        bucket_idx = self.find_bucket_idx(tag, size)
        if bucket_idx is None:
            return None, None, None
        return tag, size, bucket_idx

    def find_bucket_idx(self, tag, size):
        r"""find bucket idx for a size.

        Args:
            size(int): size of a data item
        Returns:
            int: the minimum bucket idx for this size, which match
                 `size <= bucket_schedule[idx]`.
                 -1 means no bucket match.
        """

        if size > self.buckets[tag][-1]:
            logger.warning(
                (
                    f"{size=} exceeding the maximum bucket"
                    + "[{tag}] size {self.buckets[tag][-1]}."
                )
            )
            return None

        bucket_length = len(self.buckets[tag])
        low = -1
        high = bucket_length - 1
        while low + 1 < high:
            mid = (high + low) >> 1
            if self.buckets[tag][mid] < size:
                low = mid
            else:
                high = mid
        return high

    def push_bucket(self, data_item, tag, bucket_idx):
        self.bucket_list[tag][bucket_idx].append(data_item)

    def get_item_size(self, data_item):
        """get item size."""
        try:
            tag = self.tag_fn(data_item)
            size = self.length_fn(data_item)
        except Exception as e:
            logger.warning(f"Failed to calculate data length with error message {e}")
            tag = None
            size = None
        return tag, size

    def collate_batch(self, data_item):
        """
        push data_item to bucket_list for collate batch.
        Args:
            data_item(any): data item.
        Returns:
            batch_data(any): collated batch data if batch is full else None.
        """
        tag, size, bucket_idx = self.find_bucket(data_item)
        if tag is None or size is None or bucket_idx is None:
            self.throw_num += 1
            if self.throw_num % self.bucket_skip_warning_num == 100:
                logger.warning(
                    f"Cannot find suitable bucket. You have already "
                    f"skipped {self.throw_num} data_item"
                )
            return None

        bsz = len(self.bucket_list[tag][bucket_idx]) + 1
        self.push_bucket(data_item, tag, bucket_idx)

        if bsz >= self.batch_size:
            batch_data = self.bucket_list[tag][bucket_idx]
            self.clear(tag, bucket_idx)
            return batch_data
        else:
            return None

    def clear(self, tag, bucket_idx):
        """clear data buffer"""
        self.bucket_list[tag][bucket_idx] = []


class TokenBucketBatcher:
    def __init__(
        self,
        buckets: List[int] = None,
        dynamic_batch: bool = True,
        maximum_bucket_size: int = None,
        batch_size: int = None,
        bucket_schedule_key: str = "num_total_tokens",
        bucket_skip_warning_num: int = 10000,
        frame_rate: int = 50,  # estimated  audio HZ：25 lyrics/phonem token HZ: 15
        sample_rate: int = 24000,
    ):

        if buckets is None:
            # in seconds
            self.buckets = [i * frame_rate for i in range(10, 240, 10)]
        else:
            self.buckets = buckets

        logger.info(f"buckets: {self.buckets}")

        if dynamic_batch and maximum_bucket_size is None:
            raise ValueError(
                "Expecting maximum_bucket_size be provided when dynamic_batch is True."
            )

        if not dynamic_batch and batch_size is None:
            raise ValueError(
                "Expecting batch_size be provided when dynamic_batch is False."
            )

        self.bucket_schedule_key = bucket_schedule_key
        self.dynamic_batch = dynamic_batch
        self.maximum_bucket_size = maximum_bucket_size
        self.bucket_num = len(self.buckets)
        self.bucket_list = [[] for _ in range(self.bucket_num)]
        self.bucket_size = [0 for _ in range(self.bucket_num)]
        self.bucket_max_size = [0 for _ in range(self.bucket_num)]
        self.throw_num = 0
        self.sample_rate = sample_rate
        self.bucket_skip_warning_num = bucket_skip_warning_num

    def get_item_size(self, data_item):
        """Get item size in seconds from audio data or direct length value.

        Args:
            data_item (dict): Dictionary containing audio data or length

        Returns:
            int or None: Duration in seconds, None if invalid data
        """
        size = data_item.get(self.bucket_schedule_key, None)
        if size is None:
            logger.warning(f"Missing key: {self.bucket_schedule_key}")
            return None
        return size

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

    def _max_batch_size(self, bucket_idx, current_size):
        return max(self.bucket_max_size[bucket_idx], current_size)

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
            if self.throw_num % self.bucket_skip_warning_num == 0:
                logger.warning(
                    f"Cannot find suitable bucket. You have already "
                    f"skipped {self.throw_num} data_item"
                )
            return None

        max_batch_size = self._max_batch_size(bucket_idx, size)
        bsz = len(self.bucket_list[bucket_idx]) + 1

        if self.dynamic_batch:
            total_size = bsz * max_batch_size

            if total_size >= self.maximum_bucket_size:
                batch_data = self.bucket_list[bucket_idx]
                self.clear(bucket_idx)
                self.push_bucket(data_item, size, bucket_idx)
                return batch_data
        else:
            if bsz == self.batch_size:
                batch_data = self.bucket_list[bucket_idx]
                self.clear(bucket_idx)
                self.push_bucket(data_item, size, bucket_idx)
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


class BalancedTokenBucketBatcher:
    def __init__(
        self,
        buckets: List[int] = None,
        dynamic_batch: bool = True,
        maximum_bucket_size: int = None,
        batch_size: int = None,
        bucket_schedule_key: str = "num_total_tokens",
        balance_key: str = None,
        bucket_skip_warning_num: int = 10000,
        frame_rate: int = 50,
        sample_rate: int = 24000,
    ):
        if buckets is None:
            self.buckets = [i * frame_rate for i in range(10, 240, 10)]
        else:
            self.buckets = buckets

        logger.info(f"buckets: {self.buckets}")

        if dynamic_batch and maximum_bucket_size is None:
            raise ValueError(
                "Expecting maximum_bucket_size be provided when dynamic_batch is True."
            )

        if not dynamic_batch and batch_size is None:
            raise ValueError(
                "Expecting batch_size be provided when dynamic_batch is False."
            )

        self.bucket_schedule_key = bucket_schedule_key
        self.dynamic_batch = dynamic_batch
        self.maximum_bucket_size = maximum_bucket_size
        self.batch_size = batch_size
        self.bucket_num = len(self.buckets)

        # Each bucket is a dictionary where the key is from `balance_key` and the value is a list of items.
        self.bucket_list: List[defaultdict[str, List[Any]]] = [
            defaultdict(list) for _ in range(self.bucket_num)
        ]

        self.bucket_size = [0 for _ in range(self.bucket_num)]
        self.bucket_max_size = [0 for _ in range(self.bucket_num)]

        # --- RENAMED ATTRIBUTE ---
        self.balance_key = balance_key
        if self.balance_key is None:
            logger.warning(
                "balance_key is not provided. The batcher will not perform balancing."
            )

        self.throw_num = 0
        self.sample_rate = sample_rate
        self.bucket_skip_warning_num = bucket_skip_warning_num

    def get_item_size(self, data_item):
        size = data_item.get(self.bucket_schedule_key, None)
        if size is None:
            logger.warning(f"Missing key: {self.bucket_schedule_key}")
            return None
        return size

    def find_bucket_idx(self, size):
        if size > self.buckets[-1]:
            logger.warning(
                f"{size=} exceeding the maximum bucket size {self.buckets[-1]}."
            )
            return None

        # (Binary search logic remains the same)
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
        if self.balance_key:
            # Place data into the corresponding sub-list based on the balance key's value
            category = data_item.get(self.balance_key, "unknown")
            self.bucket_list[bucket_idx][category].append(data_item)
        else:
            # If balance_key is not provided, fall back to a default category
            if "default" not in self.bucket_list[bucket_idx]:
                self.bucket_list[bucket_idx]["default"] = []
            self.bucket_list[bucket_idx]["default"].append(data_item)

        self.bucket_size[bucket_idx] += size
        self.bucket_max_size[bucket_idx] = max(self.bucket_max_size[bucket_idx], size)

    def _max_batch_size(self, bucket_idx, current_size):
        return max(self.bucket_max_size[bucket_idx], current_size)

    def _create_balanced_batch(self, bucket_idx: int) -> List[Any]:
        """
        Creates a balanced batch from a bucket using a Round-Robin strategy.
        This function consumes the items from the bucket.
        """
        category_dict = self.bucket_list[bucket_idx]
        if not category_dict:
            return []

        # If balancing is disabled, just flatten all lists
        if not self.balance_key:
            batch = [item for sublist in category_dict.values() for item in sublist]
            return batch

        balanced_batch = []
        # Convert the dictionary to a list of (category, items_list) for polling
        category_queues = list(category_dict.items())

        # Use an index to simulate taking an element from each queue in turn
        item_idx = 0
        while True:
            items_added_in_this_round = 0
            for category, items in category_queues:
                if item_idx < len(items):
                    balanced_batch.append(items[item_idx])
                    items_added_in_this_round += 1

            if items_added_in_this_round == 0:
                # If a full round adds no items, all queues are exhausted
                break
            item_idx += 1

        return balanced_batch

    def collate_batch(self, data_item):
        size, bucket_idx = self.find_bucket(data_item)
        if size is None:
            self.throw_num += 1
            if self.throw_num % self.bucket_skip_warning_num == 0:
                logger.warning(
                    f"Cannot find suitable bucket. You have already "
                    f"skipped {self.throw_num} data_item"
                )
            return None

        # Calculate the total number of items across all categories in the current bucket
        current_total_items = sum(
            len(items) for items in self.bucket_list[bucket_idx].values()
        )
        bsz = current_total_items + 1

        max_batch_size = self._max_batch_size(bucket_idx, size)

        batch_is_ready = False
        if self.dynamic_batch:
            total_size = bsz * max_batch_size
            if total_size >= self.maximum_bucket_size:
                batch_is_ready = True
        else:
            if self.batch_size is not None and bsz >= self.batch_size:
                batch_is_ready = True

        if batch_is_ready:
            # 1. Create a batch using the balanced sampling algorithm
            batch_data = self._create_balanced_batch(bucket_idx)
            # 2. Clear the bucket
            self.clear(bucket_idx)
            # 3. Add the current item to the now-empty bucket
            self.push_bucket(data_item, size, bucket_idx)
            return batch_data

        # If the batch is not full, just push the data item
        self.push_bucket(data_item, size, bucket_idx)
        return None

    def collect_last_batch(self):
        all_remaining_batches = []
        for i in range(self.bucket_num):
            # Check if there are any items in any category list for this bucket
            if any(self.bucket_list[i].values()):
                # Apply balanced sampling to all remaining buckets
                batch = self._create_balanced_batch(i)
                if batch:
                    all_remaining_batches.append(batch)
        self.clear()  # Clear all data after collection
        return all_remaining_batches

    def clear(self, bucket_idx=None):
        if bucket_idx is None:
            # Clear all buckets
            self.bucket_list = [defaultdict(list) for _ in range(self.bucket_num)]
            self.bucket_size = [0 for _ in range(self.bucket_num)]
            self.bucket_max_size = [0 for _ in range(self.bucket_num)]
        else:
            # Clear a specific bucket
            assert bucket_idx >= 0
            self.bucket_list[bucket_idx].clear()
            self.bucket_size[bucket_idx] = 0
            self.bucket_max_size[bucket_idx] = 0

    def find_bucket(self, data_item):
        """Find a suitable bucket for a data item."""
        size = self.get_item_size(data_item)
        if size is None:
            return None, None
        bucket_idx = self.find_bucket_idx(size)
        if bucket_idx is None:
            return None, None
        return size, bucket_idx


class HierarchicalTokenBucketBatcher:
    """
    A streaming batcher that supports hierarchical bucketing strategy.

    It now intelligently handles instrumental tracks (where text_length is 0)
    by assigning them a configurable, healthy target ratio.
    """

    def __init__(
        self,
        buckets: List[int],
        bucket_schedule_key: str = "audio_shape",
        bucket_secondary_key: str = "text_length",
        instrumental_target_ratio: float = 3.0,
        batch_size: int = None,
        maximum_bucket_size: int = None,
        dynamic_batch: bool = True,
        drop_last: bool = True,
        bucket_skip_warning_num: int = 10000,
        **kwargs,
    ):
        """
        Initializes the batcher.
        Args:
            instrumental_target_ratio (float): The T/L ratio to assign to instrumental tracks.
            ... (other parameters)
        """
        if dynamic_batch and maximum_bucket_size is None:
            raise ValueError(
                "maximum_bucket_size must be provided for dynamic batching."
            )
        if not dynamic_batch and batch_size is None:
            raise ValueError("batch_size must be provided for fixed batching.")

        self.buckets = sorted(buckets)
        self.batch_size = batch_size
        self.maximum_bucket_size = maximum_bucket_size
        self.dynamic_batch = dynamic_batch
        self.bucket_schedule_key = bucket_schedule_key
        self.bucket_secondary_key = bucket_secondary_key
        self.instrumental_target_ratio = instrumental_target_ratio
        self.drop_last = drop_last
        self.bucket_skip_warning_num = bucket_skip_warning_num

        self.bucket_num = len(self.buckets) + 1
        self.bucket_list = [[] for _ in range(self.bucket_num)]
        self.bucket_max_len = [0 for _ in range(self.bucket_num)]
        self.throw_num = 0

    def _get_item_keys(self, data_item: Dict[str, Any]) -> Tuple[int, int, float]:
        """Gets primary key, secondary key, and ratio from a data item."""
        primary_val = data_item.get(self.bucket_schedule_key)
        secondary_val = data_item.get(self.bucket_secondary_key)

        if primary_val is None or secondary_val is None:
            return None, None, None

        # --- NEW LOGIC FOR RATIO CALCULATION ---
        if secondary_val == 0:
            # This is an instrumental track. Assign the target ratio.
            ratio = self.instrumental_target_ratio
        else:
            # This is a vocal track. Calculate the true ratio.
            ratio = primary_val / secondary_val

        return primary_val, secondary_val, ratio

    def _find_bucket_idx(self, size: int) -> int:
        """Finds the corresponding bucket index for a given size."""
        for i, boundary in enumerate(self.buckets):
            if size <= boundary:
                return i
        return len(self.buckets)

    def _push_to_bucket(
        self, bucket_idx: int, data_item: Dict[str, Any], ratio: float, primary_val: int
    ):
        """Pushes the data item, its ratio, and its length into a bucket."""
        self.bucket_list[bucket_idx].append((data_item, ratio))
        self.bucket_max_len[bucket_idx] = max(
            self.bucket_max_len[bucket_idx], primary_val
        )

    def collate_batch(self, data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes a single data item and returns a batch when a bucket is ready.
        """
        primary_val, _, ratio = self._get_item_keys(data_item)

        if primary_val is None:
            self.throw_num += 1
            if self.throw_num % self.bucket_skip_warning_num == 0:
                logger.warning(
                    f"Skipped {self.throw_num} items due to missing key "
                    f"'{self.bucket_schedule_key}' or '{self.bucket_secondary_key}'"
                )
            return None

        bucket_idx = self._find_bucket_idx(primary_val)

        batch_ready = False
        if self.dynamic_batch:
            current_bsz = len(self.bucket_list[bucket_idx])
            estimated_max_len = max(self.bucket_max_len[bucket_idx], primary_val)
            estimated_total_tokens = (current_bsz + 1) * estimated_max_len

            if estimated_total_tokens >= self.maximum_bucket_size and current_bsz > 0:
                batch_ready = True
        else:
            if len(self.bucket_list[bucket_idx]) >= self.batch_size:
                batch_ready = True

        if batch_ready:
            bucket_to_process = self.bucket_list[bucket_idx]

            bucket_to_process.sort(key=lambda x: x[1])
            batch_data = [item for item, ratio in bucket_to_process]

            self.clear(bucket_idx)
            self._push_to_bucket(bucket_idx, data_item, ratio, primary_val)

            return batch_data

        self._push_to_bucket(bucket_idx, data_item, ratio, primary_val)
        return None

    def collect_last_batches(self) -> List[List[Dict[str, Any]]]:
        """Collects all remaining data in buckets as final batches."""
        all_last_batches = []
        for bucket_idx in range(self.bucket_num):
            remaining_items = self.bucket_list[bucket_idx]
            if remaining_items:
                if not self.drop_last and len(remaining_items) > 0:
                    remaining_items.sort(key=lambda x: x[1])
                    batch_data = [item for item, ratio in remaining_items]
                    all_last_batches.append(batch_data)

        self.clear()
        return all_last_batches

    def clear(self, bucket_idx: int = None):
        """Clears the data buffers."""
        if bucket_idx is not None:
            self.bucket_list[bucket_idx] = []
            self.bucket_max_len[bucket_idx] = 0
        else:
            self.bucket_list = [[] for _ in range(self.bucket_num)]
            self.bucket_max_len = [0 for _ in range(self.bucket_num)]
