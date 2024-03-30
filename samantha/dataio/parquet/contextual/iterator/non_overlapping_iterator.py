import logging
import random

from .base_iterator import _BaseContextualIterator
from .utils import merge_contextual_sample

logger = logging.getLogger(__name__)


class NonOverlappingIterator(_BaseContextualIterator):
    def __init__(
        self, min_context_num, max_context_num, time_interval_threshold, **kwargs
    ):
        self.min_context_num = min_context_num
        self.max_context_num = max_context_num
        self.time_interval_threshold = time_interval_threshold

    def __call__(self, items, cluster, sample_template):
        basis_item_cluster = cluster.features["index"]
        sorted_uttids = [item_info.uttid for item_info in basis_item_cluster.item_infos]
        item_cnt = 0
        i = 0
        while i < len(sorted_uttids):
            context_len = random.randint(self.min_context_num, self.max_context_num)
            uttid = sorted_uttids[i]
            sample_items = {}
            is_valid = True
            for name, sub_items in items.items():
                if uttid not in sub_items:
                    is_valid = False
                    break
                else:
                    sample_items[name] = [sub_items[uttid]]
            if not is_valid:
                continue
            # the context can be accepted must fit these conditions:
            # condition1: the time interval of continous 2 items
            #             should < time_interval_threshold
            # condition2: max num of context should < context_len (NOTE: not equal, unlike other strategies)
            # find context
            last_end_time = sample_items["index"][0].get("end_time", -1)
            j = 0
            for j in range(
                min(context_len, len(sorted_uttids) - i - 1)
            ):  # forward search
                context_uttid = sorted_uttids[i + j + 1]
                context_index_item = items["index"][context_uttid]
                context_start_time = context_index_item.get("start_time", -1)
                # check time interval
                if (
                    context_start_time < 0
                    or last_end_time < 0
                    or context_start_time - last_end_time > self.time_interval_threshold
                ):
                    break
                is_context_valid = all(
                    context_uttid in sub_items for _, sub_items in items.items()
                )
                if not is_context_valid:
                    continue
                # append context to sample items
                for name, sub_items in items.items():
                    sample_items[name] = sample_items[name] + [sub_items[context_uttid]]
                last_end_time = sample_items["index"][-1].get("end_time", -1)
            i = i + j + 2  # continue search from next item to avoid overlapping
            # construct sample
            sample = merge_contextual_sample(sample_items)
            item_cnt += 1
            if sample is not None:
                sample.update(sample_template)
                yield sample
