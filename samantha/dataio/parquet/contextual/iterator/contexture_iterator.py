import logging
import random

from .base_iterator import _BaseContextualIterator
from .utils import merge_contextual_sample

logger = logging.getLogger(__name__)


class ContextualIterator(_BaseContextualIterator):
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
        for i in range(len(sorted_uttids)):
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
            # condition2: context length should equal to context_len
            # find context
            last_start_time = sample_items["index"][0].get("start_time", -1)
            for j in range(min(context_len, i)):
                context_uttid = sorted_uttids[i - j - 1]
                context_index_item = items["index"][context_uttid]
                context_end_time = context_index_item.get("end_time", -1)
                # check time interval
                if (
                    context_end_time < 0
                    or last_start_time < 0
                    or last_start_time - context_end_time > self.time_interval_threshold
                ):
                    break
                is_context_valid = all(
                    context_uttid in sub_items for _, sub_items in items.items()
                )
                if not is_context_valid:
                    continue
                # append context to sample items
                for name, sub_items in items.items():
                    sample_items[name] = [sub_items[context_uttid]] + sample_items[name]
                last_start_time = sample_items["index"][0].get("start_time", -1)

            # construct sample
            if len(sample_items["index"]) != context_len + 1:
                continue
            sample = merge_contextual_sample(sample_items)
            item_cnt += 1
            if sample is not None:
                sample.update(sample_template)
                yield sample
        if item_cnt < len(sorted_uttids) - self.max_context_num:
            logger.warning(
                f"total {len(sorted_uttids)} items, but yield {item_cnt} items"
            )
