import logging
import random

from .base_iterator import _BaseContextualIterator
from .utils import merge_contextual_sample

logger = logging.getLogger(__name__)


class SameSpeakerIterator(_BaseContextualIterator):
    def __init__(
        self, min_context_num, max_context_num, time_interval_threshold, **kwargs
    ):
        self.min_context_num = min_context_num
        self.max_context_num = max_context_num
        self.time_interval_threshold = time_interval_threshold

    def __call__(self, items, cluster, sample_template):
        # sourcery skip: low-code-quality
        basis_item_cluster = cluster.features["index"]
        basis_items = items["index"]
        item_speakers = {
            item_info.uttid: basis_items[item_info.uttid]["speaker_id"]
            for item_info in basis_item_cluster.item_infos
            if cluster.parent_uttid == item_info.parent_uttid
        }
        raw_sorted_uttids = [
            item_info.uttid
            for item_info in basis_item_cluster.item_infos
            if cluster.parent_uttid == item_info.parent_uttid
            and item_info.uttid in item_speakers
        ]
        sorted_uttids = sorted(
            raw_sorted_uttids,
            key=lambda x: (item_speakers[x], raw_sorted_uttids.index(x)),
        )
        start_pos = 0
        item_cnt = 0
        while start_pos < len(sorted_uttids):
            context_len = random.randint(self.min_context_num, self.max_context_num)
            uttid = sorted_uttids[start_pos]
            sample_items = {
                name: [sub_items[uttid]] for name, sub_items in items.items()
            }
            last_end_time = sample_items["index"][0].get("end_time", -1)
            basis_speaker = sample_items["index"][0].get("speaker_id", "")

            # the context can be accepted must fit these conditions:
            # condition1: all item should have the same speaker
            # condition2: the time interval of continous 2 items
            #             should < time_interval_threshold
            # condition3: context length should equal to context_len
            next_start_pos = start_pos + 1
            for j in range(context_len):
                context_pos = start_pos + j + 1
                if context_pos >= len(sorted_uttids):
                    next_start_pos = context_pos
                    break

                context_uttid = sorted_uttids[context_pos]
                context_index_item = items["index"][context_uttid]
                context_start_time = context_index_item.get("start_time", -1)
                context_speaker = context_index_item.get("speaker_id", "")
                # check time interval & speaker
                if (
                    context_start_time < 0
                    or last_end_time < 0
                    or context_start_time - last_end_time > self.time_interval_threshold
                ) or (
                    basis_speaker == ""
                    or context_speaker == ""
                    or basis_speaker != context_speaker
                ):
                    next_start_pos = context_pos
                    break
                # append context to sample items
                for name, sub_items in items.items():
                    sample_items[name].append(sub_items[context_uttid])
                last_end_time = sample_items["index"][-1].get("end_time", -1)
                next_start_pos = context_pos + 1

            start_pos = next_start_pos
            # construct sample
            if len(sample_items["index"]) != context_len + 1:
                continue
            sample = merge_contextual_sample(sample_items)
            item_cnt += len(sample_items["index"])
            if sample is not None:
                sample.update(sample_template)
                yield sample
        if item_cnt < len(sorted_uttids) - self.max_context_num:
            logger.warning(
                f"total {len(sorted_uttids)} items, but yield {item_cnt} items"
            )
