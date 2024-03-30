import logging
import random

from .base_iterator import _BaseContextualIterator
from .utils import merge_contextual_sample

logger = logging.getLogger(__name__)


class GreedyIterator(_BaseContextualIterator):
    def __init__(
        self,
        min_num_speakers,
        max_num_speakers,
        max_duration,
        time_interval_threshold,
        split_context,
        **kwargs,
    ):
        self.min_num_speakers = min_num_speakers
        self.max_num_speakers = max_num_speakers
        self.max_duration = max_duration
        self.time_interval_threshold = time_interval_threshold
        self.split_context = split_context

    def __call__(self, items, cluster, sample_template):
        # sourcery skip: low-code-quality
        basis_item_cluster = cluster.features["index"]
        basis_items = items["index"]
        item_speakers = {
            item_info.uttid: basis_items[item_info.uttid]["speaker_id"]
            for item_info in basis_item_cluster.item_infos
            if cluster.parent_uttid == item_info.parent_uttid
        }
        sorted_uttids = sorted(
            [
                item_info.uttid
                for item_info in basis_item_cluster.item_infos
                if cluster.parent_uttid == item_info.parent_uttid
                and item_info.uttid in item_speakers
            ]
        )
        start_pos = 0
        item_cnt = 0
        speaker_set = set()
        while start_pos < len(sorted_uttids):
            uttid = sorted_uttids[start_pos]
            sample_items = {
                name: [sub_items[uttid]] for name, sub_items in items.items()
            }
            last_end_time = sample_items["index"][0].get("end_time", -1)
            basis_speaker = sample_items["index"][0].get("speaker_id", "")
            speaker_set.clear()
            speaker_set.add(basis_speaker)

            # the context can be accepted must fit these conditions:
            # condition1: the total number of speakers must be within [self.min_num_speakers, self.max_num_speakers]
            # condition2: the time interval of continous 2 items
            #             should < time_interval_threshold
            # condition3: the total duration of this session is within self.max_duration
            current_pos = next_start_pos = start_pos + 1
            last_speaker_change_pos = None
            session_duration = 0.0
            while current_pos < len(sorted_uttids):
                current_uttid = sorted_uttids[current_pos]
                current_index_item = items["index"][current_uttid]
                current_start_time = current_index_item.get("start_time", -1)
                current_duration = (
                    current_index_item.get("end_time", -1) - current_start_time
                )
                current_speaker = current_index_item.get("speaker_id", "")
                # break if exceeded time interval
                if (
                    current_start_time < 0
                    or last_end_time < 0
                    or current_start_time - last_end_time > self.time_interval_threshold
                ):
                    next_start_pos = (
                        last_speaker_change_pos
                        if last_speaker_change_pos
                        else current_pos
                    )
                    break
                # check when speaker changes
                if (
                    basis_speaker == ""
                    or current_speaker == ""
                    or basis_speaker != current_speaker
                ):
                    if len(speaker_set) < self.max_num_speakers:
                        # continue if not reached max_num_speakers, otherwise end this round
                        speaker_set.add(current_speaker)
                        basis_speaker = current_speaker
                        last_speaker_change_pos = current_pos
                    else:
                        # in multi-speaker scenario, restart from last speaker change point
                        next_start_pos = (
                            last_speaker_change_pos
                            if last_speaker_change_pos
                            else current_pos
                        )
                        break
                # check total duration
                if session_duration + current_duration >= self.max_duration:
                    next_start_pos = (
                        last_speaker_change_pos
                        if last_speaker_change_pos
                        else current_pos
                    )
                    break
                else:
                    session_duration += current_duration
                # append context to sample items
                for name, sub_items in items.items():
                    sample_items[name].append(sub_items[current_uttid])
                last_end_time = sample_items["index"][-1].get("end_time", -1)
                current_pos += 1
                next_start_pos = current_pos + 1

            start_pos = next_start_pos
            if len(speaker_set) < self.min_num_speakers:
                continue
            # construct sample
            sample = merge_contextual_sample(sample_items)
            if self.split_context and len(speaker_set) < self.max_num_speakers:
                n_utts = len(sample["contextual_speaker_list"])
                if n_utts == 1:
                    continue
                ctx_split_point = random.randint(0, n_utts - 2)
                for i_ctx in range(ctx_split_point):
                    sample["contextual_speaker_list"][i_ctx] += "_ctx"
            item_cnt += len(sample_items["index"])
            if sample is not None:
                sample.update(sample_template)
                yield sample
        if item_cnt < len(sorted_uttids):
            logger.warning(
                f"total {len(sorted_uttids)} items, but yield {item_cnt} items"
            )
