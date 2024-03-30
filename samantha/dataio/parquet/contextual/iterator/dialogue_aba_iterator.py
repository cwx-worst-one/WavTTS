import logging

from .base_iterator import _BaseContextualIterator
from .utils import check_time_interval, get_time_interval, merge_contextual_sample

logger = logging.getLogger(__name__)


class DialogueABAIterator(_BaseContextualIterator):
    def __init__(
        self,
        main_speaker_round,
        max_num_speakers,
        time_interval_threshold,
        dialogue_aba_source_a_time_threshold,
        dialogue_aba_source_b_time_threshold,
        dialogue_aba_target_a_time_threshold,
        **kwargs,
    ):
        self.main_speaker_round = main_speaker_round
        self.max_num_speakers = max_num_speakers
        self.time_interval_threshold = time_interval_threshold
        self.dialogue_aba_source_a_time_threshold = dialogue_aba_source_a_time_threshold
        self.dialogue_aba_source_b_time_threshold = dialogue_aba_source_b_time_threshold
        self.dialogue_aba_target_a_time_threshold = dialogue_aba_target_a_time_threshold

    def __call__(self, items, cluster, sample_template):
        basis_item_cluster = cluster.features["index"]
        basis_items = items["index"]
        basis_cluster_items = [
            basis_items[item_info.uttid] for item_info in basis_item_cluster.item_infos
        ]
        raw_samples_info = self.find_dialog_samples_info(basis_cluster_items)
        for raw_sample_info in raw_samples_info:
            fixed_sample_info = self.fix_dialog_sample_info(
                basis_items, raw_sample_info
            )
            if fixed_sample_info is None:
                continue
            sample_items = {
                name: [items[name][uttid] for uttid in fixed_sample_info]
                for name in items.keys()
            }
            sample = merge_contextual_sample(sample_items)
            if sample is not None:
                sample.update(sample_template)
                yield sample

    def find_dialog_samples_info(self, data):
        subarrays = []
        start_index = 0
        while start_index < len(data):
            speaker_id = data[start_index]["speaker_id"]
            end_index = start_index
            contain_other_speakers = False
            start_other_speaker_index = -1
            end_other_speaker_index = -1
            main_speaker_round = 1
            cur_pos = start_index + 1
            while cur_pos < len(data):
                if data[cur_pos]["speaker_id"] != speaker_id:
                    if start_other_speaker_index == -1:
                        start_other_speaker_index = cur_pos
                    contain_other_speakers = True
                    cur_pos += 1
                    continue
                if not contain_other_speakers:
                    cur_pos += 1
                    continue
                end_index = cur_pos
                end_other_speaker_index = end_index - 1
                for j in range(end_index + 1, len(data)):
                    if data[j]["speaker_id"] != speaker_id:
                        break
                    end_index = j
                cur_pos = end_index + 1
                main_speaker_round += 1
                if main_speaker_round >= self.main_speaker_round:
                    break

            speakers = {
                data[m]["speaker_id"] for m in range(start_index, end_index + 1)
            }
            if (
                end_index > start_index + 1
                and contain_other_speakers
                and main_speaker_round >= self.main_speaker_round
                and len(speakers) >= 2
                and (
                    self.max_num_speakers == -1
                    or len(speakers) <= self.max_num_speakers
                )
            ):
                subarrays.append(
                    (
                        [item["uttid"] for item in data[start_index : end_index + 1]],
                        # data[start_index:end_index+1]["uttid"],
                        start_other_speaker_index - start_index,
                        end_other_speaker_index - start_index,
                    )
                )
            if start_other_speaker_index == -1:
                break
            start_index = start_other_speaker_index
        return subarrays

    def fix_dialog_sample_info(self, basis_items, raw_sample):
        uttids = raw_sample[0]
        start_other_speaker_index = raw_sample[1]
        end_other_speaker_index = raw_sample[2]
        # check time interval for each neighbor pair in source_b
        for i in range(start_other_speaker_index, end_other_speaker_index):
            if not self.check_time_interval(
                basis_items[uttids[i]], basis_items[uttids[i + 1]]
            ):
                return None

        # check source_b total time
        source_b_time = 0
        for i in range(start_other_speaker_index, end_other_speaker_index + 1):
            source_b_time += self.get_time_interval(basis_items[uttids[i]])
            if source_b_time > self.dialogue_aba_source_b_time_threshold:
                return None
        result = uttids[start_other_speaker_index : end_other_speaker_index + 1]

        # find valid source_a
        source_a_time = 0
        source_a_cnt = 0
        for i in range(start_other_speaker_index):
            cur_index = start_other_speaker_index - i - 1
            source_a_time += get_time_interval(basis_items[uttids[cur_index]])
            if source_a_time > self.dialogue_aba_source_a_time_threshold:
                break
            if not check_time_interval(
                basis_items[uttids[cur_index]],
                basis_items[uttids[cur_index + 1]],
                self.time_interval_threshold,
            ):
                break
            result.insert(0, uttids[cur_index])
            source_a_cnt += 1
        if source_a_cnt <= 0:
            return None

        # find valid target_a
        target_a_time = 0
        target_a_cnt = 0
        for cur_index in range(end_other_speaker_index + 1, len(uttids)):
            target_a_time += self.get_time_interval(basis_items[uttids[cur_index]])
            if target_a_time > self.dialogue_aba_target_a_time_threshold:
                break
            if not self.check_time_interval(
                basis_items[uttids[cur_index - 1]], basis_items[uttids[cur_index]]
            ):
                break
            result.append(uttids[cur_index])
            target_a_cnt += 1
        return None if target_a_cnt <= 0 else result
