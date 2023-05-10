"""for measuring partial result stability of streaming asr"""

from core.utils.misc import infer_text_format


def merge_stable_metric(stable_metric_list):
    """to form stable metric for a testset"""
    utt_num = len(stable_metric_list)
    unstable_word_num = sum(m.tot_unstable_word_num for m in stable_metric_list)
    final_hyp_word_num = sum(m.final_hyp_word_num for m in stable_metric_list)
    unstable_seg_num = sum(m.tot_unstable_seg_num for m in stable_metric_list)

    upwr = unstable_word_num / final_hyp_word_num
    upsr = unstable_seg_num / utt_num
    return upwr, upsr


class StreamingStableMetricOneSample:
    """
    streaming stable metric for one sample
    UPWR - unstable partial word ratio
    UPSR - unstable partial segment ratio
    https://arxiv.org/pdf/2005.09137.pdf
    """

    def __init__(self, ms_per_packet, ms_per_frame, tgt_dict, filter_list, formator):
        '''init.'''
        # duration per packet, by milliseconds
        self.ms_per_packet = ms_per_packet
        self.ms_per_frame = ms_per_frame
        self.packet_to_come = 1
        self.last_result_word = []
        self.tgt_dict = tgt_dict
        # text normalization
        self.filter_list = filter_list
        self.formator = formator
        # UPWR - unstable partial word ratio
        self.tot_unstable_word_num = 0
        self.final_hyp_word_num = 1  # count one more so no worry about dividing by 0
        self.upwr = 0.0
        # UPSR - unstable partial segment ratio
        self.tot_unstable_seg_num = 0
        self.upsr = 0.0

    def _calc_uwn(self, cur_result_word):
        """calculate unstable word number between two packets"""
        word_idx = 0
        while (
            word_idx < len(self.last_result_word)
            and word_idx < len(cur_result_word)
            and self.last_result_word[word_idx] == cur_result_word[word_idx]
        ):
            word_idx += 1
        # e.g. "some" -> "something" or "something good" is still considered stable
        if word_idx == len(self.last_result_word) - 1 and len(self.last_result_word) <= len(
            cur_result_word
        ):
            if (
                self.last_result_word[word_idx][: len(self.last_result_word[word_idx])]
                == cur_result_word[word_idx][: len(self.last_result_word[word_idx])]
            ):
                return len(self.last_result_word) - word_idx - 1
        return len(self.last_result_word) - word_idx

    def update(self, cur_result_token, cur_frame_idx=-1, las_rescore=False):
        """
        update metric when streaming or doing final rescore
        """
        assert (
            cur_frame_idx >= 0 or las_rescore
        ), 'has to be cur_frame_idx > 0 for streaming or las_rescore == True for final rescore'
        # see if new packet arrived for streaming
        if cur_frame_idx > 0:
            if cur_frame_idx * self.ms_per_frame < self.packet_to_come * self.ms_per_packet:
                return
        # if no frame index and las_rescore true, then update immediately
        elif not las_rescore:
            return

        # compare word results between two packets
        cur_result_token = [int(t) for t in cur_result_token]
        cur_result_str = self.tgt_dict.string(cur_result_token)
        cur_result_word = infer_text_format(cur_result_str, self.filter_list, self.formator)
        unstable_word_num = self._calc_uwn(cur_result_word)
        if unstable_word_num > 0:
            self.tot_unstable_seg_num += 1
            self.tot_unstable_word_num += unstable_word_num
        self.last_result_word = list(cur_result_word)
        self.final_hyp_word_num = len(cur_result_word) + 1
        self.packet_to_come += 1

        # update upwr & upsr
        self.upwr = self.tot_unstable_word_num / self.final_hyp_word_num
        self.upsr = self.tot_unstable_seg_num / 1.0
