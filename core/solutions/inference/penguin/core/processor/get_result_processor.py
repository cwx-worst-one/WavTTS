""" penguin get_result processor"""
# pylint: disable=abstract-method,too-many-branches
from core.solutions.inference.penguin.core.processor.processor import Processor
from core.solutions.inference.penguin.core.register import Registers


@Registers.processor.register("get_result")
class GetResult(Processor):
    """get_result processor"""

    def __init__(self, config):
        """init"""
        super().__init__()
        self.vocab_path = config.vocab_path
        self.prefetch = config.prefetch
        self.output_speed = config.output_speed
        self.down_sample = config.down_sample
        self.fixed_prefix = config.fixed_prefix
        self.use_las_g2p = config.use_las_g2p
        self.vocab_shift = config.vocab_shift
        self.use_ce_timestamp_conf = config.use_ce_timestamp_conf
        if self.use_las_g2p:
            g2p_vocab_path = config.g2p_vocab_path
            self.g2p_vocab = []
            files = open(g2p_vocab_path)
            for line in files.readlines():
                line = line.strip('\n')
                self.g2p_vocab.append(line)
        self.output_streaming_stable_metric = config.output_streaming_stable_metric

    def __call__(self, name_and_seq):
        '''call'''
        return self.process(name_and_seq)

    def process(self, name_and_seq):
        '''process'''
        wav_name = name_and_seq[0]
        extra = name_and_seq[2]
        prefetch_lists = []
        if "prefetch_twopass" in extra:
            prefetch_lists = extra["prefetch_twopass"]
        elif "prefetch" in extra:
            prefetch_lists = extra["prefetch"]
        prob_seq = None
        if "prob_seq" in extra:
            prob_seq = extra["prob_seq"]
        pronounce_seq = None
        g2p_vocab = None
        if "pronounce_seq" in extra:
            pronounce_seq = extra["pronounce_seq"]
            g2p_vocab = self.g2p_vocab
        total_time = -1
        if self.output_speed and "total_time" in extra:
            total_time = extra["total_time"]
            if total_time <= 0:
                total_time = -1  # avoid divide zero
        vocab = []
        files = open(self.vocab_path)
        for line in files.readlines():
            line = line.strip('\n')
            vocab.append(line)
        prefetch_results = []
        if self.prefetch and prefetch_lists is not None:
            for frame, label_seq in prefetch_lists:
                label_str, _, _, _ = self.token_id2text(vocab, label_seq, None, None, None)
                prefetch_results.append([frame, label_str])
        label_seq = name_and_seq[1]
        output, output_prob, output_pronounce, word_count = self.token_id2text(
            vocab, label_seq, prob_seq, g2p_vocab, pronounce_seq
        )
        speed = -1
        if self.output_speed:
            speed = word_count / (total_time * self.down_sample) * 6000.0
        fixed_prefix_info = []
        if "fixed_prefix_info" in extra:
            fixed_prefix_info = extra["fixed_prefix_info"]
            if len(fixed_prefix_info) != 0:
                fixed_prefix_info = list(
                    map(
                        lambda x: (x[0], self.token_id2text(vocab, x[1], None, None, None)[0]),
                        fixed_prefix_info,
                    )
                )
        streaming_stable_info = ()
        if self.output_streaming_stable_metric:
            label_seq_by_segs = extra['label_seq_by_segs']
            word_seq_by_segs = [
                self.token_id2text(vocab, label_seq)[0].split() for label_seq in label_seq_by_segs
            ]
            # unstable_word_num, unstable_seg_num, final_hyp_word_num, upwr,
            # upsr
            streaming_stable_info = self.calc_streaming_stable_metric(word_seq_by_segs)

        output_ce_timestamp_conf = ""
        if self.use_ce_timestamp_conf:
            # new_label_seq = [tmp_seq for tmp_seq in label_seq \
            # if vocab[tmp_seq-4] not in [' ', ''] ]
            filter_list = ['^', '@@ ', '<s>', '</s>', '<pad>', '<unk>', '@@', '', ' ']
            output_ce_timestamp_conf = self.ce_id2char(vocab, extra['ce_out_list'], filter_list)

        return [
            wav_name,
            {
                "label_str": output,
                "score": extra['score'],
                "output_prob": output_prob,
                "output_pronounce": output_pronounce,
                "prefetch_results": prefetch_results,
                "speed": speed,
                "fixed_prefix_results": fixed_prefix_info,
                "streaming_stable_info": streaming_stable_info,
                "output_ce_timestamp_conf": output_ce_timestamp_conf,
            },
        ]

    def calc_streaming_stable_metric(self, word_seq_by_segs):
        '''calculate streaming stable metric'''
        unstable_word_num, unstable_seg_num, final_hyp_word_num = 0, 0, 0
        for i_seg in range(1, len(word_seq_by_segs)):
            last_result_word = word_seq_by_segs[i_seg - 1]
            cur_result_word = word_seq_by_segs[i_seg]
            if last_result_word == cur_result_word:
                continue
            cur_unstable_word_num = self.calc_unstable_words_num(last_result_word, cur_result_word)
            if cur_unstable_word_num > 0:
                unstable_word_num += cur_unstable_word_num
                unstable_seg_num += 1
        # count one more so no worry about dividing by 0 later
        final_hyp_word_num = len(word_seq_by_segs[i_seg]) + 1
        upwr = unstable_word_num / final_hyp_word_num
        upsr = unstable_seg_num / 1.0
        return (
            unstable_word_num,
            unstable_seg_num,
            final_hyp_word_num,
            upwr,
            upsr,
        )

    # pylint: disable=no-self-use
    def calc_unstable_words_num(self, last_result_word, cur_result_word):
        '''calculate unstable words num'''
        word_idx = 0
        while (
            word_idx < len(last_result_word)
            and word_idx < len(cur_result_word)
            and last_result_word[word_idx] == cur_result_word[word_idx]
        ):
            word_idx += 1
        # e.g. "some" -> "something" or "something good"
        # is still considered stable
        if word_idx == len(last_result_word) - 1 and len(last_result_word) <= len(cur_result_word):
            if (
                last_result_word[word_idx][: len(last_result_word[word_idx])]
                == cur_result_word[word_idx][: len(last_result_word[word_idx])]
            ):
                return len(last_result_word) - word_idx - 1
        return len(last_result_word) - word_idx

    def ce_id2char(self, vocab, out_list, filter_list, frame_shift=10, frame_length=40):
        '''post process for timestamp'''
        hyp_token = []
        tmp_list = ["", 0, 0, 0.0]
        for idx, token_info in enumerate(out_list):
            if token_info[0] < self.vocab_shift:
                continue
            token_info[0] = vocab[token_info[0] - self.vocab_shift]
            token = token_info[0]
            if token not in filter_list:
                if tmp_list[0] == "":
                    tmp_list = token_info
                else:
                    tmp_list[0] += token_info[0]
                    tmp_list[2] = token_info[2]
                    tmp_list[3] += token_info[3]
                if token[-1] != '@' or idx == len(out_list) - 1:
                    hyp_token.append(tmp_list)
                    tmp_list = ["", 0, 0, 0.0]

        # pylint: disable=consider-using-enumerate
        for i in range(0, len(hyp_token)):
            hyp_token[i][0] = hyp_token[i][0].replace('@@', '')
            start_index = hyp_token[i][1]
            end_index = hyp_token[i][2]
            duration = end_index - start_index
            hyp_token[i][3] /= duration
            refine_start = hyp_token[i][1] * frame_length - frame_shift
            hyp_token[i][1] = refine_start if refine_start > 0 else 0
            refine_end = hyp_token[i][2] * frame_length - frame_shift
            hyp_token[i][2] = refine_end if refine_end > 0 else 0
        return str(hyp_token)

    def token_id2text(self, vocab, label_seq, prob_seq=None, g2p_vocab=None, pronounce_seq=None):
        '''convert token_id to text'''
        output = ""
        pre_word_piece = False
        prev_char = " "
        word_prob = 0
        word_bpe_count = 0
        output_prob = ""
        word_count = 0
        output_pronounce = ""
        if prob_seq is not None:
            assert len(label_seq) == len(prob_seq), "word prob sequence length not match"
        if pronounce_seq is not None:
            assert len(label_seq) == len(pronounce_seq), "word pronounce sequence length not match"
        for idx, label in enumerate(label_seq):
            label = round(label)
            if (
                label < self.vocab_shift
                or vocab[label - self.vocab_shift] == ""
                or (prev_char == " " and vocab[label - self.vocab_shift] == " ")
            ):
                if label < self.vocab_shift and pre_word_piece:
                    output += ' '
                    word_count += 1
                    pre_word_piece = False
                continue
            word = vocab[label - self.vocab_shift]
            if pronounce_seq is not None:
                pronounce_idx = pronounce_seq[idx]
                pronounce = g2p_vocab[pronounce_idx]
            if len(word) > 2 and word[-2:] == "@@":
                # begin/middle of a word
                word = word[:-2]
                output += word
                prev_char = word
                pre_word_piece = True
                if prob_seq is not None:
                    word_prob += prob_seq[idx]
                    word_bpe_count += 1
            else:
                word_count += 1
                if pre_word_piece:
                    # end of a word
                    output += word + ' '
                    prev_char = ' '
                    if prob_seq is not None:
                        word_prob += prob_seq[idx]
                        word_bpe_count += 1
                        output_prob += str(word_prob / word_bpe_count) + ' '
                    if pronounce_seq is not None:
                        output_pronounce += pronounce + ' '
                elif word == ' ':
                    output += word
                    prev_char = word
                    word_count -= 1  # empty word
                else:
                    # single word
                    output += word + ' '
                    prev_char = ' '
                    if prob_seq is not None:
                        output_prob += str(prob_seq[idx]) + ' '
                    if pronounce_seq is not None:
                        output_pronounce += pronounce + ' '

                pre_word_piece = False
                word_prob = 0
                word_bpe_count = 0

        if len(output) != 0 and output[-1] == ' ':
            output = output[:-1]
        if len(output_prob) != 0 and output_prob[-1] == ' ':
            output_prob = output_prob[:-1]
        if len(output_pronounce) != 0 and output_pronounce[-1] == ' ':
            output_pronounce = output_pronounce[:-1]

        return output, output_prob, output_pronounce, word_count
