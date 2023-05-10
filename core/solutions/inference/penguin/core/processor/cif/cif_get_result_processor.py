""" CIF penguin get_result processor"""
# pylint: disable=abstract-method,too-many-branches
from core.solutions.inference.penguin.core.register import Registers
from core.solutions.inference.penguin.core.processor.get_result_processor import GetResult


@Registers.processor.register("cif_get_result")
class CifGetResult(GetResult):
    """CIF get_result processor"""

    def __init__(self, config):
        """init"""
        super().__init__(config)
        self.config = config
        self.vocab_path = config.vocab_path
        self.vocab_shift = config.vocab_shift
        self.use_ce_timestamp_conf = bool(config.use_ce_timestamp_conf)

    def process(self, name_and_seq):
        """process"""
        wav_name = name_and_seq[0]
        label_seq = name_and_seq[1]
        extra = name_and_seq[2]
        vocab = []
        files = open(self.vocab_path)
        for line in files.readlines():
            line = line.strip('\n')
            vocab.append(line)
        # time_stamp = name_and_seq[2]
        # vocab map <eos> to ''
        output = self.token_id2text(vocab, label_seq)
        ce_timestamp = None
        if self.use_ce_timestamp_conf:
            # new_label_seq = [tmp_seq for tmp_seq in label_seq \
            # if vocab[tmp_seq-4] not in [' ', ''] ]
            filter_list = ['^', '@@ ', '<s>', '</s>', '<pad>', '<unk>', '@@', '', ' ']
            ce_timestamp = self.ce_id2char(vocab, extra['ce_out_list'], filter_list, frame_shift=20)
        results = [wav_name, {"label_str": output, "ce_timestamp": ce_timestamp}]
        return results

    def token_id2text(self, vocab, label_seq):
        """convert id to text"""
        filter_list = ['@@', ' ', '']
        word = ''
        hyp_token = []
        for idx, token_id in enumerate(label_seq):
            if token_id < self.vocab_shift:
                continue
            token = vocab[token_id - self.vocab_shift]
            if token not in filter_list:
                word += token
                if token[-1] != '@' or idx == len(label_seq):
                    hyp_token.append(word)
                    word = ''
        res_str = ' '.join(hyp_token).replace('@@', '')
        return res_str
