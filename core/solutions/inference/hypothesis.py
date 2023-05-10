''' Hypothesis definition '''

from core.utils import FalconDict


class HotwordToken:
    """Lm token"""

    def __init__(self):
        '''init.'''
        self.fst_idx = 0
        self.state = 0
        self.total_lm_score = 0.0
        self.selected = False
        self.cur_lm_score = 0.0
        self.timestamp = 0
        self.cur_hotword_seq = []  # record the searching path
        self.matched_hotwords = []  # record the matched hotword

    def copy(self):
        '''deep copy'''
        token = HotwordToken()
        token.fst_idx = self.fst_idx
        token.state = self.state
        token.total_lm_score = self.total_lm_score
        token.selected = self.selected
        token.cur_lm_score = self.cur_lm_score
        token.timestamp = self.timestamp
        token.cur_hotword_seq = self.cur_hotword_seq[:]
        token.matched_hotwords = self.matched_hotwords[:]
        return token


class ClassLmFstHypothesis(FalconDict):
    """Hypothesis for class fst."""

    def __init__(self, **kwargs):
        """init class fst hypothesis"""
        self.tokens = [0]
        self.class_lm_fst_score = 0.0
        self.class_lm_fst_states = (0, 0)
        self.class_lm_fst_idx = 0
        super().__init__(**kwargs)

    def copy(self):
        hyp = ClassLmFstHypothesis(
            tokens=self.tokens[:],
            class_lm_fst_score=self.class_lm_fst_score,
            class_lm_fst_states=self.class_lm_fst_states[:],
            class_lm_fst_idx=self.class_lm_fst_idx,
        )
        return hyp


class Hypothesis(FalconDict):
    """Hypothesis definition for Transducer beam search"""

    def __init__(self, **kwargs):
        '''init Hypothesis'''
        self.label_seq = [0]  # yseq, predicted tokens
        self.score = 0.0
        self.pred_state = []
        self.pred_feat = None
        self.ilm_score = 0.0
        self.nnlm_score = 0.0
        self.nnlm_lprobs = None
        self.nnlm_state = []
        self.ngram_score = 0.0
        self.ngram_state = 0
        self.hotword_score = 0.0
        self.coldword_state = 0
        self.class_lm_fst_best_hyp_index = 0
        self.class_lm_fst_hyps = []
        # TODO(chenjinkun): manage lm scores and states
        self.timestamp = [-1]
        self.confidence = [0]
        self.eos_score = 0.0
        self.next_biasing_position = 0

        # for multi step search when fusion hotword fst
        self.prev_total_lm_score = 0.0
        self.lm_tokens = []
        self.matched_hotwords = []
        self.lm_token_timestamp = 0
        super().__init__(**kwargs)

    def copy(self):
        '''return a mem-copy obj'''
        hyp = Hypothesis(
            label_seq=self.label_seq[:],
            score=self.score,
            pred_state=self.pred_state[:],
            pred_feat=self.pred_feat,
            ilm_score=self.ilm_score,
            ilm_lprobs=self.ilm_lprobs,
            nnlm_score=self.nnlm_score,
            nnlm_lprobs=self.nnlm_lprobs,
            nnlm_state=self.nnlm_state[:],
            ngram_score=self.ngram_score,
            ngram_state=self.ngram_state,
            hotword_score=self.hotword_score,
            coldword_state=self.coldword_state,
            class_lm_fst_best_hyp_index=self.class_lm_fst_best_hyp_index,
            class_lm_fst_hyps=[hyp.copy() for hyp in self.class_lm_fst_hyps],
            timestamp=self.timestamp[:],
            confidence=self.confidence[:],
            eos_score=self.eos_score,
            next_biasing_position=self.next_biasing_position,
        )
        hyp.prev_total_lm_score = self.prev_total_lm_score
        hyp.lm_tokens = self.lm_tokens[:]
        hyp.matched_hotwords = self.matched_hotwords[:]
        hyp.lm_token_timestamp = self.lm_token_timestamp  # used to sort the lm tokens
        return hyp

    @property
    def label_str(self):
        '''string the tokens list'''
        return ' '.join(map(str, self.label_seq))

    def init_lm_tokens(self, fst_num, fst_states):
        """Init lm tokens"""
        assert fst_num == len(fst_states)
        self.lm_tokens = []
        for i in range(0, fst_num):
            token = HotwordToken()
            token.fst_idx = i
            token.state = fst_states[i]
            self.lm_tokens.append(token)

    def init_class_lm_fst_hyps(self, states):
        """Init class fst hypothesis."""
        self.class_lm_fst_best_hyp_index = 0
        self.class_lm_fst_hyps = [
            ClassLmFstHypothesis(tokens=self.label_seq, class_lm_fst_states=states)
        ]
