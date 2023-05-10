"""Lm solution"""
# pylint:disable=too-many-public-methods
from core.models.lm.hotword_fst import HotwordFst
from core.models.lm.ngram_fst import NgramFst
from core.models.lm.class_lm_fst import ClassLmFst
from core.models.lm.lstm_lm import *
from core.solutions.base_solution import BaseSolution
from core.solutions.inference import INFERS
from core.solutions.inference.hypothesis import HotwordToken
from .base_nnlm_solution import LSTMLMExporter


def is_overlapped_token(left, right):
    """Determine whether cur_hotword_seq overlaps"""
    if len(left) == 0 or len(right) == 0:
        return False
    length = min(len(left), len(right))
    for i in range(1, length + 1):
        suffix = left[-i:]
        prefix = right[:i]
        if suffix == prefix:
            return True
    return False


class LmSolution(BaseSolution):
    """Lm solution for asr, including training and inference, only inference ready by now."""

    def __init__(self, config):
        """Parse config.
        input: config is a dict.
        """
        super().__init__()
        self.config = config
        self.hotword_fst_list = []
        self.coldword_fst = None
        self.ngram_fst = None
        self.class_lm_fst = None
        self.nnlm = None
        self.criterion_module = None
        self.hotword_weight = self.config["hotword_fst"].get('hotword_weight', [])
        if self.hotword_weight:
            self.hotword_weight = [float(x) for x in str(self.hotword_weight).strip().split('|')]

    def init_criterion_module(self, cfg):
        """Init criterion module."""
        self.criterion_module = eval(cfg["criterion_type"])(cfg)

    def load_from_inference_cfg(self, inference_cfg):
        """Load models."""
        hotword_fst_path = inference_cfg.get('hotword_fst_path', '')
        coldword_fst_path = inference_cfg.get('coldword_fst_path', '')
        ngram_fst_path = inference_cfg.get('ngram_fst_path', '')
        nnlm_path = inference_cfg.get('nnlm_path', '')
        class_lm_fst_path = inference_cfg.get('class_lm_fst_path', '')
        if hotword_fst_path:
            path_list = hotword_fst_path.strip().split('|')
            for path in path_list:
                cur_hotword_fst = HotwordFst(self.config["hotword_fst"])
                cur_hotword_fst.load(path)
                self.hotword_fst_list.append(cur_hotword_fst)
        if coldword_fst_path:
            self.coldword_fst = HotwordFst(self.config["coldword_fst"])
            self.coldword_fst.load(coldword_fst_path)
        if ngram_fst_path:
            self.ngram_fst = NgramFst(self.config["ngram_fst"])
            self.ngram_fst.load(ngram_fst_path)
        if nnlm_path:
            nn_params = self.config["nnlm"]
            assert "tgt_vocab_size" in nn_params.keys()
            self.nnlm = eval(nn_params["nnlm_model_type"])(nn_params)
            self.nnlm.load(nnlm_path)
            self.nnlm.cuda()
        if class_lm_fst_path:
            self.class_lm_fst = ClassLmFst(self.config["class_lm_fst"])
            self.class_lm_fst.load(class_lm_fst_path)

    def hotword_fst_start(self):
        '''return start states of all hotword fsts'''
        start_states = []
        if self.hotword_fst_list:
            start_states = [fst.start() for fst in self.hotword_fst_list]
        return start_states

    def coldword_fst_start(self):
        '''return start state of coldword fst'''
        start_state = 0
        if self.coldword_fst:
            start_state = self.coldword_fst.start()
        return start_state

    def ngram_fst_start(self):
        '''return start states of ngram'''
        start_state = 0
        if self.ngram_fst:
            start_state = self.ngram_fst.start()
        return start_state

    def class_lm_fst_start(self):
        '''return start state of class_lm fst'''
        start_state = (0, 0)
        if self.class_lm_fst:
            start_state = self.class_lm_fst.start()
        return start_state

    def nnlm_start(self):
        '''return start state of nnlm'''
        nnlm_start_state = []
        if self.nnlm:
            nnlm_start_state = self.nnlm.start()
        return nnlm_start_state

    def train(self):
        """Train models, not implete yet."""
        raise NotImplementedError("Not implemented yet!!!")

    def insert_token_on_start_state(self, timestamp, lm_tokens):
        """Insert token on start states"""
        for i in range(self.get_hotword_fst_number()):
            find = False
            for token in lm_tokens:
                if token.fst_idx == i and token.state == self.hotword_fst_list[i].start():
                    find = True
                    break
            if not find:
                token = HotwordToken()
                token.fst_idx = i
                token.timestamp = timestamp
                lm_tokens.append(token)

    def prune_lm_tokens(self, tokens):
        """Prune lm tokens"""
        tokens = sorted(tokens, key=lambda token: token.total_lm_score)
        if len(tokens) > self.config.hotword_fst.lm_token_beam_size:
            return tokens[: self.config.hotword_fst.lm_token_beam_size]
        return tokens

    def get_best_token(self, is_last, lm_tokens):
        """Get best lm tokens"""
        best_score = float('inf')
        select_idx = None
        for i, ele in enumerate(lm_tokens):
            token = ele
            if is_last and not self.hotword_fst_finish_match(
                token.state, token.total_lm_score, token.fst_idx
            ):
                continue
            if token.total_lm_score < best_score:
                select_idx = i
                best_score = token.total_lm_score
        if select_idx is None:
            token = HotwordToken()
            return token
        lm_tokens[select_idx].selected = True
        return lm_tokens[select_idx].copy()

    def finish_non_greedy_search(self, is_last, best_token, lm_tokens):  # pylint: disable=R0201
        """Finish non greedy search or not"""
        if is_last:
            return True

        if (
            best_token.cur_lm_score >= 0
            and best_token.state == self.hotword_fst_list[best_token.fst_idx].start()
            and best_token.total_lm_score == 0
        ):
            return True

        exist_overlapped_token = False

        lm_tokens = sorted(lm_tokens, key=lambda x: x.timestamp)
        for i, left_token in enumerate(lm_tokens):
            for right_token in lm_tokens[i + 1 :]:
                if is_overlapped_token(left_token.cur_hotword_seq, right_token.cur_hotword_seq):
                    exist_overlapped_token = True

        all_token_finished = True
        # We won't clear tokens as long as there is a matching token or overlapped path.
        for token in lm_tokens:
            if (
                token.cur_lm_score < 0
                and token.state != self.hotword_fst_list[token.fst_idx].start()
            ):
                all_token_finished = False
        if all_token_finished and not exist_overlapped_token:
            return True
        return False

    def step_hotword(self, cur_label, hyp, is_last):
        """Search and fuse node with best token"""
        if not self.hotword_fst_list:
            return None

        self.insert_token_on_start_state(hyp.lm_token_timestamp, hyp.lm_tokens)
        new_tokens = []
        for token in hyp.lm_tokens:
            fst = self.hotword_fst_list[token.fst_idx]
            states_and_costs = fst.forward(cur_label, token.state)
            for state, cost in states_and_costs:
                new_token = HotwordToken()
                new_token.fst_idx = token.fst_idx
                new_token.state = state
                new_token.total_lm_score = (
                    token.total_lm_score + cost * self.hotword_weight[new_token.fst_idx]
                )
                new_token.cur_lm_score = cost * self.hotword_weight[new_token.fst_idx]
                new_token.cur_hotword_seq = token.cur_hotword_seq[:]
                new_token.matched_hotwords = token.matched_hotwords[:]
                new_token.timestamp = token.timestamp
                if new_token.cur_lm_score < 0:
                    new_token.cur_hotword_seq.append(cur_label)
                    if new_token.state == self.hotword_fst_list[new_token.fst_idx].start():
                        new_token.matched_hotwords.append(new_token.cur_hotword_seq)
                else:
                    new_token.cur_hotword_seq = []

                duplicated_token = None
                for i in new_tokens:
                    if i.fst_idx == new_token.fst_idx and i.state == new_token.state:
                        duplicated_token = i
                        break
                if duplicated_token is None:
                    new_tokens.append(new_token)
                elif new_token.total_lm_score < duplicated_token.total_lm_score:
                    duplicated_token.total_lm_score = new_token.total_lm_score
                    duplicated_token.cur_lm_score = new_token.cur_lm_score
                    duplicated_token.cur_hotword_seq = new_token.cur_hotword_seq[:]
                    duplicated_token.timestamp = new_token.timestamp
                    duplicated_token.matched_hotwords = new_token.matched_hotwords
                if len(new_tokens) > self.config.hotword_fst.max_active_lm_token_num:
                    break
        hyp.lm_tokens = self.prune_lm_tokens(new_tokens)
        hyp.lm_token_timestamp += 1
        best_token = self.get_best_token(is_last, hyp.lm_tokens)
        return best_token

    def step_coldword(self, label, state):
        """Step state for a certain coldword fst."""
        score = 0
        if self.coldword_fst:
            states_and_costs = self.coldword_fst.forward_single_step(label, state)
            state = states_and_costs[0][0]
            score = states_and_costs[0][1]
        return state, score

    def step_ngram(self, label, state):
        """Step state for ngram fst."""
        score = 0
        if self.ngram_fst:
            state, score = self.ngram_fst.forward(label, state)
        return state, score

    def step_nn(self, tokens, states=None):
        """Step state for nn model.
        input:
            tokens: input tokens, torch.tensor type, shape is 1 * roads_size.
            states: input states, torch.tensor type, shape is roads_size * vocab_size.
        output:
            lprobs: out probs, torch.tensor type, shape is roads_size * vocab_size.
            states: next states, torch.tensor type, shape is roads_size * vocab_size.
        """
        lprobs = []
        if self.nnlm:
            lprobs, states = self.nnlm.forward_step(tokens, states)
        else:
            raise ValueError("Please set nnlm model first.")
        return lprobs, states

    def step_class_lm_fst(self, fst_idx, label, states):
        """Step state for class fst."""
        if self.class_lm_fst:
            return self.class_lm_fst.forward(fst_idx, label, states)
        return []

    def hotword_fst_finish_match(self, state, score, fst_index):
        """Whether hotword fst completely matched."""
        finish_match = False
        if self.hotword_fst_list:
            hotword_fst = self.hotword_fst_list[fst_index]
            finish_match = hotword_fst.finish_match(state, score)
        return finish_match

    def hotword_fst_backoff(self, lm_tokens):
        """Hotword fst step backoff. Return state and fst score"""
        score = 0
        for token in lm_tokens:
            if not token.selected:
                continue
            hotword_fst = self.hotword_fst_list[token.fst_idx]
            states_and_costs = hotword_fst.forward(hotword_fst.get_backoff_idx(), token.state)
            score = states_and_costs[0][1] * self.hotword_weight[token.fst_idx]
        return score

    def get_hotword_fst_number(self):
        """Get hotword fst numbers."""
        return len(self.hotword_fst_list)

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        self._infer_names = []
        if self.nnlm is not None:
            infer = LSTMLMExporter(self.nnlm, **self.config)
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)

    def class_lm_fst_check_labels(self, labels):
        """Check labels for class fst."""
        if self.class_lm_fst:
            return self.class_lm_fst.check_labels(labels)
        return False
