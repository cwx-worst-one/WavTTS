"""ngram fst"""
import os

from core.utils import logging, dist_hdfs_get
from .base_model import FST


def load_vocab(path):
    """Load vocab file, mainly for sentence fst model."""
    vocab = {}
    if not os.path.exists(path):
        file_name = os.path.basename(path)
        path = dist_hdfs_get(path, local_file=file_name)
        if not path:
            raise ValueError("Can not get vocab file.")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            if line:
                vocab[line[0]] = int(line[1])
    return vocab


class NgramFst(FST):
    """Fst model for sentence."""

    def __init__(self, config):
        '''init.'''
        # {state: {ilabel: arc}}
        self.dict_state_arcs = {}
        self.disambiguation_idx = -1  # 12964
        self.unk_idx = -1  # 4
        self.unk_des = -1
        self.set_special_symbol(config["vocab"])

    def load(self, path):
        """Load fst model and get self.dict_state_arcs."""
        super().load(path)
        self.fst_to_dict()
        logging.info("Loading sentence fst successfully.")

    # tips: ngram fst weight is -loge(p), while hotword fst weight is log10(p)
    # return loge(p)
    def forward_with_default(self, label, state, default_score=None):
        """Forword to next state according to inlabel.
        If the default_score is set and can't find the target label in fst,
        the out fst score is default_score.
        Return next state and fst score.
        """
        fst_score = 0.0
        if default_score is not None:
            fst_score = default_score
        active_state_queue = [(state, 0.0)]
        # init destination state.
        next_state = self.unk_des
        while len(active_state_queue) > 0:
            active_state = active_state_queue.pop(0)
            cur_state_arcs = self.dict_state_arcs[active_state[0]]
            if label in cur_state_arcs:
                cur_arc = cur_state_arcs[label]
                cur_weight = -float(cur_arc.weight) + active_state[1]
                fst_score = cur_weight
                next_state = cur_arc.nextstate
                break
            # go back off.
            if self.disambiguation_idx in cur_state_arcs:
                cur_arc = cur_state_arcs[self.disambiguation_idx]
                next_state_base_weight = -float(cur_arc.weight) + active_state[1]
                active_state_queue.append((cur_arc.nextstate, next_state_base_weight))
        return next_state, fst_score

    def fst_to_dict(self):
        """Parse self.model and get self.dict_state_arcs"""
        self.dict_state_arcs = {}
        for state in self.fst.states():
            arc_info = {}
            for arc in self.fst.arcs(state):
                arc_info[arc.ilabel] = arc
                # get unk info.
                if self.unk_idx == arc.ilabel:
                    self.unk_des = arc.nextstate
            self.dict_state_arcs[state] = arc_info
        assert self.unk_des != -1

    def set_special_symbol(self, vocab_path):
        """Parse vocab and get self.disambiguation_idx, self.unk_idx."""
        if not vocab_path:
            raise ValueError("Please set vocab for ngram.")
        dict_vocab = load_vocab(vocab_path)
        if not dict_vocab:
            raise ValueError("Empty lm vocab.")
        self.disambiguation_idx = dict_vocab["#0"]
        self.unk_idx = dict_vocab["<unk>"]
        assert self.disambiguation_idx != -1 and self.unk_idx != -1

    def forward(self, label, state):
        """Forword to next state according to inlabel.
        Return next state and fst score.
        """
        _, unk_score = self.forward_with_default(self.unk_idx, state)
        out_state, out_score = self.forward_with_default(label, state, unk_score)
        return out_state, out_score
