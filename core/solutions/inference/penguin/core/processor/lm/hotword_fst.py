"""Hotword fst"""
import logging

from core.solutions.inference.penguin.core.processor.lm.base_model import FST

INT_MAX = 2147483647


class HotwordFst(FST):
    """Fst model for hotword."""

    def __init__(self):
        self.backoff_idx = INT_MAX

    def load(self, path):
        super().load(path)
        logging.info("Loading hotword fst successfully.")

    # tips: ngram fst weight is -loge(p), while hotword fst weight is log10(p)
    # return log10(p)
    def forward(self, label, state):
        """Forword fst state according to inlabel.
        Return weight from fst arc, and next state.
        """
        is_match = False
        fst_score = 0.0
        next_state = self.start()
        for arc in self.fst.arcs(state):
            if label == arc.ilabel:
                fst_score = float(arc.weight)
                next_state = arc.nextstate
                is_match = True
                break
        # add backoff
        if state != self.start() and not is_match:
            for arc in self.fst.arcs(state):
                if (arc.nextstate == self.start()) and (arc.ilabel == self.backoff_idx):
                    next_state = arc.nextstate
                    fst_score = float(arc.weight)
                    break
        return next_state, fst_score

    def finish_match(self, states, score):
        """Whether fst completely matched."""
        is_finish_match = False
        # when fst backoff, states == self.start() and score > 0
        if states == self.start() and score < 0:
            is_finish_match = True
        return is_finish_match

    def get_backoff_idx(self):
        """Get backoff idx."""
        return self.backoff_idx
