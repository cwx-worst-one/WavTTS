"""Lm solution"""
from core.solutions.inference.penguin.core.processor.lm.hotword_fst import HotwordFst


class LmSolution:
    """Lm solution for asr inference"""

    def __init__(self):
        """Init hotword fst."""
        self.hotword_fst = None

    def load(self, hotword_fst_path):
        """Load models."""
        if hotword_fst_path:
            self.hotword_fst = HotwordFst()
            self.hotword_fst.load(hotword_fst_path)

    def start(self):
        """Return start states."""
        hotword_fst_start_state = 0
        if self.hotword_fst:
            hotword_fst_start_state = self.hotword_fst.start()
        return hotword_fst_start_state

    def step_fsts(self, label, states):
        """Step fst models: hotword fst and ngram fst.
        input:
            label: the ilabel for fst.
            states: list for hotword and ngram fst states.
        return:
            list for hotword fst and ngram fst forword result.
        """
        hotwrod_fst_state = states
        hotword_fst_score = 0
        if self.hotword_fst:
            hotwrod_fst_state, hotword_fst_score = self.hotword_fst.forward(
                label, hotwrod_fst_state
            )
        return hotwrod_fst_state, hotword_fst_score
