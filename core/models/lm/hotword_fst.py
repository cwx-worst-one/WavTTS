"""Hotword fst"""

from core.utils import logging
from .base_model import FST

INT_MAX = 2147483647


class HotwordFst(FST):
    """Fst model for hotword."""

    def __init__(self, cfg):
        '''init.'''
        self.backoff_idx = INT_MAX
        self.cfg = cfg
        self.map_fst = dict()

    def load(self, path):
        super().load(path)
        # We use a map-fst to accelerate the lm fusion.
        # The map-fst is a list of dictionary. Assuming the fst to be:
        # 0 1  10 10 0.0 // arc1
        # 1 2  20 20 1.0 // arc2
        # 1 2  20 20 3.0 // arc3
        # The the map-fst will be [{10: [arc1]}, {20, [arc2, arc3]}]
        # It can extremely accelerate the lm-fusion since searching happened in the map-fst is O(1)
        # Duration of decoding 1w audios with 2 fsts on Arnold: 4~5h(before) vs 44min(after).
        for state in self.fst.states():
            ilabel_to_arcs = {}
            for arc in self.fst.arcs(state):
                if arc.ilabel not in ilabel_to_arcs:
                    ilabel_to_arcs[arc.ilabel] = []
                ilabel_to_arcs[arc.ilabel].append(arc)
            self.map_fst[state] = ilabel_to_arcs
        logging.info("Loading hotword fst successfully.")

    def forward(self, label, state):
        """Search using multi_step or single_step. You can choose this by config."""
        if self.cfg.enable_multi_step_search:
            return self.forward_multi_step(label, state)
        return self.forward_single_step(label, state)

    def forward_single_step(self, label, state):
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
        if state != self.start() and not is_match:
            for arc in self.fst.arcs(state):
                if (arc.nextstate == self.start()) and (arc.ilabel == self.backoff_idx):
                    next_state = arc.nextstate
                    fst_score = float(arc.weight)
                    break
        states_and_costs = []
        states_and_costs.append([next_state, fst_score])
        return states_and_costs

    def forward_multi_step(self, label, state):
        """Forword fst state according to inlabel.
        Return lists of pair, each pair is comprised of weight from fst arc, and next state.
        """
        is_match = False
        states_and_costs = []
        assert state in self.map_fst
        ilabel_to_arcs = self.map_fst[state]
        if label in ilabel_to_arcs.keys():
            arcs = ilabel_to_arcs[label]
            for arc in arcs:
                states_and_costs.append([arc.nextstate, float(arc.weight)])
                is_match = True
                if len(states_and_costs) > self.cfg.max_active_lm_token_num:
                    break
        # add backoff
        # Insert self-loop on start state whatever the label is.
        # Consider the following case:
        # Hotword 1633, 276,
        # Hotword 275, 888
        # Hotword 470, 1, 2
        # Label seq 1633, 276, 275, 470, 1, 2
        # Without the self-loop we can't remain the score of (1633, 276) when
        # we input 470
        if state == self.start():
            states_and_costs.append([self.start(), 0.0])
        elif not is_match:
            if self.backoff_idx in ilabel_to_arcs.keys():
                arcs = ilabel_to_arcs[self.backoff_idx]
                for arc in arcs:
                    if arc.nextstate == self.start():
                        states_and_costs.append([arc.nextstate, float(arc.weight)])
                        break
        states_and_costs = sorted(states_and_costs, key=lambda x: x[1])[
            : self.cfg.lm_token_beam_size
        ]
        return states_and_costs

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
