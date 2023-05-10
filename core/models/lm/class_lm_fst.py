"""Class lm fst"""
from .hotword_fst import HotwordFst
from .ngram_fst import load_vocab, NgramFst


class Ifst(HotwordFst):
    """Ifst build from class words."""

    def forward(self, label, state):
        """Ifst forward once time.
        match_state:
        0: matching;
        1: finish match a word, but this word may be a prefix for another word(eg: ab, abc).
        2: finish match a word, can not be word prefix.
        -1: match failed.
        """
        match_state = -1
        next_state = 0
        score = 0
        if label in self.map_fst[state]:
            arcs = self.map_fst[state][label]
            assert len(arcs) <= 2
            # with no common prefix word.
            if len(arcs) == 1:
                next_state = arcs[0].nextstate
                score = float(arcs[0].weight)
                if next_state == self.start():
                    match_state = 2
                else:
                    match_state = 0
            # with common prefix words.
            elif len(arcs) == 2:
                match_state = 1
                next_states = [arc.nextstate for arc in arcs]
                assert self.start() in next_states and len(set(next_states)) != 1
                for arc in arcs:
                    if arc.nextstate != self.start():
                        # return the state that is not finish match.
                        next_state = arc.nextstate
                        score = float(arc.weight)
                        break
        return match_state, next_state, score


class TopFst(NgramFst):
    """Top fst, build from common corpus and pattern corpus."""

    def __init__(self, config):
        super().__init__(config)
        # [non_terminal_ids], eg: [12969, 12970]
        self.non_terminal_ids = set()
        # {state: [non_terminal_id]}, eg: {2, [12969]}
        # topfst 的states，其ilabel是包含non_terminal_id
        self.state_to_non_terminal_ids = {}
        # {state: set(labels)}
        self.non_terminal_state_to_ilabels = {}

    def forward(self, label, state):
        """Top fst forward once time."""
        next_state = -1
        score = 0.0
        # For no terminal id, no ngram backoff.
        if label in self.non_terminal_ids and label not in self.state_to_non_terminal_ids.get(
            state, set()
        ):
            return next_state, score
        if (
            state in self.non_terminal_state_to_ilabels
            and label not in self.non_terminal_state_to_ilabels[state]
        ):
            return next_state, score
        next_state, score = super().forward(label, state)
        return next_state, score

    def set_non_terminal_info(self, non_terminal_ids):
        """Parse no terminal info from top fst."""
        self.non_terminal_ids = set(non_terminal_ids)
        for state, arc_info in self.dict_state_arcs.items():
            non_terminal_ids = []
            for non_terminal_id in self.non_terminal_ids:
                if non_terminal_id in arc_info:
                    non_terminal_ids.append(non_terminal_id)
                    next_state = arc_info[non_terminal_id].nextstate
                    next_state_ilabels = self.dict_state_arcs[next_state].keys()
                    self.non_terminal_state_to_ilabels[next_state] = set(next_state_ilabels)
            if non_terminal_ids:
                self.state_to_non_terminal_ids[state] = set(non_terminal_ids)


class ClassLmFst:
    """Class fst, include top fst and ifsts."""

    def __init__(self, config):
        """Init."""
        self.top_fst = TopFst(config)
        # {non_terminal_id: ifst}
        self.ifsts = {}
        self.vocab_dict = load_vocab(config["vocab"])

    def load(self, fsts_path):
        """Load models from path."""
        path_list = fsts_path.split('|')
        self.top_fst.load(path_list[0])
        for i in range(1, len(path_list)):
            non_terminal, path = path_list[i].split('*')
            # load ifst
            current_ifst = Ifst(cfg={})
            current_ifst.load(path)
            non_terminal_id = self.vocab_dict[non_terminal]
            assert non_terminal_id != 0
            self.ifsts[non_terminal_id] = current_ifst
        self.top_fst.set_non_terminal_info(list(self.ifsts.keys()))

    def start(self):
        """Return start state."""
        return self.top_fst.start(), 0

    def forward(self, fst_id, label, states):
        """Fst forward once time, and do on the fly replace for top fst and ifsts.
        output:[[fst_id, [top_fst_state, ifst_state], [top_fst_score, ifst_score], [out_labels]]]
        """
        ret = []
        top_fst_state, ifst_state = states
        if fst_id == 0:  # top fst
            top_fst_result = self.get_top_fst_result(label, top_fst_state)
            for item in top_fst_result:
                current_id, current_top_fst_state, current_top_fst_score = item
                if current_top_fst_state == -1:
                    continue
                current_ifst_state = 0
                current_ifst_score = 0
                if current_id in self.ifsts:
                    current_ifst = self.ifsts[current_id]
                    (
                        current_ifst_match_state,
                        current_ifst_state,
                        current_ifst_score,
                    ) = current_ifst.forward(label, current_ifst.start())
                    if current_ifst_match_state == 0:
                        ret.append(
                            [
                                current_id,
                                [current_top_fst_state, current_ifst_state],
                                [current_top_fst_score, current_ifst_score],
                                [current_id, label],
                            ]
                        )
                    elif current_ifst_match_state == 1:
                        ret.append(
                            [
                                current_id,
                                [current_top_fst_state, current_ifst_state],
                                [current_top_fst_score, current_ifst_score],
                                [current_id, label],
                            ]
                        )
                        ret.append(
                            [
                                0,
                                [current_top_fst_state, current_ifst.start()],
                                [current_top_fst_score, current_ifst_score],
                                [current_id, label, current_id],
                            ]
                        )
                    elif current_ifst_match_state == 2:
                        ret.append(
                            [
                                0,
                                [current_top_fst_state, current_ifst.start()],
                                [current_top_fst_score, current_ifst_score],
                                [current_id, label, current_id],
                            ]
                        )
                else:
                    ret.append(
                        [
                            0,
                            [current_top_fst_state, current_ifst_state],
                            [current_top_fst_score, current_ifst_score],
                            [label],
                        ]
                    )
        else:  # ifst
            current_top_fst_score = 0
            current_top_fst_state = top_fst_state
            current_ifst = self.ifsts[fst_id]
            current_ifst_match_state, current_ifst_state, current_ifst_score = current_ifst.forward(
                label, ifst_state
            )
            if current_ifst_match_state == 0:
                ret.append(
                    [
                        fst_id,
                        [current_top_fst_state, current_ifst_state],
                        [current_top_fst_score, current_ifst_score],
                        [label],
                    ]
                )
            elif current_ifst_match_state == 1:
                # 分出两条路径，一条回到top fst，一条继续在ifst
                ret.append(
                    [
                        0,
                        [current_top_fst_state, current_ifst.start()],
                        [current_top_fst_score, current_ifst_score],
                        [label, fst_id],
                    ]
                )
                ret.append(
                    [
                        fst_id,
                        [current_top_fst_state, current_ifst_state],
                        [current_top_fst_score, current_ifst_score],
                        [label],
                    ]
                )
            elif current_ifst_match_state == 2:
                ret.append(
                    [
                        0,
                        [current_top_fst_state, current_ifst.start()],
                        [current_top_fst_score, current_ifst_score],
                        [label, fst_id],
                    ]
                )
        return ret

    def get_top_fst_result(self, label, state):
        """Get top fst forward results, given inlabel: label & non_terminal_ids."""
        ret = []
        next_state, score = self.top_fst.forward(label, state)
        ret.append([0, next_state, score])
        for non_terminal_id in self.ifsts:
            next_state, score = self.top_fst.forward(non_terminal_id, state)
            ret.append([non_terminal_id, next_state, score])
        return ret

    def check_labels(self, labels):
        """Check labels is full match class."""
        valid = True
        current_non_terminal_id = -1
        for label in labels:
            if label in self.ifsts:
                if current_non_terminal_id == -1:
                    current_non_terminal_id = label
                elif current_non_terminal_id == label:
                    current_non_terminal_id = -1
                else:
                    return False
        if current_non_terminal_id != -1:
            return False
        return valid
