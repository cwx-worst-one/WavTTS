from __future__ import annotations
import copy
from typing import List, Optional, Union

from recipes.bigmusic.inference.fsm.states import *


class Grammar:
    def get_begin_elements(self) -> List[Element]:
        raise NotImplementedError()

    def add_next_elements(self, eles: List[Element]):
        raise NotImplementedError()

    def get_last_element(self) -> Element:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()


class Element(Grammar):
    """Grammar node
    """
    def __init__(self, state: Union[State, str]):
        if isinstance(state, str):
            self.state = SingleTokenState(state)
        else:
            self.state = state
        self.next_elements: List[Element] = []

    def add_next_elements(self, eles: List[Element]):
        return self.next_elements.extend(eles)

    def get_begin_elements(self) -> List[Element]:
        return [self]

    def __repr__(self) -> str:
        return repr(self.state)


class Sequence(Grammar):
    def __init__(self, *grammars: Union[Grammar, State, str]):
        self.grammars: List[Grammar] = []
        for g in grammars:
            if isinstance(g, (str, State)):
                self.grammars.append(Element(g))
            else:
                self.grammars.append(copy.deepcopy(g))

        for i in range(len(self.grammars) - 1):
            self.grammars[i].add_next_elements(self.grammars[i + 1].get_begin_elements())
            for j in range(i + 1, len(self.grammars) - 1):
                if isinstance(self.grammars[j], Skippable):
                    self.grammars[i].add_next_elements(self.grammars[j + 1].get_begin_elements())
                else:
                    break

    def get_begin_elements(self) -> List[Element]:
        begin_eles = []
        for g in self.grammars:
            begin_eles.extend(g.get_begin_elements())
            if not isinstance(g, Skippable):
                break
        return begin_eles

    def add_next_elements(self, eles: List[Element]):
        for i in range(len(self.grammars) - 1, -1, -1):
            self.grammars[i].add_next_elements(eles)
            if not isinstance(self.grammars[i], Skippable):
                break

    def __repr__(self) -> str:
        prefix = "  "
        result = ["["]
        for g in self.grammars:
            lines = repr(g).split("\n")
            for l in lines:
                result.append(prefix + l)
        result.append("]")
        return "\n".join(result)

    def __getitem__(self, index: int) -> Grammar:
        return self.grammars[index]


class Skippable(Grammar):
    def __init__(self, grammar: Union[Grammar, State]):
        if isinstance(grammar, State):
            self.inner_grammar = Element(grammar)
        else:
            self.inner_grammar = copy.deepcopy(grammar)

    def get_begin_elements(self) -> List[Element]:
        return self.inner_grammar.get_begin_elements()

    def add_next_elements(self, eles: List[Element]):
        self.inner_grammar.add_next_elements(eles)

    def __getitem__(self, index: int) -> Grammar:
        return self.inner_grammar[index]


class ZeroOrMore(Skippable):
    def __init__(self, inner_grammar: Grammar):
        super().__init__(inner_grammar)
        self.inner_grammar.add_next_elements(self.inner_grammar.get_begin_elements())

    def __repr__(self) -> str:
        return repr(self.inner_grammar) + "*"


class ZeroOrOne(Skippable):
    def __repr__(self) -> str:
        return repr(self.inner_grammar) + "?"


class OneOrMore(Sequence):
    def __init__(self, grammar: Grammar):
        super().__init__(copy.deepcopy(grammar), ZeroOrMore(copy.deepcopy(grammar)))

    def __repr__(self):
        return repr(self.grammars[0]) + "+"

    def __getitem__(self, index: int) -> Grammar:
        return self.grammars[0][index]


class FSM:
    def __init__(
        self,
        grammar: Grammar
    ):
        self.grammar = grammar
        self.entry_elements = self.grammar.get_begin_elements()
        self.reset()

    def consume(self, token: str) -> List[State]:
        if self.curr_element is None:
            candidates = self.entry_elements
        else:
            candidates = self.curr_element.next_elements
        for c in candidates:
            if c.state.matches(token):
                self.curr_element = c
                break
        else:
            ## No match is found
            raise ValueError(f"{token} not valid")
        return [ne.state for ne in self.curr_element.next_elements]
    
    def reset(self):
        self.curr_element: Optional[Element] = None
