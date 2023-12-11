import torch
from abc import abstractmethod
from typing import List, Union



class Filter:

    def __init__(self, enabled: bool):
        self._enabled = enabled

    @abstractmethod
    def filter(self):
        pass

    def __call__(self, args, **kwargs):
        if not self._enabled:
            return False
        return self.filter(args, **kwargs)

class WordLengthFilter(Filter):

    def __init__(self, enabled: bool, min_words: int, max_words: int):
        super().__init__(enabled=enabled)
        self._min_words = min_words
        self._max_words = max_words

    def filter(self, sentence: Union[List[str], str], delimiter=" "):
        words = sentence.split(sep=delimiter)
        if self._min_words <= len(words) <= self._max_words:
            return False
        return True

class TokenLengthFilter(Filter):

    def __init__(self, enabled: bool, min_tokens: int, max_tokens: int):
        super().__init__(enabled=enabled)
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens

    def filter(self, tokens: Union[torch.Tensor, List[int]]):
        if self._min_tokens <= len(tokens) <= self._max_tokens:
            return False
        return True

class Filters:

    def __init__(self, filters: List[Filter] = []):
        self._filters = filters

    def __call__(self, x) -> bool:
        for filter in self._filters:
            if filter(x):
                return True
        return False