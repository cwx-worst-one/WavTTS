import copy
from typing import Union

from .vocab import StyleTagVocab


class StyleTagTokenizerError(Exception):
    pass


class StyleTagTokenizer:
    DEFAULT_VOCAB_VER = "v0"

    def __init__(self, vocab: Union[StyleTagVocab, dict, str] = DEFAULT_VOCAB_VER):
        """
        Arg:
            vocab: A StyleTagVocab object, a dict that can be parsed by StyleTagVocab, or a path to a json file.
        """
        self.vocab = self.get_vocab(vocab)

    def __call__(self, category: str, tag: str) -> int:
        try:
            return self.vocab.token_to_id[category][tag]
        except KeyError as e:
            raise StyleTagTokenizerError(
                f'Tag "{str(e)}" is not in the vocab of cateogry "{category}"'
            )

    @staticmethod
    def get_vocab(vocab: Union[str, dict, StyleTagVocab]) -> StyleTagVocab:
        if isinstance(vocab, dict):
            return StyleTagVocab.from_dict(vocab)
        if isinstance(vocab, StyleTagVocab):
            return copy.deepcopy(vocab)
        try:
            return StyleTagVocab.from_version(vocab)
        except FileNotFoundError:
            return StyleTagVocab.from_json(vocab)
