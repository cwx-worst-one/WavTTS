"""
Tokenizers for phonemes (by SAMI TTS frontend), notes (leadsheet), and special tags (section, singer, etc.)
"""

import copy
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce, lru_cache
import logging
import operator
from recipes.structure.constants import substr_map, seg_map

from ..utils.sami_parser import (
    convert_labels_to_text_id,
    PHONExTONE_TO_INT_LEGACY,
    PHONExTONE_TO_INT_V2,
    PHONExTONE_TO_INT_V3,
    PHONExTONE_TO_INT_V4,
    PHONExTONE_TO_INT_V5,
)
from .leadsheet import LeadSheetTokenizerV2


logger = logging.getLogger(__file__)


class SamiPhonemeTokenizerConfig:
    VOCABS = {
        "legacy": PHONExTONE_TO_INT_LEGACY,
        "v2": PHONExTONE_TO_INT_V2,
        "v3": PHONExTONE_TO_INT_V3,     # added instrument vocab
        "v4": PHONExTONE_TO_INT_V4,     # added japanese vocab
        "v5": PHONExTONE_TO_INT_V5,     # added new section tags
    }
    DEFAULT_VOCAB = "legacy"
    MAX_DUR_SEC = 240

    @classmethod
    def get_vocab(cls, vocab: Union[str, Dict[str, int]]):
        return copy.deepcopy(cls.VOCABS[vocab]) if isinstance(vocab, str) else vocab


class SamiPhonemeTokenizerError(ValueError):
    pass


class SamiPhonemeTokenizer:
    @property
    def vocab(self) -> Dict[str, int]:
        raise NotImplementedError()
    
    @property
    def reversed_vocab(self) -> Dict[int, str]:
        return {v: k for k, v in self.vocab.items()}


class SamiPhonemeStrTokenizer(SamiPhonemeTokenizer):
    """phoneme string to a list of integers"""

    def __init__(self, vocab: Union[Dict[str, int], str] = SamiPhonemeTokenizerConfig.DEFAULT_VOCAB):
        self._vocab = SamiPhonemeTokenizerConfig.get_vocab(vocab)
        self._reversed_vocab = {v: k for k, v in self._vocab.items()}

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def reversed_vocab(self) -> Dict[int, str]:
        return self._reversed_vocab

    def _tokenize_phoneme_str(self, phonemes: str) -> List[int]:
        labels = list(filter(lambda x: x != "", phonemes.split("\n"))) if phonemes else None
        # TODO (Yilin): Refactor the conversion fn
        try:
            return convert_labels_to_text_id(labels, vocab_type="phoneme+tone", vocab=self.vocab)[0][0].tolist()
        except ValueError as e:
            raise SamiPhonemeTokenizerError(f"Phonemes {phonemes} are not valid: {e}")
    
    def _tokenize_symbol(self, symbol: str) -> List[int]:
        if self.vocab.get(symbol, None) is None:
            symbol = symbol.lower()
            for s in substr_map:
                if s in symbol:
                    symbol = substr_map[s]
                    break
        if self.vocab.get(symbol, None) is None:
            symbol = seg_map.get(symbol, "other")
        return [self.vocab[symbol]]

    def __call__(
        self,
        phonemes: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[int]:
        return self._tokenize_symbol(tag) + self._tokenize_phoneme_str(phonemes)


class SamiPhonemeSeqTokenizer(SamiPhonemeStrTokenizer):
    """List of dicts to a list of integers (tokens) and a list of coefficients (coffs)."""

    class SpecialSymbols:
        SECTION_DUR = "[SECTION_DUR]"
        SLICE_DUR = "[SLICE_DUR]"
        # Special symbols defined in sami_parser
        PHONE_PAD = "[PAD]"
        EOS = "[EOS]"
        # Special symbols defined in the leadsheet tokenizer
        START_OF_TIME = "[START_OF_TIME]"
        DURATION = "[DURATION]"
        END_OF_TIME = "[END_OF_TIME]"
        END_OF_NOTE = "[END_OF_NOTE]"
        TIME = 2000

    def __init__(self, vocab: Union[Dict[str, int], str] = SamiPhonemeTokenizerConfig.DEFAULT_VOCAB):
        self._vocab = SamiPhonemeTokenizerConfig.get_vocab(vocab)
        self._leadsheet_tokenizer = LeadSheetTokenizerV2(all_phones=[
            k for k in self._vocab.keys() if k not in [self.SpecialSymbols.PHONE_PAD, self.SpecialSymbols.EOS]
        ])
        # The internal vocab (self._vocab) is an extension of self.leadsheet_tokenizer.vocab.
        # We don't want to modify the vocab inside leadsheet_tokenizer, so we need to maintain
        # a copy of it to modify.
        self._vocab = copy.deepcopy(self._leadsheet_tokenizer.vocab)
        self._patch_vocab()
        self._validate_vocab()

    def _patch_vocab(self) -> None:
        """Add 2 special tokens: section_duration and slice_duration
        which are used to indicate the duration of a section or a slice."""
        max_token_id = max(list(self._vocab.values()))
        for key in [self.SpecialSymbols.SECTION_DUR, self.SpecialSymbols.SLICE_DUR]:
            if key in self._vocab:
                raise SamiPhonemeTokenizerError(f"{key} already exists in vocab")
            self._vocab[key] = max_token_id + 1
            max_token_id += 1
        self._reversed_vocab = {v: k for k, v in self.vocab.items()}

    def _validate_vocab(self) -> None:
        """Check if all the special symbols are in the vocab"""
        for attr in [attr for attr in dir(self.SpecialSymbols) if not callable(getattr(self.SpecialSymbols, attr)) and not attr.startswith("__")]:
            sym = getattr(self.SpecialSymbols, attr)
            if sym not in self.vocab:
                raise SamiPhonemeTokenizerError(f"Special symbol {sym} not in vocab")

    def _tokenize_phoneme_str(self, phonemes: str) -> Tuple[List[int], List[float]]:
        tokens = super()._tokenize_phoneme_str(phonemes)
        return tokens, [1.0] * len(tokens)

    def _tokenize_symbol(self, symbol: str) -> Tuple[List[int], List[float]]:
        tokens = super()._tokenize_symbol(symbol)
        return tokens, [1.0] * len(tokens)

    def _tokenize_phoneme_str_with_time(self, phonemes: str, time_span: Tuple[float, float]) -> Tuple[List[int], List[float]]:
        _check_time_span(time_span)
        tokens = super()._tokenize_phoneme_str(phonemes)
        symbols = [self._reversed_vocab[t] for t in tokens]
        start, end = time_span
        leadsheet_item = {
            "start": start,
            "end": end,
            "pitch": [],
            "phone": symbols,
        }
        tokens, _, _, token_coffs = self._leadsheet_tokenizer._tokenize([leadsheet_item])
        token_coffs = _normalize_time_coffs(tokens, token_coffs, self.vocab[self.SpecialSymbols.TIME])
        return tokens, token_coffs
    
    def _tokenize_note_with_time(self, pitch: int, time_span: Tuple[float, float])-> Tuple[List[int], List[float]]:
        _check_time_span(time_span)
        start, end = time_span
        leadsheet_item = {
            "start": start,
            "end": end,
            "pitch": pitch,
            "phone": [],
        }
        tokens, _, _, token_coffs = self._leadsheet_tokenizer._tokenize([leadsheet_item])
        token_coffs = _normalize_time_coffs(tokens, token_coffs, self.vocab[self.SpecialSymbols.TIME])
        return tokens, token_coffs
    
    def _tokenize_section_duration(self, section_duration: float) -> Tuple[List[int], List[float]]:
        return [self.vocab[self.SpecialSymbols.SECTION_DUR]], [_normalize_duration(section_duration)]

    def _tokenize_slice_duration(self, slice_duration: float) -> Tuple[List[int], List[float]]:
        return [self.vocab[self.SpecialSymbols.SLICE_DUR]], [_normalize_duration(slice_duration)]
    
    def _call_by_kwargs(self, kwargs) -> Tuple[List[int], List[float]]:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # All methods that start with _tokenize are tokenization methods.
        tokenization_methods = [
            getattr(self, func) for func in dir(self)
            if callable(getattr(self, func)) and func.startswith("_tokenize")
        ]
        for method in tokenization_methods:
            try:
                return method(**kwargs)
            except TypeError:
                continue
        raise SamiPhonemeTokenizerError(f"kwargs {kwargs} are not supported")

    def _call_one(
        self,
        phonemes: Optional[str] = None,
        pitch: Optional[int] = None,
        time_span: Optional[Tuple[int, int]] = None,
        symbol: Optional[str] = None,
        section_duration: Optional[float] = None,
        slice_duration: Optional[float] = None,
    ) -> Tuple[List[int], List[float]]:
        # We could have used **kwargs for the method arg to catch all,
        # but the current way allows to support other arguments/flags that are
        # not part of the arguments used by tokenization methods.
        return self._call_by_kwargs({
            "phonemes": phonemes,
            "pitch": pitch,
            "time_span": time_span,
            "symbol": symbol,
            "section_duration": section_duration,
            "slice_duration": slice_duration,
        })

    def __call__(self, dicts: List[Dict]) -> Tuple[List[int], List[float]]:
        if not dicts:
            return [], []
        results = [self._call_one(**d) for d in dicts]
        tokens = reduce(operator.add, [r[0] for r in results])
        coffs = reduce(operator.add, [r[1] for r in results])
        return tokens, coffs

    @classmethod
    @lru_cache
    def init_cached(cls, vocab: Union[Dict[str, int], str] = SamiPhonemeTokenizerConfig.DEFAULT_VOCAB) -> "SamiPhonemeSeqTokenizer":
        return cls(vocab)


def _clamp(val: float, range: Tuple[float, float]) -> float:
    _min, _max = range
    return min(max(val, _min), _max)


def _normalize_duration(
    val: float,
    max_duration: float = SamiPhonemeTokenizerConfig.MAX_DUR_SEC,
    max_val: float = 10,
) -> float:
    val = _clamp(val, (0, max_duration))
    return val / max_duration * max_val


def _normalize_time_coffs(tokens: List[int], token_coffs: List[float], token_match: int):
    return [_normalize_duration(tc) if t == token_match else tc for t, tc in zip(tokens, token_coffs)]


def _check_time_span(time_span: Tuple[float, float]) -> None:
    start, end = time_span
    if start < 0 or end < 0 or start > end:
        raise SamiPhonemeTokenizerError(f"Invalid time span {time_span}")
