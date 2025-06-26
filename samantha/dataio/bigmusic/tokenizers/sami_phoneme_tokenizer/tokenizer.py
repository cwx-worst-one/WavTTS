"""
Tokenizers for phonemes (by SAMI TTS frontend), notes (leadsheet), and special tags (section, singer, etc.)
"""

import copy
import logging
import operator
import random
from dataclasses import dataclass
from functools import lru_cache, reduce
from typing import Optional, Union

from .leadsheet import LeadSheetTokenizer
from .vocab import PhnVocabBuilder, SamiPhonemeVocab

logger = logging.getLogger(__file__)


class SamiPhonemeTokenizerError(ValueError):
    pass


@dataclass
class PhnStrParser:
    phn: str
    tone: str
    ws: str
    pwpp: str
    semtype: str
    word: str
    unit: str

    @classmethod
    def parse(cls, line: str, normalize: bool = True) -> "PhnStrParser":
        parser = cls(*line.split("\t"))
        if normalize:
            parser.normalize_inplace()
        return parser

    @property
    def identifier(self) -> str:
        return self.phn[:2]

    def format(self) -> str:
        return "\t".join(
            [
                self.phn,
                self.tone,
                self.ws,
                self.pwpp,
                self.semtype,
                self.word,
                self.unit,
            ]
        )

    def normalize_inplace(self) -> None:
        if self.identifier == "JP" and int(self.tone) >= 15:
            # shift japanese tone from 15, 16 to 0, 1
            self.tone = str(int(self.tone) - 15)


class SamiPhonemeTokenizer:
    DEFAULT_VOCAB_VER = "v0"
    # This parameter is for the xval coefficient normalization, which should be fixed during model training.
    # It's not supposed to change along with the configuration of song slice duration.
    MAX_DUR_SEC = 240

    @property
    def token_to_id(self) -> dict[str, int]:
        raise NotImplementedError()

    @property
    def id_to_token(self) -> dict[int, str]:
        return {v: k for k, v in self.token_to_id.items()}

    @staticmethod
    def get_vocab(vocab: Union[str, dict, SamiPhonemeVocab]) -> SamiPhonemeVocab:
        if isinstance(vocab, dict):
            return SamiPhonemeVocab.from_dict(vocab)
        if isinstance(vocab, SamiPhonemeVocab):
            return copy.deepcopy(vocab)
        try:
            return SamiPhonemeVocab.from_version(vocab)
        except FileNotFoundError:
            return SamiPhonemeVocab.from_json(vocab)


class SamiPhonemeStrTokenizer(SamiPhonemeTokenizer):
    """phoneme string to a list of integers"""

    def __init__(
        self, vocab: Union[dict[str, int], str] = SamiPhonemeTokenizer.DEFAULT_VOCAB_VER
    ):
        self._vocab_obj = self.get_vocab(vocab)
        self._token_to_id = self._vocab_obj.token_to_id
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}

    @property
    def token_to_id(self) -> dict[str, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> dict[int, str]:
        return self._id_to_token

    def validate_vocab(self):
        pass
        # TODO (Yilin): Add back the validation.
        # The previous validation was wrong, but it happened to
        # allow the v0 vocab to pass. I temporarily removed the validation
        # to make both v0 and v1 compatible with the code.
        # assert len(self.token_to_id) == len(
        #     self.id_to_token
        # ), "Duplicated tokens or input_ids in vocab"

    def _tokenize_phoneme_str(self, phonemes: str) -> dict:
        try:
            return _merge_dicts([self._tk_single_line(p) for p in phonemes.split("\n")])
        except (ValueError, TypeError, KeyError) as e:
            raise SamiPhonemeTokenizerError(f"Phonemes {phonemes} are not valid: {e}")

    def _tk_single_line(self, phoneme: str) -> dict:
        phn_str = PhnStrParser.parse(phoneme)
        vocab2id = self.token_to_id
        lang = self._vocab_obj.lang_prefixes.get(phn_str.identifier)
        if not lang:
            if (
                phn_str.phn
                in self._vocab_obj.punctuation_tokens
                + self._vocab_obj.get_all_special_tokens()
            ):
                return {"tokens": [phn_str.phn], "input_ids": [vocab2id[phn_str.phn]]}
            else:
                # Skip unknown/unsupported phonemes
                return {"tokens": [], "input_ids": []}

        if lang == "en":
            is_syl_sep = False
            is_word_sep = phn_str.pwpp != "0"
        else:
            is_syl_sep = phn_str.unit in ["S", "E"]
            is_word_sep = is_syl_sep and phn_str.ws in ["S", "E"]

        tokens = []
        token = PhnVocabBuilder(
            language=lang,
            vowels=self._vocab_obj.vowels,
            consonants=self._vocab_obj.consonants,
            tones=[],
        ).merge_phone_tone(phn_str.phn, phn_str.tone)
        tokens.append(token)

        if is_syl_sep:
            tokens.append("syl_sep")
        if is_word_sep:
            tokens.append(f"{lang}_word_sep")

        input_ids = [vocab2id[x] for x in tokens]

        return {"tokens": tokens, "input_ids": input_ids}

    def _tokenize_symbol(self, symbol: str) -> list[int]:
        return {"tokens": [symbol], "input_ids": [self.token_to_id[symbol]]}

    def __call__(
        self, phonemes: Optional[str] = None, tag: Optional[str] = None
    ) -> list[int]:
        return _merge_dicts(
            [self._tokenize_symbol(tag), self._tokenize_phoneme_str(phonemes)]
        )


class SamiPhonemeSpecialToken:
    # leadsheet tokens
    END_OF_NOTE = "[END_OF_NOTE]"
    TIME_START = "[TIME_START]"
    TIME_DURATION = "[TIME_DURATION]"

    # song slice tokens
    SECTION_DUR = "[SECTION_DUR]"
    SLICE_DUR = "[SLICE_DUR]"


class SamiPhonemeSeqTokenizer(SamiPhonemeStrTokenizer):
    """list of dicts to a list of integers (input_ids) and a list of coefficients (coffs)."""

    def __init__(
        self,
        vocab: Union[
            SamiPhonemeVocab, dict, str
        ] = SamiPhonemeTokenizer.DEFAULT_VOCAB_VER,
    ):
        super().__init__(vocab)
        self._leadsheet_tokenizer = LeadSheetTokenizer(
            token_to_id=self.token_to_id,
            time_start_token=SamiPhonemeSpecialToken.TIME_START,
            time_duration_token=SamiPhonemeSpecialToken.TIME_DURATION,
            end_of_note_token=SamiPhonemeSpecialToken.END_OF_NOTE,
            special_tokens=[
                SamiPhonemeSpecialToken.SLICE_DUR,
                SamiPhonemeSpecialToken.SECTION_DUR,
            ],
        )
        self._token_to_id = self._leadsheet_tokenizer.token_to_id
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        self.validate_vocab()  # re-validate

        self._time_input_ids = [
            self.token_to_id[SamiPhonemeSpecialToken.TIME_START],
            self.token_to_id[SamiPhonemeSpecialToken.TIME_DURATION],
        ]

    def _tokenize_phoneme_str(self, phonemes: str) -> dict:
        result = super()._tokenize_phoneme_str(phonemes)
        result["coffs"] = [1.0] * len(result["input_ids"])
        return result

    def _tokenize_symbol(self, symbol: str) -> dict:
        result = super()._tokenize_symbol(symbol)
        result["coffs"] = [1.0] * len(result["input_ids"])
        return result

    def _tokenize_phoneme_str_with_time(
        self, phonemes: str, time_span: tuple[float, float]
    ) -> dict:
        _check_time_span(time_span)
        result = super()._tokenize_phoneme_str(phonemes)
        tokens = result["tokens"]
        start, end = time_span
        leadsheet_item = {"start": start, "end": end, "pitch": [], "phone": tokens}
        input_ids, _, _, coffs = self._leadsheet_tokenizer._tokenize([leadsheet_item])
        coffs = _normalize_time_coffs(input_ids, coffs, self._time_input_ids)
        return {
            "tokens": [self._id_to_token[t] for t in input_ids],
            "input_ids": input_ids,
            "coffs": coffs,
        }

    def _tokenize_note_with_time(
        self, pitch: int, time_span: tuple[float, float]
    ) -> dict:
        _check_time_span(time_span)
        start, end = time_span
        leadsheet_item = {"start": start, "end": end, "pitch": pitch, "phone": []}
        input_ids, _, _, coffs = self._leadsheet_tokenizer._tokenize([leadsheet_item])
        coffs = _normalize_time_coffs(input_ids, coffs, self._time_input_ids)
        return {
            "tokens": [self._id_to_token[t] for t in input_ids],
            "input_ids": input_ids,
            "coffs": coffs,
        }

    def _tokenize_phoneme_str_and_note_with_time(
        self, phonemes: str, pitch: int, time_span: tuple[float, float]
    ) -> dict:
        _check_time_span(time_span)
        result = super()._tokenize_phoneme_str(phonemes)
        tokens = result["tokens"]
        start, end = time_span
        leadsheet_item = {"start": start, "end": end, "pitch": pitch, "phone": tokens}
        input_ids, _, _, coffs = self._leadsheet_tokenizer._tokenize([leadsheet_item])
        coffs = _normalize_time_coffs(input_ids, coffs, self._time_input_ids)
        return {
            "tokens": [self._id_to_token[t] for t in input_ids],
            "input_ids": input_ids,
            "coffs": coffs,
        }

    def _tokenize_section_duration(self, section_duration: float) -> dict:
        return {
            "tokens": [SamiPhonemeSpecialToken.SECTION_DUR],
            "input_ids": [self.token_to_id[SamiPhonemeSpecialToken.SECTION_DUR]],
            "coffs": [_normalize_duration(section_duration)],
        }

    def _tokenize_slice_duration(self, slice_duration: float) -> dict:
        return {
            "tokens": [SamiPhonemeSpecialToken.SLICE_DUR],
            "input_ids": [self.token_to_id[SamiPhonemeSpecialToken.SLICE_DUR]],
            "coffs": [_normalize_duration(slice_duration)],
        }

    def _drop_symbols(self, dicts: list[dict], config: dict[str, float]) -> list[dict]:
        """Drop symbols from the input dicts.
        Args:
            dicts: A list of dicts that need to be tokenized
            config: A dict of drop-out configuration. key: category name, value: dropout rate.
                The key (category name) should be the keys of the vocab json's special_tokens.
                Example:
                {
                    "section": 0.1,
                    "singer": 0.2,
                }
        """

        def get_symbol_category(symbol: str) -> Optional[str]:
            for category, tokens in self._vocab_obj.special_tokens.items():
                if symbol in tokens:
                    return category
            return None

        new_dicts = []
        for d in dicts:
            if "symbol" not in d:
                new_dicts.append(d)
                continue
            category = get_symbol_category(d["symbol"])
            if category is None:
                new_dicts.append(d)
                continue
            if random.random() < config.get(category, 0.0):
                continue  # drop the symbol
            new_dicts.append(d)

        return new_dicts

    def _drop_keys(self, dicts: list[dict], config: dict[str, float]) -> list[dict]:
        """Randomly dropout the dicts based on key
        Args:
            dicts: A list of dicts that need to be tokenized
            config: A dict of drop-out configuration. key: key name, value: dropout rate.
                Example:
                {
                    "section_duration": 0.1,
                    "slice_duration": 0.2,
                }
        """
        for key, rate in config.items():
            new_dicts = []
            for d in dicts:
                if key not in d:
                    new_dicts.append(d)
                    continue
                if random.random() < rate:
                    continue  # drop the key
                new_dicts.append(d)
            dicts = new_dicts
        return new_dicts

    def _call_by_kwargs(self, kwargs) -> dict:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # All methods that start with _tokenize are tokenization methods.
        tokenization_methods = [
            getattr(self, func)
            for func in dir(self)
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
        time_span: Optional[tuple[int, int]] = None,
        symbol: Optional[str] = None,
        section_duration: Optional[float] = None,
        slice_duration: Optional[float] = None,
    ) -> tuple[list[int], list[float]]:
        # We could have used **kwargs for the method arg to catch all,
        # but the current way allows to support other arguments/flags that are
        # not part of the arguments used by tokenization methods.
        return self._call_by_kwargs(
            {
                "phonemes": phonemes,
                "pitch": pitch,
                "time_span": time_span,
                "symbol": symbol,
                "section_duration": section_duration,
                "slice_duration": slice_duration,
            }
        )

    def __call__(
        self,
        dicts: list[dict],
        key_dropout_config: Optional[dict[str, float]] = None,
        symbol_dropout_config: Optional[dict[str, float]] = None,
    ) -> dict:
        if key_dropout_config:
            dicts = self._drop_keys(dicts, key_dropout_config)
        if symbol_dropout_config:
            dicts = self._drop_symbols(dicts, symbol_dropout_config)
        if not dicts:
            return {"tokens": [], "input_ids": [], "coffs": []}
        results = [self._call_one(**d) for d in dicts]
        return _merge_dicts(results)

    @classmethod
    @lru_cache
    def init_cached(
        cls, vocab: Union[dict[str, int], str] = SamiPhonemeTokenizer.DEFAULT_VOCAB_VER
    ) -> "SamiPhonemeSeqTokenizer":
        return cls(vocab)


class SamiPhonemeSeqPosTokenizer(SamiPhonemeSeqTokenizer):
    """List of dicts to a list of integers (tokens) and a list of coefficients (coffs)."""

    def __init__(
        self,
        vocab: Union[
            SamiPhonemeVocab, dict, str
        ] = SamiPhonemeTokenizer.DEFAULT_VOCAB_VER,
    ):
        super().__init__(vocab)
        self.blank_id = -1

    def _tokenize_phoneme_str(self, phonemes: str) -> dict:
        result = super()._tokenize_phoneme_str(phonemes)
        result["pos"] = [z for z in range(len(result["input_ids"]))]
        return result

    def __call__(
        self,
        dicts: list[dict],
        key_dropout_config: Optional[dict[str, float]] = None,
        symbol_dropout_config: Optional[dict[str, float]] = None,
    ) -> dict:
        if key_dropout_config:
            dicts = self._drop_keys(dicts, key_dropout_config)
        if symbol_dropout_config:
            dicts = self._drop_symbols(dicts, symbol_dropout_config)
        if not dicts:
            return {"tokens": [], "input_ids": [], "coffs": [], "pos": []}

        results = [self._call_one(**d) for d in dicts]
        tokens, input_ids, coeffs, pos = [], [], [], []

        utt_cnt = 0
        input_id_of_last_symbol = self.blank_id

        for result, this_dict in zip(results, dicts):

            tokens.append(result["tokens"])
            coeffs.append(result["coffs"])
            input_ids.append(result["input_ids"])

            if (
                "symbol" in this_dict
            ):  # here we assume that the symbol (section tag) is the only token in the result
                input_id_of_last_symbol = result["input_ids"][0]
            if "pos" in result:
                this_pos = [
                    [utt_cnt, phone_pos, input_id_of_last_symbol]
                    for phone_pos in result["pos"]
                ]
                pos.extend(this_pos)
                utt_cnt += 1
            else:
                pos.append([self.blank_id, self.blank_id, self.blank_id])

        tokens = reduce(operator.add, tokens)
        input_ids = reduce(operator.add, input_ids)
        coeffs = reduce(operator.add, coeffs)

        return {"tokens": tokens, "input_ids": input_ids, "coffs": coeffs, "pos": pos}


def _clamp(val: float, range: tuple[float, float]) -> float:
    _min, _max = range
    return min(max(val, _min), _max)


def _normalize_duration(
    val: float,
    max_duration: float = SamiPhonemeTokenizer.MAX_DUR_SEC,
    max_val: float = 10,
) -> float:
    val = _clamp(val, (0, max_duration))
    return val / max_duration * max_val


def _normalize_time_coffs(
    input_ids: list[int], coffs: list[float], token_match: Union[int, list[int]]
):
    def match_token(input_id: int) -> int:
        if isinstance(token_match, int):
            return input_id == token_match
        return input_id in token_match

    return [
        _normalize_duration(tc) if match_token(t) else tc
        for t, tc in zip(input_ids, coffs)
    ]


def _check_time_span(time_span: tuple[float, float]) -> None:
    start, end = time_span
    if start < 0 or end < 0 or start > end:
        raise SamiPhonemeTokenizerError(f"Invalid time span {time_span}")


def _merge_dicts(ds: list[dict[str, list]]) -> dict:
    return {k: reduce(operator.add, [d[k] for d in ds]) for k in ds[0]}
