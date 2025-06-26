import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...utils.dcbase import DCBase

_TONE_SEP = "@"

_VOCAB_DIR = Path(__file__).parent.absolute() / "vocabs"
_DEFAULT_VER = "v0"
_DEFAULT_VOCAB = _VOCAB_DIR / f"vocab.{_DEFAULT_VER}.json"
_DEFAULT_VOCAB_BUILDER = _VOCAB_DIR / f"builder.{_DEFAULT_VER}.json"


@dataclass
class SamiPhonemeVocab(DCBase):
    version: str
    token_to_id: dict[str, int]
    special_tokens: dict[str, list[str]]
    punctuation_tokens: list[str]
    vowels: list[str]
    consonants: list[str]
    lang_prefixes: dict[str, str]

    @classmethod
    def from_json(cls, json_fp: str = _DEFAULT_VOCAB) -> "SamiPhonemeVocab":
        return cls.from_dict(_read_json(json_fp))

    def to_json(self, json_fp: Optional[str] = None) -> None:
        if json_fp is None:
            json_fp = _VOCAB_DIR / f"vocab.{self.version}.json"
        return _write_json(self.to_dict(), json_fp)

    @classmethod
    def from_version(cls, version: str = _DEFAULT_VER) -> "SamiPhonemeVocab":
        return cls.from_json(_VOCAB_DIR / f"vocab.{version}.json")

    def get_all_special_tokens(self) -> list[str]:
        return [token for tokens in self.special_tokens.values() for token in tokens]


@dataclass
class VocabBuilderBase(DCBase):
    def to_placeholder_tokens(self) -> list[str]:
        raise NotImplementedError

    def to_tokens(self) -> list[str]:
        raise NotImplementedError


@dataclass
class SpecialVocabBuilder(VocabBuilderBase):
    token_type: str
    tokens: list[str]
    n_placeholders: int = 0

    def to_tokens(self) -> list[str]:
        return self.tokens + self.to_placeholder_tokens()

    def to_placeholder_tokens(self) -> list[str]:
        return [
            f"<{self.token_type}>placeholder-{i}" for i in range(self.n_placeholders)
        ]


@dataclass
class PhnVocabBuilder(VocabBuilderBase):
    language: str
    consonants: list[str]
    vowels: list[str]
    tones: list[str]
    n_placeholders: int = 0

    def to_tokens(self) -> list[str]:
        tokens = []
        for phn in self.vowels:
            for tone in self.tones:
                tokens.append(self.merge_phone_tone(phn, tone))
        for phn in self.consonants:
            tokens.append(phn)
        return tokens + self.to_placeholder_tokens()

    def merge_phone_tone(self, phone: str, tone: str) -> str:
        if phone in self.vowels:
            return f"{phone}{_TONE_SEP}{tone}"
        return phone

    def to_placeholder_tokens(self) -> list[str]:
        return [f"<{self.language}>placeholder-{i}" for i in range(self.n_placeholders)]


@dataclass
class SamiPhonemeVocabBuilder(DCBase):
    version: str
    special_vocabs: list[SpecialVocabBuilder]
    punctuation_vocabs: list[SpecialVocabBuilder]
    phn_vocabs: list[PhnVocabBuilder]

    def build(self) -> SamiPhonemeVocab:
        token_to_id = {}

        special_token_dict = {}
        for special_vocab in self.special_vocabs:
            special_token_dict[special_vocab.token_type] = special_vocab.tokens
            for token in special_vocab.tokens:
                assert token not in token_to_id, f"Duplicate token: {token}"
                token_to_id[token] = len(token_to_id)

        punc_tokens = []
        for punctuation_vocab in self.punctuation_vocabs:
            punc_tokens.extend(punctuation_vocab.tokens)
            for token in punctuation_vocab.tokens:
                assert token not in token_to_id, f"Duplicate token: {token}"
                token_to_id[token] = len(token_to_id)

        lang_prefixes = {}
        vowels = []
        consonants = []
        for phn_vocab in self.phn_vocabs:
            vowels.extend(phn_vocab.vowels)
            consonants.extend(phn_vocab.consonants)
            for token in phn_vocab.to_tokens():
                assert token not in token_to_id, f"Duplicate token: {token}"
                token_to_id[token] = len(token_to_id)
            lang_prefixes[token[:2]] = phn_vocab.language

        return SamiPhonemeVocab(
            version=self.version,
            token_to_id=token_to_id,
            special_tokens=special_token_dict,
            punctuation_tokens=punc_tokens,
            vowels=vowels,
            consonants=consonants,
            lang_prefixes=lang_prefixes,
        )

    @classmethod
    def from_json(
        cls, json_fp: str = _DEFAULT_VOCAB_BUILDER
    ) -> "SamiPhonemeVocabBuilder":
        return cls.from_dict(_read_json(json_fp))

    def to_json(self, json_fp: str = _DEFAULT_VOCAB_BUILDER) -> None:
        _write_json(self.to_dict(), json_fp)

    @classmethod
    def from_version(
        cls, version: str = _DEFAULT_VOCAB_BUILDER
    ) -> "SamiPhonemeVocabBuilder":
        return cls.from_json(_VOCAB_DIR / f"builder.{version}.json")


def _read_json(fp: str) -> dict:
    with open(fp, "r") as f:
        return json.load(f)


def _write_json(data: dict, fp: str):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
