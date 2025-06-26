import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...utils.dcbase import DCBase

_VOCAB_DIR = Path(__file__).parent.absolute() / "vocabs"
_DEFAULT_VER = "v0"
_DEFAULT_VOCAB = _VOCAB_DIR / f"vocab.{_DEFAULT_VER}.json"
_DEFAULT_VOCAB_BUILDER = _VOCAB_DIR / f"builder.{_DEFAULT_VER}.json"


@dataclass
class StyleTagVocab(DCBase):
    version: str
    token_to_id: dict[str, dict[str, int]]

    @classmethod
    def from_json(cls, json_fp: str = _DEFAULT_VOCAB) -> "StyleTagVocab":
        return cls.from_dict(_read_json(json_fp))

    def to_json(self, json_fp: Optional[str] = None) -> None:
        if json_fp is None:
            json_fp = _VOCAB_DIR / f"vocab.{self.version}.json"
        return _write_json(self.to_dict(), json_fp)

    @classmethod
    def from_version(cls, version: str = _DEFAULT_VER) -> "StyleTagVocab":
        return cls.from_json(_VOCAB_DIR / f"vocab.{version}.json")


@dataclass
class CategoryVocabBuilder(DCBase):
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
class StyleTagVocabBuilder(DCBase):
    version: str
    category_vocabs: list[CategoryVocabBuilder]

    def build(self) -> "StyleTagVocab":
        token_to_id = {}
        token_id = 0
        for vocab in self.category_vocabs:
            token_to_id[vocab.token_type] = {}
            for token in vocab.tokens:
                assert (
                    token not in token_to_id[vocab.token_type]
                ), f"Duplicate token: {token}"
                token_to_id[vocab.token_type][token] = token_id
                token_id += 1
        return StyleTagVocab(version=self.version, token_to_id=token_to_id)

    @classmethod
    def from_json(cls, json_fp: str = _DEFAULT_VOCAB_BUILDER) -> "CategoryVocabBuilder":
        return cls.from_dict(_read_json(json_fp))

    def to_json(self, json_fp: str = _DEFAULT_VOCAB_BUILDER):
        return _write_json(self.to_dict(), json_fp)

    @classmethod
    def from_version(cls, version: str = _DEFAULT_VER) -> "CategoryVocabBuilder":
        return cls.from_json(_VOCAB_DIR / f"builder.{version}.json")


def _read_json(fp: str) -> dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(data: dict, fp: str):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
