from typing import List

import torch
from whisper.tokenizer import get_tokenizer

from recipes.research.diff.transforms.text.normalizers.english import (
    EnglishTextNormalizer,
)
from recipes.research.diff.transforms.tokenizers.phoneme_tokenizer import (
    Wav2VecPhonemeTokenizer,
)


def sequence_mask(length: torch.Tensor, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class LyricsNormalizer:
    def __init__(self):
        self.normalizer = EnglishTextNormalizer()

    def normalize(self, text: List[str]):
        normalized = []
        for line in text:
            normalized.append(self.normalizer(line))
        return normalized

    def __call__(self, text: List[str]):
        text = self.normalize(text)
        return text


class LyricsTokenizer:
    _newline_char = " \n "
    _rest_token = "<REST>"

    def __init__(self):
        self.normalizer = LyricsNormalizer()

        self.tokenizer = get_tokenizer(multilingual=True, task="transcribe")

        # self.tokenizer.tokenizer.add_tokens(
        #     [
        #         self._word_delimiter,
        #         self._newline_char,
        #         self._rest_token,
        #         "<verse>",
        #         "<chorus>",
        #         "<intro>",
        #         "<bridge>",
        #         "<inst>",
        #     ]
        # )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.encoding.max_token_value + 1

    @property
    def sos_token_id(self) -> int:
        # NOTE: whisper has single token for bos, eos, pad and unk
        return self.tokenizer.eot

    @property
    def eos_token_id(self) -> int:
        # NOTE: whisper has single token for bos, eos, pad and unk
        return self.tokenizer.eot

    @property
    def pad_token_id(self) -> int:
        # NOTE: whisper has single token for bos, eos, pad and unk
        return self.tokenizer.eot

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, token_ids: torch.Tensor):
        # NOTE: errors=strict
        return self.tokenizer.decode(
            token_ids,
            # errors="strict"
        )

    def batch_decode(self, token_ids: torch.Tensor) -> List[str]:
        return [self.decode(t) for t in token_ids]

    def __call__(self, text: List[List[str]], device: torch.device):
        assert type(text[0]) == list

        batch_size = len(text)
        normalized_text = []
        batch_input_ids = []
        lengths = []
        for t in text:
            # if only 1 whitespace, treat it as a <REST> token:
            t = [t.replace(" ", self._rest_token) if t == " " else t for t in t]

            # t = self.normalizer(t)

            normalized_t = f"{self._newline_char}".join(t)

            tokens = self.encode(normalized_t)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            normalized_text.append(normalized_t)
            batch_input_ids.append(tokens)
            lengths.append(len(tokens))

        lengths = torch.tensor(lengths, device=device)
        max_length = max(lengths)
        for idx in range(batch_size):
            pad_len = max_length - len(batch_input_ids[idx])
            batch_input_ids[idx] = torch.nn.functional.pad(
                batch_input_ids[idx], (0, pad_len), value=self.pad_token_id
            )

        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        attention_mask = sequence_mask(lengths)

        return {
            "text": text,
            "normalized_text": normalized_text,
            "input_ids": batch_input_ids,
            "length": lengths,
            "attention_mask": attention_mask,
        }


class LyricsPhonemeTokenizer:
    _newline_char = "\n"
    _rest_token = "<REST>"

    def __init__(self):
        self.normalizer = None  # LyricsNormalizer()
        self.tokenizer = Wav2VecPhonemeTokenizer()
        self._word_delimiter = self.tokenizer.tokenizer.word_delimiter_token
        self.tokenizer.tokenizer.add_tokens(
            [
                self._word_delimiter,
                self._newline_char,
                self._rest_token,
                "<verse>",
                "<chorus>",
                "<intro>",
                "<bridge>",
                "<inst>",
            ]
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def batch_decode(self, token_ids: torch.Tensor):
        return self.tokenizer.decode(token_ids)

    def __call__(self, text: List[List[str]], device: torch.device):
        assert type(text[0]) == list

        normalized_text = []
        for t in text:

            # if only 1 whitespace, treat it as a <REST> token:
            t = [t.replace(" ", self._rest_token) if t == " " else t for t in t]

            # t = self.normalizer(t)

            normalized_t = f"{self._newline_char}".join(t)
            normalized_text.append(normalized_t)

        res = self.tokenizer(normalized_text).to(device)
        return {
            "text": text,
            "normalized_text": normalized_text,
            "input_ids": res["input_ids"],
            "length": res["length"],
            "attention_mask": res["attention_mask"],
        }


class LyricsPhonemeDurationTokenizer:
    _newline_char = "\n"
    _rest_token = "<REST>"

    def __init__(self):
        self.normalizer = None  # LyricsNormalizer()
        self.tokenizer = Wav2VecPhonemeTokenizer()
        self._word_delimiter = self.tokenizer.tokenizer.word_delimiter_token
        self.tokenizer.tokenizer.add_tokens(
            [
                self._word_delimiter,
                self._newline_char,
                self._rest_token,
                "<verse>",
                "<chorus>",
                "<intro>",
                "<bridge>",
                "<inst>",
            ]
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def decode(self, token_ids: torch.Tensor):
        return self.tokenizer.decode(token_ids)

    def __call__(self, text: List[List[str]], durations: torch.Tensor):
        assert type(text[0]) == list
        batch_size = len(text)

        lengths = []
        batch_phoneme_ids = []
        batch_durations = []
        normalized_text = []
        for idx in range(batch_size):
            words = text[idx]
            word_durations = durations[idx, : len(words)]

            # if only 1 whitespace, treat it as a <REST> token:
            words = [w.replace(" ", self._rest_token) if w == " " else w for w in words]
            normalized_text.append(words)

            # phonemize here and align word-level durations by averaging them across phonemes
            phomeme_ids = []
            all_durations = []
            for w, w_dur in zip(words, word_durations):
                phonemes = self.tokenizer._tokenize(w)
                phonemes = phonemes[:-1]  # TODO remove | token (?)
                phoneme_ids = [
                    self.tokenizer.tokenizer._convert_token_to_id(p) for p in phonemes
                ]
                avg_durations = [
                    w_dur / len(phoneme_ids) for _ in range(len(phoneme_ids))
                ]

                phomeme_ids.extend(phoneme_ids)
                all_durations.extend(avg_durations)

            phomeme_ids = torch.tensor(phomeme_ids)
            all_durations = torch.stack(all_durations)
            assert len(phomeme_ids) == len(all_durations)

            batch_phoneme_ids.append(phomeme_ids)
            batch_durations.append(all_durations)
            lengths.append(len(phomeme_ids))

        lengths = torch.tensor(lengths)
        max_length = max(lengths)
        for idx in range(batch_size):
            pad_len = max_length - len(batch_phoneme_ids[idx])

            batch_phoneme_ids[idx] = torch.nn.functional.pad(
                batch_phoneme_ids[idx], (0, pad_len), value=self.pad_token_id
            )
            batch_durations[idx] = torch.nn.functional.pad(
                batch_durations[idx], (0, pad_len), value=0.0
            )

        batch_phoneme_ids = torch.stack(batch_phoneme_ids, dim=0)
        batch_durations = torch.stack(batch_durations, dim=0)
        attention_mask = sequence_mask(lengths)
        # phonemes = self.decode(batch_phoneme_ids)
        return {
            "text": text,
            "normalized_text": normalized_text,
            "input_ids": batch_phoneme_ids,
            "durations": batch_durations,
            "length": lengths,
            "attention_mask": attention_mask,
        }
