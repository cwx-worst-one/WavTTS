from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch

Vocab = Dict[str, int]


def sequence_mask(length: torch.Tensor, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class TokenizerBase:

    _pad_token = "<|pad|>"
    _sos_token = "<|sos|>"
    _eos_token = "<|eos|>"

    _special_dict: Vocab = {_pad_token: 0, _sos_token: 1, _eos_token: 2}

    def __init__(self, vocab: List[str], special_dict: Optional[Vocab] = None):

        if special_dict is not None:
            self.special_dict = self._special_dict.update(special_dict)

        self.vocab = self.build_vocab(vocab)
        self.rev_vocab = {idx: word for word, idx in self.vocab.items()}
        self.pad_token_id = self._special_dict[self._pad_token]
        self.sos_token_id = self._special_dict[self._sos_token]
        self.eos_token_id = self._special_dict[self._eos_token]
        self.unk_token_id = self._special_dict[self._pad_token]
        self.special_token_ids = list(self._special_dict.values())

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def build_vocab(self, v: List[str]) -> Vocab:
        vocab = self._special_dict.copy()
        idx = len(vocab)
        for word in v:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
        return vocab

    def word_to_index(self, word):
        return self.vocab.get(word, self.unk_token_id)

    def encode(self, words: List[str]) -> List[int]:
        tokens = [self.sos_token_id]
        for word in words:
            tokens.append(self.word_to_index(word))
        tokens.append(self.eos_token_id)
        return tokens

    def batch_encode(
        self, batch_words: List[List[str]], device: torch.device
    ) -> torch.Tensor:
        batch_tokens = []
        lengths = []
        for words in batch_words:
            tokens = self.encode(words)
            batch_tokens.append(tokens)
            lengths.append(len(tokens))

        lengths = torch.tensor(lengths, device=device)
        batch_tokens = self.pad_sequence(batch_tokens, max(lengths), device)

        attention_mask = sequence_mask(lengths)
        return {
            "normalized_text": batch_words,
            "input_ids": batch_tokens,
            "length": lengths,
            "attention_mask": attention_mask,
        }

    def decode(self, token_ids: List[int], remove_special_tokens: bool) -> str:
        if remove_special_tokens:
            return "".join(
                [
                    self.rev_vocab.get(token_id, "")
                    for token_id in token_ids
                    if token_id not in self.special_token_ids
                ]
            )
        else:
            return "".join(
                [
                    self.rev_vocab.get(token_id, self.unk_token_id)
                    for token_id in token_ids
                ]
            )

    def batch_decode(
        self,
        batch_token_ids: Union[torch.Tensor, List[List[int]]],
        remove_special_tokens: bool = False,
    ) -> List[str]:
        if type(batch_token_ids) is torch.Tensor:
            batch_token_ids = batch_token_ids.tolist()
        return [
            self.decode(token_ids, remove_special_tokens)
            for token_ids in batch_token_ids
        ]

    def pad_sequence(
        self, batch_tokens: List[List[int]], max_length: int, device: torch.device
    ):
        batch_size = len(batch_tokens)
        padded_tokens = torch.full(
            (batch_size, max_length), self.pad_token_id, dtype=torch.long, device=device
        )
        for idx, tokens in enumerate(batch_tokens):
            length = len(tokens)
            padded_tokens[idx, :length] = torch.tensor(
                tokens, dtype=torch.long, device=device
            )
        return padded_tokens

    def __call__(self, batch_words: Union[List[str], str], device: torch.device):
        if isinstance(batch_words, str):
            batch_words = [batch_words]
        elif not isinstance(batch_words, list):
            raise ValueError("Input should be a string or a list of strings.")

        tokens = self.batch_encode(batch_words, device)
        return tokens
