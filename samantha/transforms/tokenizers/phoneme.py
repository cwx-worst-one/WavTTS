import logging
from typing import List, Optional, Union

import phonemizer
import torch
from transformers import PreTrainedTokenizer, Wav2Vec2PhonemeCTCTokenizer

from recipes.bigmusic.utils.format_utils import normalize_text

MAX_PHONE_LEN = 400


class PhonemeTokenizer:
    tokenizer: PreTrainedTokenizer

    def __init__(self, max_phone_len: int = MAX_PHONE_LEN):
        self._max_phone_len = max_phone_len

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.tokenizer.pad_token_id

    @property
    def max_phone_len(self) -> int:
        return self._max_phone_len

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self.tokenizer.vocab_size

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return len(self.tokenizer)

    def __call__(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self._max_phone_len,
        )["input_ids"]


class Wav2VecPhonemeTokenizer(PhonemeTokenizer):
    def __init__(self, max_phone_len: int = MAX_PHONE_LEN):
        super().__init__(max_phone_len=max_phone_len)
        self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        phonemizer.logger.get_logger().setLevel(logging.ERROR)


class LyricPhonemeTokenizer(Wav2VecPhonemeTokenizer):
    _lyric_cls_token = "<LYRICS>"

    def __init__(
        self, max_phone_len: int = MAX_PHONE_LEN, newline_character: str = "<n>"
    ):
        super().__init__(max_phone_len=max_phone_len)
        self.tokenizer._add_tokens(
            [newline_character, "<verse>", "<chorus>", "<intro>", "<bridge>", "<inst>"]
        )
        self.tokenizer.add_special_tokens({"cls_token": self._lyric_cls_token})

    def __call__(
        self, text: Union[List[str], str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if type(text) == str:
            text = [text]

        normalized_text = list(
            map(
                lambda t: normalize_text(t, enable_punctuation=True, lowercase=False),
                text,
            )
        )

        token_ids = self.tokenizer(
            normalized_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self._max_phone_len,
            add_special_tokens=True,
        )["input_ids"]

        if device:
            token_ids = token_ids.to(device)
        return {
            "token_ids": token_ids,
            "normalized_text": normalized_text,
            "text": text,
        }
