from typing import Optional

import torch
from transformers import PreTrainedTokenizer, Wav2Vec2PhonemeCTCTokenizer

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
