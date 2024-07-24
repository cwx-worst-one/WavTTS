import logging
from typing import List, Optional, Union

import phonemizer
import torch
from transformers import PreTrainedTokenizer, Wav2Vec2PhonemeCTCTokenizer


class PhonemeTokenizer:
    tokenizer: PreTrainedTokenizer

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def decode(self, tokens: Union[torch.Tensor, List[List[int]]]) -> List[str]:
        if type(tokens) == torch.Tensor:
            tokens = tokens.tolist()
        return self.tokenizer.batch_decode(tokens)

    def __call__(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt", padding=True, return_length=True
        )


class Wav2VecPhonemeTokenizer(PhonemeTokenizer):
    def __init__(self):
        super().__init__()
        self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        phonemizer.logger.get_logger().setLevel(logging.ERROR)

    def _tokenize(self, text: str):
        return self.tokenizer._tokenize(text)
