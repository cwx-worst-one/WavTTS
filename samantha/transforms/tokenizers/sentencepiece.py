import os
from pathlib import Path
from typing import Optional

import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

from samantha.utils.hdfs_helper import get


class SentencePieceTokenizer:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path

    def setup(self, data: str) -> None:
        if not os.path.exists(self._model_path):
            get(
                "hdfs://harunava/home/byte_speech_sv/models/llama/tokenizer.model",
                self._model_path,
            )

        self._model_path = "sentence_piece.tokenizer"
        self.train(data, model_prefix=self._model_path, vocab_size=65)
        self.processor = SentencePieceProcessor(model_file=self._model_path)

        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.pad_id = self.processor.pad_id()
        assert self.processor.vocab_size() == self.processor.get_piece_size()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(
        self,
        string: str,
        bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        tokens = self.processor.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long, device=device)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.processor.decode(tokens.tolist())

    @staticmethod
    def train(text: str, model_prefix: str, vocab_size=32000) -> None:
        SentencePieceTrainer.Train(
            input=text, model_prefix=model_prefix, vocab_size=vocab_size
        )
