import random
from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import Self

from samantha.data.base import LightningDataModuleBase
from samantha.transforms.tokenizers.character import CharacterTokenizer
from samantha.transforms.tokenizers.sentencepiece import SentencePieceTokenizer
from samantha.utils.logger import RankedLogger

logger = RankedLogger(rank_zero_only=True)


@dataclass
class TokenDataResult:
    x: torch.Tensor
    y: torch.Tensor


class TokenDataset(Dataset):
    def __init__(self, fp: str, max_seq_len: int):
        self.data = np.memmap(fp, dtype=np.uint16, mode="r")
        self.max_seq_len = max_seq_len
        self.total = len(self.data)

    @staticmethod
    def prepare(
        text: str,
        tokenizer: Union[SentencePieceTokenizer, CharacterTokenizer],
        out_fp: str,
    ):
        ids = np.array(tokenizer.encode(text, device="cpu"), dtype=np.uint16)
        ids.tofile(out_fp)

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        idx = random.randint(0, self.total - self.max_seq_len - 1)
        x = torch.from_numpy(
            self.data[idx : idx + self.max_seq_len + 1].astype(np.int64)
        )
        y = x[1:]
        x = x[:-1]
        return TokenDataResult(x=x, y=y)


class TokenDataModule(LightningDataModuleBase):
    def __init__(
        self,
        data_fp: str,
        tokenizer: Union[SentencePieceTokenizer, CharacterTokenizer],
        max_seq_len: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        self.data_fp = data_fp
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.prepare_data()
        train_dataset = TokenDataset("train.bin", max_seq_len=self.max_seq_len)
        validation_dataset = TokenDataset("valid.bin", max_seq_len=self.max_seq_len)
        test_dataset = TokenDataset("test.bin", max_seq_len=self.max_seq_len)
        predict_dataset = test_dataset

        super().__init__(
            train_dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def collate_fn(self, batch):
        result = TokenDataResult(x=[], y=[])
        for b in batch:
            result.x.append(b.x)
            result.y.append(b.y)

        result.x = torch.stack(result.x, dim=0)
        result.y = torch.stack(result.y, dim=0)
        return result

    def transfer_batch_to_device(
        self, batch: TokenDataResult, device: torch.device, dataloader_idx: int
    ) -> TokenDataResult:
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)
        return batch

    def prepare_data(self) -> None:
        with open(self.data_fp, "r") as f:
            text = f.read()

        self.tokenizer.setup(text)

        n = len(text)
        train_data = text[: int(n * 0.8)]
        val_data = text[int(n * 0.8) : int(n * 0.9)]
        test_data = text[int(n * 0.9) :]

        # Train our own tokenizer:
        # vocab_size = 100
        # LlamaTokenizer.train(input_file_path, ".", vocab_size=vocab_size)
        # tokenizer = LlamaTokenizer("tokenizer.model")

        # Or load a pre-trained tokenizer (e.g., LLaMA's)
        TokenDataset.prepare(train_data, self.tokenizer, "train.bin")
        TokenDataset.prepare(val_data, self.tokenizer, "valid.bin")
        TokenDataset.prepare(test_data, self.tokenizer, "test.bin")

    def state_dict(
        self, fp: str = "datamodule.ckpt", save: bool = True
    ) -> Dict[str, Any]:
        state = {"tokenizer": self.tokenizer}

        if save:
            torch.save(state, fp)

        logger.info(f"Saving datamodule checkpoint to {fp}")
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.tokenizer = state_dict["tokenizer"]

    def load_from_checkpoint(self, fp: str = "datamodule.ckpt") -> Self:
        state_dict = torch.load(fp)
        self.load_state_dict(state_dict)
        return self
