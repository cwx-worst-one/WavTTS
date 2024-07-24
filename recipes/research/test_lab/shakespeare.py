from typing import Union

import requests

from recipes.research.test_lab.token_dataset import TokenDataModule
from samantha.transforms.tokenizers.character import CharacterTokenizer
from samantha.transforms.tokenizers.sentencepiece import SentencePieceTokenizer
from samantha.utils.logger import RankedLogger

logger = RankedLogger(rank_zero_only=True)


class ShakespeareDataModule(TokenDataModule):
    _url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    def __init__(
        self,
        data_fp: str,
        tokenizer: Union[SentencePieceTokenizer, CharacterTokenizer],
        max_seq_len: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        with requests.Session() as s:
            logger.info(f"Downloading Shakespeare data from {self._url} to {data_fp}")
            response = s.get(self._url)
            with open(data_fp, "wb") as f:
                f.write(response.content)
            logger.info(f"Written data to {data_fp}")

        super().__init__(
            data_fp, tokenizer, max_seq_len, batch_size, num_workers, pin_memory
        )
