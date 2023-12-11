import logging
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer  # type: ignore

from recipes.mi1.models.tokenizers import WhiteSpaceTokenizer, NoopTokenizer
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__)

ConditionType = Tuple[torch.Tensor, torch.Tensor]  # condition, mask

class ConditionerResult(NamedTuple):
    embeds: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor

class BaseConditioner(nn.Module):
    """Base model for all conditioner modules.
    We allow the output dim to be different than the hidden dim for two reasons:
    1) keep our LUTs small when the vocab is large;
    2) make all condition dims consistent.

    Args:
        dim (int): Hidden dim of the model.
        n_embd (int): Output dim of the conditioner.
    """

    def __init__(self, dim: int, n_embd: int):
        super().__init__()
        self.dim = dim
        self.n_embd = n_embd
        self.output_proj = nn.Linear(dim, n_embd)

    def tokenize(self, *args, **kwargs) -> Any:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def forward(self, inputs: Any) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, genre, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, D] where B is the batch size, T is the length of the
                  output embedding and D is the dimension of the embedding.
                - And a mask indicating where the padding tokens.
        """
        raise NotImplementedError()

class LUTConditioner(BaseConditioner):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        cond_n_embd (int): Hidden dim of the model (text-encoder/LUT).
        n_embd (int): Output dim of the conditioner.
        tokenizer (str): Name of the tokenizer.
        pad_idx (int, optional): Index for padding token. Defaults to 0.
    """
    def __init__(self, n_bins: int, cond_n_embd: int, n_embd: int, pad_token_id: int = 0):
        super().__init__(cond_n_embd, n_embd)
        self._pad_token_id = pad_token_id
        self.embed = nn.Embedding(n_bins, cond_n_embd, padding_idx=pad_token_id)
        self.tokenizer = NoopTokenizer(n_bins, pad_idx=self.pad_token_id)

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    def tokenize(self, x: List[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.embed.weight.device
        tokens, attention_mask = self.tokenizer(x)
        tokens, attention_mask = tokens.to(device), attention_mask.to(device)
        return tokens, attention_mask

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> ConditionType:
        embeds = self.embed(token_ids)
        embeds = self.output_proj(embeds)
        embeds = (embeds * attention_mask.unsqueeze(-1))
        return ConditionerResult(
            embeds=embeds,
            token_ids=token_ids,
            attention_mask=attention_mask
        )

class ArtistConditioner(LUTConditioner):
    ...

        

class T5ConditionerResult(NamedTuple):
    embeds: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor


class T5Conditioner(BaseConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        n_embd (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """

    MODELS = [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(
        self,
        name: str,
        n_embd: int,
        finetune: bool,
        device: str,
        autocast_dtype: Optional[str] = "float32",
        word_dropout: float = 0.0,
        normalize_text: bool = False,
    ):
        assert (
            name in self.MODELS
        ), f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__(self.MODELS_DIMS[name], n_embd)
        self.device = device
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout
        if autocast_dtype is None or self.device == "cpu":
            self.autocast = torch.autocast(device_type=self.device, enabled=False)
            if self.device != "cpu":
                logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = torch.autocast(
                device_type=self.device, dtype=dtype, enabled=True
            )
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
                t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)
        if finetune:
            self.t5 = t5
        else:
            # this makes sure that the t5 models is not part
            # of the saved checkpoint
            self.__dict__["t5"] = t5.to(device)

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def tokenize(self, x: List[Optional[str]]) -> Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)
        if self.word_dropout > 0.0 and self.training:
            new_entries = []
            for entry in entries:
                words = [
                    word
                    for word in entry.split(" ")
                    if random.random() >= self.word_dropout
                ]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors="pt", padding=True, add_special_tokens=True).to(
            self.device
        )
        mask = inputs["attention_mask"]
        mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
        return inputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        embeds = embeds * mask.unsqueeze(-1)
        return T5ConditionerResult(
            embeds=embeds,
            attention_mask=mask,
            input_ids=inputs["input_ids"],
        )
