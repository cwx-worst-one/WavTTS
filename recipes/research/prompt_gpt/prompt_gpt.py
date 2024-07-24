from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer

from byteformers.models import Llama, LlamaConfig
from recipes.research.diff.transforms.text.normalizers.english import (
    EnglishTextNormalizer,
)
from samantha.data.audio.types import AudioDataResult
from samantha.models.base import DefaultTrainingBaseModule, LossDict
from samantha.optim.lr_scheduler.warmup_cosine_lr import WarmupCosine
from samantha.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class TextNormalizer:
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


class TextTokenizer:

    def __init__(self):
        self.normalizer = TextNormalizer()
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-uncased"
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.sep_token_id

    def bach_decode(
        self, token_ids: torch.Tensor, skip_special_tokens: bool = False
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def __call__(self, text: List[str], device: torch.device):
        return self.tokenizer(
            text, padding=True, return_length=True, return_tensors="pt"
        ).to(device)


@dataclass
class PromptGPTConfig:
    n_embd: int = 1536
    n_layer: int = 12
    n_head: int = 12

    max_seq_len: int = 500

    # Optimizer
    learning_rate: float = 3.0e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    warmup_steps: int = 8000
    cycle_steps: int = 20000
    min_lr: float = learning_rate * 0.1


@dataclass
class PromptGPTResult:
    logits: torch.Tensor
    loss: Optional[LossDict] = None


class PromptGPT(DefaultTrainingBaseModule):
    def __init__(self, config: PromptGPTConfig):
        super().__init__()
        self.config = config

        self.tokenizer = TextTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        model_config = LlamaConfig(
            vocab_size=self.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            use_rotary_embeddings=True,
            is_causal=True,
        )
        self.model = Llama(model_config)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.initialize_weights()

        self.max_seq_len = config.max_seq_len
        self.eos_token_id = self.tokenizer.eos_token_id

    def initialize_weights(self):
        logger.info("Initializing weights...")
        self.model.apply(self.model._init_weights)

    def forward(
        self, text_token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> PromptGPTResult:
        logits = self.model.forward(text_token_ids, attention_mask=attention_mask)
        return PromptGPTResult(logits=logits)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> LossDict:
        batch_size, seq_length, vocab_size = logits.size()

        # Shift logits and target for next-token prediction
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_target = target[:, 1:].contiguous()

        # Flatten the tensors
        shifted_logits = shifted_logits.view(-1, vocab_size)
        shifted_target = shifted_target.view(-1)

        # Compute cross entropy loss
        loss = self.criterion(shifted_logits, shifted_target)

        pred = shifted_logits.argmax(dim=-1)
        pad_mask = shifted_target != self.pad_token_id
        correct = ((pred == shifted_target) & pad_mask).float()
        total = pad_mask.float()
        accuracy = correct.sum() / (total.sum() + 1e-7)

        return {
            "loss": loss,
            "perplexity": loss.exp(),
            "accuracy": accuracy,
            "batch_size": batch_size,
        }

    def step(
        self, batch: AudioDataResult, batch_idx: int, return_loss: bool
    ) -> PromptGPTResult:

        text = [i["description"] for i in batch.index]

        tokenizer_result = self.tokenizer(text, device=self.device)

        input_ids = tokenizer_result["input_ids"]
        attention_mask = tokenizer_result["attention_mask"]
        result = self.forward(input_ids, attention_mask)

        if batch_idx % 100 == 0:
            logger.info(text)
            logger.info(self.tokenizer.bach_decode(input_ids))

        if return_loss:
            result.loss = self.loss(result.logits, input_ids)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )
        scheduler = WarmupCosine(
            optimizer,
            self.config.learning_rate,
            self.config.warmup_steps,
            self.config.cycle_steps,
            self.config.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.inference_mode()
    def sampler(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.95,
    ) -> Iterable:
        curr_inputs = (
            tokens
            if tokens.shape[1] <= self.max_seq_len
            else tokens[:, : self.max_seq_len]
        )
        sampled_tokens_ids = torch.empty(
            (tokens.shape[0], 0), device=tokens.device, dtype=torch.long
        )
        sampled_tokens_ids = torch.cat((sampled_tokens_ids, curr_inputs), dim=1)
        for _ in tqdm(range(curr_inputs.shape[1], max_new_tokens), desc="Sampling"):

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                logits = self.model(sampled_tokens_ids)
                logits = logits[:, -1]

            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = logits.softmax(dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            curr_inputs = idx_next
            sampled_tokens_ids = torch.cat((sampled_tokens_ids, idx_next), dim=1)

            if idx_next == self.eos_token_id:
                break
        return sampled_tokens_ids

    @torch.inference_mode()
    def generate(
        self,
        prompt: List[str],
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.95,
    ):
        assert len(prompt) == 1, "TODO: attention mask :)"
        max_new_tokens = self.max_seq_len

        tokenizer_result = self.tokenizer(prompt, device=self.device)

        input_ids = tokenizer_result["input_ids"]

        # remove [SEP] token from prompt:
        input_ids = input_ids[:, :-1]

        sampled_tokens_ids = self.sampler(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
        )
        return self.tokenizer.bach_decode(sampled_tokens_ids, skip_special_tokens=True)

    # def on_validation_start(self) -> None:
    #     self = self.eval()
    #     prompt = "Summer Pop with piano, guitars and synths in summer mood"
    #     with torch.no_grad():
    #         result = self.generate(prompt)
    #     logger.info(result)

    #     self = self.train()


if __name__ == "__main__":
    config = PromptGPTConfig(n_layer=8, n_embd=512, n_head=8)
    model = PromptGPT(config).to("cuda")
    logger.info(model.summarize())

    batch_size = 2

    index = {"text": "this is a test prompt."}

    batch = AudioDataResult(
        audio=None, shard=None, key=None, segment_info=None, index=[index] * batch_size
    )

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        result = model.step(batch, batch_idx=0, return_loss=True)

    with torch.no_grad():
        pred_text = model.generate("hi")
        logger.info(pred_text)
