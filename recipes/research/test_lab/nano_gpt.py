import math
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from byteformers.models import GPT2, GPT2Config, Llama, LlamaConfig
from recipes.research.test_lab.token_dataset import TokenDataResult
from samantha.models.base import DefaultTrainingBaseModule

LossDict = Dict[str, torch.Tensor]


@dataclass
class NanoGPTConfig:
    model_type: str
    n_embd: int
    n_head: int
    n_layer: int
    vocab_size: int
    max_seq_len: int
    learning_rate: float
    dropout: float
    warmup_steps: int
    weight_decay: float
    betas: Tuple[float, float]
    use_rotary_embeddings: bool
    attention_kwargs: Optional[dict] = field(default_factory=dict)


@dataclass
class NanoGPTResult:
    x: torch.Tensor
    loss: Optional[torch.Tensor] = None


class NanoGPT(DefaultTrainingBaseModule):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.max_seq_len = config.max_seq_len

        if config.model_type == "gpt":
            self.model_config = GPT2Config(
                vocab_size=self.config.vocab_size,
                max_seq_len=self.config.max_seq_len,
                n_layer=self.config.n_layer,
                n_head=self.config.n_head,
                n_embd=self.config.n_embd,
                n_inner=self.config.n_embd * 4,
                resid_pdrop=self.config.dropout,
                embd_pdrop=self.config.dropout,
                attn_pdrop=self.config.dropout,
                mlp_bias=False,
                layer_norm_bias=False,
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                use_rotary_embeddings=self.config.use_rotary_embeddings,
                attention_kwargs=self.config.attention_kwargs,
            )
            self.model = GPT2(self.model_config)
        elif config.model_type == "llama":
            self.model_config = LlamaConfig(
                vocab_size=self.config.vocab_size,
                n_layer=self.config.n_layer,
                n_head=self.config.n_head,
                n_embd=self.config.n_embd,
                is_causal=True,
                use_rotary_embeddings=self.config.use_rotary_embeddings,
            )
            self.model = Llama(self.model_config)
        else:
            raise NotImplementedError("Choose between model_type=[gpt, llama]")

        self.model.apply(self.model._init_weights)

    @property
    def extra_hparams(self):
        hparams = super().extra_hparams
        hparams.update({"global_token_size": self.global_token_size})
        return hparams

    @property
    def global_token_size(self) -> int:
        return self.global_batch_size * self.max_seq_len

    @property
    def _tokens_seen(self) -> int:
        try:
            return self.global_token_size * self.global_step
        except Exception as e:
            return 0

    def forward(self, src: torch.Tensor) -> NanoGPTResult:
        x = self.model(src)
        return NanoGPTResult(x=x)

    def step(
        self, batch: TokenDataResult, batch_idx: int, return_loss: bool
    ) -> NanoGPTResult:
        result = self.forward(batch.x)
        if return_loss:
            result.loss = self.loss(result.x, batch.y)
        return result

    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor) -> LossDict:
        logits = rearrange(logits, "b s c -> (b s) c")
        targets = rearrange(targets, "b s -> (b s)")
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        return {"loss": loss, "nll": loss, "perplexity": loss.exp()}

    def configure_optimizers(self) -> Any:
        # - Start with a warm up, ramp up then cosine
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            fused=True,
        )

        def update_lr(*_):
            config = self.config
            warmup_tokens = self.global_token_size * config.warmup_steps
            final_tokens = self.global_token_size * self.trainer.max_steps
            if self._tokens_seen < warmup_tokens:
                # linear warmup
                lr_mult = float(self._tokens_seen) / float(max(1, warmup_tokens))
                lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
            else:
                # cosine learning rate decay
                progress = float(self._tokens_seen - warmup_tokens) / float(
                    max(1, final_tokens - warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return lr_mult

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=update_lr
            ),
            "name": "learning_rate",
            "interval": "step",  # The unit of the scheduler's step size
            "frequency": 1,  # The frequency of the scheduler
        }
        return [optimizer], [lr_scheduler]

    @torch.inference_mode()
    def sampler(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.95,
    ) -> Generator:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time. Most likely you'll want to make sure to
        be in model.eval() mode of operation for this.
        """
        curr_inputs = (
            tokens
            if tokens.shape[1] <= self.config.max_seq_len
            else tokens[:, : self.config.max_seq_len]
        )
        sampled_tokens_ids = torch.empty(
            (tokens.shape[0], 0), device=tokens.device, dtype=torch.long
        )
        sampled_tokens_ids = torch.cat((sampled_tokens_ids, curr_inputs), dim=1)
        kv_cache = {}
        for _ in tqdm(range(curr_inputs.shape[1], max_new_tokens), desc="Sampling"):

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                # logits = self.model(curr_inputs, kv_cache=kv_cache, last_logit_only=True)
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
            yield curr_inputs

    @torch.inference_mode()
    def generate(self, tokenizer):
        max_new_tokens = self.config.max_seq_len
        tokens = tokenizer("\n", device=self.device)

        sampler = self.sampler(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True,
            top_k=200,
            top_p=None,
        )
        sampled_tokens = []
        for token in tqdm(sampler, total=max_new_tokens):
            sampled_tokens.append(token.cpu())

        return tokenizer.decode(sampled_tokens)
