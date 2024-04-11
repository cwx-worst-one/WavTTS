from typing import Optional, List

import torch
import pytorch_lightning as pl
from pydantic import BaseModel

from samantha.utils.ctiga.inference_params import InferenceParams
from recipes.musiclm.inference.utils import sample
from recipes.bigmusic.inference.base_observer import TokenGeneratorObserver


class GeneratorState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    allowed_indexes: Optional[List[int]] = None

    temperature: float = 1.0
    # top_k: Optional[int] = None
    # do_sample: bool = True
    top_p: float = 0.95


class TokenGenerator:
    def __init__(
        self,
        pl_module: pl.LightningModule,
        initial_state: GeneratorState,
    ):
        self.pl_module = pl_module
        if torch.cuda.is_available():
            self.pl_module.cuda(0)
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.pl_module.model.eval()

        self.state: GeneratorState = initial_state
        self.observers: List[TokenGeneratorObserver] = []
    
    def register_observer(self, observer: TokenGeneratorObserver):
        self.observers.append(observer)
    
    def remove_all_observers(self):
        self.observers = []

    def notify_on_reset(self):
        for ob in self.observers:
            ob.on_reset()

    def notify_on_generate_finished(self, logits: torch.Tensor):
        for ob in self.observers:
            ob.on_generate_finished(logits)
    
    def notify_on_sample_finished(self, next_index: int):
        for ob in self.observers:
            ob.on_sample_finished(next_index)

    def reset(self) -> None:
        self.notify_on_reset()

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
    ) -> torch.Tensor:
        logits = self.pl_module.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            inference_params=inference_params,
            last_token_only=True,
        ).logits

        ## This is required!!!!!!
        if input_ids is not None:
            inference_params.sequence_len_offset += input_ids.size(1)
        else:
            inference_params.sequence_len_offset += inputs_embeds.size(1)

        self.notify_on_generate_finished(logits)
        return logits

    def sample(self, logits: torch.Tensor) -> Optional[int]:
        # logits = self.state.logits
        if logits is None:
            return
        if self.state.allowed_indexes is None:
            allowed_indexes = list(range(logits.shape[1]))
        else:
            allowed_indexes = self.state.allowed_indexes
            logits = logits[0, self.state.allowed_indexes]
        idx_next = sample(logits, self.state.temperature, self.state.top_p, "top_p")
        # from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
        # logits = logits / self.state.temperature
        # if self.state.top_k is not None:
        #     v, i = torch.topk(logits, self.state.top_k)
        #     logits[logits < v[[-1]]] = -float("Inf")
        # probs = logits.softmax(dim=-1)
        # if self.state.do_sample:
        #     if self.state.top_p is not None:
        #         idx_next = sample_top_p(probs, self.state.top_p)
        #     else:
        #         idx_next = torch.multinomial(probs, num_samples=1)
        # else:
        #     _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx_gen = allowed_indexes[idx_next]
        self.notify_on_sample_finished(idx_gen)
        return idx_gen