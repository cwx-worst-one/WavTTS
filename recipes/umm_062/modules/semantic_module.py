from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from tqdm import tqdm
from transformers import AutoTokenizer, T5Config, T5EncoderModel, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack

from recipes.soundstorm2.lightning.bestrq import BestRQMelCTCModel


@dataclass
class TextSemanticConfig:
    sample_rate: int
    t5_model: str


class T5TextEncoder(LightningModule):
    def __init__(self, config: TextSemanticConfig):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.t5_model)
        self.model = T5EncoderModel.from_pretrained(config.t5_model)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def forward(self, text: List[str]):
        model_inputs = self.tokenizer(text, padding="longest", return_tensors="pt").to(
            self.device
        )
        return self.model(**model_inputs)


class T5SemanticAudioDecoderWithLMHead(T5PreTrainedModel):
    def __init__(self, config: T5Config, decoder_vocab_size: int):
        super().__init__(config)
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        decoder_config.pad_token_id = decoder_vocab_size
        decoder_config.decoder_start_token_id = decoder_config.pad_token_id
        decoder_config.eos_token_id = decoder_vocab_size + 1

        self.vocab_size = decoder_vocab_size + 2  # SOS/EOS
        self.semantic_embedding = nn.Embedding(self.vocab_size, config.d_model)
        self.decoder = T5Stack(decoder_config, self.semantic_embedding)
        self.lm_head = nn.Linear(config.d_model, self.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @property
    def pad_token_id(self):
        return self.decoder.config.pad_token_id

    @property
    def decoder_start_token_id(self):
        return self.decoder.config.decoder_start_token_id

    @property
    def eos_token_id(self):
        return self.decoder.config.eos_token_id

    def forward(
        self,
        encoder_hidden_states,
        semantic_audio_tokens: torch.Tensor,
        use_cache: Optional[bool] = None,
    ):
        decoder_input_ids = self.decoder._shift_right(semantic_audio_tokens)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=use_cache,
        )
        sequence_output = decoder_outputs[0]
        return self.lm_head(sequence_output)

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        max_seq_len: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size = encoder_hidden_states.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1), self.decoder_start_token_id, device=self.device
        )
        for _ in tqdm(range(max_seq_len), desc="Sampling semantic tokens..."):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden_states
            )
            sequence_output = decoder_outputs[0]
            logits = self.lm_head(sequence_output)

            last_logit = logits[:, -1]
            sampled_token_id = torch.multinomial(
                last_logit.softmax(dim=-1) / temperature, num_samples=1
            )
            decoder_input_ids = torch.cat((decoder_input_ids, sampled_token_id), dim=1)

            if sampled_token_id == self.eos_token_id:
                break

        # remove padding and eos tokens
        decoder_input_ids = decoder_input_ids[:, 1:-1]
        return decoder_input_ids


class TextSemanticStage(LightningModule):
    def __init__(self, config: TextSemanticConfig, optimizer_cls, scheduler_cls):
        super().__init__()
        self.save_hyperparameters()
        self.text_encoder = T5TextEncoder(config)
        self.text_encoder.freeze()
        self.semantic_audio_model = BestRQMelCTCModel().eval()
        self.semantic_audio_model.freeze()

        t5_config = T5Config.from_pretrained(config.t5_model)
        self.model = T5SemanticAudioDecoderWithLMHead(
            t5_config, decoder_vocab_size=self.semantic_audio_model.codebook_size
        )

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2), sync_dist=True)

    @torch.no_grad()
    def generate(self, text: List[str], max_seq_len: int, temperature: float = 1.0):
        encoder_outputs = self.text_encoder(text)
        encoder_hidden_states = encoder_outputs[0]
        return self.model.generate(
            encoder_hidden_states, max_seq_len=max_seq_len, temperature=temperature
        )

    def prepare_inputs(self, batch) -> Dict[str, torch.Tensor]:
        self.text_encoder.eval()
        self.semantic_audio_model.eval()
        with torch.no_grad():
            encoder_outputs = self.text_encoder(batch["normalized_text"])
            semantic_audio_tokens = self.semantic_audio_model(batch["audio"])
            batch.update(
                encoder_outputs=encoder_outputs,
                semantic_audio_tokens=semantic_audio_tokens,
            )
        return batch

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoder_hidden_states = inputs["encoder_outputs"][0]
        semantic_audio_tokens = inputs["semantic_audio_tokens"]
        return self.model(encoder_hidden_states, semantic_audio_tokens)

    def step(
        self, batch: Dict[str, torch.Tensor], return_loss: bool = True
    ) -> torch.Tensor:
        inputs = self.prepare_inputs(batch)
        logits = self.forward(inputs)
        if return_loss:
            return self.loss(logits, inputs["semantic_audio_tokens"])
        return logits

    def loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        eos_token = torch.full(
            (targets.shape[0], 1), self.model.eos_token_id, device=self.device
        )
        targets = torch.cat((targets, eos_token), dim=1)

        targets = targets[:, 1:]

        accuracy = (preds.argmax(2) == targets).float().mean()
        loss = F.cross_entropy(
            rearrange(preds, "b n c -> b c n"),
            targets,
            ignore_index=self.model.pad_token_id,
        )
        return {"loss": loss, "ppl": loss.exp(), "accuracy": accuracy}

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        for k, v in loss.items():
            self.log(f"{k}/train", v, rank_zero_only=True, prog_bar=True)
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        for k, v in loss.items():
            self.log(f"{k}/valid", v, rank_zero_only=True)
        return loss["loss"]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, return_loss=False)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = {
            "scheduler": self.hparams.scheduler_cls(optimizer),
            "interval": "step",
            "name": "learning_rate",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
