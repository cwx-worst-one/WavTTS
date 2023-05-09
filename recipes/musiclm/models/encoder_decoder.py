from typing import List

import torch
import torch.nn as nn
import transformers
from transformers import T5Config, T5Model

from recipes.musiclm.models.compat.t5_flash_attention import T5FlashAttention


class CompatT5Config(T5Config):
    def __init__(self, max_position_embeddings: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_embd = self.d_model
        self.n_layer = self.num_layers
        self.n_head = self.num_heads
        self.n_inner = self.d_ff
        self.max_position_embeddings = max_position_embeddings


class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_inner: int,
        n_layer: int,
        codebook_size: int,
        n_codebooks: List[int],
        max_sequence_length: int,
        dropout: float = 0.0,
        attention_kwargs: dict = {},
    ):
        """Acoustic token model as described by the AudioLM paper.
        Given representations from a set of hierarchically modeled
        quantizers, this class models a hierarchical sequence-to-sequence
        task for next-token prediction of coarse- and fine audio tokens.

        Args:
            n_embd(int): _description_
            n_head (int): _description_
            n_inner (int): _description_
            n_layer (int): _description_
            codebook_size (int): _description_
            n_codebooks (int): _description_
            max_sequence_length (int): _description_
            dropout (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        self.max_sequence_length = max_sequence_length
        self.vocab_size = self.n_codebooks * codebook_size
        self.max_flattened_seq_len = self.n_codebooks * self.max_sequence_length

        if attention_kwargs.get("use_flash", False):
            # Monkey patch T5Attention, adding Flash Attention
            transformers.models.t5.modeling_t5.T5Attention = T5FlashAttention

        config = CompatT5Config(
            vocab_size=self.vocab_size + 1,  # + 1 for the start_token_id
            decoder_start_token_id=self.vocab_size,
            num_layers=n_layer,
            num_heads=n_head,
            d_model=n_embd,
            d_ff=n_inner,
            dropout_rate=dropout,
            feed_forward_proj="gated-gelu",  # T5.1.1
            max_position_embeddings=self.max_flattened_seq_len,
        )
        self.transformer = T5Model(config)
        self.lm_head = nn.Linear(n_embd, self.codebook_size, bias=False)

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shifts inputs right by prepending a start token (required when doing
        conditional encoder-decoder generation)
        Args:
            input_ids (torch.Tensor): Token id's that need to be shifted right.

        Returns:
            torch.Tensor: Token id's with prepended start token id.
        """
        start_token = (
            torch.ones(
                input_ids.shape[0], 1, dtype=input_ids.dtype, device=input_ids.device
            )
            * self.transformer.config.decoder_start_token_id
        )
        return torch.cat((start_token, input_ids), dim=1)

    def forward(
        self, encoder_input_ids: torch.Tensor, decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        decoder_input_ids = self._shift_right(decoder_input_ids)
        x = self.transformer(
            input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids
        ).last_hidden_state
        return self.lm_head(x)

    def sample(
        self, encoder_input_ids: torch.Tensor, decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            preds = self.forward(encoder_input_ids, decoder_input_ids)
        return preds[:, -1]
