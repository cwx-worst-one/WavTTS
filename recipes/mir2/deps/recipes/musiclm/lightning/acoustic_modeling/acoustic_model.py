from typing import List, Optional

import torch
from einops import rearrange
from torch.cuda.amp import autocast
from tqdm import tqdm

from recipes.musiclm.lightning.audio_model import SoundStreamModel
from recipes.musiclm.lightning.base import (
    LitModuleBase,
    flatten_row_major,
    generate_offsets,
)


class AcousticModel(LitModuleBase):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_layer: int,
        codebook_size: int,
        n_codebooks: int,
        max_sequence_length: int,
        optimizer_class,
        scheduler_class,
        sample_rate: int,
        quantizers: List[int],
        cond_quantizers: List[int] = [],
        attention_kwargs: dict = {},
    ):
        super().__init__(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            codebook_size=codebook_size,
            n_codebooks=n_codebooks,
            max_sequence_length=max_sequence_length,
            optimizer_class=optimizer_class,
            scheduler_class=scheduler_class,
            attention_kwargs=attention_kwargs,
        )
        self.audio_model = SoundStreamModel(sample_rate).eval()
        self.audio_model.freeze()

        self.input_quantizers = sorted(
            set(self.hparams.quantizers) - set(self.hparams.cond_quantizers)
        )
        self.acoustic_start_channel = self.input_quantizers[0]
        self.acoustic_end_channel = self.input_quantizers[-1] + 1

    def preprocess_audio_tokens(
        self, token_ids: torch.Tensor, start_channel: int, end_channel: int
    ) -> torch.Tensor:
        """We need to preprocess our (b, q, s) tensors into uniquely offset,
        row-major flattened tensors. We also need to embed the token id's to our
        acoustic audio token space. We first apply offsets to make token id's that
        are within the SoundStream token space $R^(codebook_size)$ unique per quantizer.
        After that, these offsets are projected to the acoustic audio token
        space $R^(codebook_size*n_channels)$

        Args:
            token_ids (torch.Tensor): _description_
            quantizers (List[int]): _description_

        Returns:
            torch.Tensor: _description_
        """
        token_ids = token_ids[:, list(range(start_channel, end_channel))]
        offset_token_ids = self.offset(
            token_ids, start_channel, end_channel, self.audio_model.codebook_size
        )
        return flatten_row_major(offset_token_ids)

    def prepare_input_token_ids(self, audio: torch.Tensor) -> torch.Tensor:
        self.audio_model = self.audio_model.eval()
        with torch.no_grad():
            target_acoustic_tokens = self.audio_model(audio)

        input_token_ids = self.preprocess_audio_tokens(
            target_acoustic_tokens,
            start_channel=self.acoustic_start_channel,
            end_channel=self.acoustic_end_channel,
        )
        return input_token_ids

    def prepare_cond_token_ids(
        self, audio: torch.Tensor = None, acoustic_token_ids=None
    ) -> Optional[torch.Tensor]:
        cond_token_ids = None
        self.audio_model = self.audio_model.eval()

        if audio is not None:
            with torch.no_grad():
                acoustic_token_ids = self.audio_model(audio)

        cond_token_ids = self.preprocess_audio_tokens(
            acoustic_token_ids,
            start_channel=self.hparams.cond_quantizers[0],
            end_channel=self.hparams.cond_quantizers[-1] + 1,
        )
        return cond_token_ids

    def step(self, batch, return_loss: bool = True):
        """Forward pass of the model, given a batch of audio.

        Args:
            batch (_type_): Audio data
            return_loss (bool, optional): Whether to return the loss value or the
                predicted logits.

        Returns:
            _type_: Loss value or the predicted logits
        """
        audio = batch[0]

        input_token_ids = self.prepare_input_token_ids(audio)
        cond_token_ids = self.prepare_cond_token_ids(audio)

        prefixed_inputs = torch.cat((cond_token_ids, input_token_ids), dim=1)
        pred_tokens = self.forward(prefixed_inputs)

        cond_seq_len = cond_token_ids.shape[1]
        pred_tokens = pred_tokens[:, cond_seq_len:]
        if return_loss:
            return self.loss(pred_tokens, targets=input_token_ids)
        else:
            return pred_tokens

    @autocast()
    def sample_audio_tokens(
        self,
        cond_token_ids: torch.Tensor,
        temperature: float = 1.0,
        seq_len: Optional[int] = None,
        prefix: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Sample token id's in R^codebook_size
        Predicted tokens are back in the SoundStream token space R^(codebook_size).
        Args:
            cond_token_ids (Optional[torch.Tensor], optional): Defaults to None.
            temperature (float, optional): _description_. Defaults to 1.0.
            seq_len (int, optional): how many samples to generate.
                Defaults to None (use model's max sequence length).
            prefix (torch.LongTensor): prefix of previous tokens, shape (b, q, s).
                Defaults to None (no prefix).
        Returns:
            [torch.Tensor]: _description_
        """
        if seq_len is None:
            seq_len = self.hparams.max_sequence_length

        sampled_token_ids = cond_token_ids.clone()
        cond_seq_len = cond_token_ids.shape[1]
        q = len(self.input_quantizers)

        prefix_len = 0
        if prefix is not None:
            prefix_len = prefix.shape[2]
            # (b, q, s) --> (b, s * q)
            prefix = flatten_row_major(
                self.offset(
                    prefix,
                    self.acoustic_start_channel,
                    self.acoustic_end_channel,
                    self.audio_model.codebook_size,
                )
            )
            sampled_token_ids = torch.cat((sampled_token_ids, prefix), dim=1)

        prog_bar = tqdm(
            desc="Sampling acoustic tokens", total=(seq_len - prefix_len) * q
        )

        self.model.transformer.init_cache()
        curr_input = sampled_token_ids
        for _ in range(seq_len - prefix_len):
            for quantizer_idx in self.input_quantizers:
                sampled = self.sample_within_bounds(
                    curr_input,
                    codebook_size=self.audio_model.codebook_size,
                    start_channel=quantizer_idx,
                    end_channel=quantizer_idx + 1,
                    temperature=temperature,
                )
                sampled_token_ids = torch.cat((sampled_token_ids, sampled), dim=1)
                curr_input = sampled
                prog_bar.update()

        self.model.transformer.deinit_cache()
        sampled_token_ids = sampled_token_ids[:, cond_seq_len + prefix_len * q :]
        sampled_token_ids = rearrange(sampled_token_ids, "b (s q) -> b q s", q=q)

        offsets = generate_offsets(
            sampled_token_ids.shape[2],
            self.audio_model.codebook_size,
            start_channel=self.acoustic_start_channel,
            end_channel=self.acoustic_end_channel,
            device=sampled_token_ids.device,
        )
        return sampled_token_ids - offsets

    def sample_with_audio_conditioning(
        self,
        audio: Optional[torch.Tensor] = None,
        acoustic_token_ids: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        seq_len: Optional[int] = None,
        prefix: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        cond_token_ids = self.prepare_cond_token_ids(
            audio=audio, acoustic_token_ids=acoustic_token_ids
        )
        return self.sample_audio_tokens(
            cond_token_ids=cond_token_ids,
            temperature=temperature,
            seq_len=seq_len,
            prefix=prefix,
        )
