from typing import List, Optional

import torch

from recipes.musiclm.lightning.acoustic_modeling.acoustic_model import AcousticModel
from recipes.musiclm.lightning.mulan_model import QuantizedMulanModel
from recipes.musiclm.lightning.semantic_model import SemanticModel


class SemanticAcousticModel(AcousticModel):
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
            sample_rate=sample_rate,
            quantizers=quantizers,
            cond_quantizers=cond_quantizers,
            attention_kwargs=attention_kwargs,
        )

        self.semantic_model = SemanticModel().eval()
        self.semantic_model.freeze()
        self.mulan_model = QuantizedMulanModel().eval()
        self.mulan_model.freeze()

        self.mulan_start_channel = self.acoustic_end_channel
        self.mulan_end_channel = self.mulan_start_channel + 1
        self.semantic_start_channel = self.mulan_end_channel
        self.semantic_end_channel = self.semantic_start_channel + 1

    def on_train_epoch_start(self):
        self.mulan_model.post_init(device=self.device)

    def prepare_cond_token_ids(
        self,
        audio: Optional[torch.Tensor] = None,
        mulan_token_ids: Optional[torch.Tensor] = None,
        semantic_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.mulan_model = self.mulan_model.eval()
        self.semantic_model = self.semantic_model.eval()

        if audio is not None:
            with torch.no_grad():
                mulan_token_ids = self.mulan_model(audio, data_type="music")
                semantic_token_ids = self.semantic_model(audio)

        mulan_token_ids = self.offset(
            mulan_token_ids,
            start_channel=self.mulan_start_channel,
            end_channel=self.mulan_end_channel,
            codebook_size=self.mulan_model.codebook_size,
        )

        semantic_token_ids = self.offset(
            semantic_token_ids,
            start_channel=self.semantic_start_channel,
            end_channel=self.semantic_end_channel,
            codebook_size=self.semantic_model.codebook_size,
        )
        return torch.cat((mulan_token_ids, semantic_token_ids), dim=1)

    def sample_with_semantic_conditioning(
        self,
        mulan_token_ids: torch.Tensor,
        semantic_token_ids: torch.Tensor,
        temperature: float,
        seq_len: Optional[int] = None,
        prefix: Optional[torch.LongTensor] = None,
    ):
        mulan_token_ids = mulan_token_ids.to(self.device)
        semantic_token_ids = semantic_token_ids.to(self.device)
        cond_token_ids = self.prepare_cond_token_ids(
            mulan_token_ids=mulan_token_ids, semantic_token_ids=semantic_token_ids
        )
        return self.sample_audio_tokens(
            cond_token_ids=cond_token_ids,
            temperature=temperature,
            seq_len=seq_len,
            prefix=prefix,
        )

    def sample_with_audio_conditioning(
        self,
        audio,
        temperature: float,
        seq_len: Optional[int] = None,
        prefix: Optional[torch.LongTensor] = None,
    ):
        audio = audio.to(self.device)
        cond_token_ids = self.prepare_cond_token_ids(audio=audio)
        return self.sample_audio_tokens(
            cond_token_ids=cond_token_ids,
            temperature=temperature,
            seq_len=seq_len,
            prefix=prefix,
        )
