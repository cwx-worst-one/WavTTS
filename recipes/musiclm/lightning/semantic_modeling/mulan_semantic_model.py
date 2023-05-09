from typing import Optional

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from recipes.musiclm.lightning.base import LitModuleBase, generate_offsets
from recipes.musiclm.lightning.mulan_model import QuantizedMulanModel
from recipes.musiclm.lightning.semantic_model import SemanticModel


class MulanSemanticModel(LitModuleBase):
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
        self.semantic_model = SemanticModel().eval()
        self.semantic_model.freeze()
        self.mulan_model = QuantizedMulanModel().eval()
        self.mulan_model.freeze()

        self.semantic_start_channel = 0
        self.semantic_end_channel = 1
        self.mulan_start_channel = self.semantic_end_channel
        self.mulan_end_channel = self.mulan_start_channel + 1

    def on_train_epoch_start(self):
        self.mulan_model.post_init(device=self.device)

    def prepare_input_token_ids(self, audio: torch.Tensor) -> torch.Tensor:
        self.semantic_model = self.semantic_model.eval()
        with torch.no_grad():
            target_semantic_tokens = self.semantic_model(audio)

        return self.offset(
            target_semantic_tokens,
            start_channel=self.semantic_start_channel,
            end_channel=self.semantic_end_channel,
            codebook_size=self.semantic_model.codebook_size,
        )

    def prepare_cond_token_ids(
        self, cond: torch.Tensor, data_type: str
    ) -> torch.Tensor:
        self.mulan_model = self.mulan_model.eval()

        if data_type in ["text", "music"]:
            with torch.no_grad():
                cond = self.mulan_model(cond, data_type=data_type)

        # TODO: Start using embedding layers, might be easier :)
        return self.offset(
            cond,
            start_channel=self.mulan_start_channel,
            end_channel=self.mulan_end_channel,
            codebook_size=self.mulan_model.codebook_size,
        )

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
        cond_token_ids = self.prepare_cond_token_ids(audio, data_type="music")

        prefixed_inputs = torch.cat((cond_token_ids, input_token_ids), dim=1)
        pred_tokens = self.forward(prefixed_inputs)

        cond_seq_len = cond_token_ids.shape[1]
        pred_tokens = pred_tokens[:, cond_seq_len:]
        if return_loss:
            return self.loss(pred_tokens, targets=input_token_ids)
        else:
            return pred_tokens

    @autocast()
    def sample_semantic_tokens(
        self,
        cond_token_ids: torch.Tensor,
        temperature: float = 1.0,
        seq_len: Optional[int] = None,
        prefix: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Sample token id's in R^codebook_size
        Predicted tokens are back in the Semantic token space R^(codebook_size).
        Args:
            cond_token_ids (torch.Tensor): _description_.
                Defaults to None.
            temperature (float, optional): _description_.
                Defaults to 1.0.
            seq_len (int, optional): how many samples to generate.
                Defaults to None (use model's max sequence length).
            prefix (torch.LongTensor): prefix of previous tokens, shape (b, s).
                Defaults to None (no prefix).
        Returns:
            [torch.Tensor]: _description_
        """
        if seq_len is None:
            seq_len = self.hparams.max_sequence_length

        sampled_token_ids = cond_token_ids.clone()
        cond_seq_len = cond_token_ids.shape[1]

        prefix_len = 0
        if prefix is not None:
            prefix_len = prefix.shape[1]
            prefix = self.offset(
                prefix,
                self.semantic_start_channel,
                self.semantic_end_channel,
                self.semantic_model.codebook_size,
            )
            sampled_token_ids = torch.cat((sampled_token_ids, prefix), dim=1)

        self.model.transformer.init_cache()
        curr_input = sampled_token_ids
        for _ in tqdm(range(seq_len - prefix_len), desc="Sampling semantic tokens..."):
            sampled = self.sample_within_bounds(
                curr_input,
                codebook_size=self.semantic_model.codebook_size,
                start_channel=self.semantic_start_channel,
                end_channel=self.semantic_end_channel,
                temperature=temperature,
            )

            sampled_token_ids = torch.cat((sampled_token_ids, sampled), dim=1)
            curr_input = sampled

        self.model.transformer.deinit_cache()
        sampled_token_ids = sampled_token_ids[:, cond_seq_len + prefix_len :]

        offsets = generate_offsets(
            sampled_token_ids.shape[1],
            self.semantic_model.codebook_size,
            start_channel=self.semantic_start_channel,
            end_channel=self.semantic_end_channel,
            device=sampled_token_ids.device,
        )
        return sampled_token_ids - offsets

    def sample_with_conditioning(
        self,
        cond_data: torch.Tensor,
        temperature: float,
        data_type: str = "music",
        seq_len: Optional[int] = None,
        prefix: Optional[torch.LongTensor] = None,
    ):
        cond_data = cond_data.to(self.device)
        cond_token_ids = self.prepare_cond_token_ids(cond_data, data_type=data_type)
        return self.sample_semantic_tokens(
            cond_token_ids=cond_token_ids,
            temperature=temperature,
            seq_len=seq_len,
            prefix=prefix,
        )
