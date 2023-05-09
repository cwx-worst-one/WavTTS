from typing import Optional

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from recipes.musiclm.lightning.acoustic_modeling.acoustic_model import AcousticModel
from recipes.musiclm.lightning.mulan_model import QuantizedMulanModel
from recipes.musiclm.lightning.semantic_model import SemanticModel


class UnifiedModel(AcousticModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        input_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
        i_mulan: Optional[torch.Tensor] = None,
        t_mulan: Optional[torch.Tensor] = None,
        i_semantic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.mulan_model = self.mulan_model.eval()
        self.semantic_model = self.semantic_model.eval()

        with torch.no_grad():
            if input_audio is not None:
                i_mulan = self.mulan_model(input_audio, data_type="music")
                i_semantic = self.semantic_model(input_audio)

            if target_audio is not None:
                t_mulan = self.mulan_model(target_audio, data_type="music")

        i_mulan_offset = self.offset(
            i_mulan,
            start_channel=self.mulan_start_channel,
            end_channel=self.mulan_end_channel,
            codebook_size=self.mulan_model.codebook_size,
        )
        t_mulan_offset = self.offset(
            t_mulan,
            start_channel=self.mulan_start_channel,
            end_channel=self.mulan_end_channel,
            codebook_size=self.mulan_model.codebook_size,
        )
        i_semantic_offset = self.offset(
            i_semantic,
            start_channel=self.semantic_start_channel,
            end_channel=self.semantic_end_channel,
            codebook_size=self.semantic_model.codebook_size,
        )
        return torch.cat((i_mulan_offset, t_mulan_offset, i_semantic_offset), dim=1)

    def prepare_input_token_ids(self, audio: torch.Tensor) -> torch.Tensor:
        self.semantic_model = self.semantic_model.eval()
        self.audio_model = self.audio_model.eval()
        with torch.no_grad():
            t_semantic = self.semantic_model(audio)
            t_audio = self.audio_model(audio)

        t_semantic_offset = self.offset(
            t_semantic,
            start_channel=self.semantic_start_channel,
            end_channel=self.semantic_end_channel,
            codebook_size=self.semantic_model.codebook_size,
        )
        t_audio_offset = self.preprocess_audio_tokens(
            t_audio,
            start_channel=self.acoustic_start_channel,
            end_channel=self.acoustic_end_channel,
        )
        return torch.cat((t_semantic_offset, t_audio_offset), dim=1)

    def step(self, batch, return_loss: bool = True):
        # TODO: Randomly dropout the input_audio
        input_audio = batch[0]
        target_audio = batch[1]

        input_token_ids = self.prepare_input_token_ids(target_audio)
        cond_token_ids = self.prepare_cond_token_ids(input_audio, target_audio)

        cond_seq_len = cond_token_ids.shape[1]
        prefixed_inputs = torch.cat((cond_token_ids, input_token_ids), dim=1)
        pred_tokens = self.forward(prefixed_inputs)
        pred_tokens = pred_tokens[:, cond_seq_len:]
        if return_loss:
            return self.loss(pred_tokens, targets=input_token_ids)
        else:
            return pred_tokens

    @autocast()
    def sample_target_semantic(
        self, cond_token_ids: Optional[torch.Tensor] = None, temperature: float = 1.0
    ) -> torch.Tensor:
        """Sample token id's in R^codebook_size (we restrict the sampling to the codebook_size)
        of the semantic token space.

        Args:
            cond_token_ids (Optional[torch.Tensor], optional): _description_.
                Defaults to None.
            temperature (float, optional): _description_.
                Defaults to 1.0.
        Returns:
            [torch.Tensor]: _description_
        """

        sampled_t_semantic = cond_token_ids.clone()
        cond_seq_len = cond_token_ids.shape[1]

        self.model.transformer.init_cache()
        curr_input = sampled_t_semantic
        for _ in tqdm(
            range(0, self.semantic_model.n_frames),
            desc="Sampling target semantic tokens...",
        ):
            sampled = self.sample_within_bounds(
                curr_input,
                codebook_size=self.semantic_model.codebook_size,
                start_channel=self.semantic_start_channel,
                end_channel=self.semantic_end_channel,
                temperature=temperature,
            )
            sampled_t_semantic = torch.cat((sampled_t_semantic, sampled), dim=1)
            curr_input = sampled

        self.model.transformer.deinit_cache()
        sampled_t_semantic = sampled_t_semantic[:, cond_seq_len:]
        return sampled_t_semantic

    def get_mulan_tokens(self, text: str):
        text_token_id = self.mulan_model.text_to_token_ids(text, device=self.device)
        return self.mulan_model(text_token_id, data_type="text")

    def sample_ssa(
        self, input_audio: torch.Tensor, target_audio: torch.Tensor, temperature: float
    ):
        input_audio = input_audio.to(self.device)
        target_audio = target_audio.to(self.device)

        cond_token_ids = self.prepare_cond_token_ids(
            input_audio=input_audio, target_audio=target_audio
        )

        # TODO: we could also just take the ground-truth target semantic tokens
        # for now (for testing)
        # with torch.no_grad():
        #     input_token_ids = self.prepare_input_token_ids(input_audio, target_audio)
        #     t_semantic = input_token_ids[:, :247]
        t_semantic = self.sample_target_semantic(cond_token_ids, temperature)
        cond_token_ids = torch.cat((cond_token_ids, t_semantic), dim=1)
        return self.sample_audio_tokens(cond_token_ids, temperature)
