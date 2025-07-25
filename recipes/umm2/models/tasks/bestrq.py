import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from recipes.umm2.models.base import BaseStage
from recipes.umm2.models.umm_fm import (
    AudioEncoder,
    SpeechTransformModified,
    INPUT_SEQ_LEN_ALIGNMENT,
    pad_btd_to,   
)
from recipes.umm2.models.umm_conformer import (
    ConformerRotaryPositionalEmbedding,
)

from recipes.umm2.models.tasks.masked_rq_predict import (
    MaskedCrossEntropy, 
    accuracy, 
    RandomProjectionQuantizer
)

from typing import List, Dict, Optional, Tuple

class Best_RQ(BaseStage):
    def __init__(
        self, 
        config,
        takes=["audio", "audio_length"],
        provides=["mel", "rq_logits", "rq_target", "flops", "loss", "accu"],
        bypasses=[],
        lr_ratio=1.0,
        loss_weight=None,
        is_frozen=False,
    ):
        BaseStage.__init__(self, takes, provides, bypasses, lr_ratio=lr_ratio, loss_weight=loss_weight, is_frozen=is_frozen, config=config)
        self.config = config
        self.audio_encoder = AudioEncoder(config)
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)

        self.audio_transform = SpeechTransformModified(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=0,
            f_max=config.sample_rate // 2,
        )
        if config.feature_cmvn is not None:
            self.audio_transform.load_from_checkpoint(config.feature_cmvn)

        if config.rq_input_layernorm:
            self.rq_input_layernorm = nn.LayerNorm(
                config.n_mels
                * pow(config.feature_encoder_kernel, config.feature_encoder_padding),
                elementwise_affine=False,
            )

        self.unfolder = nn.Unfold(
            kernel_size=(config.feature_encoder_kernel, 1),
            dilation=1,
            padding=(config.feature_encoder_padding, 0),
            stride=(2, 1),
        )
        self.rq = RandomProjectionQuantizer(config)
        self.rq_head = nn.Linear(
            config.hidden_size,
            config.rq_codebook_size * config.rq_codebook_num,
            bias=False,
        ) 
        self.criterion = MaskedCrossEntropy()

    @torch.no_grad()
    def masking(
            self, 
            x: torch.Tensor, 
            x_length: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random masking in the time domain to the input sequence x.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, time]
            x_length (Tensor): Valid lengths for each sample, shape [batch_size]

        Returns:
            mx (Tensor): Masked tensor
            token_domain_masked_indices (Tensor): Indices for token-level masking
            mel_domain_masked_indices (Tensor): Indices for mel-level masking
        """
        mx = x.clone()
        b, t = mx.shape
        device = x.device

        # Extract config parameters
        len_raw = self.config.len_masking_raw
        len_mel = self.config.len_masking_mel
        len_token = self.config.len_masking_token
        mask_prob = self.config.mask_prob

        # Compute number of candidate masking segments
        num_segments = t // len_raw
        valid_segments_per_sample = x_length // len_raw

        # Generate segment indices and validity mask
        segment_indices = torch.arange(num_segments, device=device).unsqueeze(0)  # [1, num_segments]
        valid_segment_mask = segment_indices < valid_segments_per_sample.unsqueeze(1)  # [batch, num_segments]

        # Sample start indices for masking
        start_indices = (torch.rand(b, num_segments, device=device) < mask_prob) & valid_segment_mask

        # Ensure at least one segment is masked
        if not start_indices.any():
            i = torch.randint(0, b, (1,)).item()
            j = torch.randint(0, valid_segments_per_sample[i].item(), (1,)).item()
            start_indices[i, j] = True

        # Generate time-domain mask and apply noise
        time_mask = start_indices.repeat_interleave(len_raw, dim=1)[:, :t]  # [batch, time]
        time_domain_masked_indices = torch.nonzero(time_mask)  # [N, 2]

        masking_noise = torch.randn(len(time_domain_masked_indices), dtype=x.dtype, device=device) * 0.1
        mx[time_domain_masked_indices[:, 0], time_domain_masked_indices[:, 1]] = masking_noise

        # Generate token-level and mel-level masked indices
        token_mask = start_indices.repeat_interleave(len_token, dim=1)
        mel_mask = start_indices.repeat_interleave(len_mel, dim=1)
        token_domain_masked_indices = torch.nonzero(token_mask)
        mel_domain_masked_indices = torch.nonzero(mel_mask)

        return mx, token_domain_masked_indices, mel_domain_masked_indices


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        if x.size(-1) % rate > 0:
            return F.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(
        self, 
        x: torch.Tensor, 
        x_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        normalize = self.config.feature_cmvn is not None
        mel, mel_length = self.audio_transform(x, x_length, normalize=normalize)
        return mel, mel_length

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_feature(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wav = batch['audio']
        wav_len = batch['audio_length']
        wav = self.pad_audio(wav) 
        mel, mel_length = self.preprocessing(wav, wav_len)
        return mel , mel_length


    def get_masked_features(
        self, 
        batch: Dict[str, torch.Tensor], 
        mask_on_mel: bool = False
    ) -> Dict[str, torch.Tensor]:
        wav = batch['audio']
        wav_len = batch['audio_length']
        wav = self.pad_audio(wav) 
        mel, mel_length = self.preprocessing(wav, wav_len)
        
        if mask_on_mel:
            # Corresponds to original get_masked_mel logic
            # Masking function receives the padded wav to determine indices
            _, masked_indices, masked_mel_indices = self.masking(wav, wav_len)

            masked_mel = mel.clone() # mel was derived from padded_wav
            if masked_mel_indices.numel() > 0: # Ensure there are indices to apply noise
                mel_noise = 0.1 * torch.randn(
                    [len(masked_mel_indices), mel.shape[-1]], # mel.shape[-1] is num_mels
                    dtype=masked_mel.dtype,
                    device=masked_mel.device,
                )
                # Ensure masked_mel_indices are compatible with mel dimensions
                masked_mel[tuple(masked_mel_indices.t())] = mel_noise
            masked_mel_length = mel_length
        else:
            # Corresponds to original get_masked_wav_to_mel logic
            masked_audio, masked_indices, _ = self.masking(wav, wav_len)
            masked_mel, masked_mel_length = self.preprocessing(masked_audio, wav_len)
            
        return {"mel": mel, 
                "masked_mel": masked_mel, 
                "masked_indices": masked_indices, 
                'mel_length': mel_length,
                'masked_mel_length': masked_mel_length}


    def _unfold(self, feature):
        d = feature.size(-1)
        unfold_feature = self.unfolder(feature.unsqueeze(1))
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d)
        return unfold_feature

    def _subsample(self, feature):
        feature = self._unfold(feature)
        feature = self._unfold(feature)
        return feature
    
    @torch.no_grad()
    def get_rq_target(
        self, 
        feature: torch.Tensor
    ) -> torch.Tensor:
        rq_input = rearrange(self._subsample(feature), "b t d -> (b t) d")
        if self.config.rq_input_layernorm:
            rq_input = self.rq_input_layernorm(rq_input)
        target_tokens = self.rq(rq_input)
        target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=feature.size(0))
        return target_tokens

    def get_metrics(self, masked_logits: torch.Tensor, masked_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        metric_dict = {}
        metric_dict["loss"] = self.criterion(masked_logits, masked_target)
        metric_dict["accu"] = accuracy(masked_logits, masked_target)
        return metric_dict

    @torch.no_grad()
    def get_spec(self, batch):
        input_dict = self.get_masked_features(batch, mask_on_mel=self.config.get("mask_mel", False))
        mel = input_dict["mel"]
        masked_mel = input_dict["masked_mel"]
        return {
            "Original Mel": mel.transpose(1, 2),
            "Masked Mel": masked_mel.transpose(1, 2),
        }


    def forward(
            self, 
            batch: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:
        if self.config.use_fused_kernel:
            if not self.config.use_causal_conformer:
                return self._compute_fused_kernel(batch)
            elif self.config.use_causal_conformer:
                return self._compute_causal_conformer(batch)
        else:
            return self._compute_normal(batch)

    
    def _compute_causal_conformer(
            self, 
            batch: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:
        input_dict = self.get_masked_features(batch, mask_on_mel=self.config.get("mask_mel", False))
        masked_feature = input_dict["masked_mel"]
        masked_indices = input_dict["masked_indices"]
        masked_feature_len = input_dict["masked_mel_length"]

        encoded_masked_feature, encoded_masked_feature_masking = self.audio_encoder(masked_feature, masked_feature_len)
        hidden_states = self.encoder_input_dropout(encoded_masked_feature)

        hidden_states = self.conformer(hidden_states, encoded_masked_feature_masking)
        hidden_states = hidden_states.float() * encoded_masked_feature_masking.view(*hidden_states.shape[:-1], 1)
        fw_flops, bw_flops, _ = self.conformer.calc_flops(encoded_masked_feature_masking.sum(dim = -1).view(-1),
                                                          rmpad=self.causal_conformer_config.conformer_use_rmpad)
        flops = fw_flops + bw_flops

        logits = self.rq_head(hidden_states)
        logits = rearrange(
            logits, "b t (d c) -> b t d c", c=self.config.rq_codebook_num
        )
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(
            masked_logits, "b d c -> (b c) d", c=self.config.rq_codebook_num
        )

        feature = input_dict["mel"]
        target = self.get_rq_target(feature)
        masked_target = target[tuple(masked_indices.t())]
        masked_target = rearrange(masked_target, "b c -> (b c)")
        output_dict = {
            "mel": input_dict["mel"],
            "mel_length": input_dict["mel_length"],
            "rq_logits": logits,
            "rq_target": target,
            "flops": flops,
        }
        metric_dict = self.get_metrics(masked_logits, masked_target)
        output_dict.update(metric_dict)  # include loss and flops
        
        return output_dict    

    def _compute_fused_kernel(
            self, 
            batch: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:

        input_dict = self.get_masked_features(batch, mask_on_mel=self.config.get("mask_mel", False))

        masked_feature = input_dict["masked_mel"]
        masked_indices = input_dict["masked_indices"]
        masked_feature_len = input_dict["masked_mel_length"] 

        flops = self.audio_encoder.get_flops(*masked_feature.shape)
        encoded_masked_feature, encoded_masked_feature_masking = self.audio_encoder(masked_feature, masked_feature_len)
        hidden_states = self.encoder_input_dropout(encoded_masked_feature)
        # position_embeddings = self.embed_positions(hidden_states)

        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)
        pos_emb = self.pos_enc(hidden_states)[1]
  
        len_diff = conformer_mask.shape[-1] - encoded_masked_feature_masking.shape[-1]
        assert len_diff >= 0, \
            f"conformer_mask should be larger than encoded_masked_feature_masking by {len_diff}"
        conformer_mask = torch.nn.functional.pad(encoded_masked_feature_masking, (0, len_diff)).float()

        if self.training:
            audio_input_shape = list(hidden_states.shape)
            audio_encoder_flops, _, _ = self.encoder_layers[0].calc_flops(audio_input_shape)
            flops += audio_encoder_flops * len(self.encoder_layers)

        for layer in self.encoder_layers:
            # hidden_states = layer(
            #     hidden_states, 
            #     attn_mask=encoded_masked_feature_masking, 
            #     position_embeddings=position_embeddings
            # )
            hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]
        
        hidden_states = hidden_states[:, 0:seqlen, :]
        
        flops += (
            hidden_states.shape[0]
            * hidden_states.shape[1]
            * self.rq_head.weight.shape[0]
            * self.rq_head.weight.shape[1]
            * 2
        )

        logits = self.rq_head(hidden_states)
        logits = rearrange(
            logits, "b t (d c) -> b t d c", c=self.config.rq_codebook_num
        )
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(
            masked_logits, "b d c -> (b c) d", c=self.config.rq_codebook_num
        )

        feature = input_dict["mel"]
        target = self.get_rq_target(feature)
        masked_target = target[tuple(masked_indices.t())]
        masked_target = rearrange(masked_target, "b c -> (b c)")
        output_dict = {
            "mel": input_dict["mel"],
            "mel_length": input_dict["mel_length"],
            "rq_logits": logits,
            "rq_target": target,
            "flops": flops * 3,  # extra 2x for backward.
        }
        metric_dict = self.get_metrics(masked_logits, masked_target)
        output_dict.update(metric_dict)  # include loss and flops
        

        return output_dict

    def _compute_normal(
            self, 
            batch: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:

        input_dict = self.get_masked_features(batch, mask_on_mel=self.config.get("mask_mel", False))

        masked_feature = input_dict["masked_mel"]
        masked_indices = input_dict["masked_indices"]
        masked_feature_len = input_dict["masked_mel_length"] 

        flops = self.audio_encoder.get_flops(*masked_feature.shape)
        encoded_masked_feature, encoded_masked_feature_masking = self.audio_encoder(masked_feature, masked_feature_len)
        hidden_states = self.encoder_input_dropout(encoded_masked_feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(
            self.encoder_layers
        )
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, 
                attn_mask=encoded_masked_feature_masking, 
                position_embeddings=position_embeddings
            )

        flops += (
            hidden_states.shape[0]
            * hidden_states.shape[1]
            * self.rq_head.weight.shape[0]
            * self.rq_head.weight.shape[1]
            * 2
        )

        logits = self.rq_head(hidden_states)
        logits = rearrange(
            logits, "b t (d c) -> b t d c", c=self.config.rq_codebook_num
        )
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(
            masked_logits, "b d c -> (b c) d", c=self.config.rq_codebook_num
        )

        feature = input_dict["mel"]
        target = self.get_rq_target(feature)
        masked_target = target[tuple(masked_indices.t())]
        masked_target = rearrange(masked_target, "b c -> (b c)")
        output_dict = {
            "mel": input_dict["mel"],
            "mel_length": input_dict["mel_length"],
            "rq_logits": logits,
            "rq_target": target,
            "flops": flops * 3,  # extra 2x for backward.
        }
        metric_dict = self.get_metrics(masked_logits, masked_target)
        output_dict.update(metric_dict)  # include loss and flops
        

        return output_dict
    