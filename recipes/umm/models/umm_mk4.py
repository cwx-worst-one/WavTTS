import math

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from mariana.models.audio.conformer import ConformerLayer
from mariana.models.audio.positional_encoding import RotaryPositionalEncoding
from recipes.umm.models.rmvpe import RMVPE
from recipes.umm.models.umm_mkii import (
    ClusteredVectorQuantizer,
    Conv2dSubsampling,
    Conv2dUpsampling,
    EMAVectorQuantizer,
    EMAVectorQuantizerEntropy,
    f0_normalize,
    FiniteScalarQuantizer,
    get_vuv,
    LookupFreeQuantizer,
    RandomProjectionQuantizer,
    Transpose,
    UMMResult,
    WNConv1d,
)
from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.transforms.speech import SpeechTransform

from easydict import EasyDict

INPUT_SEQ_LEN_ALIGNMENT = 32


def pad_btd_to(inputs, align):
    seqlen = inputs.shape[1]
    pad_len = (align - seqlen % align) % align
    inputs = torch.nn.functional.pad(inputs, [0, 0, 0, pad_len])
    mask = torch.ones_like(inputs[:, :, 0]).float()
    mask[:, seqlen:] = 0
    return inputs, mask.float()


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        act_fn = config.get('act_fn', 'relu')
        if act_fn == 'relu':
            act_fn=torch.nn.ReLU
        else:
            act_fn=torch.nn.GELU
        self.feature_encoder = Conv2dSubsampling(
            config.num_channels,
            config.hidden_size,
            config.feature_encoder_kernel,
            config.feature_encoder_padding,
            use_bn=config.get("use_bn", True),
            act_fn=act_fn,
        )

    def forward(self, x):
        x = self.feature_encoder(x)
        return x

    def get_flops(self, b, t, d):
        return self.feature_encoder.get_flops(b, t, d)


class Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_encoder = AudioEncoder(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)
        conformer_config = EasyDict({
            # backbone
            'conformer_normalize_before': True,
            'conformer_attention_heads': config.num_attention_heads,
            # conformer_mask_topology: None
            'conformer_linear_units': config.intermediate_size,
            'conformer_num_blocks': config.num_hidden_layers,
            'conformer_dropout_rate': config.hidden_dropout,
            'conformer_positional_dropout_rate': 0.1,
            'conformer_attention_dropout_rate': config.attention_dropout,
            'conformer_positionwise_layer_type': 'linear',
            'conformer_activation_fn': 'gelu',
            'conformer_positionwise_conv_kernel_size': 1,
            'conformer_macaron_style': 1,
            'conformer_pos_enc_layer_type': '',  # Not actually used.
            'conformer_selfattention_layer_type':  config.get('attn_type', 'rope_selfattn'),
            'conformer_layer_order': 'mhsa_before_conv',
            'conformer_use_cnn_module': 1,
            'conformer_cnn_module': 'ConvolutionModule',
            'conformer_cnn_module_kernel': str(config.conv_depthwise_kernel_size),
            'conformer_cnn_norm_type': 'layer_norm',
            'backbone_memory_size': config.hidden_size,
            'dropout': 0.1,
            "squeeze_mem": True,
            'attn_amp_enable': True,
            'flash_attn': True,
        })
        self.encoder_layers = nn.Sequential(*[
            ConformerLayer(conformer_config, None, i) for i in range(config.num_hidden_layers)
        ])
        self.pos_enc = RotaryPositionalEncoding(
            config.hidden_size / config.num_attention_heads,
            variant_type = config.get('early_rope_variant', 0)
        )

        self.audio_transform = SpeechTransform(
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

        self.config = config
        self.profile = None
        if self.config.get('enable_profile', False):
            def trace_handler(p):
                # export trace data when traces ready (schedule cycle ends)
                trace_name = f"umm_profile_" + str(p.step_num) + ".pt.trace.json"
                p.export_chrome_trace(f'./{torch.distributed.get_rank()}_{trace_name}')
            self.profile = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=50,
                    warmup=10,
                    active=3,
                    repeat=1,
                ),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profile.start()

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate)
        pad_len = (rate - x.shape[-1] % rate) % rate
        return F.pad(x, (0, pad_len))


class Stage1(Base):
    def __init__(self, config):
        super().__init__(config)
        if config.rq_input_layernorm:
            self.rq_input_layernorm = nn.LayerNorm(
                config.num_channels
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

    def forward(self, input_dict):
        masked_feature = input_dict["masked_mel"]
        masked_indices = input_dict["masked_indices"]
        flops = 0

        flops += self.audio_encoder.get_flops(*masked_feature.shape)
        encoded_masked_feature = self.audio_encoder(masked_feature)
        hidden_states = self.encoder_input_dropout(encoded_masked_feature)
        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)

        audio_input_shape = list(hidden_states.shape)
        audio_encoder_flops, _, _ = self.encoder_layers[0].calc_flops(audio_input_shape)
        flops += audio_encoder_flops * len(self.encoder_layers)

        # Returns: (hidden_states, pos_emb)
        pos_emb = self.pos_enc(hidden_states)[1]
        for _, layer in enumerate(self.encoder_layers):
            # Returns: ((hidden_states, pos_emb), mask)
            hidden_states = layer(((hidden_states, pos_emb), conformer_mask), is_training=True)[0][0]
        # Slice
        hidden_states = hidden_states[:, 0:seqlen, :]

        flops += (
            hidden_states.shape[0] * hidden_states.shape[1] *
            self.rq_head.weight.shape[0] * self.rq_head.weight.shape[1] * 2
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
            "rq_logits": logits,
            "rq_masked_logits": masked_logits,
            "rq_target": target,
            "rq_masked_target": masked_target,
            "flops": flops * 3  # extra 2x for backward.
        }
        if self.profile is not None:
            self.profile.step()
        return output_dict

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
    @torch.cuda.amp.autocast(enabled=False)
    def get_rq_target(self, feature):
        rq_input = rearrange(self._subsample(feature), "b t d -> (b t) d")
        if self.config.rq_input_layernorm:
            rq_input = self.rq_input_layernorm(rq_input)
        target_tokens = self.rq(rq_input)
        target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=feature.size(0))
        return target_tokens

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        return {"mel": mel}

    @torch.no_grad()
    def extract_features(self, wav):
        x = self.pad_audio(wav)
        input_dict = self.preprocessing(x)
        mel = input_dict["mel"]

        hidden_states = self.audio_encoder(mel)
        hidden_states = self.encoder_input_dropout(hidden_states)
        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)

        hidden_states_list = []
        # Returns: (hidden_states, pos_emb)
        pos_emb = self.pos_enc(hidden_states)[1]
        for _, layer in enumerate(self.encoder_layers):
            # Returns: ((hidden_states, pos_emb), mask)
            hidden_states = layer(((hidden_states, pos_emb), conformer_mask), is_training=True)[0][0]
            hidden_states_list.append(hidden_states)
        # Slice
        hidden_states_list = [h[:, 0:seqlen, :] for h in hidden_states_list]
        return hidden_states_list


class Stage2(Base):
    def __init__(self, config):
        super().__init__(config)
        act_fn = config.get('act_fn', 'relu')
        if act_fn == 'relu':
            act_fn = torch.nn.ReLU
        else:
            act_fn = torch.nn.GELU
        self.mel_head = Conv2dUpsampling(
            config.hidden_size, config.n_mels, use_bn=config.get("use_bn", True), act_fn=act_fn
        )
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.add_chroma:
            self.chroma_transform = ChromaSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                n_chroma=config.n_chroma,
                normalized=False,
            )
            self.chroma_head = Conv2dUpsampling(
                config.hidden_size, config.n_chroma, use_bn=config.get("use_bn", True), act_fn=act_fn
            )
        if config.get("add_pitch", False):
            #  must be sr=16000, hop_length=160
            hop_length = config.hop_length * 16000 // config.sample_rate
            print("RMVPE hop_length (on 16k):", hop_length)
            self.rmvpe = RMVPE(hop_length=hop_length)
            self.f0_vuv_head = Conv2dUpsampling(config.hidden_size, 2, act_fn=act_fn)

    def forward(self, input_dict):
        feature = input_dict["mel"]
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)

        audio_input_shape = list(hidden_states.shape)
        audio_encoder_flops, _, _ = self.encoder_layers[0].calc_flops(audio_input_shape)
        flops += audio_encoder_flops * len(self.encoder_layers)

        # Returns: (hidden_states, pos_emb)
        pos_emb = self.pos_enc(hidden_states)[1]
        for _, layer in enumerate(self.encoder_layers):
            # Returns: ((hidden_states, pos_emb), mask)
            hidden_states = layer(((hidden_states, pos_emb), conformer_mask), is_training=True)[0][0]
        # Slice
        hidden_states = hidden_states[:, 0:seqlen, :]

        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)
        flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
        ctc_out = self.ctc_head(hidden_states)
        output_dict = {
            "mel_out": mel_out,
            "ctc_out": ctc_out,
            "flops": flops * 3,  # extra 2x for backward.
        }
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(f0_out=f0_vuv_out[:, :, 0:1])
            output_dict.update(vuv_out=f0_vuv_out[:, :, 1:])
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x, only_mel=False):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        if only_mel:
            return input_dict
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict["chroma"] = chroma
        if self.config.get("add_pitch", False):
            f0 = self.rmvpe.batch_infer(
                x, self.config.sample_rate, thred=0.03, use_viterbi=False
            )
            f0 = f0[:, :-1]
            vuv = get_vuv(f0)
            f0 = f0_normalize(f0)
            input_dict.update(f0=f0, vuv=vuv)
        return input_dict


class Stage3(Stage2):
    def __init__(self, config):
        super().__init__(config)
        if config.get("vq_type", None) == "CVQ":
            self.vq = ClusteredVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                distance=config.get("vq_distance", "cos"),
            )
        elif config.get("vq_type", None) == "FSQ":
            self.vq = FiniteScalarQuantizer(codebook_size=config.vq_codebook_size)
        elif config.get("vq_type", None) == "LFQ":
            self.vq = LookupFreeQuantizer(codebook_size=config.vq_codebook_size)
        elif config.get("vq_type", None) == "EMAEntropy":
            self.vq = EMAVectorQuantizerEntropy(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
        else:
            self.vq = EMAVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
        if config.get("vq_proj_norm", None) == "bn":
            self.vq_proj_in = nn.Sequential(
                Transpose(),
                WNConv1d(config.hidden_size, config.vq_codebook_dim, kernel_size=1)
                if config.hidden_size != config.vq_codebook_dim
                else nn.Identity(),
                nn.BatchNorm1d(config.vq_codebook_dim, affine=False, momentum=0.05),
                Transpose(),
            )
            self.vq_proj_out = nn.Sequential(
                Transpose(),
                WNConv1d(config.vq_codebook_dim, config.hidden_size, kernel_size=1)
                if config.vq_codebook_dim != config.hidden_size
                else nn.Identity(),
                Transpose(),
            )
        elif config.get("vq_proj_norm", None) == "ln":
            self.vq_proj_in = nn.Sequential(
                nn.Linear(config.hidden_size, config.vq_codebook_dim, bias=False)
                if config.hidden_size != config.vq_codebook_dim
                else nn.Identity(),
                nn.LayerNorm(config.vq_codebook_dim, elementwise_affine=False),
            )
            self.vq_proj_out = nn.Sequential(
                nn.Linear(config.vq_codebook_dim, config.hidden_size, bias=False)
                if config.vq_codebook_dim != config.hidden_size
                else nn.Identity()
            )
        else:
            self.vq_proj_in = nn.Linear(
                config.hidden_size, config.vq_codebook_dim, bias=False
            )
            self.vq_proj_out = nn.Linear(
                config.vq_codebook_dim, config.hidden_size, bias=False
            )
        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer("cnt", torch.FloatTensor([0]))

    def forward(self, input_dict, return_ids=False):
        feature = input_dict["mel"]
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)

        audio_input_shape = list(hidden_states.shape)
        audio_encoder_flops, _, _ = self.encoder_layers[0].calc_flops(audio_input_shape)
        flops += audio_encoder_flops * len(self.encoder_layers)

        # Returns: (hidden_states, pos_emb)
        pos_emb = self.pos_enc(hidden_states)[1]
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                # Slice
                hidden_states = hidden_states[:, 0:seqlen, :]
                hidden_states = self.vq_proj_in(hidden_states)
                if self.config.get("vq_proj_noise", 0) > 0:
                    noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                        0
                    ) / self.config.vq_proj_noise
                    hidden_states = (
                        hidden_states + torch.randn_like(hidden_states) * noise_scale
                    )
                    self.cnt.add_(1)
                if self.config.get("vq_type", None) == "FSQ":
                    vq_embs, vq_ids = self.vq(hidden_states)
                    vq_loss = None
                elif self.config.get("vq_type", None) == "EMAEntropy":
                    vq_embs, vq_ids, vq_loss = self.vq(
                        hidden_states, e_scale=1.0 if self.cnt < 30_000 else 0.0
                    )
                else:
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                if return_ids:
                    return vq_ids
                hidden_states = self.vq_proj_out(vq_embs)
                # Pad
                hidden_states, _ = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)

            # Returns: ((hidden_states, pos_emb), mask)
            hidden_states = layer(((hidden_states, pos_emb), conformer_mask), is_training=True)[0][0]
        # Slice
        hidden_states = hidden_states[:, 0:seqlen, :]

        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)
        flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
        ctc_out = self.ctc_head(hidden_states)
        output_dict = {
            "mel_out": mel_out,
            "ctc_out": ctc_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
            "flops": flops * 3,  # extra 2x for backward.
        }
        if self.config.get("vq_proj_noise", False):
            output_dict.update(noise_scale=noise_scale)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            output_dict.update(f0_out=f0_vuv_out[:, :, 0:1])
            output_dict.update(vuv_out=f0_vuv_out[:, :, 1:])
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        """Convert audio file to tokens (after Vector Quantization)."""
        x = self.pad_audio(wav)
        input_dict = self.preprocessing(x, only_mel=True)
        vq_ids = self.forward(input_dict, return_ids=True)
        return vq_ids
