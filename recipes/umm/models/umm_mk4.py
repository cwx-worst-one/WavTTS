import math

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from recipes.umm.transforms.speech import SpeechTransform
from mariana.models.audio.conformer import ConformerLayer, ConformerBackbone
from recipes.umm.models.umm_mkii import RandomProjectionQuantizer, Conv2dSubsampling

from easydict import EasyDict


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_encoder = Conv2dSubsampling(
            config.num_channels,
            config.hidden_size,
            config.feature_encoder_kernel,
            config.feature_encoder_padding,
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
            'conformer_pos_enc_layer_type': 'fix_rel_pos',
            'conformer_selfattention_layer_type': 'rel_selfattn',
            'conformer_layer_order': 'mhsa_before_conv',
            'conformer_use_cnn_module': 1,
            'conformer_cnn_module': 'ConvolutionModule',
            'conformer_cnn_module_kernel': str(config.conv_depthwise_kernel_size),
            'conformer_cnn_norm_type': 'layer_norm',
            'conformer_layernorm_interval': 0,
            'conformer_weight_scale': 1.0,
            'conformer_half_pooling': 0,
            'backbone_memory_size': config.hidden_size,
            'dropout': 0.1,
            "squeeze_mem": True,
            'attn_amp_enable': True,
            'flash_attn': True,
        })
        self.encoder_layers = ConformerBackbone(conformer_config)

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
        if x.size(-1) % rate > 0:
            return F.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def interfere_audio(self, wav_batch):
        interfered_batch = wav_batch.clone()
        b, t = interfered_batch.size()
        primary_indices = np.random.binomial(
            size=b, n=1, p=self.config.mix_prob
        ).astype(bool)
        for primary_i, to_mix in enumerate(primary_indices):
            if to_mix:
                r = ((torch.rand(1) * 10 - 5) / 10)[0].to(wav_batch)
                secondary_i = np.random.randint(b)
                sampled_duration = np.random.randint(1, math.floor(t / 2))
                primary_start = np.random.randint(0, t - sampled_duration)
                secondary_start = np.random.randint(0, t - sampled_duration)
                primary_clip = wav_batch[
                    primary_i, primary_start : primary_start + sampled_duration
                ]
                secondary_clip = wav_batch[
                    secondary_i, secondary_start : secondary_start + sampled_duration
                ]
                scale = (wav_batch[primary_i].square().mean()) / (
                    (wav_batch[secondary_i].square().mean() * torch.pow(10, r) + 1.0e-5)
                ).sqrt()
                interfered_batch[
                    primary_i, primary_start : primary_start + sampled_duration
                ] = (primary_clip + scale * secondary_clip)
        return interfered_batch


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
        audio_input_shape = list(hidden_states.shape)
        audio_encoder_flops, _, _ = self.encoder_layers.calc_flops(audio_input_shape)
        flops += audio_encoder_flops
        hidden_states = self.encoder_layers(hidden_states)

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

