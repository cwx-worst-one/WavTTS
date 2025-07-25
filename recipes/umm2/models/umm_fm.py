from dataclasses import dataclass
from typing import Optional, Tuple
import os
import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
import samantha
from recipes.umm2.transforms.speech import SpeechTransform, SpeechTransformModified
from recipes.umm2.models.base import BaseStage
from easydict import EasyDict
from typing import Callable, List, Optional, Union, Dict


from easydict import EasyDict
from mariana.models.audio.conformer import ConformerLayer
from mariana.models.audio.positional_encoding import RotaryPositionalEncoding
from recipes.umm2.models.tasks.vq_module import VQ, EMAVectorQuantizerEntropy
from recipes.umm2.models.tasks.rvq_module import RVQ, EMAResidualVectorQuantizerEntropy, EMAResidualVectorQuantizerRP

from recipes.umm2.models.umm_conformer import (
    ConformerRotaryPositionalEmbedding,
    ConformerEncoderLayer,
    ConformerFeedForward,
    ConformerSelfAttention,
    ConformerConvolutionModule,
    Transpose,
)

from mariana.utils.audio.audio_logger import AudioLogger
logger = AudioLogger()

INPUT_SEQ_LEN_ALIGNMENT = 32

def pad_btd_to(inputs, align):
    seqlen = inputs.shape[1]
    pad_len = (align - seqlen % align) % align
    inputs = torch.nn.functional.pad(inputs, [0, 0, 0, pad_len])
    mask = torch.ones_like(inputs[:, :, 0]).float()
    mask[:, seqlen:] = 0
    return inputs, mask.float()

@dataclass
class UMMResult:
    hidden_states: torch.Tensor
    vq_ids: torch.Tensor
    vq_hidden_states: torch.Tensor
    vq_loss: Optional[torch.Tensor] = None


def conv_flops(module, input_shape):
    output_shape = input_shape
    output_shape[1] = module.out_channels
    dims = len(input_shape) - 2
    flops = input_shape[0] * module.in_channels * module.out_channels
    for i in range(dims):
        new_kernel_size = module.dilation[i] * (module.kernel_size[i] - 1) + 1
        flops *= new_kernel_size
        output_shape[2 + i] = (
            input_shape[2 + i] + 2 * module.padding[i] - new_kernel_size
        ) // module.stride[i] + 1
        flops *= output_shape[2 + i]
    return 2 * flops, output_shape


def conv_transpose_flops(module, input_shape):
    output_shape = input_shape
    output_shape[1] = module.out_channels
    dims = len(input_shape) - 2
    flops = input_shape[0] * module.in_channels * module.out_channels
    for i in range(dims):
        new_kernel_size = module.dilation[i] * (module.kernel_size[i] - 1) + 1
        flops *= new_kernel_size
        output_shape[2 + i] = (
            (input_shape[2 + i] - 1) * module.stride[i]
            + module.output_padding[i]
            - 2 * module.padding[i]
            + new_kernel_size
        )
        flops *= input_shape[2 + i]
    return 2 * flops, output_shape


@dataclass
class ConformerEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None



class ConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_positions = ConformerRotaryPositionalEmbedding(config)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [ConformerEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        hidden_states = self.dropout(hidden_states)
        position_embeddings = self.embed_positions(hidden_states)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # create gradient checkpointing function
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), hidden_states, position_embeddings
                )
            else:
                layer_outputs = layer(
                    hidden_states, position_embeddings=position_embeddings
                )
            hidden_states = layer_outputs

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return ConformerEncoderOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )

class Conv1dUpsampling(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn=True, act_fn=nn.ReLU):
        super().__init__()
        hidden_dim = input_dim * 4
        self.conv = nn.Sequential(
            Transpose(),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            act_fn(),
            nn.ConvTranspose1d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            act_fn(),
            nn.ConvTranspose1d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            act_fn(),
            Transpose(),
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x

    def get_flops(self, b, t, d):
        return 0


class Conv2dUpsampling(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn=True, act_fn=nn.ReLU, stride=None, pad=None):
        super().__init__()
        if stride is None:  # 25Hz config
            stride = [1, 2, 2]
            pad = [3, 2, 2]

        self.conv = nn.Sequential(
            # [1, 1, 750, 1024]
            nn.Conv2d(1, 64, 7, stride[0], pad[0]),
            torch.nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            act_fn(),
            # [1, 64, 750, 1024]
            nn.ConvTranspose2d(64, 8, 6, stride[1], pad[1]),
            torch.nn.BatchNorm2d(8) if use_bn else nn.Identity(),
            act_fn(),
            # [1, 8, 1500, 2048]
            nn.ConvTranspose2d(8, 1, 6, stride[2], pad[2]),
            torch.nn.BatchNorm2d(1) if use_bn else nn.Identity(),
            act_fn(),
            # [1, 1, 3000, 4096]
        )
        print(stride, pad)
        self.linear = nn.Linear(input_dim * int(stride[0] * stride[1] * stride[2]), output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x

    def get_flops(self, b, t, d):
        flops1, out_shape1 = conv_flops(self.conv[0], [b, 1, t, d])
        flops2, out_shape2 = conv_transpose_flops(self.conv[3], out_shape1)
        flops3, _ = conv_transpose_flops(self.conv[6], out_shape2)
        return flops1 + flops2 + flops3


class Conv2dSubsampling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel: Union[int, tuple],
        padding: Union[int, tuple],
        use_bn: bool = True,
        act_fn: Callable[[], nn.Module] = nn.ReLU,
        stride: Optional[List[int]] = [2, 2]
    ):
        super().__init__()

        self.kernel_size = kernel
        self.padding_size = padding
        self.stride = stride

        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel, stride[0], padding),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            act_fn(),
            nn.Conv2d(256, 256, kernel, stride[1], padding),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            act_fn(),
        )
        self.linear = nn.Linear(input_dim * int(256 / (stride[0] * stride[1])), output_dim)

    def _compute_output_length(self, x_length: Tensor) -> Tensor:
        for s in self.stride:
            x_length = (x_length + 2 * self.padding_size - self.kernel_size) // s + 1
        return x_length

    def forward(self, x: Tensor, x_lengths: Optional[Tensor] = None):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, T, F)

        x = self.conv(x)

        if x_lengths is not None:
            x_lengths = self._compute_output_length(x_lengths)

        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return (x, x_lengths) if x_lengths is not None else x

    def get_flops(self, b: int, t: int, d: int) -> int:
        flops1, out_shape1 = conv_flops(self.conv[0], [b, 1, t, d])
        flops2, _ = conv_flops(self.conv[3], out_shape1)
        return flops1 + flops2


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.feature_encoder = Conv2dSubsampling(
            input_dim=config.n_mels,
            output_dim=config.hidden_size,
            kernel=config.feature_encoder_kernel,
            padding=config.feature_encoder_padding,
            use_bn=config.get("use_bn", True),
        )
        self.conformer_layer = (
            ConformerEncoderLayer(config)
            if config.get("first_conformer", True)
            else nn.Identity()
        )

    def _calculate_masking(self, x: Tensor, x_length: Tensor) -> Tensor:
        # x: Tensor of shape [B, T, D] where T is the time dimension
        # x_length: Tensor of shape [B], containing the valid lengths for each sample in the batch

        B = x_length.size(0)           # Batch size
        max_len = x.shape[1]           # Maximum sequence length (T)

        # Create an index tensor [0, 1, 2, ..., max_len - 1] and broadcast it
        indices = torch.arange(max_len, device=x_length.device)

        # Create a boolean mask of shape [B, T]
        # Each position is True if the index is less than the actual sequence length
        mask = indices.unsqueeze(0) < x_length.unsqueeze(1)

        return mask  # True where attention is allowed (valid positions), False for padding

    def forward(self, x: Tensor, x_length: Optional[Tensor] = None):
        if x_length is not None:
            x, x_length = self.feature_encoder(x, x_length)
            attn_mask = self._calculate_masking(x, x_length)
            if self.config.get("first_conformer", True):
                x = self.conformer_layer(x, attn_mask=attn_mask)
            return x, attn_mask
        else:
            x = self.feature_encoder(x)
            x = self.conformer_layer(x)
            return x

    def get_flops(self, b: int, t: int, d: int) -> int:
        fe_flops = self.feature_encoder.get_flops(b, t, d)
        if self.config.get("first_conformer", True):
            return fe_flops + self.conformer_layer.get_flops(b, t)
        else:
            return fe_flops


# Deprecated Do not use, it is wrong!
class UMM(BaseStage):
    def __init__(
        self,
        config,       
        takes,
        provides,
        bypasses=[],
        lr_ratio=1.0,
        loss_weight=None,
        is_frozen=False,
    ):
        BaseStage.__init__(self, takes, provides, bypasses, lr_ratio=lr_ratio, loss_weight=loss_weight, is_frozen=is_frozen)

        self.audio_encoder = AudioEncoder(config)
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)

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
                sampled_duration = np.random.randint(1, int(np.floor(t / 2)))
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

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}
        if self.config.get("interfere_audio", None):
            x_interfered = self.interfere_audio(x)
            mel_interfered = self.audio_transform(x_interfered, normalize=normalize)
            input_dict.update(mel_interfered=mel_interfered)
        return input_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_feature(self, batch):
        wav = batch['audio'].squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        input_dict = self.preprocessing(wav)
        return input_dict

    def _compute(self, batch):

        input_dict = self.get_feature(batch)  
        feature = input_dict['mel']
        flops = self.audio_encoder.get_flops(*feature.shape)
        feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(self.encoder_layers)

        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
    
        flops += hidden_states.shape[0] * hidden_states.shape[1] * 2

        output_dict = {
            "audio": batch['audio'],
            "mel": input_dict["mel"],
            "latent": hidden_states,
            "flops": flops * 3,  # extra 2x for backward.
        }
        if "no_text_label" in self.provides and "no_text_label" in batch:
            output_dict["no_text_label"] = batch["no_text_label"]

        return output_dict

# Existing fused version stage2 is trained with UMMModified_no_VQ
# need to keep until next version
class UMMModified_no_VQ(BaseStage):
    def __init__(
        self,
        config,       
        takes=[],
        provides=[],
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
    def forward(self, batch):
        if self.config.use_fused_kernel:
            if not self.config.use_causal_conformer:
                return self._compute_fused_kernel(batch)
            elif self.config.use_causal_conformer:
                return self._compute_causal_conformer(batch)
        else:
            return self._compute_normal(batch)

    def _compute_causal_conformer(self, batch):
        mel, mel_len = self.get_feature(batch)
        feature, feature_mask = self.audio_encoder(mel, mel_len)
        hidden_states = self.encoder_input_dropout(feature)   

        hidden_states = self.conformer(hidden_states, feature_mask)
        hidden_states = hidden_states.float() * feature_mask.view(*hidden_states.shape[:-1], 1)
        fw_flops, bw_flops, _ = self.conformer.calc_flops(feature_mask.sum(dim = -1).view(-1),
                                                          rmpad=self.causal_conformer_config.conformer_use_rmpad)
        flops = fw_flops + bw_flops

        batch['mel'] = mel
        batch['mel_len'] = mel_len
        batch['latent'] = hidden_states
        batch['flops'] = flops
        batch['attn_mask'] = feature_mask
        return batch 

    
    def _compute_normal(self, batch):
        mel, mel_len = self.get_feature(batch)  
        flops = self.audio_encoder.get_flops(*mel.shape)
        feature, feature_mask = self.audio_encoder(mel, mel_len)
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(self.encoder_layers)
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, 
                attn_mask=feature_mask, 
                position_embeddings=position_embeddings
            )
    
        flops += hidden_states.shape[0] * hidden_states.shape[1] * 2
        batch['mel'] = mel
        batch['mel_len'] = mel_len
        batch['latent'] = hidden_states
        batch['flops'] = flops * 3  # extra 2x for backward.
        batch['attn_mask'] = feature_mask
        return batch 

    def _compute_fused_kernel(self, batch):
        mel, mel_len = self.get_feature(batch)  
        flops = self.audio_encoder.get_flops(*mel.shape)
        feature, feature_mask = self.audio_encoder(mel, mel_len)
        hidden_states = self.encoder_input_dropout(feature)
        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)
        pos_emb = self.pos_enc(hidden_states)[1]
  
        len_diff = conformer_mask.shape[-1] - feature_mask.shape[-1]
        assert len_diff >= 0, \
            f"conformer_mask should be larger than feature_mask by {len_diff}"
        conformer_mask = torch.nn.functional.pad(feature_mask, (0, len_diff)).float()
        if self.training:
            audio_input_shape = list(hidden_states.shape)
            audio_encoder_flops, _, _ = self.encoder_layers[0].calc_flops(audio_input_shape)
            flops += audio_encoder_flops * len(self.encoder_layers)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]
        
        hidden_states = hidden_states[:, 0:seqlen, :]
    
        flops += hidden_states.shape[0] * hidden_states.shape[1] * 2
        batch['mel'] = mel
        batch['mel_len'] = mel_len
        batch['latent'] = hidden_states
        batch['flops'] = flops * 3  # extra 2x for backward.
        batch['attn_mask'] = feature_mask
        return batch 


class UMMModified(BaseStage):

    def __init__(

        self,
        config,       
        takes=[],
        provides=[],
        bypasses=[],
        lr_ratio=1.0,
        loss_weight=None,
        is_frozen=False,
    ):
        BaseStage.__init__(self, takes, provides, bypasses, lr_ratio=lr_ratio, loss_weight=loss_weight, is_frozen=is_frozen, config=config)
        logger.info(f"UMMModified config: {config}")
        self.config = config
        self.audio_encoder = AudioEncoder(config)
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.encoder_input_dropout = nn.Dropout(config.hidden_dropout)

        #Initialize audio transform    
        self._init_audio_transform()

        #Initial VQ layer
        self.vq_layer = None
        self.vq_layer_idx = None

        if config.add_vq:
            self._init_vq_module(config)

        #Consistency loss
        self.use_consistency_loss = getattr(config, 'use_consistency_loss', False)
        if self.use_consistency_loss:
            self.consistency_loss_weight = config.consistency_loss_weight
            self.consistency_chunk_ratio = config.consistency_chunk_ratio
            self.consistency_loss_fn = nn.CrossEntropyLoss()

        # ARloss
        # Haven't implement yet


    def _init_audio_transform(self):
        """Initialize audio transformation module."""
        try:
            self.audio_transform = SpeechTransformModified(
                sample_rate=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                win_length=self.config.win_length,
                hop_length=self.config.hop_length,
                f_min=0,
                f_max=self.config.sample_rate // 2,
            )
            
            # Load feature normalization if specified
            if hasattr(self.config, 'feature_cmvn') and self.config.feature_cmvn is not None:
                if not os.path.exists(self.config.feature_cmvn):
                    raise FileNotFoundError(f"Feature CMVN checkpoint not found: {self.config.feature_cmvn}")
                self.audio_transform.load_from_checkpoint(self.config.feature_cmvn)
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio transform: {str(e)}")


    def _init_vq_module(self, config):
        """Initialize vector quantization module."""
        try:
            # Validate VQ layer index
            if not (0 <= config.vq_layer_idx < config.num_hidden_layers):
                raise ValueError(
                    f"vq_layer_idx ({config.vq_layer_idx}) must be between 0 and {config.num_hidden_layers - 1}"
                )
            distance_type = getattr(config, 'vq_distance_type', 'euclidean')
            if config.rvq == 1:
                # Initialize VQ components
                quantize = EMAVectorQuantizerEntropy(
                    config.vq_codebook_size,
                    config.vq_codebook_dim, 
                    decay=config.vq_decay,
                    distance_type=distance_type
                )
                self.vq_layer = VQ(
                    config, 
                    task='vq', 
                    loss_weight=config.w_loss_vq, 
                    vq_scheme=quantize
                )
            else:
                logger.info(f"rvq: {config.rvq}")
                if config.vq_type == "EMAEntropy":
                    quantize = EMAResidualVectorQuantizerEntropy(
                        config.vq_codebook_size, # support list of codebook_size
                        config.vq_codebook_dim, # support list of vq_codebook_dim
                        decay=config.vq_decay,
                        rvq=config.rvq,
                        distance_type=distance_type
                    )
                elif config.vq_type.startswith("EMARP"):
                    quantize = EMAResidualVectorQuantizerRP(
                        config.vq_type, 
                        config.vq_codebook_size, # support list of codebook_size
                        config.vq_codebook_dim, 
                        decay=config.vq_decay,
                        rvq=config.rvq, 
                        stale_tolerance=config.stale_tolerance,
                    )
                else:
                    raise NotImplementedError(f"vq_type {config.vq_type} not supported")

                self.vq_layer = RVQ(
                    config, 
                    task='rvq', 
                    loss_weight=config.w_loss_vq, 
                    rvq_scheme=quantize
                )

            self.vq_layer_idx = config.vq_layer_idx
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VQ module: {str(e)}")

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

    def calculate_consistency_loss(
        self,
        hidden_states: torch.Tensor,
        original_vq_ids: torch.Tensor,
        feature_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the consistency loss in a fully vectorized manner, avoiding
        inefficient for-loops and leveraging batch-wide tensor operations.
        """
        device = hidden_states.device

        seq_lengths = feature_mask.sum(dim=-1)
        valid_mask = seq_lengths > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        valid_hidden_states = hidden_states[valid_mask]
        valid_vq_ids = original_vq_ids[valid_mask]
        valid_seq_lengths = seq_lengths[valid_mask]

        min_seq_len = valid_seq_lengths.min().item()
        chunk_len = int(self.consistency_chunk_ratio * min_seq_len)

        if chunk_len < 1:
            return torch.tensor(0.0, device=device)

        current_batch_size = valid_hidden_states.shape[0]
        valid_range = (valid_seq_lengths - chunk_len).float()
        start_indices = (torch.rand(current_batch_size, device=device) * valid_range).long()

        hidden_dim = valid_hidden_states.shape[-1]
        offsets = torch.arange(chunk_len, device=device).unsqueeze(0)
        indices = start_indices.unsqueeze(1) + offsets

        chunk_hidden_states = torch.gather(valid_hidden_states, 1, indices.unsqueeze(-1).expand(-1, -1, hidden_dim))
        chunk_target_vq_ids = torch.gather(valid_vq_ids, 1, indices)

        chunk_logits = self.vq_layer.vq.get_logits(chunk_hidden_states)
        logits_for_loss = chunk_logits.view(-1, self.config.vq_codebook_size)

        targets_for_loss = chunk_target_vq_ids.view(-1).long()
        final_loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        with torch.no_grad(): # 监控指标不需要计算梯度
            chunk_pred_ids = torch.argmax(chunk_logits, dim=-1)
            mismatches = (chunk_pred_ids != chunk_target_vq_ids).sum()
            total_elements = chunk_pred_ids.numel() # 获取元素总数
            mismatch_rate = mismatches / total_elements if total_elements > 0 else 0.0

        return final_loss, mismatch_rate

    def forward(self, batch):
        if self.config.use_fused_kernel:
            return self._compute_fused_kernel(batch)
        else:
            return self._compute_normal(batch)

    def _compute_normal(self, batch):

        # Extract features
        mel, mel_len = self.get_feature(batch)
        flops = self.audio_encoder.get_flops(*mel.shape)

        # Encode features
        feature, feature_mask = self.audio_encoder(mel, mel_len)
        hidden_states = self.encoder_input_dropout(feature)
        position_embeddings = self.embed_positions(hidden_states)
        
        # Initialize tracking variables
        vq_output_dict = None

        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(self.encoder_layers)

        for layer_index, layer in enumerate(self.encoder_layers):
            
            if layer_index == self.vq_layer_idx:

                vq_output_dict = self.vq_layer({
                                    'latent': hidden_states,
                                    'attn_mask' : feature_mask
                                    })
                hidden_states = vq_output_dict['quantized_out']

                if self.use_consistency_loss:
                    projected_states_before_vq = vq_output_dict['prevq_embs']
                    target_vq_ids = vq_output_dict['vq_ids']
                    consistency_loss, mismatch_rate = self.calculate_consistency_loss(
                        projected_states_before_vq,
                        target_vq_ids,
                        feature_mask
                    )
                    vq_output_dict['loss'] += consistency_loss * self.consistency_loss_weight
                    vq_output_dict['aux/loss_consistency'] = consistency_loss
                    vq_output_dict['aux/consistency_mismatch_rate'] = mismatch_rate


            hidden_states = layer(
                hidden_states, 
                attn_mask=feature_mask, 
                position_embeddings=position_embeddings
            )
    
        flops += hidden_states.shape[0] * hidden_states.shape[1] * 2

        batch['mel'] = mel
        batch['mel_len'] = mel_len
        batch['latent'] = hidden_states
        batch['flops'] = flops * 3  # extra 2x for backward.
        batch['attn_mask'] = feature_mask
        
        if vq_output_dict:
            batch.update(vq_output_dict)
        
        return batch 

    def _compute_fused_kernel(self, batch):

        # Extract features
        mel, mel_len = self.get_feature(batch)
        flops = self.audio_encoder.get_flops(*mel.shape)

        # Encode features
        feature, feature_mask = self.audio_encoder(mel, mel_len)
        hidden_states = self.encoder_input_dropout(feature)

        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)
        pos_emb = self.pos_enc(hidden_states)[1]
  
        len_diff = conformer_mask.shape[-1] - feature_mask.shape[-1]
        assert len_diff >= 0, \
            f"conformer_mask should be larger than feature_mask by {len_diff}"
        conformer_mask = torch.nn.functional.pad(feature_mask, (0, len_diff)).float()

        if self.training:
            audio_input_shape = list(hidden_states.shape)
            audio_encoder_flops, _, _ = self.encoder_layers[0].calc_flops(audio_input_shape)
            flops += audio_encoder_flops * len(self.encoder_layers)
        
        # Initialize tracking variables
        vq_output_dict = None

        for layer_index, layer in enumerate(self.encoder_layers):
            
            if layer_index == self.vq_layer_idx:
                vq_output_dict = self.vq_layer({
                                    'latent': hidden_states,
                                    'attn_mask' : conformer_mask
                                    })
                hidden_states = vq_output_dict['quantized_out']

                if self.use_consistency_loss:
                    projected_states_before_vq = vq_output_dict['prevq_embs']
                    target_vq_ids = vq_output_dict['vq_ids']
                    consistency_loss, mismatch_rate = self.calculate_consistency_loss(
                        projected_states_before_vq,
                        target_vq_ids,
                        conformer_mask
                    )
                    vq_output_dict['loss'] += consistency_loss * self.consistency_loss_weight
                    vq_output_dict['aux/loss_consistency'] = consistency_loss
                    vq_output_dict['aux/consistency_mismatch_rate'] = mismatch_rate


            hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]

        hidden_states = hidden_states[:, 0:seqlen, :]
    
        flops += hidden_states.shape[0] * hidden_states.shape[1] * 2

        batch['mel'] = mel
        batch['mel_len'] = mel_len
        batch['latent'] = hidden_states
        batch['flops'] = flops * 3  # extra 2x for backward.
        batch['attn_mask'] = feature_mask

        vq_output_dict['vq_ids'] = vq_output_dict['vq_ids'][:, 0:seqlen]
        
        if vq_output_dict:
            batch.update(vq_output_dict)
        
        return batch 

    @torch.no_grad()
    def wav2token(self, audio, audio_len=None, **kwargs):
        if self.config.use_fused_kernel:
            return self._wav2token_fused(audio, audio_len, **kwargs)
        else:
            return self._wav2token_normal(audio, audio_len, **kwargs)

    @torch.no_grad()
    def _wav2token_normal(self, audio, audio_len=None, **kwargs):
 
        if audio_len is None:
            audio_len = torch.tensor([audio.shape[-1]] * audio.shape[0], device=audio.device, dtype=torch.long)
        
        mel, mel_len = self.preprocessing(audio, audio_len)
        feature, feature_mask = self.audio_encoder(mel, mel_len)
        
        position_embeddings = self.embed_positions(feature)
        
        output_dict = {}
        hidden_states = feature

        for layer_index, layer in enumerate(self.encoder_layers):
            if self.vq_layer is not None and layer_index == self.vq_layer_idx:
                
                vq_input = {'latent': hidden_states, 'attn_mask': feature_mask, **kwargs}
                vq_output = self.vq_layer(vq_input)
                
                hidden_states = vq_output['quantized_out']
                # Store intermediate VQ-related tensors for analysis or other purposes
                output_dict['quantized_latent'] = hidden_states

            # Pass the hidden_states through the current Conformer encoder layer
            hidden_states = layer(
                hidden_states,
                attn_mask=feature_mask,
                position_embeddings=position_embeddings
            )
        
        output_dict['mel'] = mel
        output_dict['mel_len'] = mel_len
        output_dict['hidden_states'] = hidden_states
        output_dict['attn_mask'] = feature_mask
        output_dict.update(vq_output)
        
        return output_dict

    def _wav2token_fused(self, audio, audio_len=None, **kwargs):

        if audio_len is None:
            audio_len = torch.tensor([audio.shape[-1]] * audio.shape[0], device=audio.device, dtype=torch.long)
        
        mel, mel_len = self.preprocessing(audio, audio_len)
        feature, feature_mask = self.audio_encoder(mel, mel_len)
        
        output_dict = {}
        hidden_states = feature

        seqlen = hidden_states.shape[1]
        hidden_states, conformer_mask = pad_btd_to(hidden_states, INPUT_SEQ_LEN_ALIGNMENT // 4)
        pos_emb = self.pos_enc(hidden_states)[1]

        len_diff = conformer_mask.shape[-1] - feature_mask.shape[-1]
        assert len_diff >= 0, \
            f"conformer_mask should be larger than feature_mask by {len_diff}"
        conformer_mask = torch.nn.functional.pad(feature_mask, (0, len_diff)).float()

        for layer_index, layer in enumerate(self.encoder_layers):
            if self.vq_layer is not None and layer_index == self.vq_layer_idx:
                
                vq_input = {'latent': hidden_states, 'attn_mask': conformer_mask, **kwargs}
                vq_output = self.vq_layer(vq_input)
                
                hidden_states = vq_output['quantized_out']
                # Store intermediate VQ-related tensors for analysis or other purposes
                output_dict['quantized_latent'] = hidden_states[:, :seqlen, :]

            # Pass the hidden_states through the current Conformer encoder layer
            hidden_states = layer(hidden_states, pos_emb, conformer_mask, 
                use_fused_block=True, fuse_dropout_residual_layernorm=False)[0]

        hidden_states = hidden_states[:, :seqlen, :]
        
        output_dict['mel'] = mel
        output_dict['mel_len'] = mel_len
        output_dict['hidden_states'] = hidden_states
        output_dict['attn_mask'] = feature_mask
        
        vq_output['vq_ids'] = vq_output['vq_ids'][:, :seqlen]
        output_dict.update(vq_output)
        
        return output_dict
