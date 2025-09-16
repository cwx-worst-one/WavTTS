import os
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from recipes.sacodec.models.components.conformer_next import ConformerRotaryPositionalEmbedding, ConformerFeedForward, ConformerSelfAttention
from transformers.configuration_utils import PretrainedConfig


class TransposeLast(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)
    
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    https://arxiv.org/pdf/2301.00808
    (AS) - Used in apcodec / ConvNext_v2, not vocos
    """
    def __init__(self, dim, use_2d=False):
        super().__init__()
        self.use_2d = use_2d
        if use_2d:
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        else:
            self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        dim = (1, 2) if self.use_2d else 1
        Gx = torch.norm(x, p=2, dim=dim, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """  # noqa: E501

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            x = x.transpose(1, -1)
            x = F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
            x = x.transpose(1, -1)
            return x
        ### old way of doing it. Takes up way more memory. 40GB vs 50GB
        # elif self.data_format == "channels_first":
        #     u = x.mean(1, keepdim=True)
        #     s = (x - u).pow(2).mean(1, keepdim=True)
        #     x = (x - u) / torch.sqrt(s + self.eps)
        #     if len(x.shape) == 3:
        #         x = self.weight[:, None] * x + self.bias[:, None]
        #     elif len(x.shape) == 4:
        #         x = self.weight[:, None, None] * x + self.bias[:, None, None]
        #     return x
class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3,
        layer_scale_init_value: float = 1e-6,
        use_2d=False,
        use_grn=False
    ):
        super().__init__()
        self.use_2d = use_2d
        intermediate_dim = int(dim * mlp_ratio)
        if use_2d:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=(5, 7), padding=(2, 3), groups=dim)  # depthwise conv
        else:
            self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.norm2 = GRN(intermediate_dim) if use_grn else nn.LayerNorm(intermediate_dim, eps=1e-6)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, -1) ## (B, C, T) -> (B, T, C), or (B, C, F, T) -> (B, F, T, C)
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, -1)
        x = residual + x
        return x

class ConvNextBackbone(nn.Module):
    def __init__(
        self,
        # input_channels: int,
        dim: int,
        num_layers: int,
        use_2d=False,
        mlp_ratio=3,
        use_grn=False
    ):
        super().__init__()
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    use_2d=use_2d,
                    mlp_ratio=mlp_ratio,
                    use_grn=use_grn
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: bs x ch x l
        # out: bs x ch x l
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.norm(x)
        return x


class ConformerConfig(PretrainedConfig):
    model_type = "Conformer"

    def __init__(
        self,
        # normalize inputs
        # input_channels=128, # n_mels
        
        # shared encoder
        hidden_size=1536,
        mlp_ratio=3,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        rotary_embedding_base=10000,

        # ROPE
        rope_enhance_pos=7500, # random at training, fixed at validation
        # rope_enhance_pos = 0, # fixed rope at training and validation

        # ConvNext
        use_grn: bool=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        # shared encoder
        # self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.rotary_embedding_base = rotary_embedding_base
        # ROPE
        self.rope_enhance_pos = rope_enhance_pos

        # ConvNext
        self.use_grn = use_grn

    def get(self, name, default):
        if hasattr(self, name):
            return self.__getattribute__(name)
        else:
            return default


class ConformerNextEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config: ConformerConfig):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout
        mlp_ratio = config.mlp_ratio

        # Feed-forward 1
        self.ffn1 = ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = ConformerSelfAttention(config)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        # ConvNeXt Convolution
        self.conv_module = ConvNeXtBlock(
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            use_grn=config.use_grn
        )

        # Feed-forward 2
        self.ffn2 = ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, hidden_states, position_embeddings: Optional[torch.Tensor] = None
    ):
        # hidden_states = b x l x c

        # 1. Feed-Forward 1 layer
        hidden_states = self.ffn1(hidden_states) * 0.5 + hidden_states

        # 2. Self-Attention layer
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, position_embeddings=position_embeddings
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        # residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2) # b x l x c -> b x c x l
        hidden_states = self.conv_module(hidden_states) # convnext needs transpose.
        hidden_states = hidden_states.transpose(1, 2) # b x l x c -> b x c x l
        # hidden_states = residual + hidden_states # ConvNextBlock already adds residual

        # 4. Feed-Forward 2 Layer
        hidden_states = self.ffn2(hidden_states) * 0.5 + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

class ConformerNextBlock(nn.Module):
    """
    Combination of conformer and ConvNext
    """
    def __init__( # VOCOS
        self,
        config: ConformerConfig,
        channels_first=True
    ):
        super().__init__()
        self.config = config
        self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        self.layers = nn.ModuleList(
            [ConformerNextEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.channels_first = channels_first
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, hidden_states):
        if self.channels_first:
            hidden_states = hidden_states.transpose(1, 2)
        # inpput = B, L, D
        position_embeddings = self.embed_positions(hidden_states)
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states, position_embeddings=position_embeddings
            )
            hidden_states = layer_outputs
        if self.channels_first:
            hidden_states = hidden_states.transpose(1, 2)
        return hidden_states
    

def pixel_shuffle_1d(x, factor):
    batch_size = x.shape[0]
    short_channel_len = x.shape[1]
    short_width = x.shape[2]

    long_channel_len = short_channel_len // factor
    long_width = factor * short_width

    x = x.contiguous().view([batch_size, factor, long_channel_len, short_width])
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(batch_size, long_channel_len, long_width)

    return x


def pixel_unshuffle_1d(x, factor):
    batch_size = x.shape[0]
    long_channel_len = x.shape[1]
    long_width = x.shape[2]

    short_channel_len = long_channel_len * factor
    short_width = long_width // factor

    x = x.contiguous().view([batch_size, long_channel_len, short_width, factor])
    x = x.permute(0, 3, 1, 2).contiguous()
    x = x.view([batch_size, short_channel_len, short_width])
    return x

class DCDownBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        downsample: bool = False, 
        shortcut: bool = True,
        kernel_size=7,
        padding=3,
        factor=2,
        use_2d=False
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = factor
        self.stride = 1 if downsample else self.factor

        channel_factor = self.factor**2 if use_2d else self.factor

        self.group_size = in_channels * channel_factor // out_channels
        self.shortcut = shortcut

        if downsample:
            assert out_channels % channel_factor == 0
            out_channels = out_channels // channel_factor

        conv_cls = nn.Conv2d if use_2d else nn.Conv1d

        self.conv = weight_norm(conv_cls(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
        ))

        self.reduction_fn = F.pixel_unshuffle if use_2d else pixel_unshuffle_1d

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = self.reduction_fn(x, self.factor)

        if self.shortcut:
            y = self.reduction_fn(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DCUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_type: str = "interpolate", # interpolate | pixel_shuffle | transpose | none
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
        kernel_size=7,
        padding=3,
        factor=2,
        use_2d=False
    ) -> None:
        super().__init__()

        self.upsample_type = upsample_type
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = factor
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        channel_factor = self.factor**2 if use_2d else self.factor
        self.repeats = out_channels * channel_factor // in_channels

        if upsample_type == "pixel_shuffle":
            out_channels = out_channels * channel_factor
        
        if use_2d:
            if upsample_type == "transpose":
                self.conv = weight_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, factor, padding=padding))
            else:
                self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding))

            self.reduction_fn = F.pixel_shuffle
        else:
            if upsample_type == "transpose":
                self.conv = weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, factor, padding=padding))
            else:
                self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, 1, padding))

            self.reduction_fn = pixel_shuffle_1d

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.upsample_type == "interpolate":
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        elif self.upsample_type == "pixel_shuffle":
            x = self.conv(hidden_states)
            x = self.reduction_fn(x, self.factor)
        else:
            x = self.conv(hidden_states)
            
        if self.shortcut == 2:
            y = hidden_states.repeat_interleave(self.out_channels // self.in_channels, dim=1, output_size=x.shape[1])
            y = F.interpolate(y, scale_factor=self.factor, mode=self.interpolation_mode)
            hidden_states = x + y
        elif self.shortcut == 1:
            y = hidden_states.repeat_interleave(self.repeats, dim=1, output_size=hidden_states.shape[1] * self.repeats)
            y = self.reduction_fn(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states
    
def get_transpose_padding(kernel_size, stride=2):
    assert kernel_size >= stride, "Kernel size must be larget than stride"
    return int((kernel_size - stride)//2)

class DownUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio=1,
        padding=3,
        kernel_size=7,
        upsample_type="interpolate",
        shortcut=True,
        use_2d=False
    ):
        super().__init__()
        self.in_channels = in_channels

        if ratio >= 1:
            self.down_block = DCDownBlock(
                in_channels, out_channels, factor=round(ratio), 
                shortcut=shortcut, kernel_size=kernel_size, padding=padding, 
                use_2d=use_2d
            )
        elif ratio < 1:
            up_ratio = round(1/ratio)
            if upsample_type == "transpose":
                # Hack: setting kernel and padding of upsample layer to match even padding
                is_even = up_ratio % 2 == 0
                kernel_size = 6 if is_even else 7
                padding = get_transpose_padding(kernel_size, stride=up_ratio)

            self.down_block = DCUpBlock(
                in_channels, out_channels, upsample_type=upsample_type, factor=up_ratio,
                shortcut=shortcut, kernel_size=kernel_size, padding=padding, 
                use_2d=use_2d
            )
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: bs x ch x l
        # out: bs x ch x l
        x = self.down_block(x)
        x = self.norm(x)
        return x

class ConvNextChannelReduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)) # channel reduction
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.conv = ConvNeXtBlock(dim=out_channels)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.norm(x)
        x = self.conv(x)
        return x