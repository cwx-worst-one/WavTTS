from typing import Sequence, Optional, Callable
import torch
import numpy as np
import math

from .a_unet import TimeSpkConditioningPlugin
from .a_unet.apex import (
    XUNet,
    XCondUNet,
    XBlock,
    XCondBlock,
    ResnetItem as R,
    AttentionItem as A,
    CrossAttentionItem as C,
    CrossAttentionItem2 as C2,
    ModulationItem as M,
    AddCondItem as AddC,
    FeedForwardItem as F,
    SkipCat
)

class SinusoidalScalarEmbedding(torch.nn.Module):
    def __init__(self, max_length: int, features: int):
        super().__init__()
        assert features % 2 == 0
        div_term = torch.exp(  # [d_model]
            torch.arange(0, features // 2, dtype=torch.float32)
            * -(math.log(max_length) / features)
        )
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        idx = torch.arange(max_length).unsqueeze(0).unsqueeze(-1)
        pos_embed = idx * div_term
        pos_embed = torch.stack([torch.sin(pos_embed), torch.cos(pos_embed)], dim=-1)
        pos_embed = pos_embed.view(*pos_embed.shape[:2], -1)
        self.register_buffer('pos_embed', pos_embed)  

    def forward(self, length):
        return self.pos_embed[:, :length, :]

# Use AddC instead of CrossAttenion for conditioning
def CondUNet(
    dim: int, # 1, 2 or 3
    in_channels: int,
    channels: Sequence[int],
    factors: Sequence[int],
    items: Sequence[int],
    attentions: Sequence[int],
    attention_features: int,
    attention_heads: int,
    embedding_features: Optional[int] = None,
    spk_features: Optional[int] = None,
    skip_t: Callable = SkipCat,
    resnet_groups: int = 8,
    modulation_features: int = 1024,
    embedding_max_length: int = 10000,
    out_channels: Optional[int] = None,
    use_positional_embedding = True,
    aux_head_dim: Optional[int] = None,
    use_ctiga: bool = False,
    ):

    if use_positional_embedding:
        positional_embedding = SinusoidalScalarEmbedding
    else:
        positional_embedding = None

    # Check lengths
    num_layers = len(channels)
    sequences = (channels, factors, items, attentions)
    assert all(len(sequence) == num_layers for sequence in sequences)

    # Define UNet type with time conditioning and CFG plugins
    UNet = TimeSpkConditioningPlugin(XCondUNet)

    return UNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        blocks=[
            XCondBlock( # args of items
                channels=channels,
                factor=factor,
                items=([R, M, AddC] + [A] * n_att) * n_items,
            ) for channels, factor, n_items, n_att in zip(*sequences)
        ],
        skip_t=skip_t,
        attention_features=attention_features,
        attention_heads=attention_heads,
        embedding_features=embedding_features,
        spk_features = spk_features,
        modulation_features=modulation_features,
        resnet_groups=resnet_groups,
        positional_embedding_t=SinusoidalScalarEmbedding,
        max_length=embedding_max_length,
        aux_head_dim=aux_head_dim,
        use_ctiga=use_ctiga,
    )

def DualCondUNet(
    dim: int, # 1, 2 or 3
    in_channels: int,
    channels: Sequence[int],
    factors: Sequence[int],
    items: Sequence[int],
    attentions: Sequence[int],
    attention_features: int,
    attention_heads: int,
    embedding_features: Optional[int] = None,
    spk_features: Optional[int] = None,
    skip_t: Callable = SkipCat,
    resnet_groups: int = 8,
    modulation_features: int = 1024,
    embedding_max_length: int = 10000,
    out_channels: Optional[int] = None,
    use_positional_embedding: bool = True,
    use_ctiga: bool = False,
    ):

    if use_positional_embedding:
        positional_embedding = SinusoidalScalarEmbedding
    else:
        positional_embedding = None

    # Check lengths
    num_layers = len(channels)
    sequences = (channels, factors, items, attentions)
    assert all(len(sequence) == num_layers for sequence in sequences)

    # Define UNet type with time conditioning and CFG plugins
    UNet = TimeSpkConditioningPlugin(XCondUNet)

    return UNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        blocks=[
            XCondBlock( # args of items
                channels=channels,
                factor=factor,
                items=([R, M, AddC, C2] + [A] * n_att) * n_items,
            ) for channels, factor, n_items, n_att in zip(*sequences)
        ],
        skip_t=skip_t,
        attention_features=attention_features,
        attention_heads=attention_heads,
        embedding_features=embedding_features,
        spk_features = spk_features,
        modulation_features=modulation_features,
        resnet_groups=resnet_groups,
        positional_embedding_t=positional_embedding,
        max_length=embedding_max_length,
        use_ctiga=use_ctiga,
    )

def DualCondUNet2(
    dim: int, # 1, 2 or 3
    in_channels: int,
    channels: Sequence[int],
    factors: Sequence[int],
    items: Sequence[int],
    attentions: Sequence[int],
    attention_features: int,
    attention_heads: int,
    embedding_features: Optional[int] = None,
    spk_features: Optional[int] = None,
    skip_t: Callable = SkipCat,
    resnet_groups: int = 8,
    modulation_features: int = 1024,
    embedding_max_length: int = 10000,
    ffn_multiplier: int = 2,
    out_channels: Optional[int] = None,
    use_positional_embedding: bool = True,
    use_ctiga: bool = False,
    ):

    if use_positional_embedding:
        positional_embedding = SinusoidalScalarEmbedding
    else:
        positional_embedding = None

    # Check lengths
    num_layers = len(channels)
    sequences = (channels, factors, items, attentions)
    assert all(len(sequence) == num_layers for sequence in sequences)

    # Define UNet type with time conditioning and CFG plugins
    UNet = TimeSpkConditioningPlugin(XCondUNet)

    return UNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        blocks=[
            XCondBlock( # args of items
                channels=channels,
                factor=factor,
                items=([R, M, AddC] + [A, C2, F] * n_att) * n_items,
            ) for channels, factor, n_items, n_att in zip(*sequences)
        ],
        skip_t=skip_t,
        attention_features=attention_features,
        attention_heads=attention_heads,
        embedding_features=embedding_features,
        spk_features = spk_features,
        modulation_features=modulation_features,
        resnet_groups=resnet_groups,
        positional_embedding_t=positional_embedding,
        max_length=embedding_max_length,
        attention_multiplier=ffn_multiplier,
        use_ctiga=use_ctiga,
    )



MODEL_MAP = {
        "CondUNet": CondUNet,
        "DualCondUNet": DualCondUNet,
        "DualCondUNet2": DualCondUNet2
        }
