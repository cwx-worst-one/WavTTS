import copy
import math
import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.checkpoint import checkpoint

from typing import Optional, Tuple
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from torch.nn.utils import weight_norm, remove_weight_norm

from recipes.diffusion.models.nn import (
    avg_pool_nd, 
    conv_nd, 
    linear, 
    normalization, 
    timestep_embedding, 
    zero_module,
)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self, 
        query_dim, 
        context_dim=None, 
        heads=8, 
        dim_head=64, 
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, rotary_emb=None):
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        if rotary_emb is not None: 
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout_p,
                is_causal=False,
            )

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(
        self,  
        dim, 
        heads, 
        dim_head,
        depth=1,
        attn_dropout=0., 
        ff_dropout=0.,   
        use_checkpoint=False,
        context=False
    ):
        super(TransformerBlock, self).__init__()

        layers = nn.ModuleList([])
        for _ in range(depth):
            
            layer = nn.ModuleDict(
                    {
                        'norm_0': RMSNorm(dim),
                        'attn': Attention(
                            query_dim=dim, 
                            heads=heads, 
                            dim_head=dim_head, 
                            dropout=attn_dropout
                        ),
                        'norm_1': RMSNorm(dim),
                        'ff': FeedForward(dim=dim, dropout=ff_dropout)
                    }
                )     
            if context:
                layer['norm_context'] = RMSNorm(dim)
            layers.append(layer)

        self.layers = layers
        self.use_checkpoint = use_checkpoint
        self.context = context
    
    def _forward_checkpoint(self, x, context=None, rotary_emb=None):
        for idx, transformer in enumerate(self.layers):
            x = checkpoint(
                transformer['attn'], 
                checkpoint(transformer['norm_0'], x, use_reentrant=False), 
                checkpoint(transformer['norm_context'], context, use_reentrant=False) if self.context else None,
                rotary_emb, 
                use_reentrant=False
            ) + x

            x = checkpoint(
                transformer['ff'], 
                checkpoint(transformer['norm_1'], x, use_reentrant=False), 
                use_reentrant=False
            ) + x

        return x

    def _forward(self, x, context=None, rotary_emb=None):
        for idx, transformer in enumerate(self.layers):
            x = transformer['attn'](
                transformer['norm_0'](x), 
                transformer['norm_context'](context) if self.context else None, 
                rotary_emb
            ) + x
            x = transformer['ff'](transformer['norm_1'](x)) + x
        return x
    
    def forward(self, x, context=None, rotary_emb=None):
        if self.use_checkpoint:
            return self._forward_checkpoint(x, context, rotary_emb)
        else:
            return self._forward(x, context, rotary_emb)

# dual-path blocks
class TNTBlocks(nn.Module):
    def __init__(self, 
            input_dim,
            context_dim,    
            fine_dim,
            fine_heads,
            fine_head_dim,
            coarse_dim, 
            coarse_heads,
            coarse_head_dim,
            depth=1, 
            dropout=0,
            mulan_cfg_prob=0.1,
            semantic_cfg_prob=0.1,
            use_checkpoint=False
        ):
        super().__init__()
        self.mulan_cfg_prob = mulan_cfg_prob
        self.semantic_cfg_prob = semantic_cfg_prob

        # null embedding for cfg
        self.semantic_null_embedding = nn.Parameter(torch.randn(input_dim))
        self.mulan_null_embedding = nn.Parameter(torch.randn(input_dim))

        self.fine_rotary_embedding = RotaryEmbedding(dim=fine_head_dim)
        self.coarse_rotary_embedding = RotaryEmbedding(dim=coarse_head_dim)

        layers = nn.ModuleList()
        for i in range(depth):
            get_cts = nn.Sequential(
                RMSNorm(coarse_dim),
                nn.Linear(coarse_dim, fine_dim),
                Rearrange('b t (n d) -> (b t) n d', n=1)
            )

            fine_to_coarse = nn.Sequential(
                RMSNorm(fine_dim),
                Rearrange('... n d -> ... (n d)'),
                nn.Linear(fine_dim, coarse_dim),
            )

            layers.append(
                nn.ModuleList([
                    fine_to_coarse,
                    TransformerBlock(
                        dim=coarse_dim, 
                        heads=coarse_heads,
                        dim_head=coarse_head_dim,
                        depth=1,
                        attn_dropout=dropout,
                        ff_dropout=dropout,
                        use_checkpoint=False,
                    ),
                    TransformerBlock(
                        dim=fine_dim, 
                        heads=fine_heads, 
                        dim_head=fine_head_dim,
                        depth=1,
                        attn_dropout=dropout,
                        ff_dropout=dropout,
                        use_checkpoint=False,
                        context=True,
                    ),
                    get_cts, 
                    TransformerBlock(
                        dim=fine_dim, 
                        heads=fine_heads, 
                        dim_head=fine_head_dim,
                        depth=1,
                        attn_dropout=dropout,
                        ff_dropout=dropout,
                        use_checkpoint=use_checkpoint
                    ),
                ]
            ))
        self.layers = layers

    def forward(self, 
        x, 
        time_emb, 
        semantic_context_emb, 
        mulan_context_emb, 
        mulan_force_cfg=None,
        semantic_force_cfg=None,
    ):
        # input shape: b, d, fine_len, course_len
        # apply transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        b, d, lf, lc = x.shape

        # get mask for cfg
        mulan_cfg_prob = self.mulan_cfg_prob if mulan_force_cfg is None else mulan_force_cfg
        semantic_cfg_prob = self.semantic_cfg_prob if semantic_force_cfg is None else semantic_force_cfg
        mulan_prob_keep_mask = prob_mask_like((b, 1, 1), 1. - mulan_cfg_prob, device=x.device)
        semantic_prob_keep_mask = prob_mask_like((b, 1, 1), 1. - semantic_cfg_prob, device=x.device)

        # get null embeddings for mulan context
        mulan_null_coarse_emb = repeat(self.mulan_null_embedding, 'd -> b l d', b=b, l=mulan_context_emb.shape[1])
        mulan_context_emb = torch.where(
            mulan_prob_keep_mask,
            mulan_context_emb,
            mulan_null_coarse_emb
        )
        
        # get null embeddings for semantic context
        semantic_null_coarse_emb = repeat(self.semantic_null_embedding, 'd -> b l d', b=b, l=semantic_context_emb.shape[1])
        semantic_context_emb = torch.where(
            semantic_prob_keep_mask,
            semantic_context_emb,
            semantic_null_coarse_emb
        )
        # concat at dim1
        coarse_context = torch.cat([mulan_context_emb, semantic_context_emb], dim=1)

        time_emb = torch.mean(rearrange(time_emb, 'b d lf lc-> (b lc) lf d'), axis=1, keepdim=True)
        
        fine_emb = rearrange(x, 'b d lf lc -> (b lc) lf d')
        fine_emb = F.pad(fine_emb, (0, 0, 1, 0), value=0) # pad class token to the first positions
        cts = F.pad(time_emb, (0, 0, 0, lf), value=0)
        fine_emb = fine_emb + cts
        coarse_emb = 0

        for idx, (fine_to_coarse, coarse_transformer, context_cross_attn, get_cts, fine_transformer) in enumerate(self.layers):
            coarse_emb_residual = fine_to_coarse(fine_emb[:, 0:1])
            coarse_emb_residual = rearrange(coarse_emb_residual, '(b t) d -> b t d', b=b)
            coarse_emb = coarse_emb + coarse_emb_residual
            coarse_emb = coarse_transformer(coarse_emb, rotary_emb=self.coarse_rotary_embedding)
            # context cross-attn
            coarse_emb = context_cross_attn(coarse_emb, context=coarse_context, rotary_emb=self.coarse_rotary_embedding)

            cts = get_cts(coarse_emb)
            cts = F.pad(cts, (0, 0, 0, lf), value=0)
            fine_emb = fine_emb + cts
            fine_emb = fine_transformer(fine_emb, rotary_emb=self.fine_rotary_embedding)
        
        # remove cts and reshape
        output = fine_emb[:, 1:]
        output = rearrange(output, '(b lc) lf d -> b d lf lc', b=b, lc=lc)

        return output

# base module for deep DPT
class TNTDiffusionNetwork(nn.Module):
    def __init__(self,
            input_dim=256,
            feature_dim=1024,
            context_dim=512,
            depth=8,
            num_chunks=1,
            segment_size=64,
            segment_stride=32,
            dropout=0,
            mulan_cfg_prob=0.1,
            semantic_cfg_prob=0.1,
            use_checkpoint=False,
        ):
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.num_chunks = num_chunks

        self.segment_size = segment_size
        self.segment_stride = segment_stride

        self.dpp = DualPathProcessing(segment_size, segment_stride)
        self.time_slerp_points = nn.Embedding(2, 256)
        self.time_embed = nn.Sequential(
            RMSNorm(256),
            nn.Linear(256, feature_dim),
            nn.GELU(approximate='tanh'),
            Linear(feature_dim, feature_dim),
        )
        
        if context_dim == 1:
            emb_first_layer = nn.Sequential(
                nn.Embedding(1024, feature_dim),
                RMSNorm(context_dim),
            )
        else:
            emb_first_layer = nn.Sequential(
                RMSNorm(context_dim),
                Linear(context_dim, feature_dim),
            )

        self.semantic_context_embed = nn.Sequential(
            emb_first_layer,
            nn.GELU(approximate='tanh'),
            Linear(feature_dim, feature_dim),
        )
        self.mulan_context_embed = nn.Sequential(
            nn.Embedding(1024, feature_dim),
            RMSNorm(context_dim),
            nn.GELU(approximate='tanh'),
            Linear(feature_dim, feature_dim),
        )
        # bottleneck
        self.input_map = nn.Sequential(
            nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False),
        )

        # DPT model
        self.blocks = TNTBlocks(
            input_dim=feature_dim,
            context_dim=context_dim,
            fine_dim=feature_dim,
            fine_heads=12,
            fine_head_dim=int(feature_dim / 12),
            coarse_dim=feature_dim,
            coarse_heads=12,
            coarse_head_dim=int(feature_dim / 12),
            depth=depth,
            dropout=dropout,
            mulan_cfg_prob=mulan_cfg_prob,
            semantic_cfg_prob=semantic_cfg_prob,
            use_checkpoint=use_checkpoint,
        )
        
        self.output = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.input_dim, 1, bias=False)
        )

    def forward(
        self, 
        x, 
        timesteps=None, 
        mulan_context=None,
        semantic_context=None, 
        mulan_force_cfg=None,
        semantic_force_cfg=None,
    ):
        batch_size, input_dim, seq_length = x.shape
        if timesteps.ndim != 3:
            timesteps = timesteps.view(batch_size, 1, 1).repeat(1, 1, seq_length)
        # input: (B, D, T)
        # temporal embedding
        t_start_emb = self.time_slerp_points(torch.zeros([batch_size, 1], device=x.device, dtype=torch.long))
        t_end_emb = self.time_slerp_points(torch.ones([batch_size, 1], device=x.device, dtype=torch.long))
        low_norm = t_start_emb / torch.norm(t_start_emb, dim=-1, keepdim=True)
        high_norm = t_end_emb / torch.norm(t_end_emb, dim=-1, keepdim=True)
        omega = torch.acos((low_norm*high_norm).sum(-1, keepdim=True))
        so = torch.sin(omega)
        t_emb = (torch.sin((1.0-timesteps.view(batch_size, seq_length, 1))*omega) / so) * t_start_emb\
            + (torch.sin(timesteps.view(batch_size, seq_length, 1)*omega) / so) * t_end_emb
        t_emb = self.time_embed(t_emb)
    
        # context embedding
        semantic_context_emb = self.semantic_context_embed(semantic_context.detach())
        mulan_context_emb = self.mulan_context_embed(mulan_context.detach())
        
        t_emb = self.dpp.unfold(rearrange(t_emb, 'b t d -> b d t'))

        # split the encoder output into overlapped, longer segments
        x = self.input_map(x) 
        x = self.dpp.unfold(x)
        out = self.blocks(
            x, 
            time_emb=t_emb, 
            semantic_context_emb=semantic_context_emb,
            mulan_context_emb=mulan_context_emb,
            mulan_force_cfg=mulan_force_cfg,
            semantic_force_cfg=semantic_force_cfg,
        ).view(batch_size, self.feature_dim, self.segment_size, -1)  # b, d, lf, lc      

        # overlap-and-add of the outputs
        out = self.dpp.fold(out)  # B, N, T
        out = self.output(out)

        out = out.view(batch_size, input_dim, seq_length)

        return out

class DualPathProcessing(nn.Module):
    """
    Perform Dual-Path processing via overlap-add as in DPRNN [1].
    Args:
        chunk_size (int): Size of segmenting window.
        stride (int): segmentation hop size.
    References
        [1] Yi Luo, Zhuo Chen and Takuya Yoshioka. "Dual-path RNN: efficient
        long sequence modeling for time-domain single-channel speech separation"
        https://arxiv.org/abs/1910.06379
    """

    def __init__(self, chunk_size, stride):
        super(DualPathProcessing, self).__init__()
        self.chunk_size = chunk_size
        self.stride = stride
        self.n_orig_frames = None

    def unfold(self, x):
        r"""
        Unfold the feature tensor from $(batch, channels, time)$ to
        $(batch, channels, chunksize, nchunks)$.
        Args:
            x (:class:`torch.Tensor`): feature tensor of shape $(batch, channels, time)$.
        Returns:
            :class:`torch.Tensor`: spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.
        """
        # x is (batch, chan, frames)
        batch, chan, frames = x.size()
        assert x.ndim == 3
        
        # pad to be evenly divisible by chunk_size
        self.pad_len = 0
        if frames % self.chunk_size != 0:
            pad_len = self.chunk_size - frames % self.chunk_size
            x = torch.nn.functional.pad(x, (0, pad_len))
            self.pad_len = pad_len

        self.n_orig_frames = x.shape[-1]
        
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(0, 0),
            stride=(self.stride, 1),
        )
    
        unfolded = rearrange(unfolded, 'b (c l1) l2 -> b c l1 l2', c=chan)
        
        return unfolded

    def fold(self, x, output_size=None):
        r"""
        Folds back the spliced feature tensor.
        Input shape $(batch, channels, chunksize, nchunks)$ to original shape
        $(batch, channels, time)$ using overlap-add.
        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            output_size (int, optional): sequence length of original feature tensor.
                If None, the original length cached by the previous call of
                :meth:`unfold` will be used.
        Returns:
            :class:`torch.Tensor`:  feature tensor of shape $(batch, channels, time)$.
        .. note:: `fold` caches the original length of the input.
        """
        output_size = output_size if output_size is not None else self.n_orig_frames
        # x is (batch, chan, chunk_size, n_chunks)
        batch, chan, chunk_size, n_chunks = x.size()
        to_unfold = rearrange(x, 'b c l1 l2 -> b (c l1) l2')
        
        x = torch.nn.functional.fold(
            to_unfold,
            (output_size, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(0, 0),
            stride=(self.stride, 1),
        ).squeeze(-1)

        if self.pad_len != 0:
            x = x[..., :-self.pad_len] # remove padding

        return x

if __name__ == '__main__':
    model = TNTDiffusionNetwork(
        input_dim=16,
        feature_dim=512,
        context_dim=1024,
        depth=2,
        num_chunks=4,
        segment_size=64,
        segment_stride=64,
        dropout=0,
        use_checkpoint=True
    )
    # for k, v in model.named_parameters():
    #     print(k)
    xt = torch.randn(2, 16, 2500)
    t = torch.randn(2, 1, 2500)
    semantic = torch.randn(2, 250, 1024)
    out = model(xt, t, context=semantic)