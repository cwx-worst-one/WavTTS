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

class BiLSTMproj(nn.Module):

    def __init__(self, enc_dim):
        """
        Locally Recurrent Layer (Sec 2.2.1 in https://arxiv.org/abs/2101.05014).
            It consists of a bi-directional LSTM followed by a linear projection.
        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
        """
        super().__init__()
        # Bi-LSTM with learnable (h_0, c_0) state
        self.rnn = nn.LSTM(enc_dim, enc_dim,
                           1, dropout=0, batch_first=True, bidirectional=True, bias=True)
        self.cell_init = nn.Parameter(torch.rand(1, 1, enc_dim))
        self.hidden_init = nn.Parameter(torch.rand(1, 1, enc_dim))

        # Linear projection layer
        self.proj = nn.Linear(enc_dim * 2, enc_dim, bias=False)

    def forward(self, intra_segs):
        """
        Process through a locally recurrent layer along the intra-segment
            direction.
        Parameters:
        	frames (tensor): A batch of intra-segments in shape `[B*S, K, D]`, where
                `B` is the batch size, `S` is the number of segments, 'K' is the
                segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
            lr_output (tensor): A batch of processed segments with the same shape as the input.
        """
        batch_size_seq_len = intra_segs.size(0)
        cell = self.cell_init.repeat(2, batch_size_seq_len, 1)
        hidden = self.hidden_init.repeat(2, batch_size_seq_len, 1)
        rnn_output, _ = self.rnn(intra_segs, (hidden, cell))
        lr_output = self.proj(rnn_output)
        return lr_output

class BiGRUproj(nn.Module):

    def __init__(self, enc_dim):
        """
        Locally Recurrent Layer (Sec 2.2.1 in https://arxiv.org/abs/2101.05014).
            It consists of a bi-directional LSTM followed by a linear projection.
        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
        """
        super().__init__()
        # Bi-LSTM with learnable (h_0, c_0) state
        self.rnn = nn.GRU(enc_dim, enc_dim,
                          1, dropout=0, batch_first=True, bidirectional=True)
        self.cell_init = nn.Parameter(torch.rand(1, 1, enc_dim))

        # Linear projection layer
        self.proj = nn.Linear(enc_dim * 2, enc_dim)

    def forward(self, intra_segs):
        """
        Process through a locally recurrent layer along the intra-segment
            direction.
        Parameters:
        	frames (tensor): A batch of intra-segments in shape `[B*S, K, D]`, where
                `B` is the batch size, `S` is the number of segments, 'K' is the
                segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
            lr_output (tensor): A batch of processed segments with the same shape as the input.
        """
        batch_size_seq_len = intra_segs.size(0)
        cell = self.cell_init.repeat(2, batch_size_seq_len, 1)
        rnn_output = self.rnn(intra_segs, cell)[0]
        lr_output = self.proj(rnn_output)
        return lr_output

class BiSRUproj(nn.Module):

    def __init__(self, enc_dim):
        """
        Locally Recurrent Layer (Sec 2.2.1 in https://arxiv.org/abs/2101.05014).
            It consists of a bi-directional LSTM followed by a linear projection.
        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
        """
        from recipes.diffusion.models.sru import SRU
        super().__init__()
        self.rnn = SRU(enc_dim, enc_dim, 2, dropout=0.2, bidirectional=True, rescale=True)
        self.cell_init = nn.Parameter(torch.rand(2, 1, enc_dim * 2))

        # Linear projection layer
        self.proj = nn.Linear(enc_dim * 2, enc_dim)

    def forward(self, intra_segs):
        """
        Process through a locally recurrent layer along the intra-segment
            direction.
        Parameters:
        	frames (tensor): A batch of intra-segments in shape `[B*S, K, D]`, where
                `B` is the batch size, `S` is the number of segments, 'K' is the
                segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
            lr_output (tensor): A batch of processed segments with the same shape as the input.
        """
        batch_size_seq_len = intra_segs.size(0)
        cell = self.cell_init.repeat(1, batch_size_seq_len, 1)
        rnn_output = self.rnn(intra_segs.transpose(0, 1).contiguous(), cell)[0]
        lr_output = self.proj(rnn_output.transpose(0, 1).contiguous())
        return lr_output

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
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


class SequenceProcessingBlock(nn.Module):
    """
    Container module for a single Transformer layer.
    args: input_size: int, dimension of the input feature. The input should have shape (batch, seq_len, input_size).
    """
    def __init__(self, input_size, dropout, method='rnn', cross_attn=False, use_checkpoint=False):
        super(SequenceProcessingBlock, self).__init__()
        self.method = method
        self.cross_attn = cross_attn
        self.input_norm = None

        if method == 'gru':
            self.input_norm = RMSNorm(input_size)
            self.seq_model = BiGRUproj(input_size)
        elif method == 'sru':
            self.input_norm = RMSNorm(input_size)
            self.seq_model = BiSRUproj(input_size)
        elif method == 'roformer':
            self.seq_model = TransformerBlock(
                dim=input_size, 
                heads=12, 
                dim_head=int(input_size / 12),
                depth=1,
                attn_dropout=dropout,
                ff_dropout=dropout,
                use_checkpoint=use_checkpoint
            )

        self.dropout = Dropout(dropout)

        if cross_attn:
            self.emb_layers = TransformerBlock(
                dim=input_size, 
                heads=12, 
                dim_head=int(input_size / 12),
                depth=1,
                attn_dropout=dropout,
                ff_dropout=dropout,
                use_checkpoint=use_checkpoint,
                context=True,
            )
        else:
            self.emb_layers = nn.Sequential(
                RMSNorm(input_size),
                nn.GELU(approximate='tanh'),
                Linear(
                    input_size,
                    2 * input_size,
                ),
            )
        # TODO: weird arrangement
        self.out_layers = nn.Sequential(
            RMSNorm(input_size),
            Rearrange("b n d -> b d n"),
	        nn.GELU(approximate='tanh'),
            nn.Conv1d(input_size, input_size, 7, padding=3, bias=False),
            self.dropout,
            Rearrange("b n d -> b d n")
        )

    def forward(self, x, emb, rotary_emb=None):
        # input shape: batch, seq, dim
        if self.input_norm is not None:
            x = self.input_norm(x)

        if self.method == 'roformer':
            x = self.seq_model(x, rotary_emb=rotary_emb)
        else:
            x = self.seq_model(x)

        if self.cross_attn:
            x = x + self.dropout(self.emb_layers(x, context=emb, rotary_emb=rotary_emb))
        else:
            emb_out = self.emb_layers(emb)
            scale, shift = torch.chunk(emb_out, 2, dim=-1)
            x = x * (1 + scale) + shift
        
        x = self.out_layers(x) + x

        return x

# dual-path blocks
class DualPathBlocks(nn.Module):

    def __init__(self, 
            input_size, 
            segment_size, 
            output_size,
            merge_scale=[2, 4, 8, 16, 16, 8, 4, 2],
            intra_seq2seq='lstm', 
            inter_seq2seq='lstm',
            num_blocks=1, 
            context_dim=512, 
            dropout=0,
            cfg_prob=0.1,
            use_checkpoint=False
        ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.merge_scale = merge_scale
        self.segment_size = segment_size
        self.num_blocks = num_blocks
        self.cfg_prob = cfg_prob

        # null embedding for cfg
        self.null_emb = nn.Parameter(torch.randn(input_size))
        # dual-path fine-coarse models
        self.row_fine_model = nn.ModuleList([])
        self.col_coarse_model = nn.ModuleList([])
 
        for i in range(num_blocks):
            self.row_fine_model.append(
                SequenceProcessingBlock(input_size, dropout, method=intra_seq2seq, use_checkpoint=use_checkpoint)
            )
            self.col_coarse_model.append(
                SequenceProcessingBlock(input_size, dropout, method=inter_seq2seq, cross_attn=True, use_checkpoint=use_checkpoint)
            )

        # output layer
        self.output = nn.Sequential(
            Rearrange("b d l1 l2 -> b l1 l2 d"),
            RMSNorm(input_size),
            Rearrange("b l1 l2 d -> b d l1 l2"),
            nn.GELU(approximate='tanh'), 
            nn.Conv2d(input_size, output_size, 1)
        )

    def forward(self, x, glb_emb, seq_emb, rotary_emb=None, force_cfg=None):
        # input shape: b, d, fine_len, course_len
        # apply transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        b, d, lf, lc = x.shape
        output = x
        upper = []
        num_chunks = glb_emb.shape[1]
        # get mask for cfg
        cfg_prob = self.cfg_prob if force_cfg is None else force_cfg
        prob_keep_mask = prob_mask_like((b, 1, 1), 1. - cfg_prob, device = x.device)

        # get null embeddings
        fine_context = seq_emb.mean(1, keepdim=True)
        null_fine_emb = repeat(self.null_emb, 'd -> b 1 d', b=b)
        fine_context = torch.where(
            prob_keep_mask,
            fine_context,
            null_fine_emb
        )
        glb_emb += fine_context
        glb_emb = glb_emb.repeat_interleave(lc//num_chunks, 1).reshape(b * lc, 1, d)
        
        null_coarse_emb = repeat(self.null_emb, 'd -> b l d', b=b, l=seq_emb.shape[1])
        coarse_context = torch.where(
            prob_keep_mask,
            seq_emb,
            null_coarse_emb
        )
        
        for i in range(self.num_blocks):
            row_input = rearrange(output, 'b d lf lc -> (b lc) lf d')
            row_output = self.row_fine_model[i](row_input, glb_emb)
            row_output = rearrange(row_output, '(b lc) lf d -> b d lf lc', lc=lc)
            if (i + 1) <= int(self.num_blocks // 2):
                upper.append(row_output)

            output = output + row_output

            col_input = rearrange(output, 'b d lf lc -> (b lc) d lf')
            col_input = F.avg_pool1d(
                col_input, 
                kernel_size=self.merge_scale[i] * 2, 
                stride=self.merge_scale[i], 
                padding=self.merge_scale[i]//2
            )

            merged_size = col_input.shape[-1]
            col_input = rearrange(col_input, '(b lc) d m -> (b m) lc d', b=b, m=merged_size)   
            col_output = self.col_coarse_model[i](
                col_input, 
                coarse_context.repeat_interleave(merged_size, 0),
                rotary_emb,
            )

            col_output = rearrange(col_output, '(b m) lc d -> (b lc) d m', m=merged_size)
            col_output = col_output.repeat_interleave(self.merge_scale[i], -1)
            col_output = rearrange(col_output, '(b lc) d lf -> b d lf lc', lc=lc)
            if (i + 1) > int(self.num_blocks // 2):
                col_output = col_output + upper.pop(-1)

            output = output + col_output

        output = self.output(output) # B, output_size, dim1, dim2

        return output

# base module for deep DPT
class DualPathDiffusionNetwork(nn.Module):
    def __init__(self,
            input_dim=256,
            feature_dim=1024,
            num_blocks=8,
            segment_size=64,
            segment_stride=32,
            context_dim=512,
            dropout=0,
            num_chunks=1,
            merge_scale=[2, 4, 8, 16, 16, 8, 4, 2],
            intra_seq2seq='lstm',
            inter_seq2seq='lstm',
            cfg_prob=0.1,
            learn_sigma=False,
            use_checkpoint=False,
        ):
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.num_chunks = num_chunks

        self.num_blocks = num_blocks
        self.segment_size = segment_size
        self.segment_stride = segment_stride
        self.learn_sigma = learn_sigma

        self.eps = 1e-8

        self.rotary_emb = RotaryEmbedding(dim=int(feature_dim / 12))

        self.dpp = DualPathProcessing(segment_size, segment_stride)
        self.time_slerp_points = nn.Embedding(2, 256)
        self.time_embed = nn.Sequential(
            RMSNorm(256),
            nn.Linear(256, feature_dim),
            nn.GELU(approximate='tanh'),
            Linear(feature_dim, feature_dim),
        )
        if self.context_dim == 129:  # Only happens with w2v-bert + mulan tokens
            self.context_embed = nn.Sequential(
                nn.Embedding(1024, feature_dim),
                nn.GELU(approximate='tanh'),
                Linear(feature_dim, feature_dim),
                RMSNorm(feature_dim)
            )
            self.context_embed_mulan = nn.Sequential(
                nn.Dropout(dropout),
                zero_module(Linear(128, feature_dim)),
                nn.GELU(approximate='tanh'),
                Linear(feature_dim, feature_dim),
                RMSNorm(feature_dim)
            )
        else:
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

            self.context_embed = nn.Sequential(
                emb_first_layer,
                nn.GELU(approximate='tanh'),
                Linear(feature_dim, feature_dim),
            )
        # bottleneck
        self.input_map = nn.Sequential(
            nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False),
        )

        # DPT model
        self.blocks = DualPathBlocks(
            self.feature_dim, 
            self.segment_size, 
            self.feature_dim,
            merge_scale=merge_scale,
            intra_seq2seq=intra_seq2seq, 
            inter_seq2seq=inter_seq2seq,
            num_blocks=num_blocks, 
            context_dim=context_dim,  
            dropout=dropout,
            cfg_prob=cfg_prob,
            use_checkpoint=use_checkpoint
        )
        
        self.output = nn.Sequential(
            Rearrange('b d l -> b l d'),
            RMSNorm(self.feature_dim),
            Rearrange('b l d -> b d l'),
            nn.Conv1d(self.feature_dim, self.input_dim, 1, bias=False)
        )

    def forward(self, x, timesteps=None, context=None, force_cfg=None):
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
        glb_emb = t_emb.chunk(self.num_chunks, 1)

        # context embedding
        seq_emb = self.context_embed(context.detach())
        glb_emb = torch.cat([e.mean(1, keepdim=True) for e in glb_emb], 1) 
        
        x = self.input_map(x) + rearrange(t_emb, 'b l d -> b d l')

        # split the encoder output into overlapped, longer segments
        x = self.dpp.unfold(x)
        out = self.blocks(
            x, 
            glb_emb, 
            seq_emb, 
            self.rotary_emb,
            force_cfg,
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
        self.n_orig_frames = x.shape[-1]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size//2, 0),
            stride=(self.stride, 1),
        )
        unfolded = rearrange(unfolded, 'b (c l1) l2 -> b c l1 l2', c=chan)
        
        # pad to be evenly divisible by 16
        self.pad_len = 0
        if unfolded.shape[-1] % 16 != 0:
            pad_len =  16 - unfolded.shape[-1] % 16
            unfolded = torch.nn.functional.pad(unfolded, (0, pad_len))
            self.pad_len = pad_len
        
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
        if self.pad_len != 0:
            x = x[..., :-self.pad_len] # remove padding
        to_unfold = rearrange(x, 'b c l1 l2 -> b (c l1) l2')
        x = torch.nn.functional.fold(
            to_unfold,
            (output_size, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size//2, 0),
            stride=(self.stride, 1),
        )
        # force float div for torch jit
        x /= float(self.chunk_size) / self.stride

        return x.squeeze(-1)

if __name__ == '__main__':
    model = DualPathDiffusionNetwork(
        input_dim=16,
        feature_dim=512,
        num_blocks=8,
        num_chunks=4,
        context_dim=1024,
        segment_size=80,
        segment_stride=40,
        dropout=0,
        merge_scale=[2, 4, 8, 16, 16, 8,  4, 2],
        intra_seq2seq='gru',
        inter_seq2seq='roformer',
        use_checkpoint=True
    )
    xt = torch.randn(2, 16, 2500)
    t = torch.randn(2, 1, 2500)
    semantic = torch.randn(2, 250, 1024)
    out = model(xt, t, context=semantic)