import math
import numpy as np
import torch
from torch import nn, einsum
from typing import Optional, Tuple
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.utils import weight_norm, remove_weight_norm


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d



class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class RMSNorm(nn.Module):
    def __init__(self, dim, feat_dim=-1, eps=1e-8):
        super().__init__()
        self.rms = dim**-0.5
        self.feat_dim = feat_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, unscaled=False):
        norm = torch.norm(x, dim=self.feat_dim, keepdim=True) * self.rms
        if unscaled:
            return x / norm.clamp(min=self.eps)
        g = self.scale
        if self.feat_dim != -1:
            while g.ndim <= self.feat_dim:
                g = g[None]
            while g.ndim < x.ndim:
                g = g.unsqueeze(-1)
        return x / norm.clamp(min=self.eps) * g


class BiSRUproj(nn.Module):

    def __init__(self, enc_dim):
        """
        Locally Recurrent Layer (Sec 2.2.1 in https://arxiv.org/abs/2101.05014).
            It consists of a bi-directional LSTM followed by a linear projection.
        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
        """
        from sru import SRU
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


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(0, seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class RoFormerSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.num_attention_heads = nhead
        self.attention_head_size = int(d_model / nhead)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(d_model, self.all_head_size, bias=False)
        self.key = nn.Linear(d_model, self.all_head_size, bias=False)
        self.value = nn.Linear(d_model, self.all_head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        sinusoidal_pos=None,
        sinusoidal_pos_context=None,
        context=None,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # rotary query
        query_layer = self.apply_rotary(query_layer, sinusoidal_pos)

        if context is not None:
            key_layer = self.transpose_for_scores(self.key(context))
            value_layer = self.transpose_for_scores(self.value(context))

            # rotary key_layer & value_layer
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos_context)
            value_layer = self.apply_rotary(value_layer, sinusoidal_pos_context)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            # rotary key_layer & value_layer
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos)
            value_layer = self.apply_rotary(value_layer, sinusoidal_pos)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->RoFormer
class RoFormerSelfOutput(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model, bias=True)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_tensor)
        return hidden_states


class RoFormerAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.self = RoFormerSelfAttention(d_model, nhead, dim_feedforward, dropout)
        self.output = RoFormerSelfOutput(d_model, nhead, dim_feedforward, dropout)
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    # End Copy
    def forward(
        self,
        hidden_states,
        sinusoidal_pos=None,
        sinusoidal_pos_context=None,
        context=None,
    ):
        self_outputs = self.self(
            hidden_states,
            sinusoidal_pos,
            sinusoidal_pos_context,
            context
        )
        outputs = self.output(self_outputs, hidden_states)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->RoFormer
class RoFormerIntermediate(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.dense = nn.Linear(d_model, dim_feedforward, bias=True)
        self.intermediate_act_fn = NewGELUActivation()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->RoFormer
class RoFormerOutput(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super().__init__()
        self.dense = nn.Linear(dim_feedforward, d_model, bias=True)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + input_tensor)
        return hidden_states


class RoFormerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0, max_position_embeddings=1000):
        super().__init__()
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            max_position_embeddings,
            d_model // nhead,
        )
        self.attention = RoFormerAttention(d_model, nhead, dim_feedforward, dropout)
        #self.crossattention = RoFormerAttention(d_model, nhead, dim_feedforward, dropout)
        self.intermediate = RoFormerIntermediate(d_model, nhead, dim_feedforward, dropout)
        self.output = RoFormerOutput(d_model, nhead, dim_feedforward, dropout)

    def forward(
        self,
        hidden_states,
        sinusoidal_pos=None,
        context=None,
    ):
        # [sequence_length, embed_size_per_head] -> sin & cos [batch_size, num_heads, sequence_length, embed_size_per_head // 2]
        sinusoidal_pos = self.embed_positions(hidden_states.shape[1])[
            None, None, :, :
        ].chunk(2, dim=-1)

        if context is None:
            self_attention_outputs = self.attention(
                hidden_states,
                sinusoidal_pos,
            )
            attention_output = self_attention_outputs

        else:
            sinusoidal_pos_context = self.embed_positions(context.shape[1])[
                None, None, :, :
            ].chunk(2, dim=-1)
            cross_attention_outputs = self.attention(
                hidden_states,
                sinusoidal_pos,
                sinusoidal_pos_context,
                context
            )
            attention_output = cross_attention_outputs

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SequenceProcessingBlock(nn.Module):
    """
    Container module for a single Transformer layer.
    args: input_size: int, dimension of the input feature. The input should have shape (batch, seq_len, input_size).
    """
    def __init__(self, input_size, dropout, method='rnn', cross_attn=False):
        super(SequenceProcessingBlock, self).__init__()
        self.method = method
        self.cross_attn = cross_attn
        if method == 'sru':
            self.seq_model = BiSRUproj(input_size)
        elif method == 'roformer':
            self.seq_model = RoFormerLayer(input_size, input_size // 64, input_size * 4, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = RMSNorm(input_size)
        self.act_fn = NewGELUActivation()
        if cross_attn:
            self.emb_layers = RoFormerLayer(input_size, input_size // 64, input_size * 4, dropout=dropout)
        else:
            self.emb_layers = nn.Sequential(
                self.act_fn,
                nn.Linear(
                    input_size,
                    2 * input_size,
                ),
            )
        self.mod_norm = RMSNorm(input_size)
        self.out_layers = nn.Sequential(
	    self.act_fn,
            nn.Conv1d(input_size, input_size, 7, padding=3, bias=False),

        )

    def forward(self, x, emb):
        # input shape: batch, seq, dim
        x = self.seq_model(x)
        if self.cross_attn:
            #emb = self.context_condenser(emb.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            #emb = self.context_norm(emb)
            x = self.mod_norm(x + self.dropout(self.emb_layers(x, context=emb)))
        else:
            emb_out = self.emb_layers(emb)
            scale, shift = torch.chunk(emb_out, 2, dim=-1)
            x = self.mod_norm(x * (1 + scale) + shift)
        x = self.out_norm(x + self.dropout(self.out_layers(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()))
        return x


# dual-path blocks
class DualPathBlocks(nn.Module):

    def __init__(self, input_size, segment_size, output_size,
            intra_seq2seq='lstm', inter_seq2seq='lstm',
            num_blocks=1, context_dim=512, dropout=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.segment_size = segment_size
        self.num_blocks = num_blocks

        # dual-path fine-coarse models
        self.row_fine_model = nn.ModuleList([])
        self.col_coarse_model = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_blocks):
            self.row_fine_model.append(SequenceProcessingBlock(input_size, dropout, method=intra_seq2seq))
            self.col_coarse_model.append(SequenceProcessingBlock(input_size, dropout, method=inter_seq2seq, cross_attn=True))
            self.row_norm.append(RMSNorm(input_size, feat_dim=1))
            self.col_norm.append(RMSNorm(input_size, feat_dim=1))

        # output layer
        self.act_fn = NewGELUActivation()
        self.output = nn.Sequential(self.act_fn, nn.Conv2d(input_size, output_size, 1))

    def forward(self, x, glb_emb, seq_emb, cfg=False):
        batch_size, D, dim1, dim2 = x.shape
        output = x
        upper = []
        seq_emb = seq_emb.permute(0, 3, 1, 2).reshape(batch_size * dim2, D, dim1)
        glb_emb = glb_emb.repeat_interleave(dim2, 0).reshape(batch_size * dim2, 1, D)
        for i in range(self.num_blocks):

            # Coarse-path processing
            col_input = output.permute(0, 3, 1, 2).reshape(batch_size * dim2, D, dim1)
            merge_scale = 2 ** min(i + 1, self.num_blocks - i)
            print(i, merge_scale, output.shape, col_input.shape, output.device)
            col_input = F.avg_pool1d(col_input, kernel_size=merge_scale * 2, stride=merge_scale, padding=merge_scale//2)
            seq_emb_i = F.avg_pool1d(seq_emb, kernel_size=merge_scale * 2, stride=merge_scale, padding=merge_scale//2)
            merged_size = col_input.shape[-1]
            col_input = col_input.view(batch_size, dim2, D, merged_size).permute(0, 2, 3, 1).reshape(batch_size * merged_size, dim2, D)
            seq_emb_i = seq_emb_i.view(batch_size, dim2, D, merged_size).permute(0, 2, 3, 1).reshape(batch_size * merged_size, dim2, D)
            col_output = self.col_coarse_model[i](col_input, seq_emb_i)
            if merge_scale == 2 ** (i + 1):
                upper.append(col_output)
            else:
                col_output = col_output + upper[-1]
                del upper[-1]
            col_output = col_output.view(batch_size, merged_size, dim2, D).permute(0, 2, 3, 1).reshape(batch_size * dim2, D, merged_size)
            col_output = col_output.repeat_interleave(merge_scale, -1)
            col_output = col_output.view(batch_size, dim2, D, dim1).permute(0, 2, 3, 1).reshape(batch_size, D, dim1, dim2)
            col_output = self.col_norm[i](col_output)
            output = output + col_output

            # Fine-path processing
            row_input = output.permute(0, 3, 2, 1).reshape(batch_size * dim2, dim1, D)  # B*dim2, dim1, N
            row_output = self.row_fine_model[i](row_input, glb_emb)
            row_output = row_output.view(batch_size, dim2, dim1, D).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output


        output = self.output(output) # B, output_size, dim1, dim2

        return output


# base module for deep DPT
class DPDNet(nn.Module):
    def __init__(self,
            input_dim=256,
            feature_dim=1024,
            num_blocks=8,
            segment_size=64,
            segment_stride=32,
            dropout=0,
            cond_dim=768,
            cond_bn_dim=16,
            intra_seq2seq='sru',
            inter_seq2seq='roformer',
        ):
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.num_blocks = num_blocks
        self.segment_size = segment_size
        self.segment_stride = segment_stride

        self.eps = 1e-8

        self.act_fn = NewGELUActivation()
        self.dpp = DualPathProcessing(segment_size, segment_stride)
        self.time_slerp_points = nn.Embedding(2, 256)
        self.time_embed = nn.Sequential(
            nn.Linear(256, feature_dim),
            self.act_fn,
            RMSNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )

        # Learnable bottleneck (1536 -> N dim)
        self.context_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_bn_dim),
            self.act_fn,
            RMSNorm(cond_bn_dim),
            nn.Linear(cond_bn_dim, feature_dim),
        )

        self.context_upsampler = NearestUpsample(feature_dim, feature_dim)
        self.context_norm = RMSNorm(feature_dim, feat_dim=1)
        # bottleneck
        self.input_map = nn.Conv1d(self.input_dim, self.feature_dim, 7, padding=3, bias=False)
        self.input_norm = RMSNorm(self.feature_dim, feat_dim=1)

        # DPT model
        self.blocks = DualPathBlocks(self.feature_dim, self.segment_size, self.feature_dim,
                                     intra_seq2seq=intra_seq2seq, inter_seq2seq=inter_seq2seq,
                                     num_blocks=num_blocks, context_dim=1, dropout=dropout)
        
        self.output_norm = RMSNorm(self.feature_dim, feat_dim=1)
        self.output = nn.Conv1d(self.feature_dim, self.input_dim, 7, padding=3, bias=False)

    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder
        for p in self.autoencoder.encoder.parameters():
            p.requires_grad = False
        for p in self.autoencoder.mean_logvar_conv.parameters():
            p.requires_grad = False
        for p in self.autoencoder.decoder.parameters():
            p.requires_grad = False

    def forward(self, x, timesteps=None, context=None, prompt=None, cfg=False, dump_feat=False):
        in_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)
        batch_size, input_dim, seq_length = x.shape
        if timesteps.ndim != 3:
            timesteps = timesteps.view(batch_size, 1, 1).repeat(1, 1, seq_length)

        #if self.training:
        #    if np.random.rand() < 0.25:
        #        cfg = True

        # input: (B, D, T)
        t_start_emb = self.time_slerp_points(torch.zeros([batch_size, 1], device=x.device, dtype=torch.long))
        t_end_emb = self.time_slerp_points(torch.ones([batch_size, 1], device=x.device, dtype=torch.long))
        low_norm = t_start_emb/torch.norm(t_start_emb, dim=-1, keepdim=True)
        high_norm = t_end_emb/torch.norm(t_end_emb, dim=-1, keepdim=True)
        omega = torch.acos((low_norm*high_norm).sum(-1, keepdim=True))
        so = torch.sin(omega)
        t_emb = (torch.sin((1.0-timesteps.view(batch_size, seq_length, 1))*omega) / so) * t_start_emb\
            + (torch.sin(timesteps.view(batch_size, seq_length, 1)*omega) / so) * t_end_emb
        t_emb = self.time_embed(t_emb).mean(1, keepdim=True)

        seq_emb = self.context_embed(context)
        seq_emb = self.context_upsampler(seq_emb.transpose(1, 2).contiguous(), seq_length)
        seq_emb = self.context_norm(seq_emb)
        if prompt is not None:
            p_emb = prompt.view(batch_size, 1, self.feature_dim)
            if cfg:
                p_emb = 0 * p_emb
        else:
            p_emb = 0

        x = self.input_map(x) # (B, D, L)-->(B, N, L)
        x = self.input_norm(x)
        # split the encoder output into overlapped, longer segments
        x = self.dpp.unfold(x)
        seq_emb = self.dpp.unfold(seq_emb)
        glb_emb = (t_emb + p_emb).view(batch_size, self.feature_dim, 1, 1)
        out = self.blocks(x + seq_emb + glb_emb, glb_emb, seq_emb, cfg)
        out = out.view(batch_size, self.feature_dim, self.segment_size, -1)  # B, N, L, K

        # overlap-and-add of the outputs
        out = self.dpp.fold(out)  # B, N, T
        out = self.act_fn(out)
        out = self.output_norm(out)
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
            padding=(self.chunk_size, 0),
            stride=(self.stride, 1),
        )

        return unfolded.reshape(
            batch, chan, self.chunk_size, -1
        )  # (batch, chan, chunk_size, n_chunks)

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
        to_unfold = x.reshape(batch, chan * self.chunk_size, n_chunks)
        x = torch.nn.functional.fold(
            to_unfold,
            (output_size, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.stride, 1),
        )

        # force float div for torch jit
        x /= float(self.chunk_size) / self.stride

        return x.reshape(batch, chan, self.n_orig_frames)


class NearestUpsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_f = weight_norm
        up_scale = 10
        self.conv = nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=2 * up_scale - 1,
                      padding_mode='zeros',
                      padding=up_scale - 1)

    def forward(self, x, size):
        x_size = x.size(2)
        x = F.interpolate(x, size=size, mode='nearest')
        x = self.conv(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            RMSNorm(bottleneck, feat_dim=1),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = RMSNorm(width*scale, feat_dim=1)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(RMSNorm(width, feat_dim=1))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = RMSNorm(planes, feat_dim=1)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class ECAPA_TDNN(nn.Module):
    def __init__(self, in_dim, C, out_dim):
        super().__init__()

        self.conv1  = nn.Conv1d(in_dim, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = RMSNorm(C, feat_dim=1)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            RMSNorm(256, feat_dim=1),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = RMSNorm(3072, feat_dim=1)
        self.fc6 = nn.Linear(3072, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        #print(x.shape, flush=True)
        return x

