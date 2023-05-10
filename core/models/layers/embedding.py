''' embedding '''
import math
from typing import List

import torch
from torch import nn
import torch.onnx.operators

from core.models import utils


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False

    def forward(self, inp, incremental_state=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = inp.data.new(1, 1).fill_(int(self.padding_idx + inp.size(1)))
            else:
                positions = utils.make_positions(
                    inp,
                    self.padding_idx,
                    onnx_trace=self.onnx_trace,
                )
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        return self.num_embeddings


class RelLearnedPositionalEmbedding(nn.Embedding):
    """
    This model learns positional embedding. Different from LearnedPositionalEmbedding,
    this model returns the relative positional embedding
    """

    def __init__(
        self,
        num_buckets,
        num_heads,
        max_distance,
    ):
        super().__init__(num_buckets, num_heads, None)
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.max_distance = max_distance

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        '''get relative positions'''
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(
                relative_positions, torch.zeros_like(relative_positions)
            )

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def forward(self, inp, **_kwargs):
        '''Input is fed to the transformer model. The size should be [bsz x seqlen x dim]
        Return:
            relative positional embedding. the size is [n_heads, seqlen, seqlen]
        '''
        query_length = inp.size(1)
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(query_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position, bidirectional=True
        )
        relative_position_bucket = relative_position_bucket.to(self.weight.device)
        values = super().forward(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def prepare_for_onnx_export_(self):
        '''prepare_for_onnx_export_'''
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, inp, incremental_state=None, timestep=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(inp)
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(inp, self.padding_idx, onnx_trace=self.onnx_trace)
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    @staticmethod
    def max_positions():
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


def _pre_hook(
    state_dict, prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, _error_msgs
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.position_encoding until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class AbsPositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.position_encoding = None
        self.reverse = reverse
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.position_encoding is not None:
            if self.position_encoding.size(1) >= x.size(1):
                if (
                    self.position_encoding.dtype != x.dtype
                    or self.position_encoding.device != x.device
                ):
                    self.position_encoding = self.position_encoding.to(
                        dtype=x.dtype, device=x.device
                    )
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.position_encoding = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.position_encoding[:, : x.size(1)]
        # TODO: The return value is different from LearnedPositionalEmbedding
        return self.dropout(x)

    def jit_forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.tensor_slice(self.position_encoding, x)
        return self.dropout(x)

    @staticmethod
    @torch.jit.script
    def tensor_slice(a, b):
        '''tensor slice'''
        a = a[:, : b.size(1)]
        return a


class ScaledPositionalEncoding(AbsPositionalEncoding):
    """Scaled positional encoding module.

    See also: Sec. 3.2  https://arxiv.org/pdf/1809.08895.pdf

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x + self.alpha * self.position_encoding[:, : x.size(1)]
        # TODO: The return value is different from LearnedPositionalEmbedding
        return self.dropout(x)


class SimpleSinusoidalPositionalEmbedding(nn.Module):
    '''PositionalEmbedding'''

    def __init__(self, demb):
        '''constructor'''
        super(__class__, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        '''forward'''
        # sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        sinusoid_inp = pos_seq.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        return pos_emb[:, None, :]


class RelPositionalEncoding(torch.nn.Module):
    """Relative position encoding module
    :param d_model:embeding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    https://arxiv.org/pdf/1901.02860.pdf
    """

    def __init__(self, d_model, dropout_rate=0, max_len=5000, dim=0):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.position_encoding = None
        self.dim = dim
        if dim == 0:
            self.extend_pe(torch.tensor(0.0).expand(max_len, 1))
        elif dim == 1:
            self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        else:
            raise NotImplementedError

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.position_encoding is not None:
            # self.position_encoding contains both positive and negative parts
            # the length of self.position_encoding is 2 * input_len - 1
            if self.position_encoding.size(1) >= x.size(self.dim) * 2 - 1:
                if (
                    self.position_encoding.dtype != x.dtype
                    or self.position_encoding.device != x.device
                ):
                    self.position_encoding = self.position_encoding.to(
                        dtype=x.dtype, device=x.device
                    )
                return

        pe_positive = torch.zeros(x.size(self.dim), self.d_model)
        pe_negative = torch.zeros(x.size(self.dim), self.d_model)
        position = torch.arange(0, x.size(self.dim), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.position_encoding = pe.to(device=x.device, dtype=x.dtype)

    @staticmethod
    @torch.jit.script
    def jit_extend_pe(
        position_encoding: torch.Tensor, x: torch.Tensor, dim: int, d_model: int
    ) -> List[torch.Tensor]:
        """
        jit script for extend pe
        position_encoding contains both positive and negative parts
        the length of position_encoding is 2 * input_len - 1
        """
        if position_encoding.size(1) < x.size(dim) * 2 - 1:
            pe_positive = torch.zeros(x.size(dim), d_model)
            pe_negative = torch.zeros(x.size(dim), d_model)
            position = torch.arange(0, x.size(dim), dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
            )
            pe_positive[:, 0::2] = torch.sin(position * div_term)
            pe_positive[:, 1::2] = torch.cos(position * div_term)
            pe_negative[:, 0::2] = torch.sin(
                torch.tensor(-1.0, dtype=torch.float32) * position * div_term
            )
            pe_negative[:, 1::2] = torch.cos(
                torch.tensor(-1.0, dtype=torch.float32) * position * div_term
            )

            pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
            pe_negative = pe_negative[1:].unsqueeze(0)
            result = [pe_positive, pe_negative]
        else:
            result = []
        return result

    @staticmethod
    @torch.jit.script
    def jit_slice(pe: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
        """jit script for slice"""
        max_len = int((pe.size(1) + 1) / 2)
        return pe[
            :,
            max_len - x.size(dim) : max_len + x.size(dim) - 1,
        ]

    def forward(self, x, export_max_len=None):
        """Compute positional encoding
        Aegs:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: Encoder tensor. Its shape is (batch, time, ...)
        """
        # for export ONNX for panther fused OP.
        # export model for panther inference
        if (
            (torch.jit.is_scripting() or torch.jit.is_tracing())
            and not self.training
            and export_max_len is not None
        ):
            self.position_encoding = None
            if self.dim == 0:
                self.extend_pe(torch.tensor(0.0).expand(export_max_len, 1))
            elif self.dim == 1:
                self.extend_pe(torch.tensor(0.0).expand(1, export_max_len))
            return self.dropout(x * self.xscale), self.dropout(
                self.position_encoding.to(x.device)
            ).squeeze(0)

        if self.training:
            self.extend_pe(x)
        else:
            jit_results = self.jit_extend_pe(self.position_encoding, x, self.dim, self.d_model)
            if len(jit_results) != 0:
                [pe_positive, pe_negative] = jit_results
                position_encoding = torch.cat([pe_positive, pe_negative], dim=1)
            else:
                position_encoding = self.position_encoding
            position_encoding = position_encoding.to(dtype=x.dtype, device=x.device)
            self.position_encoding = position_encoding
        x = x * self.xscale
        if self.training:
            max_len = (self.position_encoding.size(1) + 1) // 2
            pos_emb = self.position_encoding[
                :,
                max_len - x.size(self.dim) : max_len + x.size(self.dim) - 1,
            ]
            if self.dim == 1:
                pos_emb = pos_emb.expand(x.shape[0], *pos_emb.shape[1:]).contiguous()
        else:
            # max_len = (self.position_encoding.size(1) + 1) // 2
            pos_emb = self.jit_slice(self.position_encoding, x, self.dim)
            pos_emb = pos_emb.to(x.device)
            # self.position_encoding[
            # :,
            # max_len - x.size(self.dim) : max_len + x.size(self.dim) -1,
            # ]
        # TODO: The return value is different from LearnedPositionalEmbedding
        return self.dropout(x), self.dropout(pos_emb)

    def forward_step(self, x):
        """Compute positional encoding
        Aegs:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: Encoder tensor. Its shape is (batch, time, ...)
        """
        x = x * self.xscale

        pos_emb = self.position_encoding.to(x.device)
        return self.dropout(x), self.dropout(pos_emb)


class IncrementalReducedEmbedding(nn.Module):
    """IncrementalReducedEmbedding"""

    def __init__(self, predictor_emb_size, predictor_head, limited_context):
        super().__init__()
        self.predictor_emb_size = predictor_emb_size
        self.predictor_head = predictor_head
        self.limited_context = limited_context
        assert self.limited_context is not None

    def forward(self, prev_emb, pos_emb):
        """forward"""
        assert prev_emb[0].size() == pos_emb.size()
        out_states = prev_emb
        prev = out_states[:, -(self.limited_context - 1) :, :]
        pos = pos_emb[-(self.limited_context - 1) :, :]
        reduced_embedding_out = (
            torch.mul(prev.permute(2, 0, 1).contiguous(), (torch.mul(prev, pos).sum(dim=2)))
            .permute(1, 2, 0)
            .contiguous()
        ).sum(dim=1) / (self.predictor_head * (self.limited_context - 1))
        return reduced_embedding_out, out_states

    def forward_step(self, prev_emb, pos_emb, states, prev0_emb):
        """forward_step"""
        assert self.limited_context
        if states is None:
            states = prev0_emb.unsqueeze(0).expand(
                prev_emb.size(0), self.limited_context - 2, prev0_emb.size(1)
            )
        else:
            states = states.view(-1, self.limited_context - 2, self.predictor_emb_size)
        out_states = torch.cat((states, prev_emb.unsqueeze(dim=1)), 1)
        prev = out_states
        pos = pos_emb[-(self.limited_context - 1) :, :]
        reduced_embedding_out = (
            torch.mul(prev, (torch.mul(prev, pos).sum(dim=2, keepdim=True)))
        ).sum(dim=1) / (self.predictor_head * (self.limited_context - 1))
        out_states = out_states[:, -(self.limited_context - 2) :]
        out_states = out_states.view(-1, (self.limited_context - 2) * self.predictor_emb_size)
        return reduced_embedding_out, out_states
