'''multi_head_attn'''
# pylint:disable=too-many-lines

import math
import torch
from torch import nn
import torch.nn.functional as F
from core.extensions import AmpEnable, multi_heads_chunk_wise_attention, fused_multihead_attn
from core.extensions.panther_symbol import MultiheadAttentionSymbolicExport


class BatchedMatMul(nn.Module):
    '''
    Equal to torch.bmm. This implementation is for model
    quantization to accelerate the bmm operation.
    '''

    def forward(self, x, y):
        '''forward function'''
        return torch.matmul(x, y)


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        clamp_inf=False,
        merge_qkv=True,
        **_kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.merge_qkv = self.kdim == embed_dim and self.vdim == embed_dim and merge_qkv
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.bias = bias

        if self.merge_qkv:
            self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        else:
            self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # atten = q*k
        self.bmm1 = BatchedMatMul()
        # output = atten * v
        self.bmm2 = BatchedMatMul()

        self.clamp_inf = clamp_inf
        assert not add_bias_kv, 'add_bias_kv is not supported'
        assert not add_zero_attn, 'add_zero_attn is not supported'

        self._fused_cfg = {
            "embed_dim": embed_dim,
            "attn_heads": num_heads,
            "attn_drop": dropout,
            "hidden_drop": 0.0,
        }

        self._register_load_state_dict_pre_hook(self.compatible_load_hook)

        self.reset_parameters()

    def compatible_load_hook(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        '''
        compatible for load MultiheadAttention.
        '''
        if self.merge_qkv and (prefix + 'q_proj.weight') in state_dict:
            names = ['q_proj', 'k_proj', 'v_proj']
            weights = [state_dict.pop('{}{}.weight'.format(prefix, name), None) for name in names]
            state_dict[prefix + 'in_proj.weight'] = torch.cat(weights, dim=0)
            if self.bias:
                biases = [state_dict.pop('{}{}.bias'.format(prefix, name), None) for name in names]
                state_dict[prefix + 'in_proj.bias'] = torch.cat(biases, dim=0)
        if self.merge_qkv and (prefix + 'in_proj_weight') in state_dict:
            weight = state_dict.pop(prefix + 'in_proj_weight')
            state_dict[prefix + 'in_proj.weight'] = weight
            if self.bias:
                bias = state_dict.pop(prefix + 'in_proj_bias')
                state_dict[prefix + 'in_proj.bias'] = bias
        if not self.merge_qkv and (prefix + 'in_proj_weight') in state_dict:
            weight = state_dict.pop(prefix + 'in_proj_weight')
            state_dict[prefix + 'q_proj.weight'] = weight[0 : self.embed_dim, :]
            state_dict[prefix + 'k_proj.weight'] = weight[self.embed_dim : 2 * self.embed_dim, :]
            state_dict[prefix + 'v_proj.weight'] = weight[
                2 * self.embed_dim : 3 * self.embed_dim, :
            ]
            if self.bias:
                bias = state_dict.pop(prefix + 'in_proj_bias')
                state_dict[prefix + 'q_proj.bias'] = bias[0 : self.embed_dim]
                state_dict[prefix + 'k_proj.bias'] = bias[self.embed_dim : 2 * self.embed_dim]
                state_dict[prefix + 'v_proj.bias'] = bias[2 * self.embed_dim : 3 * self.embed_dim]
        if prefix + 'out_proj_weight' in state_dict:
            weight = state_dict.pop(prefix + 'out_proj_weight')
            state_dict[prefix + 'out_proj.weight'] = weight
            if self.bias:
                bias = state_dict.pop(prefix + 'out_proj_bias')
                state_dict[prefix + 'out_proj.bias'] = bias

    def reset_parameters(self):
        '''reset_parameters'''
        if self.merge_qkv:
            nn.init.xavier_uniform_(self.in_proj.weight)
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.bias:
            if self.merge_qkv:
                nn.init.constant_(self.in_proj.bias, 0.0)
            else:
                nn.init.constant_(self.k_proj.bias, 0.0)
                nn.init.constant_(self.v_proj.bias, 0.0)
                nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def prepare_for_onnx_export_(self):
        '''
        compatible function with old code.
        TODO(liyong): remove this in about two weeks.
        '''

    def forward_qkv(self, query, key=None, value=None, prev_k=None, prev_v=None, static_kv=False):
        """forward qkv"""
        _, bsz, _ = query.size()

        if self.merge_qkv and key is None and value is None:
            q, k, v = self.in_proj_qkv(query)
        else:
            key = query if key is None else key
            value = key if value is None else value
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if prev_k is not None:
            prev_key = prev_k.view(bsz * self.num_heads, -1, self.head_dim)
            k = prev_key if static_kv else torch.cat([prev_key, k], dim=1)
        if prev_v is not None:
            prev_value = prev_v.view(bsz * self.num_heads, -1, self.head_dim)
            v = prev_value if static_kv else torch.cat([prev_value, v], dim=1)
        return q, k, v  # (bsz * num_heads, tgt_len, head_dim)

    def apply_attn_weights_mask(self, attn_weights, bsz, attn_mask=None, key_padding_mask=None):
        """apply mask"""
        _, tgt_len, src_len = attn_weights.size()
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            # attn_mask's dim is 3 now.
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None and key_padding_mask.numel() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if self.training and self.clamp_inf:
            attn_weights = torch.clamp(attn_weights, float('-inf'), 4e4)
        return attn_weights

    def get_attn_output(
        self,
        attn_weights,
        v,
        bsz,
        embed_dim,
    ):
        """get attention output"""
        _, tgt_len, _ = attn_weights.size()
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = self.bmm2(attn_probs.to(v.dtype), v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        return attn

    def get_attn_weights(
        self,
        q,
        k,
        v,
        bsz,
        tgt_len,
        embed_dim,
        need_weights=True,
        key_padding_mask=None,
        attn_mask=None,
    ):
        '''get attention weights'''
        src_len = k.size(1)
        attn_weights = self.bmm1(q * self.scaling, k.transpose(1, 2))
        attn_weights = self.apply_attn_weights_mask(attn_weights, bsz, attn_mask, key_padding_mask)
        inf_mask = attn_weights.max(-1).values == float("-inf")
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.masked_fill(inf_mask.unsqueeze(2), 0.0).clone()
        attn = self.get_attn_output(attn_weights, v, bsz, embed_dim)

        if need_weights:
            attn_weights = attn_weights.view(bsz, -1, tgt_len, src_len).transpose(1, 0)
            attn_weights = attn_weights.mean(dim=0)  # average attention weights over heads
        else:
            attn_weights = None

        return attn, attn_weights

    def _forward_fused(
        self,
        query,
        key_padding_mask=None,
        attn_mask=None,
        **_kwargs,
    ):
        """fused forward for mha"""
        attn = fused_multihead_attn(
            [
                self.in_proj.weight,
                self.in_proj.bias,
                self.out_proj.weight,
                self.out_proj.bias,
            ],
            query,
            key_padding_mask,
            attn_mask,
            **self._fused_cfg,
        )
        return attn.transpose(0, 1), None

    def forward(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=True,
        fused=True,
        **_kwargs,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # for infer and onnx export
        if (torch.jit.is_scripting() or torch.jit.is_tracing()) and not self.training:
            key = query if key is None else key
            value = key if value is None else value
            if attn_mask is not None:
                attn_mask = attn_mask.squeeze(0)
            if self.merge_qkv and key is query and value is query and attn_mask is None:
                return (
                    MultiheadAttentionSymbolicExport.apply(
                        query,
                        key_padding_mask,
                        self.in_proj.weight.t(),
                        self.in_proj.bias,
                        self.out_proj.weight.t(),
                        self.out_proj.bias,
                        self.num_heads,
                        self.embed_dim,
                        self.dropout,
                        need_weights,
                    ),
                    None,
                )
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj.weight if self.merge_qkv else torch.empty([0]),
                self.in_proj.bias,
                None,  # self.bias_k,
                None,  # self.bias_v,
                False,  # self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=not self.merge_qkv,
                q_proj_weight=None if self.merge_qkv else self.q_proj.weight,
                k_proj_weight=None if self.merge_qkv else self.k_proj.weight,
                v_proj_weight=None if self.merge_qkv else self.v_proj.weight,
            )

        if self.training and fused and not need_weights and key is None and value is None:
            return self._forward_fused(query, key_padding_mask, attn_mask)
        q, k, v = self.forward_qkv(query, key, value)
        attn, attn_weights = self.get_attn_weights(
            q, k, v, bsz, tgt_len, embed_dim, need_weights, key_padding_mask, attn_mask
        )

        return attn, attn_weights

    def forward_step(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        attn_mask=None,
        static_kv=False,
        need_weights=True,
        prev_k=None,
        prev_v=None,
        prev_mask=None,
        **_kwargs,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if prev_k is not None and static_kv:
            key = value = None

        q, k, v = self.forward_qkv(query, key, value, prev_k, prev_v, static_kv)

        if prev_mask is not None:
            key_padding_mask = prev_mask

        attn, attn_weights = self.get_attn_weights(
            q, k, v, bsz, tgt_len, embed_dim, need_weights, key_padding_mask, attn_mask
        )

        k = k.view(bsz, self.num_heads, -1, self.head_dim)
        v = v.view(bsz, self.num_heads, -1, self.head_dim)

        return attn, attn_weights, k, v, key_padding_mask

    def in_proj_qkv(self, query):
        '''in_proj_qkv'''
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        '''in_proj_q'''
        if self.merge_qkv:
            return self._in_proj(query, end=self.embed_dim)
        return self.q_proj(query)

    def in_proj_k(self, key):
        '''in_proj_k'''
        if self.merge_qkv:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        return self.k_proj(key)

    def in_proj_v(self, value):
        '''in_proj_v'''
        if self.merge_qkv:
            return self._in_proj(value, start=2 * self.embed_dim)
        return self.v_proj(value)

    def _in_proj(self, inp, start=0, end=None):
        '''_in_proj'''
        if self.merge_qkv:
            return self.in_proj(inp)[..., start:end]
        weight = self.in_proj.weight
        bias = self.in_proj.bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(inp, weight, bias)


class PosMultiHeadAttention(MultiheadAttention):
    """Multi-head attention with relative positional embedding

    See "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"
    for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        clamp_inf=False,
        gru_rel_pos=False,
        **kwargs,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            clamp_inf,
            **kwargs,
        )
        self.gru_rel_pos = gru_rel_pos
        if gru_rel_pos:
            self.grep_linear = nn.Linear(self.head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(
        self,
        query,
        pos_emb,
        key=None,
        value=None,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=False,
        **_kwargs,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
        """
        # pylint: disable=invalid-name
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        key = query if key is None else key
        src_len = key.size()[0]
        value = key if value is None else value

        # B x H x T x T
        pos_emb = (
            pos_emb.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)
        )

        if attn_mask is not None:
            raise NotImplementedError

        attn_mask_rel_pos = pos_emb
        if self.gru_rel_pos:
            query_layer = query.transpose(0, 1)
            new_x_shape = query_layer.size()[:-1] + (self.num_heads, -1)
            query_layer = query_layer.view(*new_x_shape)
            query_layer = query_layer.permute(0, 2, 1, 3)
            _B, _H, _L, __ = query_layer.size()

            gate_a, gate_b = torch.sigmoid(
                self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)
            ).chunk(2, dim=-1)
            gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
            attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, -1, 1) * pos_emb
        attn_mask_rel_pos = attn_mask_rel_pos.view((-1, tgt_len, tgt_len))

        x, _ = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj.weight if self.merge_qkv else torch.empty([0]),
            self.in_proj.bias,
            None,  # self.bias_k,
            None,  # self.bias_v,
            False,  # self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask_rel_pos,
            use_separate_proj_weight=not self.merge_qkv,
            q_proj_weight=None if self.merge_qkv else self.q_proj.weight,
            k_proj_weight=None if self.merge_qkv else self.k_proj.weight,
            v_proj_weight=None if self.merge_qkv else self.v_proj.weight,
        )
        return x


class LiRelMultiHeadAttention(MultiheadAttention):
    """Lite Relative MultiHeadSelfAttention with relative encoding.
    Lighter than Transformer-XL style, untrainable and less params.
    https://arxiv.org/pdf/2010.11395.pdf
    """

    @staticmethod
    def rel_shift(x):
        """Compute relative position encoding
        :param torch.Tensor x: (batch, time, size)
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
        tgt_len, bsz, embed_dim = x.size()
        q, k, v = self.forward_qkv(x)
        q = q.contiguous().view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.contiguous().view(bsz, self.num_heads, tgt_len, self.head_dim)

        n_batch_pos = pos_emb.size(0)
        p = pos_emb.view(n_batch_pos, -1, self.num_heads, self.head_dim)
        p = p.transpose(1, 2)  # (batch, heads, 2time-1, h_d)
        matrix_qk = torch.matmul(q, k.transpose(-2, -1))
        matrix_qp = torch.matmul(q, p.transpose(-2, -1))
        matrix_qp = self.rel_shift(matrix_qp)
        attn_weights = (matrix_qk + matrix_qp) * self.scaling
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)

        attn_weights = self.apply_attn_weights_mask(attn_weights, bsz, attn_mask, key_padding_mask)

        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)

        attn = self.get_attn_output(attn_weights, v, bsz, embed_dim)

        return attn


class EmformerAttention(MultiheadAttention):
    '''Emformer Attention'''

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0,
        bias=True,
        clamp_inf=False,
        chunk_inference=False,
    ):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, clamp_inf=clamp_inf)
        self.chunk_inference = chunk_inference

    def attn_weight(self, q, k):
        '''compute attntion weight'''
        tgt_len, bsz, _ = q.size()
        q = q * self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weight
        # (B * H, C + 1, L_C + C + M)
        attn_output_weights = self.bmm1(q, k.transpose(1, 2))
        return attn_output_weights

    def forward(
        self, x, s, mem_bank=None, key_padding_mask=None, attn_mask=None, left_k=None, left_v=None
    ):
        '''
        Args:
            x (Tensor): (T, B, U)
            mem_bank (Tensor): (M, B, U)
            key_padding_mask (Tensor): [x_1, x_2, ..., x_t, m_1, m_2, ... , m_m]
            attn_mask : (T+M, T+M)
        Returns:
            attn_output(Tensor): (T, B, U)
            mem_out(Tensor): (M, B, U)
        '''
        if not self.training and self.chunk_inference:
            return self.inference(x, s, left_k, left_v, mem_bank, key_padding_mask=None)
        # (T, B, U)
        tgt_len, bsz, embed_dim = x.size()
        # M = ceil(T / C) - 1
        # (M, B, U)
        s = s[:-1, :, :]
        if mem_bank is None:
            mem_bank = s
        mem_len = s.size(0)
        assert mem_len == mem_bank.size(0)
        attn_len = tgt_len + mem_len
        q = F.linear(
            torch.cat((x, s), 0),
            self.in_proj.weight[: self.embed_dim, :],
            self.in_proj.bias[: self.embed_dim],
        )  # (T+M, B, U)
        k = F.linear(
            torch.cat((x, mem_bank), 0),
            self.in_proj.weight[self.embed_dim : 2 * self.embed_dim, :],
            self.in_proj.bias[self.embed_dim : 2 * self.embed_dim],
        )  # (T+M, B, U)
        v = F.linear(
            torch.cat((x, mem_bank), 0),
            self.in_proj.weight[2 * self.embed_dim :, :],
            self.in_proj.bias[2 * self.embed_dim :],
        )  # (T+M, B, U)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn_output_weights = self.attn_weight(q, k)
        attn_output_weights = self.apply_attn_weights_mask(
            attn_output_weights, bsz, attn_mask, key_padding_mask
        )
        attn_weights_float = F.softmax(attn_output_weights, dim=-1, dtype=torch.float32)
        attn_output_weights = attn_weights_float.type_as(attn_output_weights)

        attn_output = self.get_attn_output(attn_output_weights, v, bsz, embed_dim)
        assert attn_output.size(0) == attn_len
        # (T, B, D)
        attn_output, mem_out = attn_output.split([tgt_len, mem_len], dim=0)
        attn_output = self.out_proj(attn_output)
        return attn_output, mem_out

    def inference(self, x, s, left_k, left_v, mem_bank, key_padding_mask=None, attn_mask=None):
        '''chunk inference
        Args:
            x: (C, B, U)
            left_k: (L_C, B, U)
            left_v: (L_C, B, U)
            mem_bank: (M, B, U)
        Returns:
            attn_output: (C, B, U)
            next_k: (L_C, B, U)
            next_v: (L_C, B, U)
            next_m: (1, B, U)
        '''
        csz, bsz, _ = x.size()
        q = F.linear(
            torch.cat((x, s), 0),
            self.in_proj.weight[: self.embed_dim, :],
            self.in_proj.bias[: self.embed_dim],
        )  # (C + 1 * mem_rate)
        if mem_bank is None:
            k, v = x, x
        else:
            k = torch.cat((x, mem_bank), 0)
            v = torch.cat((x, mem_bank), 0)
        k = F.linear(
            k,
            self.in_proj.weight[self.embed_dim : 2 * self.embed_dim, :],
            self.in_proj.bias[self.embed_dim : 2 * self.embed_dim],
        )
        v = F.linear(
            v, self.in_proj.weight[2 * self.embed_dim :, :], self.in_proj.bias[2 * self.embed_dim :]
        )
        next_k = k[:csz, :, :]
        next_v = v[:csz, :, :]
        if left_k is not None and left_v is not None:
            k = torch.cat((left_k, k), 0)  # (L_C + C + M)
            v = torch.cat((left_v, v), 0)
        attn_mask = None
        # attention output
        attn_output_weights = self.attn_weight(q, k)
        attn_output_weights = self.apply_attn_weights_mask(
            attn_output_weights, bsz, attn_mask, key_padding_mask
        )
        attn_weights_float = F.softmax(attn_output_weights, dim=-1, dtype=torch.float32)
        attn_output_weights = attn_weights_float.type_as(attn_output_weights)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn_output = self.get_attn_output(attn_output_weights, v, bsz, self.embed_dim)
        next_m = attn_output[csz:, :, :]
        next_out = attn_output[:csz, :, :]
        next_out = self.out_proj(next_out)
        assert next_m.size(0) == len(s)
        assert next_out.size(0) == csz
        return next_out, next_k, next_v, next_m


class MemoryMaskMultiheadAttention(MultiheadAttention):
    """MultiheadAttention with memory mask"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, mask_topology=None, merge_qkv=True):
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, merge_qkv=merge_qkv
        )
        self.multi_stream_mask = False
        self.stream_mask_index = 0
        (
            self.self_att_memory_mask,
            self.left_kernel_size,
            self.right_kernel_size,
        ) = self.regist_memory_mask(mask_topology)
        self.stream_mode = False
        self.stream_self_att_memory_mask = None
        self.stream_left_kernel_size = None
        self.stream_right_kernel_size = None

    def init_dual_mode_stream_mask(self, stream_mask_topology):
        '''init_dual_mode_stream_mask'''
        (
            self.stream_self_att_memory_mask,
            self.stream_left_kernel_size,
            self.stream_right_kernel_size,
        ) = self.regist_stream_memory_mask(stream_mask_topology)

    def set_stream_mode(self, stream_mode=True):
        '''set_stream_mode'''
        self.stream_mode = stream_mode

    def set_stream_index(self, stream_index=0):
        '''set_stream_mode'''
        self.stream_mask_index = stream_index

    def compatible_load_hook(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        '''
        compatible for load MultiheadAttention.
        '''
        if '{}linear_q.weight'.format(prefix) in state_dict:
            qw = state_dict.pop(prefix + 'linear_q.weight')
            kw = state_dict.pop(prefix + 'linear_k.weight')
            vw = state_dict.pop(prefix + 'linear_v.weight')
            qb = state_dict.pop(prefix + 'linear_q.bias')
            kb = state_dict.pop(prefix + 'linear_k.bias')
            vb = state_dict.pop(prefix + 'linear_v.bias')
            in_w = torch.cat([qw, kw, vw], dim=0)
            in_b = torch.cat([qb, kb, vb], dim=0)
            state_dict[prefix + 'in_proj.weight'] = in_w
            state_dict[prefix + 'in_proj.bias'] = in_b
        if '{}linear_out.weight'.format(prefix) in state_dict:
            state_dict[prefix + 'out_proj.weight'] = state_dict.pop(prefix + 'linear_out.weight')
            state_dict[prefix + 'out_proj.bias'] = state_dict.pop(prefix + 'linear_out.bias')
        if self.merge_qkv and (prefix + 'in_proj_weight') in state_dict:
            weight = state_dict.pop(prefix + 'in_proj_weight')
            state_dict[prefix + 'in_proj.weight'] = weight
            if self.bias:
                bias = state_dict.pop(prefix + 'in_proj_bias')
                state_dict[prefix + 'in_proj.bias'] = bias
        if not self.merge_qkv and (prefix + 'in_proj_weight') in state_dict:
            weight = state_dict.pop(prefix + 'in_proj_weight')
            state_dict[prefix + 'q_proj.weight'] = weight[0 : self.embed_dim, :]
            state_dict[prefix + 'k_proj.weight'] = weight[self.embed_dim : 2 * self.embed_dim, :]
            state_dict[prefix + 'v_proj.weight'] = weight[
                2 * self.embed_dim : 3 * self.embed_dim, :
            ]
            if self.bias:
                bias = state_dict.pop(prefix + 'in_proj_bias')
                state_dict[prefix + 'q_proj.bias'] = bias[0 : self.embed_dim]
                state_dict[prefix + 'k_proj.bias'] = bias[self.embed_dim : 2 * self.embed_dim]
                state_dict[prefix + 'v_proj.bias'] = bias[2 * self.embed_dim : 3 * self.embed_dim]
        if prefix + 'out_proj_weight' in state_dict:
            weight = state_dict.pop(prefix + 'out_proj_weight')
            state_dict[prefix + 'out_proj.weight'] = weight
            if self.bias:
                bias = state_dict.pop(prefix + 'out_proj_bias')
                state_dict[prefix + 'out_proj.bias'] = bias

    @staticmethod
    def regist_memory_mask(mask_topology):
        '''regist_memory_mask'''
        max_length = 3000
        temp_mask = torch.ones((max_length, max_length))
        if mask_topology is not None:
            left_kernel_size, right_kernel_size = mask_topology
            if left_kernel_size >= 0:
                left_temp_mask = torch.triu(temp_mask, diagonal=-left_kernel_size)
            else:
                left_temp_mask = temp_mask
            if right_kernel_size >= 0:
                right_temp_mask = torch.tril(temp_mask, diagonal=right_kernel_size)
            else:
                right_temp_mask = temp_mask
            memory_mask = (left_temp_mask * right_temp_mask).bool()
        else:
            left_kernel_size, right_kernel_size = max_length // 2, max_length // 2
            memory_mask = temp_mask.bool()
        return memory_mask, left_kernel_size, right_kernel_size

    def regist_stream_memory_mask(self, mask_topology):
        '''regist_stream_memory_mask'''
        if not isinstance(mask_topology[0], list):
            return self.regist_memory_mask(mask_topology)
        # multi mask for multi latency
        self.multi_stream_mask = True
        memory_mask, left_kernel_size, right_kernel_size = [], [], []
        for mt in mask_topology:
            mm, lkz, rkz = self.regist_memory_mask(mt)
            memory_mask.append(mm)
            left_kernel_size.append(lkz)
            right_kernel_size.append(rkz)
        return memory_mask, left_kernel_size, right_kernel_size

    def forward_qkv(self, query, key=None, value=None):
        """Transform query, key and value.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :return torch.Tensor transformed query, key and value

        """
        n_batch = query.size(0)
        q = self.in_proj_q(query).view(n_batch, -1, self.num_heads, self.head_dim)
        k = self.in_proj_k(key).view(n_batch, -1, self.num_heads, self.head_dim)
        v = self.in_proj_v(value).view(n_batch, -1, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    @staticmethod
    @torch.jit.script
    def jit_memory_mask_slice(memory_mask, attn_weights):
        """jit memory_mask slice"""
        tgt_len = attn_weights.size(2)
        src_len = attn_weights.size(3)
        return memory_mask[0:tgt_len, 0:src_len]

    def get_self_att_memory_mask(self):
        '''get self_att_memory_mask'''
        if self.stream_mode and self.stream_self_att_memory_mask is not None:
            if self.multi_stream_mask:
                self_att_memory_mask = self.stream_self_att_memory_mask[self.stream_mask_index]
            else:
                self_att_memory_mask = self.stream_self_att_memory_mask
        else:
            self_att_memory_mask = self.self_att_memory_mask
        return self_att_memory_mask

    def apply_attn_weights_mask(self, attn_weights, do_attn_mask=True, key_padding_mask=None):
        if do_attn_mask:
            self_att_memory_mask = self.get_self_att_memory_mask().to(attn_weights.device)
            _, _, tgt_len, src_len = attn_weights.size()
            if self.training:
                self_attn_memory_mask = self_att_memory_mask[0:tgt_len, 0:src_len]
            else:
                self_attn_memory_mask = self.jit_memory_mask_slice(
                    self_att_memory_mask, attn_weights
                )
            attn_weights = attn_weights.masked_fill(
                ~self_attn_memory_mask.bool().unsqueeze(0).unsqueeze(0), float('-inf')
            )
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(mask.squeeze(1).unsqueeze(-1), 0.0)
        return attn_weights

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        do_attn_mask=True,
        need_weights=False,
        **kwargs,
    ):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, 1, time2) or (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attention output (batch, time1, d_model)
        """
        bsz, tgt_len, _ = query.size()
        src_len = key.size(1)
        q, k, v = self.forward_qkv(query, key, value)
        attn_weights = kwargs.get('attn_weights', None)
        if attn_weights is None:
            q = q.reshape(-1, q.size(-2), q.size(-1))
            k = k.reshape(-1, k.size(-2), k.size(-1))

            attn_weights = self.bmm1(q * self.scaling, k.transpose(1, 2))
            _, tgt_len, src_len = attn_weights.size()
            attn_weights = attn_weights.reshape(bsz, -1, tgt_len, src_len)
            attn_weights = self.apply_attn_weights_mask(
                attn_weights,
                do_attn_mask=do_attn_mask,
                key_padding_mask=key_padding_mask,
            )
            attn_weights = attn_weights.reshape(-1, tgt_len, src_len)
        v = v.reshape(-1, src_len, self.head_dim)
        attn = self.get_attn_output(attn_weights, v, bsz, self.embed_dim)
        if need_weights:
            return attn.transpose(0, 1), attn_weights
        return attn.transpose(0, 1), None

    @property
    def attn_left_kernel_size(self):
        """get attn_left_kernel_size with dual-mode"""
        if self.stream_mode:
            return self.stream_left_kernel_size
        return self.left_kernel_size

    @property
    def attn_right_kernel_size(self):
        """get attn_right_kernel_size with dual-mode"""
        if self.stream_mode:
            return self.stream_right_kernel_size
        return self.right_kernel_size


class MemoryMaskRelPositionMultiHeadedAttention(MemoryMaskMultiheadAttention):
    """Multi-Head Attention layer with relative position encoding.

    Paper: https://arxiv.org/abs/1901.02860

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, mask_topology=None, merge_qkv=True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            mask_topology=mask_topology,
            merge_qkv=merge_qkv,
        )
        # linear transformation for positional ecoding
        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
        self.pos_bmm = BatchedMatMul()

    @staticmethod
    def rel_shift(x, zero_triu=False):
        """Compute relative positinal encoding.

        :param torch.Tensor x: (batch, time, size)
        :param bool zero_triu: return the lower triangular part of the matrix
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view(x.size())

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    @staticmethod
    @torch.jit.script
    def jit_bd_slice(bd, ac):
        """jit for bd slice"""
        return bd[:, :, :, : ac.size(3)]

    def forward(
        self,
        query,
        key,
        value,
        pos_emb,
        key_padding_mask=None,
        do_attn_mask=True,
        need_weights=False,
        **kwargs,
    ):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor pos_emb: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attention output  (batch, time1, d_model)
        """
        bsz, tgt_len, _ = query.size()
        src_len = key.size(1)
        q, k, v = self.forward_qkv(query, key, value)
        attn_weights = kwargs.get('attn_weights', None)
        if attn_weights is None:
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.num_heads, self.head_dim)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)

            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            with AmpEnable(enabled=False):
                # compute attention score
                # first compute matrix a and matrix c
                # as described in https://arxiv.org/abs/1901.02860 Section 3.3
                # (batch, head, time1, time2)
                matrix_ac = self.bmm1(q_with_bias_u, k.transpose(-2, -1).float())

                # compute matrix b and matrix d
                # (batch, head, time1, time2)
                matrix_bd = self.pos_bmm(q_with_bias_v, p.transpose(-2, -1).float())
                matrix_bd = self.rel_shift(matrix_bd)
                if self.training:
                    matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(3)]
                else:
                    matrix_bd = self.jit_bd_slice(matrix_bd, matrix_ac)

                attn_weights = (matrix_ac + matrix_bd) / math.sqrt(
                    self.head_dim
                )  # (batch, head, time1, time2)

                attn_weights = self.apply_attn_weights_mask(
                    attn_weights,
                    do_attn_mask=do_attn_mask,
                    key_padding_mask=key_padding_mask,
                )
                attn_weights = attn_weights.view(-1, tgt_len, src_len)
        v = v.reshape(-1, src_len, self.head_dim)
        attn = self.get_attn_output(attn_weights, v, bsz, self.embed_dim)
        if need_weights:
            return attn.transpose(0, 1), attn_weights
        return attn.transpose(0, 1), None

    def forward_step(
        self, x, cache, mask_cache, pos_emb, total_right_context, key_padding_mask=None
    ):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor pos_emb: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attention output  (batch, time1, d_model)
        """
        # pe_max_len = int((pos_emb.size(1) + 1) / 2)
        # if cache is None:
        #     cache = x
        #     query, key, value = x, x, x
        #     pos_emb = pos_emb[:, pe_max_len - key.size(1) : pe_max_len + key.size(1) - 1]
        # else:
        assert cache.size(0) == x.size(0)
        assert cache.size(2) == x.size(2)
        assert cache.size(1) == self.left_kernel_size
        # assert pe_max_len >= cache.size(1)
        key_padding_mask = torch.cat([mask_cache, key_padding_mask], dim=2)
        cache = torch.cat([cache, x], dim=1)
        query, key, value = x, cache, cache
        # pos_emb = pos_emb[:, pe_max_len - key.size(1) : pe_max_len + key.size(1) - 1]

        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.num_heads, self.head_dim)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = self.bmm1(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = self.pos_bmm(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        matrix_bd = self.jit_bd_slice(matrix_bd, matrix_ac)

        attn_weights = (matrix_ac + matrix_bd) / math.sqrt(
            self.head_dim
        )  # (batch, head, time1, time2)
        # attn_weights = (matrix_ac + matrix_bd) / math.sqrt(self.head_dim)
        attn_weights = self.apply_attn_mask_forward_step(
            attn_weights, key_padding_mask=key_padding_mask
        )
        # apply attention mask

        bsz, _, tgt_len, src_len = attn_weights.size()
        attn_weights = attn_weights.view(-1, tgt_len, src_len)
        v = v.reshape(-1, src_len, self.head_dim)
        attn = self.get_attn_output(attn_weights, v, bsz, self.embed_dim).transpose(0, 1)
        if total_right_context > 0:
            cache = cache[:, -self.left_kernel_size - total_right_context : -total_right_context, :]
            mask_cache = key_padding_mask[
                :, :, -self.left_kernel_size - total_right_context : -total_right_context
            ]
        else:
            cache = cache[:, -self.left_kernel_size :, :]
            mask_cache = key_padding_mask[:, :, -self.left_kernel_size :]
        return attn, cache, mask_cache

    def apply_attn_mask_forward_step(self, attn_weights, key_padding_mask=None):
        '''apply_attn_weights_mask_forward_step'''
        _, _, tgt_len, src_len = attn_weights.size()
        assert tgt_len <= src_len

        self_att_memory_mask = self.get_self_att_memory_mask().to(attn_weights.device)
        self_att_memory_mask = self_att_memory_mask[src_len - tgt_len : src_len, 0:src_len]

        attn_weights = attn_weights.masked_fill(
            ~self_att_memory_mask.bool().unsqueeze(0).unsqueeze(0), float('-inf')
        )
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

        return attn_weights


class RelPartialLearnableMultiHeadAttn(nn.Module):
    '''RelPartialLearnableMultiHeadAttn'''

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):
        '''constructor'''
        super(__class__, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    @staticmethod
    def _rel_shift(x, zero_triu=False):
        '''rel shift'''
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view(x.size())

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    @staticmethod
    @torch.jit.script
    def apply_attn_mask(attn_score: torch.Tensor, attn_mask: torch.Tensor):
        """jit attn mask"""
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[None, :, :, None] > 0, -float('inf'))
                    .type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[:, :, :, None] > 0, -float('inf'))
                    .type_as(attn_score)
                )
        return attn_score

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        '''forward'''
        # pylint:disable=too-many-branches
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        # qlen x bsz x n_head x d_head
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        # qlen x n_head x d_head
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        #### compute attention score
        # qlen x bsz x n_head x d_head
        rw_head_q = w_head_q + r_w_bias.type_as(w_head_q)
        # AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        # qlen x klen x bsz x n_head
        ac = torch.matmul(rw_head_q.permute(1, 2, 0, 3), w_head_k.permute(1, 2, 3, 0))
        ac = ac.permute(2, 3, 0, 1)

        rr_head_q = w_head_q + r_r_bias.type_as(w_head_q)
        # BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        # qlen x klen x bsz x n_head
        bd = torch.matmul(rr_head_q.permute(1, 2, 0, 3), r_head_k.permute(1, 2, 0))
        bd = bd.permute(2, 3, 0, 1)
        bd = self._rel_shift(bd)

        # [qlen x klen x bsz x n_head]
        attn_score = ac + bd
        attn_score.mul_(self.scale)

        #### compute attention probability
        if self.training:
            if attn_mask is not None and attn_mask.any().item():
                if attn_mask.dim() == 2:
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attn_mask[None, :, :, None].bool(), -float('inf'))
                        .type_as(attn_score)
                    )
                elif attn_mask.dim() == 3:
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attn_mask[:, :, :, None].bool(), -float('inf'))
                        .type_as(attn_score)
                    )
        else:
            attn_score = self.apply_attn_mask(attn_score, attn_mask)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1).type_as(attn_score)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        # attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = (attn_prob.unsqueeze(-1) * w_head_v.unsqueeze(0)).sum(1)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class MultiHeadChunkwiseAttention(nn.Module):
    '''MultiheadAttention that support chunk hoping'''

    def __init__(
        self,
        hidden_size,
        num_heads,
        total_key_depth,
        total_value_depth,
        dropout_rate,
        attention_type,
        num_history_chunk,
        use_layer_norm=False,
        pre_layer_norm=False,
        layer_norm_eps=1e-5,
        extra_dropout=True,
        fused=True,
    ):
        super().__init__()
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        assert (
            total_key_depth % num_heads == 0
        ), "Key depth {} must be divisible by the number of attention heads {}".format(
            total_key_depth, num_heads
        )
        assert (
            total_value_depth % num_heads == 0
        ), "Value depth {} must be divisible by the number of attention heads {}".format(
            total_value_depth, num_heads
        )

        self.q_proj = nn.Linear(hidden_size, total_key_depth, bias=False)
        self.k_proj = nn.Linear(hidden_size, total_key_depth, bias=False)
        self.v_proj = nn.Linear(hidden_size, total_value_depth, bias=False)
        self.atten_dropout = nn.Dropout(dropout_rate)
        # out net
        out_net = [nn.Linear(total_value_depth, hidden_size, bias=False)]
        if isinstance(extra_dropout, float):
            extra_dropout_rate = extra_dropout
        elif extra_dropout:
            extra_dropout_rate = dropout_rate
        else:
            extra_dropout_rate = 0.0
        if extra_dropout_rate != 0.0:
            out_net.append(nn.Dropout(extra_dropout_rate))

        if fused:
            self.fused_cfg = {
                'norm_eps': layer_norm_eps,
                'score_drop_rate': dropout_rate,
                'output_drop_rate': extra_dropout_rate,
                'num_heads': num_heads,
                'norm_before': pre_layer_norm,
                'chunk_len': 0,
                'chunk_beg': 0,
            }

        self.out_net = nn.Sequential(*out_net)
        key_depth_per_head = total_key_depth // num_heads
        self.q_scale = key_depth_per_head**-0.5
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.num_history_chunk = num_history_chunk
        self.use_layer_norm = use_layer_norm
        self.pre_layer_norm = pre_layer_norm
        self.fused = fused
        self.bmm1 = BatchedMatMul()
        self.bmm2 = BatchedMatMul()

    def split_heads(self, head):
        '''
        Split channels (dimension 2) into multiple heads (becomes dimension 1)

        Args:
            head: a Tensor with shape [batch, length, channels]

        Returns:
            a Tensor with shape [batch, num_heads, length, channels / num_heads]
        '''
        bsize, slen, hdim = head.shape
        head = head.reshape(bsize, slen, self.num_heads, hdim // self.num_heads)
        return head.permute(0, 2, 1, 3)

    def _forward_fused(self, inputs, bias, cache, chunk_cache):
        '''forward'''
        params = [
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.out_net[0].weight,
            self.layer_norm.weight,
            self.layer_norm.bias,
        ]
        if chunk_cache is not None:
            aug_length = chunk_cache['len']
            if self.num_history_chunk:
                start_pos = aug_length // self.num_history_chunk
            else:
                start_pos = aug_length
            self.fused_cfg['chunk_len'] = aug_length
            self.fused_cfg['chunk_beg'] = start_pos
        expand_bias = bias.expand(inputs.shape[0], -1, inputs.shape[1], -1)
        # pylint: disable=no-member
        return multi_heads_chunk_wise_attention(
            inputs, expand_bias, cache, chunk_cache, params, self.fused_cfg
        )

    def forward(self, inputs, bias, cache, chunk_cache):
        '''forward'''
        # pylint:disable=too-many-branches
        if self.use_layer_norm and self.training and self.fused:
            return self._forward_fused(inputs, bias, cache, chunk_cache)

        if self.use_layer_norm and self.pre_layer_norm:
            atten_inp = self.layer_norm(inputs)
        else:
            atten_inp = inputs
        q = self.q_proj(atten_inp)
        k = self.k_proj(atten_inp)
        v = self.v_proj(atten_inp)

        if cache is not None:
            if self.attention_type != 'dot_product':
                raise NotImplementedError(
                    "Caching is not guaranteed to work with attention types other than"
                    " dot_product"
                )
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring for details.")
            if cache.get('k', None) is None:
                cache['k'] = k
            else:
                k = cache['k'] = torch.cat([cache['k'], k], dim=1)

            if cache.get('v', None) is None:
                cache['v'] = v
            else:
                v = cache['v'] = torch.cat([cache['v'], v], dim=1)

        q = self.split_heads(q) * self.q_scale
        k = self.split_heads(k)
        v = self.split_heads(v)

        if self.attention_type == 'dot_product':
            if chunk_cache is not None:
                aug_length = chunk_cache['len']
                org_k = k
                org_v = v

                if chunk_cache.get('k', None) is None:
                    k = F.pad(k, (0, 0, aug_length, 0))
                else:
                    k = torch.cat([chunk_cache['k'], k], dim=2)

                if chunk_cache.get('v', None) is None:
                    v = F.pad(v, (0, 0, aug_length, 0))
                else:
                    v = torch.cat([chunk_cache['v'], v], dim=2)

                if self.num_history_chunk:
                    start_pos = aug_length // self.num_history_chunk
                    end_pos = start_pos + aug_length
                    chunk_cache['k'] = k[:, :, start_pos:end_pos]
                    chunk_cache['v'] = v[:, :, start_pos:end_pos]
                else:
                    chunk_cache['k'] = org_k[:, :, :aug_length]
                    chunk_cache['v'] = org_v[:, :, :aug_length]

            logits = self.bmm1(q, k.transpose(2, 3))
            if bias is not None:
                logits += bias
            weights = F.softmax(logits, dim=-1)
            weights = self.atten_dropout(weights)
            outputs = self.bmm2(weights, v)
        else:
            raise NotImplementedError('{} attention not support yet'.format(self.attention_type))

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        outputs = outputs.reshape((outputs.shape[0], outputs.shape[1], -1))
        outputs = self.out_net(outputs) + inputs
        if self.use_layer_norm and not self.pre_layer_norm:
            outputs = self.layer_norm(outputs)
        return outputs, cache, chunk_cache
