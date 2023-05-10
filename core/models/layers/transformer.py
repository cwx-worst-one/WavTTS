''' Transformer '''
import torch
from torch import nn
import torch.nn.functional as F
from core.extensions import fused_transformer, TpTransformerFunc, get_amp_level, mpu, pnn
from core.models import utils
from core.models.layers.feed_forward import PositionwiseFeedForward, MoEFFN
from core.models.layers.adapter import *
from core.models.layers.multi_head_attn import (
    MultiheadAttention,
    PosMultiHeadAttention,
    LiRelMultiHeadAttention,
    EmformerAttention,
    MultiHeadChunkwiseAttention,
)
from core.models.layers.normalization import LayerNorm
from core.utils import get_rank, get_communicator
from core.utils.dict import FalconDict
from .active_function import get_activation_fn


class BiTransformerLayer(nn.Module):
    '''BiTransformerLayer'''

    def __init__(
        self,
        embed_dim,
        attention_heads,
        ffn_embed_dim,
        attention_dropout,
        hidden_dropout,
        activation_dropout=0.0,
        activation='relu',
        normalize_before=True,
        clamp_inf=False,
        squeeze_mem=False,
        layernorm_eps=1e-5,
        act_glu=False,
        moe_args=None,
        lnum=None,
        adapter_mode='mode',
        adapter_type='none',
        adapter_embed_dim=-1,
        **_kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clamp_inf = clamp_inf
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
            clamp_inf=self.clamp_inf,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=layernorm_eps)
        self.dropout = hidden_dropout
        self.activation_dropout = activation_dropout
        self.normalize_before = normalize_before
        self.act_glu = act_glu
        self.moe_args = moe_args
        self.moe_ffn = False
        self.layer_num = lnum
        self.init_ffn(ffn_embed_dim, activation)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=layernorm_eps)
        self.aux_loss = 0.0
        self.init_adapter_ffn(adapter_type, adapter_embed_dim)
        if adapter_mode.endswith('_attn'):
            self.init_adapter_attn(adapter_type, adapter_embed_dim)
        self.adapter_mode = adapter_mode

        self._cfg = dict(
            embed_dim=embed_dim,
            attn_heads=attention_heads,
            ffn_hidden=ffn_embed_dim,
            attn_drop=attention_dropout,
            hidden_drop=hidden_dropout,
            act_drop=self.activation_dropout,
            eps=layernorm_eps,
            activation=activation,
            normalize_before=normalize_before,
            clamp_inf=clamp_inf,
            squeeze_mem=squeeze_mem,
        )

    def init_ffn(self, ffn_embed_dim, activation):
        '''init ffn'''
        moe_layers = []
        if self.moe_args is not None:
            moe_layers = eval(self.moe_args.get("moe_layers", "[]"))
        if self.moe_args is None or self.layer_num not in moe_layers:
            self.init_dense_ffn(ffn_embed_dim, activation)
        else:
            self.init_moe_ffn(ffn_embed_dim)

    def init_dense_ffn(self, ffn_embed_dim, activation):
        '''init dense ffn as usual'''
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim * (2 if self.act_glu else 1))
        self.activation_fn = get_activation_fn(activation=activation, act_glu=self.act_glu)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        utils.xavier_init(self.fc1)
        utils.xavier_init(self.fc2)

    def init_moe_ffn(self, ffn_embed_dim):
        '''init MoE FFN'''
        assert self.act_glu is False
        self.ffn = MoEFFN(
            self.embed_dim,
            ffn_embed_dim,
            dropout_rate=self.activation_dropout,
            activation_fn=self.moe_args.get('activation', 'relu'),
            num_expert=self.moe_args.get('moe_expert', 8),
            topk=self.moe_args.get('moe_topk', 2),
            output_dropout_prob=self.moe_args.get('output_dropout_prob', 0.0),
            moe_loss_scale=self.moe_args.get('moe_loss_scale', 0.0),
            z_loss_scale=self.moe_args.get('z_loss_scale', 0.0),
            noisy_gate_policy=self.moe_args.get('noisy_gate_policy', None),
            use_lego=self.moe_args.get('use_lego', False),
        )
        self.moe_ffn = True

    def init_adapter_ffn(self, adapter_type, adapter_embed_dim):
        if adapter_type != 'none':
            self.adapter_ffn = eval(adapter_type)(self.embed_dim, adapter_embed_dim)
        else:
            self.adapter_ffn = None

    def init_adapter_attn(self, adapter_type, adapter_embed_dim):
        if adapter_type != 'none':
            self.adapter_attn = eval(adapter_type)(self.embed_dim, adapter_embed_dim)
        else:
            self.adapter_attn = None

    def forward(
        self,
        x,
        encoder_padding_mask=None,
        attn_mask=None,
        output_layer_result=False,
        fused=False,
        eval_fused=False,
        batch_first=False,
        **_kwargs,
    ):
        """forward"""
        # pylint: disable=too-many-boolean-expressions
        if (
            ((self.training and fused) or (not self.training and eval_fused))
            and not self.act_glu
            and not self.moe_ffn
        ):
            return pnn.fused_transformer(
                [
                    self.self_attn.in_proj.weight,
                    self.self_attn.in_proj.bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.self_attn_layer_norm.weight,
                    self.self_attn_layer_norm.bias,
                    self.fc1.weight,
                    self.fc1.bias,
                    self.fc2.weight,
                    self.fc2.bias,
                    self.final_layer_norm.weight,
                    self.final_layer_norm.bias,
                ],
                x,
                encoder_padding_mask,
                attn_mask,
                batch_first=batch_first,
                output_layer_result=output_layer_result,
                training=self.training,
                **self._cfg,
            )
        # B x T x H -> T x B x H
        x = x.transpose(0, 1) if batch_first else x

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if self.adapter_mode == 'parallel_with_attn' and self.adapter_attn is not None:
            x_ = self.adapter_attn(x)
        x, _ = self.self_attn(x, key_padding_mask=encoder_padding_mask, attn_mask=attn_mask)
        if self.adapter_mode == 'sequential_with_attn' and self.adapter_attn is not None:
            x_ = self.adapter_attn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        if self.adapter_mode.startswith('parallel') and self.adapter_ffn is not None:
            x_ = self.adapter_ffn(x)
        x = self.ffn_forward(x)
        if self.adapter_mode.startswith('sequential') and self.adapter_ffn is not None:
            x_ = self.adapter_ffn(x)
        layer_result = x
        if self.training and self.clamp_inf:
            x = torch.clamp(x, -4e4, 4e4)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.adapter_ffn is not None:
            x = x_ + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        # T x B x H -> B x T x H
        x = x.transpose(0, 1) if batch_first else x
        layer_result = layer_result.transpose(0, 1) if batch_first else layer_result
        return (x, None, layer_result) if output_layer_result else x

    def ffn_forward(self, x):
        '''forward of FFN'''
        if self.moe_ffn:
            x = self.ffn(x)
            self.aux_loss = self.ffn.aux_loss
            return x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        '''maybe_layer_norm'''
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        return x


class ParallelTransformer(torch.nn.Module):
    '''
    Tensor Parallel or Model Parallel Transformer.
    see this url for more info.
    https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21496-megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism.pdf
    '''

    def __init__(
        self,
        embed_dim,
        attention_heads,
        ffn_embed_dim,
        attention_dropout,
        hidden_dropout,
        activation_dropout=0.0,
        activation='relu',
        normalize_before=True,
        clamp_inf=False,
        layernorm_eps=1e-5,
        tensor_parallel=8,
        distribute_saved_activations=False,
        **_kwargs,
    ):
        super().__init__()
        # assert tensor_parallel > 1
        assert attention_heads % tensor_parallel == 0 and ffn_embed_dim % tensor_parallel == 0

        head_dim = embed_dim // attention_heads
        local_head_num = attention_heads // tensor_parallel
        local_proj_dim = local_head_num * head_dim
        local_ffn_dim = ffn_embed_dim // tensor_parallel
        self.tensor_parallel = tensor_parallel
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.ffn_embed_dim = ffn_embed_dim

        self.self_attn_in_proj_weight = torch.nn.Parameter(
            torch.empty(3 * local_proj_dim, embed_dim)
        )
        self.self_attn_in_proj_bias = torch.nn.Parameter(torch.empty(3 * local_proj_dim))
        self.self_attn_out_proj_weight = torch.nn.Parameter(torch.empty(embed_dim, local_proj_dim))
        self.self_attn_out_proj_bias = torch.nn.Parameter(torch.empty(embed_dim))
        self.fc1_weight = torch.nn.Parameter(torch.empty(local_ffn_dim, embed_dim))
        self.fc1_bias = torch.nn.Parameter(torch.empty(local_ffn_dim))
        self.fc2_weight = torch.nn.Parameter(torch.empty(embed_dim, local_ffn_dim))
        self.fc2_bias = torch.nn.Parameter(torch.empty(embed_dim))
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim, eps=layernorm_eps)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=layernorm_eps)

        # set tp flag for subsequent special processing
        self.self_attn_in_proj_weight.__tp__ = True
        self.self_attn_in_proj_bias.__tp__ = True
        self.self_attn_out_proj_weight.__tp__ = True
        self.fc1_weight.__tp__ = True
        self.fc1_bias.__tp__ = True
        self.fc2_weight.__tp__ = True

        self.reset_parameters()
        self._register_load_state_dict_pre_hook(self.compatible_load_hook)
        self._cfg = dict(
            embed_dim=embed_dim,
            head_dim=head_dim,
            attn_heads=local_head_num,
            ffn_hidden=local_ffn_dim,
            attn_drop=attention_dropout,
            hidden_drop=hidden_dropout,
            act_drop=activation_dropout,
            eps=layernorm_eps,
            activation=activation,
            normalize_before=normalize_before,
            clamp_inf=clamp_inf,
            tensor_parallel=tensor_parallel,
            recompute=True,
            communicator=None,
            # inputs of TpTransformerFunc is redundance in one host machine.
            # we can use zero redundance to save global memory usage.
            distribute_saved_activations=distribute_saved_activations,
        )

    def reset_parameters(self):
        """Initialize the weights specific to the BERT Model."""
        local_group_offset = get_rank() % self.tensor_parallel
        local_head_num = self.attention_heads // self.tensor_parallel
        st = local_group_offset * local_head_num
        ed = st + local_head_num

        # attn in proj weight
        in_proj_full_weight = torch.zeros(
            3 * self.embed_dim,
            self.embed_dim,
            dtype=self.self_attn_in_proj_weight.dtype,
            device=self.self_attn_in_proj_weight.device,
        )
        in_proj_full_weight.data.normal_(mean=0.0, std=0.02)
        val = in_proj_full_weight.reshape(3, -1, self.head_dim, self.embed_dim)
        val = val[:, st:ed, :, :].reshape(-1, self.embed_dim)
        self.self_attn_in_proj_weight.data.copy_(val)

        # attn out proj weight
        out_proj_full_weight = torch.zeros(
            self.embed_dim,
            self.embed_dim,
            dtype=self.self_attn_out_proj_weight.dtype,
            device=self.self_attn_out_proj_weight.device,
        )
        out_proj_full_weight.data.normal_(mean=0.0, std=0.02)
        val = out_proj_full_weight.reshape(self.embed_dim, -1, self.head_dim)
        val = val[:, st:ed, :].reshape(self.embed_dim, -1)
        self.self_attn_out_proj_weight.data.copy_(val)

        local_ffn = self.ffn_embed_dim // self.tensor_parallel
        st = local_group_offset * local_ffn
        ed = st + local_ffn

        # fc1 weight
        fc1_full_weight = torch.zeros(
            self.ffn_embed_dim,
            self.embed_dim,
            dtype=self.fc1_weight.dtype,
            device=self.fc1_weight.device,
        )
        fc1_full_weight.data.normal_(mean=0.0, std=0.02)
        val = fc1_full_weight[st:ed, :]
        self.fc1_weight.data.copy_(val)

        # fc2 weight
        fc2_full_weight = torch.zeros(
            self.embed_dim,
            self.ffn_embed_dim,
            dtype=self.fc2_weight.dtype,
            device=self.fc2_weight.device,
        )
        fc2_full_weight.data.normal_(mean=0.0, std=0.02)
        val = fc2_full_weight[:, st:ed]
        self.fc2_weight.data.copy_(val)

        # bias
        self.self_attn_in_proj_bias.data.zero_()
        self.self_attn_out_proj_bias.data.zero_()
        self.fc1_bias.data.zero_()
        self.fc2_bias.data.zero_()

    def forward(
        self,
        x,
        encoder_padding_mask=None,
        attn_mask=None,
        output_layer_result=False,
        batch_first=False,
        layer_result_norm='',
        **_kwargs,
    ):
        '''fwd.'''
        ws = [
            self.self_attn_in_proj_weight,
            self.self_attn_in_proj_bias,
            self.self_attn_out_proj_weight,
            self.self_attn_out_proj_bias,
            self.self_attn_layer_norm.weight,
            self.self_attn_layer_norm.bias,
            self.fc1_weight,
            self.fc1_bias,
            self.fc2_weight,
            self.fc2_bias,
            self.final_layer_norm.weight,
            self.final_layer_norm.bias,
        ]
        if not batch_first:
            x = x.permute(1, 0, 2)
        self._cfg['fp16'] = get_amp_level() in ('O1', 'O2', 'O3')
        self._cfg['training'] = self.training
        self._cfg['output_layer_result'] = output_layer_result
        self._cfg['layer_result_norm'] = layer_result_norm
        if self._cfg['communicator'] is None:
            comm = get_communicator(self._cfg['tensor_parallel'])
            self._cfg['communicator'] = comm
            self._cfg['local_rank'] = get_rank() % self._cfg['tensor_parallel']
            self._cfg['comm_stream'] = torch.cuda.Stream()
            self._cfg['comm_event'] = [torch.cuda.Event() for _ in range(4)]
        y, layer_result = TpTransformerFunc.apply(
            self._cfg, x, encoder_padding_mask, attn_mask, *ws
        )
        if output_layer_result:
            return y, None, layer_result
        return y

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
        """compatible load BiTransformer layer"""
        old_keys = [
            'self_attn.in_proj.weight',
            'self_attn.in_proj.bias',
            'self_attn.out_proj.weight',
            'self_attn.out_proj.bias',
            'fc1.weight',
            'fc1.bias',
            'fc2.weight',
            'fc2.bias',
        ]
        local_group_offset = get_rank() % self.tensor_parallel
        for key in old_keys:
            old_key = prefix + key
            new_key = prefix + key.replace('.', '_')
            if old_key not in state_dict:
                continue
            val = state_dict.pop(old_key)
            if key in (
                'self_attn.in_proj.weight',
                'self_attn.in_proj.bias',
                'self_attn.out_proj.weight',
            ):
                local_head_num = self.attention_heads // self.tensor_parallel
                st = local_group_offset * local_head_num
                ed = st + local_head_num
                if key == 'self_attn.in_proj.weight':
                    val = val.reshape(3, -1, self.head_dim, self.embed_dim)
                    val = val[:, st:ed, :, :].reshape(-1, self.embed_dim)
                elif key == 'self_attn.in_proj.bias':
                    val = val.reshape(3, -1, self.head_dim)
                    val = val[:, st:ed, :].reshape(-1)
                else:  # self_attn.out_proj.weight
                    val = val.reshape(self.embed_dim, -1, self.head_dim)
                    val = val[:, st:ed, :].reshape(self.embed_dim, -1)
            elif key in ('fc1.weight', 'fc1.bias', 'fc2.weight'):
                local_ffn = self.ffn_embed_dim // self.tensor_parallel
                st = local_group_offset * local_ffn
                ed = st + local_ffn
                if key == 'fc1.weight':
                    val = val[st:ed, :]
                elif key == 'fc1.bias':
                    val = val[st:ed]
                else:  # fc2.weight
                    val = val[:, st:ed]
            state_dict[new_key] = val


class RelTransformerLayer(BiTransformerLayer):
    """Relative Position Transfomer"""

    def __init__(
        self,
        embed_dim,
        attention_heads,
        ffn_embed_dim,
        attention_dropout,
        hidden_dropout,
        activation_dropout=0.0,
        activation='relu',
        normalize_before=True,
        clamp_inf=False,
        attention_model='LiRelMultiHeadAttention',
        gru_rel_pos=False,
        **_kwargs,
    ):
        super().__init__(
            embed_dim,
            attention_heads,
            ffn_embed_dim,
            attention_dropout,
            hidden_dropout,
            activation_dropout=activation_dropout,
            activation=activation,
            normalize_before=normalize_before,
            clamp_inf=clamp_inf,
        )

        if attention_model == 'LiRelMultiHeadAttention':
            cls = LiRelMultiHeadAttention
        elif attention_model == 'PosMultiHeadAttention':
            cls = PosMultiHeadAttention
        self.self_attn = cls(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
            bias=True,
            clamp_inf=self.clamp_inf,
            gru_rel_pos=gru_rel_pos,
        )

    def forward(
        self,
        x,
        pos_emb,
        encoder_padding_mask=None,
        attn_mask=None,
        output_layer_result=False,
        batch_first=False,
        **_kwargs,
    ):
        """forward
        Args:
            x: input (T, B, D)
            pos_emb: positional embedding (B, T, T)
            encoder_padding_mask:
            attn_mask:
            output_pos_emb: If True, also output the positional embedding
        Return:
            output: the layer output
            pos_emb: the positional embedding
        """
        # B x T x H -> T x B x H
        x = x.transpose(0, 1) if batch_first else x

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x = self.self_attn(x, pos_emb, key_padding_mask=encoder_padding_mask, attn_mask=attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        layer_result = x
        if self.training and self.clamp_inf:
            x = torch.clamp(x, -4e4, 4e4)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        # T x B x H -> B x T x H
        x = x.transpose(0, 1) if batch_first else x
        layer_result = layer_result.transpose(0, 1) if batch_first else layer_result
        return (x, None, layer_result) if output_layer_result else x


class EmformerLayer(BiTransformerLayer):
    '''EmformerLayer'''

    def __init__(
        self,
        embed_dim,
        attention_heads,
        ffn_embed_dim,
        attention_dropout,
        hidden_dropout,
        activation_dropout,
        activation,
        normalize_before,
        clamp_inf,
        chunk_size,
        left_context,
        chunk_inference,
        memory_chunk_size,
    ):

        super().__init__(
            embed_dim,
            attention_heads,
            ffn_embed_dim,
            attention_dropout,
            hidden_dropout,
            activation_dropout=0.0,
            activation='relu',
            normalize_before=True,
            clamp_inf=False,
        )

        self.self_attn = EmformerAttention(
            embed_dim=self.embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            bias=True,
            clamp_inf=self.clamp_inf,
            chunk_inference=chunk_inference,
        )
        self.chunk_inference = chunk_inference
        self.chunk_size = chunk_size
        self.memory_chunk_size = memory_chunk_size
        self.chunk_memory_rate = self.chunk_size // memory_chunk_size
        self.left_context = left_context
        self.avg_pool = nn.AvgPool1d(
            kernel_size=memory_chunk_size, stride=memory_chunk_size, ceil_mode=True
        )

    def generate_summary(self, x):
        '''generate summary'''
        # x[T, B, U]
        x = x.permute(1, 2, 0)  # (B, U, T)
        output = self.avg_pool(x)
        return output.permute(2, 0, 1)  # [m1, m2, m(C-1)]

    def chunk_self_attn(self, x, mem_bank, encoder_padding_mask):
        '''chunk self attention'''
        tgt_len = x.size(0)
        if mem_bank is None:
            mem_bank = self.generate_summary(x)
        left_k = None
        left_v = None
        out_x = torch.Tensor().to(x)
        out_mem = torch.Tensor().to(x)
        chunk_size = self.chunk_size
        chunk_nums = x.size(0) // chunk_size
        tag = x.size(0) % chunk_size
        for chunk_id in range(chunk_nums):
            left_bound, right_bound = chunk_id * chunk_size, (chunk_id + 1) * chunk_size
            x_in = x[left_bound:right_bound, :, :]
            mem_in = mem_bank[: chunk_id * self.chunk_memory_rate, :, :]
            # left + chunk + membank
            acoustic_mask = encoder_padding_mask[
                :, max(0, left_bound - self.left_context) : right_bound
            ]
            membank_mask = encoder_padding_mask[
                :, tgt_len : tgt_len + chunk_id * self.chunk_memory_rate
            ]
            padding_mask = torch.cat((acoustic_mask, membank_mask), 1)
            s = self.generate_summary(x_in)
            if self.left_context:
                next_out, left_k, left_v, next_m = self.self_attn.inference(
                    x_in,
                    s,
                    left_k=left_k,
                    left_v=left_v,
                    mem_bank=mem_in,
                    key_padding_mask=padding_mask,
                )
            else:
                next_out, _, _, next_m = self.self_attn.inference(
                    x_in,
                    s,
                    left_k=left_k,
                    left_v=left_v,
                    mem_bank=mem_in,
                    key_padding_mask=padding_mask,
                )
            out_x = torch.cat((out_x, next_out), 0)
            out_mem = torch.cat((out_mem, next_m), 0)
        if tag:
            x_in = x[right_bound:, :, :]
            mem_in = mem_bank[: (chunk_id + 1) * self.chunk_memory_rate, :, :]
            tail = x_in.size(0) // self.memory_chunk_size
            if tail != 0:
                membank_mask = encoder_padding_mask[:, tgt_len:-tail]
            else:
                membank_mask = encoder_padding_mask[:, tgt_len:]
            acoustic_mask = encoder_padding_mask[
                :, max(0, right_bound - self.left_context) : tgt_len
            ]
            padding_mask = torch.cat((acoustic_mask, membank_mask), 1)
            s = self.generate_summary(x_in)
            next_out, _, _, next_m = self.self_attn.inference(
                x_in,
                s,
                left_k=left_k,
                left_v=left_v,
                mem_bank=mem_in,
                key_padding_mask=padding_mask,
            )
            out_x = torch.cat((out_x, next_out), 0)
        return out_x, out_mem

    def forward(self, x, mem_bank, encoder_padding_mask, attn_mask):
        # math.ceil(self.max_length / self.memory_chunk_size) - 1
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if self.chunk_inference:
            x, mem_bank = self.chunk_self_attn(x, mem_bank, encoder_padding_mask)
        else:
            s = self.generate_summary(x)
            x, mem_bank = self.self_attn(
                x, s, mem_bank, key_padding_mask=encoder_padding_mask, attn_mask=attn_mask
            )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        if self.training and self.clamp_inf:
            x = torch.clamp(x, -4e4, 4e4)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, mem_bank


class TransformerEncoderLayer(BiTransformerLayer):
    """Encoder layer block."""


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.fused_transformer = args.get('fused_transformer', True)
        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not self.cross_self_attention,
        )
        self.act_glu = getattr(args, 'decoder_act_glu', False)
        self.dropout = args.dropout
        self.activation_fn = get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu'), act_glu=self.act_glu
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, 'encoder_embed_dim', None),
                vdim=getattr(args, 'encoder_embed_dim', None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(
            self.embed_dim, args.decoder_ffn_embed_dim * (2 if self.act_glu else 1)
        )
        self.fc2 = nn.Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        utils.xavier_init(self.fc1)
        utils.xavier_init(self.fc2)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        '''prepare_for_onnx_export_'''
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_attn=False,
        need_head_weights=False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        if self.cross_self_attention and not (
            incremental_state is not None
            and "prev_key" in self.self_attn.get_input_buffer(incremental_state)
        ):
            if self_attn_mask is not None:
                self_attn_mask = torch.cat(
                    (x.new(x.size(0), encoder_out.size(0)).zero_(), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    encoder_padding_mask = self_attn_padding_mask.new(
                        encoder_out.size(1), encoder_out.size(0)
                    ).zero_()
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
            fused=self.fused_transformer,
        )
        x = self.forward_residual(x, residual, self.self_attn_layer_norm)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)

            attn_mask = None
            if encoder_padding_mask.dim() == 3:
                attn_mask = encoder_padding_mask.bool()
                encoder_padding_mask = None
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                fused=self.fused_transformer,
            )
            x = self.forward_residual(x, residual, self.encoder_attn_layer_norm)

        residual = x
        x = self.forward_ffn(x)
        x = self.forward_residual(x, residual, self.final_layer_norm)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn.get_input_buffer(incremental_state)
            if self_attn_padding_mask is not None:
                self_attn_state = (
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                )
            else:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def forward_step(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        state=None,
        need_attn=False,
        need_head_weights=False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        self_k, self_v, self_mask, encode_k, encode_v, encode_mask = state

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        y = x

        x, attn, self_k, self_v, self_mask = self.self_attn.forward_step(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            prev_k=self_k,
            prev_v=self_v,
            prev_mask=self_mask,
        )
        x = self.forward_residual(x, residual, self.self_attn_layer_norm)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)

            x, attn, encode_k, encode_v, encode_mask = self.encoder_attn.forward_step(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                prev_k=encode_k,
                prev_v=encode_v,
                prev_mask=encode_mask,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.forward_residual(x, residual, self.encoder_attn_layer_norm)

        residual = x
        x = self.forward_ffn(x)
        x = self.forward_residual(x, residual, self.final_layer_norm)
        new_state = [self_k, self_v, self_mask, encode_k, encode_v, encode_mask]
        return x, attn, new_state

    def forward_ffn(self, x):
        '''forward ffn'''
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        return x

    def forward_residual(self, x, residual, ln):
        '''forward residual'''
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(ln, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        '''maybe_layer_norm'''
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        return x

    def make_generation_fast_(self, need_attn=False):
        '''make_generation_fast_'''
        self.need_attn = need_attn


class CifSelfAttentionModule(nn.Module):  # pylint: disable=abstract-method
    '''CifSelfAttentionModule'''

    def __init__(self, args, sign):
        super().__init__()
        self.args = FalconDict(args)
        self.ffn = nn.ModuleList()
        self.atten = nn.ModuleList()

        num_layers = args['num_{}_layers'.format(sign)]
        attention_type = args.get('{}_self_attention_type'.format(sign), None)
        fused = args.get('use_fused_kernel', True)

        # TODO(mst) remove these asserts
        assert self.args.layer_preprocess_sequence == 'n' and self.args.norm_type == 'layer'
        assert self.args.layer_postprocess_sequence == 'da'
        assert self.args.ffn_layer != 'none'

        def _get_ffn_activation(ffn_layer):
            if ffn_layer == 'dense_relu_dense':
                return 'relu'
            if ffn_layer == 'dense_swish_dense':
                return 'swish'
            raise NotImplementedError('unsupport ffn_layer {}'.format(ffn_layer))

        for _ in range(num_layers):
            self.atten.append(
                MultiHeadChunkwiseAttention(
                    hidden_size=self.args.hidden_size,
                    num_heads=self.args.num_heads,
                    total_key_depth=self.args.attention_key_channels or self.args.hidden_size,
                    total_value_depth=self.args.attention_value_channels or self.args.hidden_size,
                    dropout_rate=self.args.attention_dropout,
                    attention_type=attention_type or self.args.self_attention_type,
                    num_history_chunk=self.args.num_history_chunk,
                    use_layer_norm=True,
                    pre_layer_norm=True,
                    layer_norm_eps=self.args.norm_epsilon,
                    extra_dropout=self.args.layer_prepostprocess_dropout,
                    fused=fused,
                )
            )
            self.ffn.append(
                PositionwiseFeedForward(
                    input_dim=self.args.hidden_size,
                    hidden_units=self.args.filter_size,
                    dropout_rate=self.args.relu_dropout,
                    activation_fn=_get_ffn_activation(self.args.ffn_layer),
                    use_layer_norm=True,
                    pre_layer_norm=True,
                    layer_norm_eps=self.args.norm_epsilon,
                    extra_dropout=self.args.layer_prepostprocess_dropout,
                    fused=fused,
                )
            )
        self.process = LayerNorm(
            self.args.hidden_size,
            eps=self.args.norm_epsilon,
            fused=fused,
        )

    @staticmethod
    def attention_bias_proximal(length, device=None):
        '''
        Bias for self-attention to encourage attention to close positions.

        Args:
            length: an integer scalar.

        Returns:
            a Tensor with shape [1, 1, length, length]
        '''
        r = torch.arange(0, length, dtype=torch.float)
        if device is not None:
            r = r.to(device)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        out = -torch.log(1.0 + diff.abs())
        return out.unsqueeze(0).unsqueeze(0)
