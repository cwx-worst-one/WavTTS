''' Conformer layer '''
import torch
from torch import nn

from core.models.layers.cnn import Conv1dLinear, ConvolutionModule, MultiLayeredConv1d
from core.models.layers.feed_forward import FFN, PositionwiseFeedForward, MoEFFN
from core.models.layers.multi_head_attn import (
    MemoryMaskMultiheadAttention,
    MemoryMaskRelPositionMultiHeadedAttention,
    MultiHeadChunkwiseAttention,
)
from core.models.layers.normalization import LayerNorm, DualModeNorm
from core.utils.dict import FalconDict
from core.extensions.panther_symbol import panther_conformer
from core.extensions import fused_mem_mask_rel_attn, fused_conformer_convolution, fused_ffn
from core.utils.math import get_stream_align_size
from core.models.layers.adapter import *
from core.models import utils


class ConformerLayer(nn.Module):
    """ConformerLayer module.
    :param args: config
    :param mask_topology: mask topology for streaming encoder
    :param lnum: layer index
    """

    def __init__(self, args, mask_topology, lnum=None):
        """Construct an ConformerLayer object."""
        # pylint:disable=too-many-branches,too-many-statements
        super().__init__()

        self.args = args
        self.support_panther_fusion_export = True
        size = self.attention_dim = args.backbone_memory_size
        self.layer_order = args.get("conformer_layer_order", "mhsa_before_conv")
        assert self.layer_order in ("mhsa_before_conv", "conv_before_mhsa")
        self.dual_mode = args.get("conformer_dual_mode", False)
        self.support_panther_fusion_export = (
            self.support_panther_fusion_export and self.layer_order == "mhsa_before_conv"
        )

        self.attention_heads = args.conformer_attention_heads
        self.linear_units = args.conformer_linear_units
        dropout_rate = args.conformer_dropout_rate
        attention_dropout_rate = args.conformer_attention_dropout_rate

        concat_after = False
        positionwise_layer_type = args.conformer_positionwise_layer_type
        self.activation_fn = args.conformer_activation_fn
        positionwise_conv_kernel_size = args.conformer_positionwise_conv_kernel_size

        macaron_style = args.conformer_macaron_style
        pos_enc_layer_type = args.conformer_pos_enc_layer_type
        selfattention_layer_type = args.conformer_selfattention_layer_type

        use_cnn_module = args.conformer_use_cnn_module
        cnn_module_kernel = args.conformer_cnn_module_kernel
        self.cnn_module_norm_type = args.get("conformer_cnn_norm_type", "batch_norm")

        normalize_before = args.conformer_normalize_before
        merge_qkv = args.get('merge_qkv', True)
        moe_args = args.get('moe_args', None)
        self.moe_ffn = False
        self.aux_loss = 0.0
        moe_layers = []
        if moe_args is not None:
            moe_layers = eval(moe_args.get("moe_layers", "[]"))
            if lnum in moe_layers:
                self.moe_ffn = True

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MemoryMaskMultiheadAttention
            encoder_selfattn_layer_args = (
                self.attention_dim,
                self.attention_heads,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type in ("rel_pos", "fix_rel_pos")
            encoder_selfattn_layer = MemoryMaskRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                self.attention_dim,
                self.attention_heads,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        self.support_panther_fusion_export = (
            self.support_panther_fusion_export and selfattention_layer_type == "rel_selfattn"
        )

        if positionwise_layer_type == "linear":
            positionwise_layer = FFN
            positionwise_layer_args = (
                self.attention_dim,
                self.linear_units,
                dropout_rate,
                self.activation_fn,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                self.attention_dim,
                self.linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
                self.activation_fn,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                self.attention_dim,
                self.linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
                self.activation_fn,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        self.support_panther_fusion_export = (
            self.support_panther_fusion_export and positionwise_layer_type == "linear"
        )

        if args.conformer_cnn_module == 'ConvolutionModule':
            convolution_layer = ConvolutionModule
            convolution_layer_args = (
                self.attention_dim,
                cnn_module_kernel,
                self.activation_fn,
                self.cnn_module_norm_type,
                self.dual_mode,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + args.conformer_cnn_module)

        self.self_attn = encoder_selfattn_layer(
            *encoder_selfattn_layer_args, mask_topology, merge_qkv
        )
        if self.moe_ffn:
            self.feed_forward = MoEFFN(
                self.attention_dim,
                self.linear_units,
                dropout_rate=dropout_rate,
                activation_fn=self.activation_fn,
                num_expert=moe_args.get('moe_expert', 8),
                topk=moe_args.get('moe_topk', 2),
                output_dropout_prob=moe_args.get('output_dropout_prob', 0.0),
                moe_loss_scale=moe_args.get('moe_loss_scale', 0.0),
                z_loss_scale=moe_args.get('z_loss_scale', 0.0),
                noisy_gate_policy=moe_args.get('noisy_gate_policy', None),
                use_lego=moe_args.get('use_lego', False),
            )
        else:
            self.feed_forward = positionwise_layer(*positionwise_layer_args)
        self.feed_forward_macaron = (
            positionwise_layer(*positionwise_layer_args) if macaron_style else None
        )
        self.conv_module = convolution_layer(*convolution_layer_args) if use_cnn_module else None
        self.ff_scale = 1.0
        self.norm_ff = DualModeNorm(nn.LayerNorm(size), self.dual_mode)  # for the FNN module
        self.norm_mha = DualModeNorm(nn.LayerNorm(size), self.dual_mode)  # for the MHA module
        self.stream_mode = False
        self.stream_mask_index = 0
        if self.feed_forward_macaron is not None:
            self.norm_ff_macaron = DualModeNorm(nn.LayerNorm(size), self.dual_mode)
            self.ff_scale = 0.5
        if self.conv_module is not None:
            self.norm_conv = DualModeNorm(nn.LayerNorm(size), self.dual_mode)  # for the CNN module
            self.norm_final = DualModeNorm(
                nn.LayerNorm(size), self.dual_mode
            )  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

        self.layer_num = lnum
        self.group_layer_idx = args.get('group_layer_idx', None)
        self.return_attn = False
        if self.group_layer_idx:
            self.group_layer_idx = set(eval(self.group_layer_idx))
            self.return_attn = self.layer_num in self.group_layer_idx
            # For exporting QSVDMHA with reuse_attn
            setattr(self.self_attn, 'return_attn', self.return_attn)
        self.dropout_rate = dropout_rate
        self.fused_conformer = args.get('fused_conformer', True)
        if self.group_layer_idx:
            self.fused_conformer = False
        self.left_kernel_size, self.right_kernel_size = (
            self.self_attn.left_kernel_size,
            self.self_attn.right_kernel_size,
        )
        if use_cnn_module:
            self.conv_left_kernel_size, self.conv_right_kernel_size = (
                self.conv_module.left_kernel_size,
                self.conv_module.right_kernel_size,
            )
        else:
            self.conv_left_kernel_size, self.conv_right_kernel_size = 0, 0

        self._fused_attn_cfg = {
            'embed_dim': self.attention_dim,
            'attn_heads': self.attention_heads,
            'attn_drop': attention_dropout_rate,
            'hidden_drop': self.dropout_rate,
            'eps': self.norm_mha.eps,
            'norm_mode': 'before' if self.normalize_before else 'after',
            'batch_first': True,
        }
        self._fused_ffn_cfg = {
            'embed_dim': self.attention_dim,
            'ffn_hidden': self.linear_units,
            'hidden_drop': self.dropout_rate,
            'act_drop': self.dropout_rate,
            'eps': self.norm_ff.eps,
            'do_residual': True,
            'batch_first': True,
            'squeeze_mem': self.args.get('squeeze_mem', False),
            'activation': self.activation_fn,
            'norm_mode': 'before' if self.normalize_before else 'after',
            'res_weight': self.ff_scale,
        }
        if use_cnn_module:
            self._fused_conv_cfg = {
                'norm_type': self.conv_module.norm_type,
                'activation': self.activation_fn,
                'norm_eps': self.conv_module.norm.eps,
                'ln_eps': self.norm_conv.eps,
                'bn_momentum': getattr(self.conv_module.norm, 'momentum', 0),
                'drop_rate': self.dropout_rate,
                'embed_dim': self.attention_dim,
                'left_kernel_size': self.conv_left_kernel_size,
                'right_kernel_size': self.conv_right_kernel_size,
                'do_norm': True,
                'pre_norm': self.normalize_before,
                'dual_mode': self.dual_mode,
            }
        else:
            self._fused_conv_cfg = None

        # For export QConformer pos_embdding base RelPositionalEncoding
        self.dim = args.get('position_encoding_dim', 1)
        self.export_rel_pos_embeding_len = args.get('export_rel_pos_embeding_len', 65)
        self.memory_size = args.backbone_memory_size
        adapter_type = args.get('adapter_type', 'none')
        adapter_embed_dim = args.get('adapter_embed_dim', -1)
        self.adapter_mode = args.get('adapter_mode', 'none')
        self.init_adapter_ffn(adapter_type, adapter_embed_dim)
        self.init_adapter_attn(adapter_type, adapter_embed_dim)

    def set_stream_mode(self, stream_mode=False):
        '''set stream mode for dual mode'''
        if not self.dual_mode:
            return
        self.stream_mode = stream_mode
        if self.conv_module is not None:
            self.conv_module.set_stream_mode(stream_mode)
        self.self_attn.set_stream_mode(stream_mode)
        self.norm_ff.set_stream_mode(stream_mode)
        self.norm_mha.set_stream_mode(stream_mode)
        if self.feed_forward_macaron is not None:
            self.norm_ff_macaron.set_stream_mode(stream_mode)
        if self.conv_module is not None:
            self.norm_conv.set_stream_mode(stream_mode)
            self.norm_final.set_stream_mode(stream_mode)

    def set_stream_index(self, stream_index=0):
        '''set_stream_mode'''
        self.stream_mask_index = stream_index
        self.self_attn.set_stream_index(stream_index)

    def init_adapter_ffn(self, adapter_type, adapter_embed_dim):
        self.adapter_ffn = None
        self.adapter_ffn_macaron = None
        if adapter_type != 'none':
            self.adapter_ffn = eval(adapter_type)(self.memory_size, adapter_embed_dim)
            if self.feed_forward_macaron is not None:
                self.adapter_ffn_macaron = eval(adapter_type)(self.memory_size, adapter_embed_dim)

    def init_adapter_attn(self, adapter_type, adapter_embed_dim):
        self.adapter_attn = None
        if adapter_type != 'none' and self.adapter_mode.endswith('_attn'):
            self.adapter_attn = eval(adapter_type)(self.memory_size, adapter_embed_dim)

    def forward(self, inputs, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features, w/o pos_emb
        tuple((batch, max_time_in, size), (1, max_time_in, size))
        or (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, 1, max_time_in) or (batch, time1, time2)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # pylint:disable=too-many-branches
        if self.group_layer_idx:
            x_input, mask, attn_weights = inputs
        else:
            x_input, mask = inputs
            attn_weights = None
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # for export ONNX for panther fused OP.
        if (
            torch.jit.is_tracing()
            and not self.training
            and self.args.get('export_fused_conformer', False)
            and self.support_panther_fusion_export
        ):
            return self.panther_fused_export(x, pos_emb, mask)

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            x_ = 0
            if self.adapter_mode.startswith('parallel'):
                x_ = self.adapter_ffn_macaron(x) if self.adapter_ffn_macaron is not None else 0
            x = self.ffn_macaron_forward(x)
            if self.adapter_mode.startswith('sequential'):
                x_ = self.adapter_ffn_macaron(x) if self.adapter_ffn_macaron is not None else 0
            x = x + x_

        # convolution module
        if self.conv_module is not None and self.layer_order == 'conv_before_mhsa':
            x = self.conv_module_forward(x, mask)

        x_ = 0
        # multi-headed self-attention module
        if self.adapter_mode.startswith('parallel'):
            x_ = self.adapter_attn(x) if self.adapter_attn is not None else 0
        x, attn_weights = self.self_attn_forward(x, mask, pos_emb, cache, attn_weights)
        if self.adapter_mode.startswith('sequential'):
            x_ = self.adapter_attn(x) if self.adapter_attn is not None else 0
        x = x + x_

        # convolution module
        if self.conv_module is not None and self.layer_order == 'mhsa_before_conv':
            x = self.conv_module_forward(x, mask)

        # feed forward module
        x_ = 0
        if self.adapter_mode.startswith('parallel'):
            x_ = self.adapter_ffn(x) if self.adapter_ffn is not None else 0
        x = self.ffn_forward(x)
        if self.adapter_mode.startswith('sequential'):
            x_ = self.adapter_ffn(x) if self.adapter_ffn is not None else 0
        x = x + x_

        # preprocess for output
        if self.conv_module is not None:
            x = self.norm_final(x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        if pos_emb is not None:
            if self.group_layer_idx:
                return (x, pos_emb), mask, attn_weights
            return (x, pos_emb), mask
        if self.group_layer_idx:
            return x, mask, attn_weights
        return x, mask

    def panther_fused_export(self, x, pos_emb, mask):
        '''export ONNX using panther fused CUDA OP'''
        x = panther_conformer(
            x,
            mask,
            pos_emb,
            self.norm_ff_macaron,
            self.feed_forward_macaron,
            self.norm_mha,
            self.self_attn,
            self.norm_conv,
            self.conv_module,
            self.norm_ff,
            self.feed_forward,
            self.norm_final,
            activation_fn=self.activation_fn,
            num_heads=self.attention_heads,
            embed_dim=self.attention_dim,
            linear_dim=self.linear_units,
            ff_scale=self.ff_scale,
        )
        if pos_emb is not None:
            return (x, pos_emb), mask
        return x, mask

    def ffn_macaron_forward(self, x):
        '''forward of macaron FFN'''
        if self.training and self.fused_conformer:
            ws = [
                self.feed_forward_macaron.w_1.weight,
                self.feed_forward_macaron.w_1.bias,
                self.feed_forward_macaron.w_2.weight,
                self.feed_forward_macaron.w_2.bias,
                self.norm_ff_macaron.weight,
                self.norm_ff_macaron.bias,
            ]
            return fused_ffn(ws, x, **self._fused_ffn_cfg)
        residual = x
        if self.normalize_before:
            x = self.norm_ff_macaron(x)
        x = self.dropout(self.feed_forward_macaron(x, fused=self.fused_conformer))
        x = residual + self.ff_scale * x
        if not self.normalize_before:
            x = self.norm_ff_macaron(x)
        return x

    def self_attn_forward(self, x, mask, pos_emb, cache, attn_weights):
        '''forward of self attention'''
        if self.training and self.fused_conformer and cache is None and pos_emb is not None:
            # When exporting QSVD Conformer, self_attn will be replaced by panther module
            self_att_memory_mask = self.self_attn.get_self_att_memory_mask().to(x.device)
            ws = [
                self.self_attn.in_proj.weight,
                self.self_attn.in_proj.bias,
                self.self_attn.linear_pos.weight,
                self.self_attn.pos_bias_u,
                self.self_attn.pos_bias_v,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm_mha.weight,
                self.norm_mha.bias,
            ]
            return (
                fused_mem_mask_rel_attn(
                    ws, x, pos_emb, mask, self_att_memory_mask, **self._fused_attn_cfg
                ),
                None,
            )

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]
        if pos_emb is not None:
            x_att, attn_weights = self.self_attn(
                x_q,
                x,
                x,
                pos_emb,
                key_padding_mask=mask,
                need_weights=self.return_attn,
                attn_weights=attn_weights,
            )
        else:
            x_att, attn_weights = self.self_attn(
                x_q,
                x,
                x,
                key_padding_mask=mask,
                need_weights=self.return_attn,
                attn_weights=attn_weights,
            )
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = self.concat_linear(x_concat)
        else:
            x = self.dropout(x_att)
        x = residual + x
        if not self.normalize_before:
            x = self.norm_mha(x)
        return x, attn_weights

    def self_attn_forward_step(
        self, x, att_cache, att_mask_cache, pos_emb, required_right_context, mask
    ):
        '''forward step of self attention'''
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_q = x
        # pylint: disable=unbalanced-tuple-unpacking
        if pos_emb is not None:
            x_att, att_cache, att_mask_cache = self.self_attn.forward_step(
                x_q, att_cache, att_mask_cache, pos_emb, required_right_context, mask
            )
        else:
            x_att, att_cache, att_mask_cache = self.self_attn.forward_step(
                x_q, att_cache, att_mask_cache, required_right_context, mask
            )
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = self.concat_linear(x_concat)
        else:
            x = self.dropout(x_att)
        x = residual + x
        if not self.normalize_before:
            x = self.norm_mha(x)
        return x, att_cache, att_mask_cache

    def conv_module_forward(self, x, mask):
        '''forward of conv_module'''
        if self.training and self.fused_conformer:
            ws = [
                self.conv_module.pointwise_conv1.weight.unsqueeze(2),
                self.conv_module.pointwise_conv1.bias,
                self.conv_module.depthwise_conv.weight,
                self.conv_module.depthwise_conv.bias,
                self.conv_module.norm.weight,
                self.conv_module.norm.bias,
                getattr(self.conv_module.norm, 'running_mean', torch.Tensor()),
                getattr(self.conv_module.norm, 'running_var', torch.Tensor()),
                self.conv_module.pointwise_conv2.weight.unsqueeze(2),
                self.conv_module.pointwise_conv2.bias,
                self.norm_conv.weight,
                self.norm_conv.bias,
            ]
            return fused_conformer_convolution(
                ws, x, mask, stream_mode=self.stream_mode, **self._fused_conv_cfg
            )
        residual = x
        if self.normalize_before:
            x = self.norm_conv(x)
        x = self.dropout(self.conv_module(x, conv_memory=None, mask=mask))
        x = residual + x
        if not self.normalize_before:
            x = self.norm_conv(x)
        return x

    def conv_module_forward_step(self, x, mask, conv_cache, required_right_context):
        '''forward step of conv_module'''
        residual = x
        if self.normalize_before:
            x = self.norm_conv(x)
        x, conv_cache = self.conv_module.forward_step(x, conv_cache, required_right_context, mask)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.norm_conv(x)
        return x, conv_cache

    def ffn_forward(self, x):
        '''forward of FFN'''
        if self.training and self.fused_conformer and not self.moe_ffn:
            ws = [
                self.feed_forward.w_1.weight,
                self.feed_forward.w_1.bias,
                self.feed_forward.w_2.weight,
                self.feed_forward.w_2.bias,
                self.norm_ff.weight,
                self.norm_ff.bias,
            ]
            return fused_ffn(ws, x, **self._fused_ffn_cfg)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = self.dropout(self.feed_forward(x, fused=self.fused_conformer))
        if self.moe_ffn:
            self.aux_loss = self.feed_forward.aux_loss
        x = residual + self.ff_scale * x

        if not self.normalize_before:
            x = self.norm_ff(x)
        return x

    def forward_step(self, inputs, required_right_context, att_cache, att_mask_cache, conv_cache):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features, w/o pos_emb
        tuple((batch, max_time_in, size), (1, max_time_in, size))
        or (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, 1, max_time_in) or (batch, time1, time2)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x_input, mask = inputs
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            x = self.ffn_macaron_forward(x)

        # convolution module
        if self.conv_module is not None and self.layer_order == 'conv_before_mhsa':
            x, conv_cache = self.conv_module_forward_step(
                x, mask, conv_cache, required_right_context
            )

        # multi-headed self-attention module
        x, att_cache, att_mask_cache = self.self_attn_forward_step(
            x,
            att_cache,
            att_mask_cache,
            pos_emb,
            required_right_context,
            mask,
        )

        # convolution module
        if self.conv_module is not None and self.layer_order == 'mhsa_before_conv':
            x, conv_cache = self.conv_module_forward_step(
                x, mask, conv_cache, required_right_context
            )

        # feed forward module
        x = self.ffn_forward(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask, att_cache, att_mask_cache, conv_cache
        return x, mask, att_cache, att_mask_cache, conv_cache

    @property
    def state_size(self):
        '''streamed state size.'''
        # align_16(conv_res: E * R) + align_16(conv_state: E * (L + R))
        # + align_16(attn_res: E * R) + align_16(attn_q: E * R)
        # + align_16(attn_k: E * (L + R)) + align_16(attn_v: E * (L + R))
        # + align_16(attn_len：1)
        state_size = (
            get_stream_align_size(self.conv_right_kernel_size * self.attention_dim)
            + get_stream_align_size(
                (self.conv_left_kernel_size + self.conv_right_kernel_size) * self.attention_dim
            )
            + get_stream_align_size(self.right_kernel_size * self.attention_dim)
            + get_stream_align_size(self.right_kernel_size * self.attention_dim)
            + 2
            * get_stream_align_size(
                (self.left_kernel_size + self.right_kernel_size) * self.attention_dim
            )
            + get_stream_align_size(1)
        )
        return state_size


class CifConformerV2Module(nn.Module):  # pylint: disable=abstract-method
    '''CifConformerV2Module'''

    def __init__(self, args, sign):
        super().__init__()
        self.args = FalconDict(args)
        self.macaron_ffn1 = nn.ModuleList()
        self.conv_module = nn.ModuleList()
        self.atten = nn.ModuleList()
        self.macaron_ffn2 = nn.ModuleList()

        num_layers = args['num_{}_layers'.format(sign)]
        attention_type = args.get('{}_self_attention_type'.format(sign), None)
        fused = args.get('use_fused_kernel', True)

        # TODO(mst) remove these asserts
        assert self.args.layer_preprocess_sequence == 'n' and self.args.norm_type == 'layer'
        assert self.args.layer_postprocess_sequence == 'da'
        assert self.args.ffn_layer != 'none'

        for _ in range(num_layers):
            self.macaron_ffn1.append(
                PositionwiseFeedForward(
                    input_dim=self.args.hidden_size,
                    hidden_units=self.args.filter_size,
                    dropout_rate=self.args.relu_dropout or self.args.layer_prepostprocess_dropout,
                    activation_fn='swish',
                    use_layer_norm=True,
                    pre_layer_norm=True,
                    layer_norm_eps=self.args.norm_epsilon,
                    extra_dropout=self.args.layer_prepostprocess_dropout,
                    residual_out_scaler=0.5,
                    fused=fused,
                )
            )
            self.conv_module.append(
                ConvolutionModule(
                    channels=self.args.hidden_size,
                    kernel_size=str(self.args.conformer_conv_width),
                    activation_fn='swish',
                    norm_type=self.args.conformer_conv_norm_type,
                    dropout_rate=self.args.layer_prepostprocess_dropout,
                    use_layer_norm_before=True,
                    layer_norm_eps=self.args.norm_epsilon,
                    use_residual=True,
                )
            )
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
            self.macaron_ffn2.append(
                PositionwiseFeedForward(
                    input_dim=self.args.hidden_size,
                    hidden_units=self.args.filter_size,
                    dropout_rate=self.args.relu_dropout or self.args.layer_prepostprocess_dropout,
                    activation_fn='swish',
                    use_layer_norm=True,
                    pre_layer_norm=True,
                    layer_norm_eps=self.args.norm_epsilon,
                    extra_dropout=self.args.layer_prepostprocess_dropout,
                    residual_out_scaler=0.5,
                    fused=fused,
                )
            )
        self.process = LayerNorm(
            self.args.hidden_size,
            eps=self.args.norm_epsilon,
            fused=fused,
        )
