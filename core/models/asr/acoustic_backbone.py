'''
acoustic_backbone.py.
'''
# pylint:disable=too-many-lines
import math
import torch
from torch import nn
import torch.nn.functional as F
from core.models.layers.fsmn_layer import DFSMNLayer
from core.models.layers.transformer import (
    BiTransformerLayer,
    ParallelTransformer,
    RelTransformerLayer,
    EmformerLayer,
    CifSelfAttentionModule,
)
from core.models.layers.embedding import (
    AbsPositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from core.models.layers.lstmp_layer import LSTMP
from core.models.layers.tdnn_layer import TDNN
from core.models.layers.conformer import ConformerLayer, CifConformerV2Module
from core.models.layers.time_reduce_layer import AvgPoolTimeReduce
from core.extensions import checkpoint_wrapper
from core.utils import logging


def rnnt_transpose(inputs, out_shape, frontend_shape="BTN"):
    'rnnt transpose'
    if out_shape == frontend_shape:
        return inputs
    pos = [0, 1, 2]
    for i in range(3):
        for j in range(3):
            if out_shape[i] == frontend_shape[j]:
                pos[i] = j
    inputs = inputs.permute(int(pos[0]), int(pos[1]), int(pos[2])).contiguous()
    return inputs


def get_tgt_len(inputs, backbone_shape="BTN"):
    'get tgt_len'
    for i in range(3):
        if backbone_shape[i] == "T":
            return inputs.size(i)
    return 0


class IdentityBackbone(nn.Module):
    """
    A fake backbone for extracting features
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fake_model = nn.Linear(1, 1)

    def forward(self, inputs, mask, attn_mask=None, frontend_shape=None):
        return inputs


class IdentityBackbone(nn.Module):
    """
    A fake backbone for extracting features
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fake_model = nn.Linear(1, 1)

    def forward(self, inputs, mask, attn_mask=None, frontend_shape=None):
        return inputs


class DFSMNBackboneLN(nn.Module):
    '''
    DFSMNBackboneLN can make float16 training more stable without damaging cer.
    '''

    def __init__(self, args):
        super().__init__()
        args = self._compatible_configs(args)
        self.backbone_topology = eval(args.backbone_topology)
        self.backbone_layer_num = len(self.backbone_topology)
        self.dfsmn_layers = nn.ModuleList()
        for layer_idx in range(self.backbone_layer_num):
            topology = self.backbone_topology[layer_idx]
            left_kernel_size = topology[0]
            right_kernel_size = topology[1]
            dilation = topology[2]
            self.dfsmn_layers.append(
                DFSMNLayer(
                    args.backbone_hidden_size,
                    args.backbone_memory_size,
                    left_kernel_size,
                    right_kernel_size,
                    dilation=dilation,
                    dropout=args.dropout,
                    weight_scale=args.get('backbone_weight_scale', 1.0),
                    bias=args.get('backbone_bias', True),
                    fc_layernorm=args.get('backbone_fc_ln', True),
                    memory_first=args.get('backbone_memory_first', True),
                )
            )
        self.clamp_residual = args.get('clamp_residual', False)
        self.fused_dfsmn = args.get('fused_dfsmn', True)
        self._register_load_state_dict_pre_hook(self.compatible_load_hook)

    @staticmethod
    def compatible_load_hook(
        state_dict, prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, _error_msgs
    ):
        '''
        compatible for load DFSMN backbone.
        '''
        for key in list(state_dict.keys()):
            if key.startswith(prefix + 'module_list'):
                new_key = key.replace('module_list', 'dfsmn_layers')
                state_dict[new_key] = state_dict.pop(key, None)

    @property
    def state_size(self):
        '''streamed state size.'''
        state_size = sum(layer.state_size for layer in self.dfsmn_layers)
        return state_size

    def _compatible_configs(self, args):
        '''compatible for vad&kws ce_encoder configs'''
        if 'dfsmn_topo' in args:
            dfsmn_cfg = args.copy()
            dfsmn_cfg['backbone_topology'] = args.dfsmn_topo
            dfsmn_cfg['backbone_memory_size'] = args.dfsmn_memory_size
            dfsmn_cfg['backbone_hidden_size'] = args.dfsmn_hidden_size
            dfsmn_cfg['dropout'] = args.dfsmn_dropout
            dfsmn_cfg['backbone_fc_ln'] = args.dfsmn_layer_norm
            dfsmn_cfg['backbone_bias'] = False
            dfsmn_cfg['backbone_memory_first'] = False
            return dfsmn_cfg
        return args

    def forward(
        self,
        front_end_out,
        acoustic_mask=None,
        frontend_shape="BTN",
        selected_layer_idx=-1,
        **_kwargs,
    ):
        """forward"""
        dfsmn_out = front_end_out
        selected_dfsmn_out = None
        acoustic_mask = acoustic_mask.unsqueeze(2) if acoustic_mask is not None else None
        dfsmn_out = rnnt_transpose(dfsmn_out, "BTN", frontend_shape)
        for layer_idx in range(self.backbone_layer_num):
            dfsmn_out = self.dfsmn_layers[layer_idx](dfsmn_out, fused=self.fused_dfsmn)
            if selected_layer_idx == layer_idx:
                selected_dfsmn_out = dfsmn_out
            if self.clamp_residual:
                dfsmn_out = torch.clamp(dfsmn_out, -5e4, 5e4)  # clamp 'inf'
            if acoustic_mask is not None:
                dfsmn_out = dfsmn_out * acoustic_mask
        if selected_dfsmn_out is not None:
            return dfsmn_out, selected_dfsmn_out
        return dfsmn_out

    def forward_step(
        self,
        front_end_out,
        acoustic_mask=None,
        states=None,
        x_sign=None,
        frontend_shape="BTN",
        **_kwargs,
    ):
        """forward"""
        dfsmn_out = front_end_out
        acoustic_mask = acoustic_mask.unsqueeze(1) if acoustic_mask is not None else None
        dfsmn_out = rnnt_transpose(dfsmn_out, "BNT", frontend_shape)
        states_offset = 0
        for layer_idx in range(self.backbone_layer_num):
            states_len = self.dfsmn_layers[layer_idx].state_size
            cur_states = states[:, states_offset : states_offset + states_len]
            dfsmn_out, cur_states, _ = self.dfsmn_layers[layer_idx].forward_step(
                dfsmn_out, x_mask=acoustic_mask, x_sign=x_sign, states=cur_states
            )
            states[:, states_offset : states_offset + states_len] = cur_states
            states_offset += states_len
            if acoustic_mask is not None:
                dfsmn_out = dfsmn_out * acoustic_mask
        dfsmn_out = rnnt_transpose(dfsmn_out, "BTN", "BNT")
        return dfsmn_out, states


class TransformerBackbone(nn.Module):
    '''TransformerBackbone'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.args = args

        self.tp_size = args.get('tensor_parallel_size', 1)
        self.chunk_mask = args.get('chunk_mask', False)
        self.after_norm = None
        self.position_encoding = None
        if args.get('backbone_pos_embd', False):
            self.position_encoding = AbsPositionalEncoding(
                args.backbone_memory_size, args.positional_dropout_rate
            )
        if args.self_attn_layer_norm_before and args.get('backbone_after_norm', False):
            self.after_norm = nn.LayerNorm(args.backbone_memory_size)
        backbone_topology = args.get('backbone_topology', None)
        if backbone_topology:
            backbone_topology = eval(args.backbone_topology)
            self.backbone_layer_num = len(backbone_topology)
        else:
            self.backbone_layer_num = args.backbone_layer_num
        self.transformers = nn.ModuleList()
        self.causal_transformer = args.get('causal_transformer', False)
        self.fused_transformer = args.get('fused_transformer', True)
        if self.causal_transformer:
            if self.chunk_mask:
                logging.info("Streaming Chunk-wise Transformer is Loading")
                self.streaming_chunk_size = args.get("streaming_chunk_size", 8)
            else:
                logging.info("streaming topology is %r", backbone_topology)
                self.attn_mask_list = {}
                self.attn_mask_keys = []
                self.max_length = 512
        for layer_idx in range(self.backbone_layer_num):
            if self.causal_transformer and not self.chunk_mask:
                topology = backbone_topology[layer_idx]
                left_kernel_size = topology[0]
                right_kernel_size = topology[1]
                dilation = topology[2]
                mask_key = '%d_%d_%d' % (left_kernel_size, right_kernel_size, dilation)
                self.attn_mask_keys.append(mask_key)
                if mask_key not in self.attn_mask_list:
                    self.attn_mask_list[mask_key] = {}
                    self.attn_mask_list[mask_key]['left'] = left_kernel_size
                    self.attn_mask_list[mask_key]['right'] = right_kernel_size
                    self.attn_mask_list[mask_key]['dilation'] = dilation
                    self.attn_mask_list[mask_key]['mask'] = self.gen_attn_mask(
                        left_kernel_size, right_kernel_size, dilation
                    )
            num_chkpt_layers = args.get('num_chkpt_layers', 0)
            if self.tp_size > 1:
                act_glu = args.get('backbone_act_glu', False)
                moe_args = args.get('moe_args', None)
                squeeze_mem = layer_idx >= num_chkpt_layers and args.get('squeeze_mem', True)
                assert act_glu == False
                assert moe_args is None
                assert squeeze_mem == False
                transformer_layer = ParallelTransformer(
                    args.backbone_memory_size,
                    args.self_attn_heads,
                    args.backbone_hidden_size,
                    args.self_attn_dropout,
                    args.dropout,
                    activation_dropout=args.self_attn_activation_dropout,
                    activation=args.self_attn_activation_fn,
                    normalize_before=args.self_attn_layer_norm_before,
                    clamp_inf=args.get('backbone_clamp_inf', False),
                    tensor_parallel=self.tp_size,
                )
            else:
                transformer_layer = BiTransformerLayer(
                    args.backbone_memory_size,
                    args.self_attn_heads,
                    args.backbone_hidden_size,
                    args.self_attn_dropout,
                    args.dropout,
                    activation_dropout=args.self_attn_activation_dropout,
                    activation=args.self_attn_activation_fn,
                    normalize_before=args.self_attn_layer_norm_before,
                    act_glu=args.get('backbone_act_glu', False),
                    clamp_inf=args.get('backbone_clamp_inf', False),
                    moe_args=args.get('moe_args', None),
                    lnum=layer_idx,
                    squeeze_mem=layer_idx >= num_chkpt_layers and args.get('squeeze_mem', True),
                )
            if layer_idx < num_chkpt_layers:
                transformer_layer = checkpoint_wrapper(transformer_layer)
            self.transformers.append(transformer_layer)

    @property
    def aux_loss(self):
        '''aux_loss'''
        loss = 0.0
        for layer in self.transformers:
            if layer.moe_ffn:
                loss += layer.aux_loss
        return loss

    @torch.no_grad()
    def update_attn_mask(self):
        '''update_attn_mask when self.max_length change'''
        for mask_val in self.attn_mask_list.values():
            mask_val['mask'] = self.gen_attn_mask(
                mask_val['left'],
                mask_val['right'],
                mask_val['dilation'],
            )

    @torch.no_grad()
    def gen_attn_mask(self, left_kernel_size, right_kernel_size, dilation):
        '''gen_attn_mask'''
        temp_mask = torch.ones((self.max_length, self.max_length))
        if left_kernel_size >= 0:
            left_temp_mask = torch.triu(temp_mask, diagonal=-left_kernel_size)
        else:
            left_temp_mask = temp_mask
        if right_kernel_size >= 0:
            right_temp_mask = torch.tril(temp_mask, diagonal=right_kernel_size)
        else:
            right_temp_mask = temp_mask
        memory_mask = left_temp_mask * right_temp_mask
        if dilation == 1:
            final_mask = memory_mask
        elif dilation == 2:
            final_mask = torch.cat(
                [memory_mask.unsqueeze(2), (temp_mask * 0).unsqueeze(2)], dim=2
            ).contiguous()
            mask_list = []
            for ii in range(self.max_length):
                mask_list.append(
                    final_mask.view(self.max_length, -1)[ii : ii + 1, ii : ii + self.max_length]
                )
            final_mask = torch.cat(mask_list, dim=0)
        else:
            raise ValueError("dilation %d does not supportedin TransformerBackbone" % (dilation))
        bool_mask = (0 * temp_mask).masked_fill((1 - final_mask).bool(), 1).bool()
        return bool_mask.unsqueeze(0).cuda()

    @staticmethod
    def streaming_encoder_mask(xs: torch.Tensor, masks: torch.Tensor, chunk_size: int = 8):
        '''streaming_encoder_mask'''
        masks = masks.bool()
        olens = masks.squeeze(1).sum(1)
        bsz, tgt = masks.size(0), masks.size(-1)
        new_masks = torch.zeros(bsz, tgt, tgt).to(xs.device)
        cc = chunk_size
        for i in range(bsz):
            max_len = olens[i]
            n_seg = max_len // cc if max_len % cc == 0 else max_len // cc + 1
            for j in range(n_seg):
                new_masks[
                    i,
                    j * cc : min((j + 1) * cc, max_len),
                    max((j - 1) * cc, 0) : min((j + 1) * cc, max_len),
                ] = 1

        return new_masks

    def forward(
        self, front_end_out, acoustic_mask=None, attn_mask=False, frontend_shape="BTN", **_kwargs
    ):
        """forward"""
        transformer_input = front_end_out
        backbone_shape = 'BTN'
        transformer_input = rnnt_transpose(transformer_input, backbone_shape, frontend_shape)
        if self.position_encoding:
            transformer_input = self.position_encoding(transformer_input)
        tgt_len = get_tgt_len(transformer_input, backbone_shape)
        chunk_attn_mask = None
        if self.causal_transformer:
            if self.chunk_mask:
                chunk_attn_mask = self.streaming_encoder_mask(
                    transformer_input, acoustic_mask, self.streaming_chunk_size
                ).eq(0)
            elif attn_mask and tgt_len > self.max_length:
                self.max_length = tgt_len
                self.update_attn_mask()
        if acoustic_mask is not None and not self.chunk_mask:
            transformer_mask = (1 - acoustic_mask).bool()
        else:
            transformer_mask = None
        for layer_idx in range(self.backbone_layer_num):
            if attn_mask and self.causal_transformer and not self.chunk_mask:
                mask_key = self.attn_mask_keys[layer_idx]
                causal_attn_mask = self.attn_mask_list[mask_key]['mask'][:, :tgt_len, :tgt_len]
            elif self.chunk_mask:
                causal_attn_mask = chunk_attn_mask
            else:
                causal_attn_mask = None
            transformer_input = self.transformers[layer_idx](
                transformer_input,
                transformer_mask,
                causal_attn_mask,
                fused=self.fused_transformer,
                batch_first=True,
            )
        if self.after_norm:
            transformer_input = self.after_norm(transformer_input)
        transformer_input = rnnt_transpose(transformer_input, "BTN", backbone_shape)
        return transformer_input


class OfflineTransformerBackbone(TransformerBackbone):
    '''reserved class'''


class LSTMPBackbone(nn.Module):
    '''LSTMPBackbone.'''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.backbone_layer_num = args.backbone_layer_num
        self.backbone_hidden_size = args.backbone_hidden_size
        self.lstmp_layers = nn.ModuleList()
        for _ in range(self.backbone_layer_num):
            self.lstmp_layers.append(
                LSTMP(
                    args.backbone_memory_size,
                    args.backbone_hidden_size,
                    bidirectional=bool(args.backbone_bilstm),
                    dropout=args.dropout,
                    residual=bool(args.backbone_residual),
                )
            )
        self.apply_mask = args.get('backbone_mask', False)

    def forward(self, front_end_out, acoustic_mask=None, frontend_shape="BTN", **_kwargs):
        """forward"""
        lstmp_input = front_end_out
        acoustic_mask = acoustic_mask.unsqueeze(2) if acoustic_mask is not None else None
        lstmp_input = rnnt_transpose(lstmp_input, "BTN", frontend_shape)
        for layer_idx in range(self.backbone_layer_num):
            lstmp_input = self.lstmp_layers[layer_idx](lstmp_input)
            if self.apply_mask and (acoustic_mask is not None):
                lstmp_input = lstmp_input * acoustic_mask
        return lstmp_input

    def forward_step(
        self, front_end_out, acoustic_mask=None, states=None, frontend_shape="BTN", **_kwargs
    ):
        '''forward step.'''
        lstmp_input = rnnt_transpose(front_end_out, "BTN", frontend_shape)
        assert lstmp_input.dim() == 3
        assert acoustic_mask is None or acoustic_mask.dim() == 2
        acoustic_mask = acoustic_mask.unsqueeze(2) if acoustic_mask is not None else None

        # split global_state_in
        if isinstance(states, torch.Tensor):
            new_states = []
            state_offset = 0
            for _ in range(self.backbone_layer_num):
                h0 = states[:, state_offset : state_offset + self.backbone_hidden_size]
                state_offset += self.backbone_hidden_size
                c0 = states[:, state_offset : state_offset + self.backbone_hidden_size]
                state_offset += self.backbone_hidden_size
                new_states.append((h0.unsqueeze(0), c0.unsqueeze(0)))
            states = new_states

        new_states = []
        for layer_idx in range(self.backbone_layer_num):
            lstmp_input, (h1, c1) = self.lstmp_layers[layer_idx].forward_step(
                lstmp_input, states[layer_idx]
            )
            if self.apply_mask and (acoustic_mask is not None):
                lstmp_input = lstmp_input * acoustic_mask
            new_states.append((h1.squeeze(0), c1.squeeze(0)))
        return lstmp_input, new_states

    @property
    def state_size(self):
        '''streamed state size.'''
        return self.backbone_layer_num * 2 * self.backbone_hidden_size


class TDNNLSTMPBackbone(LSTMPBackbone):
    '''TDNN LSTMPBackbone.'''

    def __init__(self, args):
        '''init.'''
        super().__init__(args)
        self.tdnn_backbone_topology = eval(args.tdnn_backbone_topology)
        assert len(self.tdnn_backbone_topology) == self.backbone_layer_num
        self.tdnn_layers = nn.ModuleList()
        for layer_idx in range(self.backbone_layer_num):
            topology = self.tdnn_backbone_topology[layer_idx]
            left_kernel_size = topology[0]
            right_kernel_size = topology[1]
            dilation = topology[2]
            if dilation > 0:
                self.tdnn_layers.append(
                    TDNN(
                        args.backbone_memory_size,
                        args.backbone_memory_size,
                        left_kernel_size,
                        right_kernel_size,
                        dilation=dilation,
                        normalization_fn='layer_norm',
                    )
                )
            else:
                self.tdnn_layers.append(None)

    def forward(self, front_end_out, acoustic_mask=None, frontend_shape="BTN", **_kwargs):
        '''forward'''
        front_end_out = rnnt_transpose(front_end_out, "BTN", frontend_shape)
        acoustic_mask = acoustic_mask.unsqueeze(2) if acoustic_mask is not None else None
        lstmp_input = front_end_out
        for layer_idx in range(self.backbone_layer_num):
            if self.tdnn_layers[layer_idx] is not None:
                lstmp_input = lstmp_input.transpose(1, 2).contiguous()
                lstmp_input = self.tdnn_layers[layer_idx](lstmp_input)
                lstmp_input = lstmp_input.transpose(1, 2).contiguous()
            lstmp_input = self.lstmp_layers[layer_idx](lstmp_input)
            if self.apply_mask and (acoustic_mask is not None):
                lstmp_input = lstmp_input * acoustic_mask
        return lstmp_input


class RelTransformerBackbone(nn.Module):
    """RelTransformerBackbone"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.backbone_layer_num = args.backbone_layer_num
        self.transformers = nn.ModuleList()
        assert args.front_end_type in ['VGGFrontEndNoFsmn', 'Conv2dPooling']
        if args.position_encoding_type == 'RelPositionalEncoding':
            self.pos_enc = RelPositionalEncoding(args.backbone_memory_size)
        else:
            raise RuntimeError("No such PositionalEncoding")
        self.causal_transformer = args.get('causal_transformer', False)
        self.look_ahead = args.get('look_ahead', 0)
        self.chunk_mask = args.get('chunk_mask', False)
        if self.chunk_mask:
            logging.info("Streaming Chunk-wise Rel-Transformer is Loading")
            self.mask_width = args.chunk_size
            self.mask_mem = args.mem_size
            self.max_length = 648
            self.look_ahead = args.get('look_ahead', 0)
            self.skip_rate = args.get('skip_rate', 1)
            self.attn_mask = self.get_mask()

        for _ in range(self.backbone_layer_num):
            self.transformers.append(
                RelTransformerLayer(
                    args.backbone_memory_size,
                    args.self_attn_heads,
                    args.backbone_hidden_size,
                    args.self_attn_dropout,
                    args.dropout,
                    activation_dropout=args.self_attn_activation_dropout,
                    activation=args.self_attn_activation_fn,
                    normalize_before=args.self_attn_layer_norm_before,
                    clamp_inf=args.get('backbone_clamp_inf', False),
                )
            )

    @torch.no_grad()
    def get_mask(self):
        """get chunk mask"""
        mask = torch.full([self.max_length, self.max_length], 1.0)
        look_ahead = self.look_ahead
        mask_chunk = self.mask_width + look_ahead
        mask_mem = self.mask_mem
        columns = self.max_length // mask_chunk

        if columns == 0:
            mask[:, :] = 0.0
        else:
            for chunk in range(columns):
                # chunk
                mask[
                    chunk * mask_chunk : (chunk + 1) * mask_chunk,
                    chunk * mask_chunk : (chunk + 1) * mask_chunk,
                ] = 0.0
                # memory
                mask[
                    chunk * mask_chunk : (chunk + 1) * mask_chunk,
                    max(chunk * mask_chunk - mask_mem - look_ahead, 0) : max(
                        chunk * mask_chunk - look_ahead, 0
                    ) : self.skip_rate,
                ] = 0.0
            # chunk
            mask[(chunk + 1) * mask_chunk :, (chunk + 1) * mask_chunk :] = 0.0
            # memory
            mask[
                (chunk + 1) * mask_chunk :,
                max((chunk + 1) * mask_chunk - mask_mem - look_ahead, 0) : max(
                    (chunk + 1) * mask_chunk - look_ahead, 0
                ) : self.skip_rate,
            ] = 0.0
        return mask.unsqueeze(0).cuda().bool()

    def hard_copy(self, x, pos, mask):
        """hard copy for lookahead"""
        # (T, B, F)
        expand_x = torch.Tensor().to(x)
        expand_pos = torch.Tensor().to(pos)
        expand_mask = torch.Tensor().to(mask)
        pos_l = torch.Tensor().to(pos)
        pos_r = torch.Tensor().to(pos)
        times = x.size(0) // self.mask_width
        tag = x.size(0) % self.mask_width
        mid = x.size(0) - 1
        if times == 0:
            return x, pos, mask
        for t in range(times):
            expand_x = torch.cat(
                (expand_x, x[t * self.mask_width : (t + 1) * self.mask_width + self.look_ahead]), 0
            )
            pos_r = torch.cat(
                (
                    pos_r,
                    pos[
                        :,
                        mid
                        + t * self.mask_width : mid
                        + (t + 1) * self.mask_width
                        + self.look_ahead,
                    ],
                ),
                1,
            )
            pos_l = torch.cat(
                (
                    pos[
                        :,
                        max(mid - (t + 1) * self.mask_width - self.look_ahead + 1, 0) : mid
                        - t * self.mask_width
                        + 1,
                    ],
                    pos_l,
                ),
                1,
            )
            expand_mask = torch.cat(
                (
                    expand_mask,
                    mask[:, t * self.mask_width : (t + 1) * self.mask_width + self.look_ahead],
                ),
                -1,
            )
        if tag > self.look_ahead:
            expand_x = torch.cat((expand_x, x[(t + 1) * self.mask_width :]), 0)
            pos_r = torch.cat((pos_r, pos[:, mid + (t + 1) * self.mask_width :]), 1)
            pos_l = torch.cat((pos[:, : mid - (t + 1) * self.mask_width + 1], pos_l), 1)
            expand_mask = torch.cat((expand_mask, mask[:, (t + 1) * self.mask_width :]), -1)
        expand_pos = torch.cat((pos_l, pos_r[:, 1:]), 1)
        return expand_x.to(x), expand_pos.to(pos), expand_mask.to(mask)

    def remove_hard_copy(self, x):
        """remove hard copy for output"""
        width = self.mask_width + self.look_ahead
        times = x.size(0) // width
        tag = x.size(0) % width
        contract_x = torch.Tensor().to(x)
        if times == 0:
            return x
        for t in range(times):
            contract_x = torch.cat((contract_x, x[t * width : t * width + self.mask_width]), 0)
        if tag == 0:
            contract_x = torch.cat((contract_x, x[t * width + self.mask_width :]), 0)
        else:
            contract_x = torch.cat((contract_x, x[(t + 1) * width :]), 0)
        return contract_x

    def forward(
        self, front_end_out, acoustic_mask=None, attn_mask=False, frontend_shape="BTN", **_kwargs
    ):
        """forward"""
        front_end_out = rnnt_transpose(front_end_out, "TBN", frontend_shape)
        transformer_input, pos_emb = self.pos_enc(front_end_out)
        if acoustic_mask is not None:
            transformer_mask = (1 - acoustic_mask).bool()
        else:
            transformer_mask = None
        if self.chunk_mask and self.look_ahead:
            transformer_input, pos_emb, transformer_mask = self.hard_copy(
                transformer_input, pos_emb, transformer_mask
            )
        tgt_len = transformer_input.size(0)
        if attn_mask and self.chunk_mask and tgt_len > self.max_length:
            self.max_length = tgt_len
            self.attn_mask = self.get_mask()
        for layer_idx in range(self.backbone_layer_num):
            if attn_mask and self.chunk_mask:
                chunk_attn_mask = self.attn_mask[:, :tgt_len, :tgt_len]
            else:
                chunk_attn_mask = None
            transformer_input = self.transformers[layer_idx](
                transformer_input, pos_emb, transformer_mask, chunk_attn_mask
            )
        if self.chunk_mask and self.look_ahead:
            transformer_input = self.remove_hard_copy(transformer_input)
        transformer_input = rnnt_transpose(transformer_input, "BTN", "TBN")
        return transformer_input


class ConformerBackbone(nn.Module):
    '''ConformerBackbone'''

    def __init__(self, args):
        # pylint:disable=too-many-branches
        super().__init__()
        attention_dim = args.backbone_memory_size
        num_blocks = args.conformer_num_blocks
        positional_dropout_rate = args.conformer_positional_dropout_rate

        pos_enc_layer_type = args.conformer_pos_enc_layer_type

        normalize_before = args.conformer_normalize_before
        self.normalize_before = normalize_before
        self.export_fused_conformer = args.get('export_fused_conformer', False)

        # mask
        conformer_mask_topology = args.conformer_mask_topology
        if conformer_mask_topology is not None:
            conformer_mask_topology = eval(conformer_mask_topology)
            assert (
                len(conformer_mask_topology) == num_blocks
            ), 'conformer_mask_topology not match num_blocks'
            self.export_rel_pos_embeding_len = 0
            for attn_left, attn_right in conformer_mask_topology:
                self.export_rel_pos_embeding_len = max(
                    self.export_rel_pos_embeding_len, attn_left, attn_right
                )
            self.export_rel_pos_embeding_len += 1
        else:
            conformer_mask_topology = [None] * num_blocks
            self.export_rel_pos_embeding_len = None
        dual_mode = args.get("conformer_dual_mode", False)
        if dual_mode:
            stream_conformer_mask_topology = args.get('stream_conformer_mask_topology', None)
            assert stream_conformer_mask_topology is not None
            stream_conformer_mask_topology = eval(stream_conformer_mask_topology)
            assert len(conformer_mask_topology) == len(
                stream_conformer_mask_topology
            ), 'stream_conformer_mask_topology must match conformer_mask_topology'
            self.export_rel_pos_embeding_len = 0
            if isinstance(stream_conformer_mask_topology[0][0], list):
                streaming_index = args.get('streaming_index', 0)
                stream_conformer_mask_topology = [
                    m[streaming_index] for m in stream_conformer_mask_topology
                ]
            for attn_left, attn_right in stream_conformer_mask_topology:
                self.export_rel_pos_embeding_len = max(
                    self.export_rel_pos_embeding_len, attn_left, attn_right
                )
            self.export_rel_pos_embeding_len += 1

        # subsample internal
        self.subsample_inter = args.get("subsample_inter", False)
        if self.subsample_inter:
            self.subsample_layer_index = args.get("subsample_layer_index", None)
            assert self.subsample_layer_index < num_blocks
            self.subsampler = AvgPoolTimeReduce()
            self.pos_enc_subsample = RelPositionalEncoding(
                attention_dim,
                positional_dropout_rate,
                dim=1,
            )

        # group attention reuse
        self.group_layer_idx = args.get('group_layer_idx', None)
        if self.group_layer_idx:
            self.group_layer_idx = set(eval(self.group_layer_idx))

        # net
        # if export to ONNX, conformer_positional_max_len is equal to
        # Max(attn_left_conext, attn_right_context) + 1
        positional_max_len = args.get('conformer_positional_max_len', 5000)
        if pos_enc_layer_type == "abs_pos":
            self.pos_enc = AbsPositionalEncoding(attention_dim, positional_dropout_rate)
        elif pos_enc_layer_type == "scaled_abs_pos":
            self.pos_enc = ScaledPositionalEncoding(attention_dim, positional_dropout_rate)
        elif pos_enc_layer_type in ("rel_pos", "fix_rel_pos"):
            assert args.conformer_selfattention_layer_type == "rel_selfattn"
            self.pos_enc = RelPositionalEncoding(
                attention_dim,
                positional_dropout_rate,
                dim=1,
                max_len=positional_max_len,
            )
        elif pos_enc_layer_type == "none":
            self.pos_enc = None
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        assert args.conformer_layernorm_interval <= 0
        assert not args.conformer_half_pooling
        encoders = []
        for lnum in range(num_blocks):
            encoders.append(ConformerLayer(args, conformer_mask_topology[lnum], lnum))
            if dual_mode:
                encoders[-1].self_attn.init_dual_mode_stream_mask(
                    stream_conformer_mask_topology[lnum]
                )
        self.encoders = nn.Sequential(*encoders)

        if self.normalize_before:
            self.after_norm = nn.LayerNorm(attention_dim)

        if args.conformer_weight_scale < 1.0:
            for _, param in self.named_parameters():
                param.data.mul_(args.conformer_weight_scale)

    @property
    def aux_loss(self):
        '''aux_loss'''
        loss = 0.0
        for layer in self.encoders:
            if layer.moe_ffn:
                loss += layer.aux_loss
        return loss

    def set_stream_mode(self, stream_mode=False):
        '''set_stream_mode'''
        for encoder in self.encoders:
            encoder.set_stream_mode(stream_mode)

    def set_stream_index(self, stream_index=0):
        '''set_stream_mode'''
        for encoder in self.encoders:
            encoder.set_stream_index(stream_index)

    def forward(self, front_end_out, acoustic_mask=None, frontend_shape="BTN", **_kwargs):
        """Encode input sequence.

        :param torch.Tensor front_end_out: input tensor # (B, T, N)
        :param torch.Tensor acoustic_mask: input mask   # (B, T)
        :rtype torch.Tensor:
        """
        # pylint:disable=too-many-branches
        front_end_out = rnnt_transpose(front_end_out, "BTN", frontend_shape)
        if (
            (torch.jit.is_scripting() or torch.jit.is_tracing())
            and self.pos_enc.__class__.__name__ == 'RelPositionalEncoding'
            and not self.training
            and self.export_fused_conformer
        ):
            conformer_input = self.pos_enc(front_end_out, self.export_rel_pos_embeding_len)
        else:
            conformer_input = self.pos_enc(front_end_out)
        if acoustic_mask is None:
            conformer_mask = None
        else:
            conformer_mask = acoustic_mask.unsqueeze(1)
        attn_weights = None
        for i, layer in enumerate(self.encoders):
            if self.subsample_inter and i == self.subsample_layer_index:
                if isinstance(conformer_input, tuple):
                    x = conformer_input[0]
                else:
                    x = conformer_input
                x = self.subsampler(x)
                conformer_input = self.pos_enc_subsample(x)
                conformer_mask = conformer_mask[:, :, ::2]
            if self.group_layer_idx:
                if i in self.group_layer_idx:
                    conformer_input, conformer_mask, attn_weights = layer(
                        [conformer_input, conformer_mask, None]
                    )
                else:
                    conformer_input, conformer_mask, _ = layer(
                        [conformer_input, conformer_mask, attn_weights]
                    )
            else:
                conformer_input, conformer_mask = layer([conformer_input, conformer_mask])
        if isinstance(conformer_input, tuple):
            conformer_input = conformer_input[0]

        if self.normalize_before:
            conformer_input = self.after_norm(conformer_input)
        return conformer_input  # , conformer_mask.squeeze(1)

    def forward_step(
        self,
        front_end_out,
        required_right_context,
        cache_list=None,
        acoustic_mask=None,
        frontend_shape="BTN",
    ):
        """Encode input sequence.

        :param torch.Tensor front_end_out: input tensor # (B, T, N)
        :param torch.Tensor acoustic_mask: input mask   # (B, T)
        :rtype torch.Tensor:
        """
        front_end_out = rnnt_transpose(front_end_out, "BTN", frontend_shape)
        conformer_input = self.pos_enc.forward_step(front_end_out)
        if acoustic_mask is None:
            conformer_mask = None
        else:
            conformer_mask = acoustic_mask.unsqueeze(1)
        if isinstance(conformer_input, tuple):
            x, pos_emb = conformer_input
            pe_max_len = int((pos_emb.size(1) + 1) / 2)
            attn_history_size = self.encoders[0].self_attn.left_kernel_size
            assert pe_max_len >= attn_history_size
            key_size = x.size(1) + attn_history_size
            pos_emb = pos_emb[:, pe_max_len - key_size : pe_max_len + key_size - 1]
            conformer_input = x, pos_emb

        for i, m in enumerate(self.encoders):
            att_cache, att_mask_cache, conv_cache = cache_list[i]
            conformer_input, conformer_mask, att_cache, att_mask_cache, conv_cache = m.forward_step(
                [conformer_input, conformer_mask],
                required_right_context,
                att_cache,
                att_mask_cache,
                conv_cache,
            )
            cache_list[i] = [att_cache, att_mask_cache, conv_cache]
        conformer_out = conformer_input

        if isinstance(conformer_out, tuple):
            conformer_out = conformer_out[0]

        if self.normalize_before:
            conformer_out = self.after_norm(conformer_out)
        return (
            conformer_out,
            required_right_context,
            cache_list,
            acoustic_mask,
        )  # , conformer_mask.squeeze(1)

    @property
    def state_size(self):
        '''streamed state size.'''
        state_size = sum(layer.state_size for layer in self.encoders)
        return state_size


class MaskedConformerBackbone(ConformerBackbone):
    '''
    MaskedConformerBackbone
    compatible with old code.
    '''


class EmformerBackbone(nn.Module):
    """EmformerBackbone"""

    def __init__(self, args):
        '''init'''
        super().__init__()
        self.backbone_layer_num = args.backbone_layer_num
        self.transformers = nn.ModuleList()
        self.chunk_size = args.chunk_size
        self.memory_chunk_size = args.get('memory_chunk_size', self.chunk_size)
        assert self.chunk_size % self.memory_chunk_size == 0
        self.left_context = args.get('left_context', 0)
        self.left_memory_bank = args.left_memory_bank
        self.chunk_memory_rate = self.chunk_size // self.memory_chunk_size
        self.chunk_inference = args.get('chunk_inference', False)
        self.max_length = 648
        self.attn_mask = self.get_attn_mask()
        self.x_len = None
        self.m_len = None
        self.attn_len = None

        for _ in range(self.backbone_layer_num):
            self.transformers.append(
                EmformerLayer(
                    args.backbone_memory_size,
                    args.self_attn_heads,
                    args.backbone_hidden_size,
                    args.self_attn_dropout,
                    args.dropout,
                    activation_dropout=args.self_attn_activation_dropout,
                    activation=args.self_attn_activation_fn,
                    normalize_before=args.self_attn_layer_norm_before,
                    clamp_inf=args.get('backbone_clamp_inf', False),
                    chunk_size=args.chunk_size,
                    memory_chunk_size=self.memory_chunk_size,
                    left_context=self.left_context,
                    chunk_inference=self.chunk_inference,
                )
            )

    def get_transformer_mask(self, acoustic_mask):
        '''get transformer mask
        Ags:
            acoustic_mask: (B, x_len)
        Returns:
            transformer_mask: (B, attn_len)
        '''
        tail_mask = acoustic_mask[:, :: self.memory_chunk_size][:, -1:]
        mem_mask = acoustic_mask[:, :: self.memory_chunk_size][:, 1 : self.m_len]
        acoustic_mask = torch.cat((acoustic_mask, mem_mask, tail_mask), -1)
        return acoustic_mask

    def get_attn_mask(self):
        '''get attn mask'''
        # 4 parts for mask
        # (Q + S) * (K + M)
        # Q-K, Q-M, S-K, S-M
        chunk_size = self.chunk_size
        mem_len = math.ceil(self.max_length / self.memory_chunk_size) - 1
        attn_len = self.max_length + mem_len
        mask = torch.full([attn_len, attn_len], True)
        for chunk in range(mem_len // self.chunk_memory_rate):
            # 1. Q-K
            left_bound = chunk * chunk_size
            right_bound = (chunk + 1) * chunk_size
            # chunk
            mask[left_bound:right_bound, left_bound:right_bound] = False
            # left_context
            mask[
                left_bound:right_bound, max(left_bound - self.left_context, 0) : left_bound
            ] = False
            # 2. Q-M
            flag = self.max_length + chunk * self.chunk_memory_rate
            mask[
                left_bound:right_bound, max(flag - self.left_memory_bank, self.max_length) : flag
            ] = False
            # 3. S-K
            mask[flag : flag + self.chunk_memory_rate, left_bound:right_bound] = False
            mask[
                flag : flag + self.chunk_memory_rate,
                max(left_bound - self.left_context, 0) : left_bound,
            ] = False
            flag += self.chunk_memory_rate
            # 4. S-M
            mask[
                flag : flag + self.chunk_memory_rate,
                max(flag - self.left_memory_bank, self.max_length) : flag,
            ] = False
        # last Q-K
        mask[
            right_bound : self.max_length, max(right_bound - self.left_context, 0) : self.max_length
        ] = False
        # last Q-M
        mask[
            right_bound : self.max_length, max(flag + 1 - self.left_memory_bank, self.max_length) :
        ] = False
        return mask.unsqueeze(0).cuda()

    def get_input_length(self, transformer_input):
        '''get x length and mem length'''
        self.x_len = transformer_input.size(0)
        self.m_len = math.ceil(self.x_len / self.memory_chunk_size) - 1
        self.attn_len = self.x_len + self.m_len

    def forward(self, front_end_out, acoustic_mask=None, attn_mask=False, frontend_shape="BTN"):
        """forward"""
        front_end_out = rnnt_transpose(front_end_out, "TBN", frontend_shape)
        transformer_input = front_end_out
        self.get_input_length(transformer_input)
        if acoustic_mask is not None:
            transformer_mask = self.get_transformer_mask(acoustic_mask)
            transformer_mask = (1 - transformer_mask).bool()
        else:
            transformer_mask = None

        # len of front_end_out and S
        if self.x_len > self.max_length:
            self.max_length = self.x_len
            self.attn_mask = self.get_attn_mask()
        end_length = self.max_length + self.m_len
        memory = None
        attn_mask1 = self.attn_mask[:, : self.x_len, : self.x_len]
        attn_mask2 = self.attn_mask[:, : self.x_len, self.max_length : end_length]
        attn_mask3 = self.attn_mask[:, self.max_length : end_length, : self.x_len]
        attn_mask4 = self.attn_mask[:, self.max_length : end_length, self.max_length : end_length]
        attn_mask = torch.cat(
            (torch.cat((attn_mask1, attn_mask2), -1), torch.cat((attn_mask3, attn_mask4), -1)), 1
        )
        for layer_idx in range(self.backbone_layer_num):
            transformer_input, memory = self.transformers[layer_idx](
                transformer_input,
                memory,
                transformer_mask,
                attn_mask,
            )
        return rnnt_transpose(transformer_input, "BTN", "TBN")


class CifSelfAttentionEncoder(CifSelfAttentionModule):
    '''CifSelfAttentionEncoder'''

    def __init__(self, args):
        super().__init__(args, sign='encoder')

    def augmented_bias(self, bias, layer, inp_shape, aug_length):
        '''update encoder_self_attention_bias with augmented_bias'''
        recal_bias_layer = [0] + self.args.sa_pooling_layers
        if layer in recal_bias_layer:
            device = bias.device
            if self.args.proximity_bias:
                aug_r = torch.arange(start=aug_length, end=0, step=-1, dtype=torch.float)
                inp_r = torch.arange(start=0, end=inp_shape[1], step=1, dtype=torch.float)
                if device is not None:
                    aug_r = aug_r.to(device)
                    inp_r = inp_r.to(device)
                proximity = aug_r.unsqueeze(0) + inp_r.unsqueeze(1)
                proximity = -torch.log(1.0 + proximity.abs())
                proximity = proximity.unsqueeze(0).unsqueeze(0)
                proximity = proximity.expand(inp_shape[0], -1, -1, -1)

                bias = torch.cat([proximity, bias], dim=-1)
            else:
                raise NotImplementedError('non-proximity bias not support yet')
        return bias

    def forward(self, inputs, ignore_padding, prev_kv_cache, **kwargs):
        '''
        Args:
          inputs: a Tensor
          bias: bias Tensor for self-attention
          ignore_padding: used for the calculation of down-sampled self-attention bias.
          prev_kv_cache: used for the streaming MultiHeadChunkwiseAttention
        '''
        assert kwargs.get('prev_conv_memory', None) is None

        # encoder self attention bias
        bias = ignore_padding
        if self.args.proximity_bias:
            bias = bias + self.attention_bias_proximal(ignore_padding.shape[3], bias.device)

        enc_inp = inputs
        for layer in range(self.args.num_encoder_layers):
            if prev_kv_cache is not None:
                aug_length = prev_kv_cache[layer]['len']
                bias = self.augmented_bias(bias, layer, enc_inp.shape, aug_length)
                enc_inp, _, layer_cache = self.atten[layer](
                    enc_inp, bias, None, prev_kv_cache[layer]
                )
                prev_kv_cache[layer] = layer_cache
            else:
                enc_inp, _, _ = self.atten[layer](enc_inp, bias, None, None)

            enc_inp = self.ffn[layer](enc_inp)

            if layer + 1 in self.args.sa_pooling_layers:
                enc_inp = enc_inp.unsqueeze(1)
                enc_inp = F.max_pool2d(enc_inp, (2, 1), (2, 1), ceil_mode=True)
                enc_inp = enc_inp.squeeze(1)

                ignore_padding = F.max_pool2d(ignore_padding, (1, 2), (1, 2), ceil_mode=True)
                bias = ignore_padding

                if self.args.proximity_bias:
                    bias = bias + self.attention_bias_proximal(ignore_padding.shape[3], bias.device)
                else:
                    raise NotImplementedError('non-proximity bias not support yet')

        not_padding = (ignore_padding == 0.0).int()
        not_padding = not_padding.squeeze(1).squeeze(1)

        enc_inp = self.process(enc_inp)
        if self.args.extra_ce_encoder:
            return enc_inp, ignore_padding, not_padding, prev_kv_cache
        return enc_inp, not_padding, prev_kv_cache


class CifConformerV2Encoder(CifConformerV2Module):
    '''CifConformerV2Encoder'''

    def __init__(self, args):
        super().__init__(args, sign='encoder')

    def augmented_bias(self, bias, layer, inp_shape, aug_length):
        '''update encoder_self_attention_bias with augmented_bias'''
        recal_bias_layer = [0] + self.args.sa_pooling_layers
        if layer in recal_bias_layer:
            device = bias.device
            augmented_bias = torch.zeros(
                inp_shape[0], 1, 1, aug_length, dtype=torch.float32, device=device
            )
            bias = torch.cat([augmented_bias, bias], dim=-1)
        return bias

    def forward(self, inputs, ignore_padding, prev_kv_cache, prev_conv_memory=None):
        '''
        Args:
          inputs: a Tensor
          bias: bias Tensor for self-attention
          ignore_padding: used for the calculation of down-sampled self-attention bias.
          prev_kv_cache: used for the streaming MultiHeadChunkwiseAttention
          prev_conv_memory: used for the streaming chunkwise conv calculation
        '''
        # encoder self attention bias
        bias = ignore_padding

        enc_inp = inputs
        for layer in range(self.args.num_encoder_layers):
            # before mha part
            enc_inp = self.macaron_ffn1[layer](enc_inp)

            if prev_conv_memory is not None:
                enc_inp, updated_conv_memory = self.conv_module[layer](
                    enc_inp, conv_memory=prev_conv_memory[layer]
                )
                prev_conv_memory[layer] = updated_conv_memory
            else:
                enc_inp = self.conv_module[layer](enc_inp, None, None)

            # mha part
            if prev_kv_cache is not None:
                aug_length = prev_kv_cache[layer]['len']
                bias = self.augmented_bias(bias, layer, enc_inp.shape, aug_length)
                enc_inp, _, layer_cache = self.atten[layer](
                    enc_inp, bias, None, prev_kv_cache[layer]
                )
                prev_kv_cache[layer] = layer_cache
            else:
                enc_inp, _, _ = self.atten[layer](enc_inp, bias, None, None)

            # after mha part
            enc_inp = self.macaron_ffn2[layer](enc_inp)

            if layer + 1 in self.args.sa_pooling_layers:
                enc_inp = enc_inp.unsqueeze(1)
                enc_inp = F.max_pool2d(enc_inp, (2, 1), (2, 1), ceil_mode=True)
                enc_inp = enc_inp.squeeze(1)

                ignore_padding = F.max_pool2d(ignore_padding, (1, 2), (1, 2), ceil_mode=True)
                bias = ignore_padding

        not_padding = (ignore_padding == 0.0).int()
        not_padding = not_padding.squeeze(1).squeeze(1)

        enc_inp = self.process(enc_inp)
        return enc_inp, not_padding, prev_kv_cache
