''' RnntLasRescoreModel and RnntDeliberationModel '''
# pylint:disable=too-many-lines,too-many-branches
from collections import OrderedDict
import copy
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from core.models.asr.las_hyp_encoder import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.asr.las_decoder import *
from core.criterions import *
from core.solutions.inference.las_beam_search import BaseBeamSearch
from core.models.layers.embedding import AbsPositionalEncoding
from core.models.layers.time_reduce_layer import *
from core.solutions.base_solution import register_solution
from core.solutions.inference import (
    BaseInfer,
    INFERS,
)
from core.utils import logging, get_local_rank, FalconDict
from .base_rnnt_model import BaseRnntModel


class LasEncoderExporter(BaseInfer):
    '''export las extra encoder to onnx'''

    NAME = 'las_encoder'
    INPUTS = [
        FalconDict(name='backbone_out', type=torch.float32, shape=['B', 'T', -1]),
    ]
    OUTPUTS = [
        FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T', -1]),
    ]

    ORIGIN_TEST_ATOL = 1e-3
    OPTIMIZED_TEST_ATOL = 1e-3

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.las_acoustic_backbone_module = model.las_acoustic_backbone_module
        self.las_backbone_final_norm = model.las_backbone_final_norm
        self.las_rescore_with_mask = kwargs.get("las_rescore_with_mask", False)
        self.backbone_memory_size = kwargs.get("backbone_memory_size", -1)
        assert self.backbone_memory_size > 0
        self._inputs[0].shape[2] = self.backbone_memory_size
        if self.las_rescore_with_mask:
            self._inputs.append(
                FalconDict(name='backbone_mask', type=torch.float32, shape=['B', 'T'])
            )

        self._convert_stream_flag = False
        self._conv_w_list = None
        self.eval()

    @torch.no_grad()
    def forward(self, backbone_out, backbone_mask=None):
        '''
        additional las encoder layers above the rnnt encoder layers
        '''
        if self.las_acoustic_backbone_module is None:
            return backbone_out
        encoder_out = self.las_acoustic_backbone_module(
            backbone_out, backbone_mask, frontend_shape="BTN"
        )
        if self.las_backbone_final_norm is not None:
            encoder_out = self.las_backbone_final_norm(encoder_out)
        return encoder_out

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        backbone_out = self._generate_input_data(0, dynamic_axis=[1, 128, -1])
        datas = [backbone_out]
        if self.las_rescore_with_mask:
            backbone_mask = self._generate_input_data(1, method='ones', dynamic_axis=[1, 128])
            datas.append(backbone_mask)
        return tuple(datas)


class RnntLasLstmDecoderExporter(BaseInfer):
    '''export las lstm decoder to onnx
    1. replace lstmcell by lstm to export onnx. Using lstmcell
        during training for efficiency, and use lstm for onnx export.
    2. remove useless input: attention state
    3. use jit.script to handle the dynamic length of an input sequence
    '''

    NAME = 'las_decoder'
    INPUTS = [
        FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='prev_char', type=torch.long, shape=['B', 'T']),
        FalconDict(name='prev_char_bw', type=torch.long, shape=['B', 'T']),
    ]
    OUTPUTS = [
        FalconDict(name='logprobs', type=torch.float32, shape=['B', 'T']),
        FalconDict(name='logprobs_bw', type=torch.float32, shape=['B', 'T']),
    ]

    ORIGIN_TEST_ATOL = 1
    ORIGIN_TEST_RTOL = 5e-1
    ORIGIN_TEST_MSE = 0.3
    OPTIMIZED_TEST_ATOL = 1
    OPTIMIZED_TEST_RTOL = 5e-1
    OPTIMIZED_TEST_MSE = 0.3

    def __init__(self, args, decoder_module, decoder_bw_module, **kwargs):
        kwargs['need_jit_script'] = True
        super().__init__(kwargs)
        self._inputs[0].shape[2] = args.backbone_memory_size
        self.encoder_proj_fc = torch.jit.trace(
            decoder_module.encoder_proj_fc,
            torch.rand(1, 512, args.backbone_memory_size, device='cuda'),
        )
        self.embed_tokens = torch.jit.trace(
            decoder_module.embed_tokens, torch.ones(1, 128, device='cuda').long()
        )
        self.attention = torch.jit.trace(
            decoder_module.attention,
            (
                torch.rand(1, 512, args.backbone_memory_size, device='cuda'),
                torch.rand(1, 512, args.atten_hidden_size, device='cuda'),
                torch.rand(1, args.embedding_size, device='cuda'),
                torch.rand(1, args.decoder_lstm_hidden_size, device='cuda'),
            ),
        )
        self.lstm = [
            torch.jit.trace(
                decoder_module.lstm[0],
                (
                    torch.rand(1, 1, args.decoder_input_size + args.embedding_size, device='cuda'),
                    (
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                    ),
                ),
            )
        ]
        self.lstm += [
            torch.jit.trace(
                decoder_module.lstm[layer],
                (
                    torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                    (
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                    ),
                ),
            )
            for layer in range(1, args.decoder_lstm_layer_num)
        ]
        self.lstm = nn.ModuleList(self.lstm)
        self.pred_fc = torch.jit.trace(
            decoder_module.pred_fc,
            torch.rand(
                1,
                127,
                args.decoder_lstm_hidden_size + args.backbone_memory_size + args.embedding_size,
                device='cuda',
            ),
        )

        # backward las decoder
        self.encoder_proj_fc_bw = torch.jit.trace(
            decoder_bw_module.encoder_proj_fc,
            torch.rand(1, 512, args.backbone_memory_size, device='cuda'),
        )
        self.embed_tokens_bw = torch.jit.trace(
            decoder_bw_module.embed_tokens, torch.ones(1, 128, device='cuda').long()
        )
        self.attention_bw = torch.jit.trace(
            decoder_bw_module.attention,
            (
                torch.rand(1, 512, args.backbone_memory_size, device='cuda'),
                torch.rand(1, 512, args.atten_hidden_size, device='cuda'),
                torch.rand(1, args.embedding_size, device='cuda'),
                torch.rand(1, args.decoder_lstm_hidden_size, device='cuda'),
            ),
        )
        self.lstm_bw = [
            torch.jit.trace(
                decoder_bw_module.lstm[0],
                (
                    torch.rand(1, 1, args.decoder_input_size + args.embedding_size, device='cuda'),
                    (
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                    ),
                ),
            )
        ]
        self.lstm_bw += [
            torch.jit.trace(
                decoder_bw_module.lstm[layer],
                (
                    torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                    (
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                        torch.rand(1, 1, args.decoder_lstm_hidden_size, device='cuda'),
                    ),
                ),
            )
            for layer in range(1, args.decoder_lstm_layer_num)
        ]
        self.lstm_bw = nn.ModuleList(self.lstm_bw)
        self.pred_fc_bw = torch.jit.trace(
            decoder_bw_module.pred_fc,
            torch.rand(
                1,
                127,
                args.decoder_lstm_hidden_size + args.backbone_memory_size + args.embedding_size,
                device='cuda',
            ),
        )

        self.decoder_lstm_hidden_size = torch.tensor(decoder_module.decoder_lstm_hidden_size)
        self.decoder_lstm_layer_num = torch.tensor(decoder_module.decoder_lstm_layer_num)
        self._convert_stream_flag = False
        self._conv_w_list = None
        self.eval()

    def forward(self, encoder_out, prev_char, prev_char_bw):
        '''
        Forward for RNN-T Las decoder module
        '''
        prev_tgt = prev_char[:, :-1]
        prev_tgt_bw = prev_char_bw[:, :-1]
        encoder_proj = self.encoder_proj_fc(encoder_out)
        encoder_proj_bw = self.encoder_proj_fc_bw(encoder_out)
        prev_emb = self.embed_tokens(prev_tgt)
        prev_emb_bw = self.embed_tokens_bw(prev_tgt_bw)

        # repeat encoder beams times for efficiency
        bsz = prev_tgt.shape[0]
        encoder_out = encoder_out.repeat(bsz, 1, 1)
        encoder_proj = encoder_proj.repeat(bsz, 1, 1)
        encoder_proj_bw = encoder_proj_bw.repeat(bsz, 1, 1)

        lstms_state0 = torch.zeros(
            self.decoder_lstm_layer_num, 1, bsz, self.decoder_lstm_hidden_size, device='cuda'
        )
        lstms_state1 = torch.zeros(
            self.decoder_lstm_layer_num, 1, bsz, self.decoder_lstm_hidden_size, device='cuda'
        )

        total_concat_out = torch.zeros(
            bsz,
            1,
            prev_emb.shape[-1] + encoder_out.shape[-1] + self.decoder_lstm_hidden_size,
            device='cuda',
        )

        for token_idx in range(prev_tgt.shape[1]):
            att_ctx = self.attention(
                encoder_out, encoder_proj, prev_emb[:, token_idx, :], lstms_state0[-1][0]
            )
            lstm_input = torch.cat([att_ctx, prev_emb[:, token_idx, :]], dim=1).unsqueeze(1)
            new_lstms_state0 = torch.zeros(1, 1, bsz, self.decoder_lstm_hidden_size, device='cuda')
            new_lstms_state1 = torch.zeros(1, 1, bsz, self.decoder_lstm_hidden_size, device='cuda')
            for index, layer in enumerate(self.lstm):
                lstm_out, (lstm_state0, lstm_state1) = layer(
                    lstm_input, (lstms_state0[index], lstms_state1[index])
                )
                lstm_input = lstm_out
                new_lstms_state0 = torch.cat([new_lstms_state0, lstm_state0.unsqueeze(0)])
                new_lstms_state1 = torch.cat([new_lstms_state1, lstm_state1.unsqueeze(0)])
            concat_out = torch.cat([prev_emb[:, token_idx, :], att_ctx, lstm_out[:, 0, :]], dim=1)
            total_concat_out = torch.cat([total_concat_out, concat_out.unsqueeze(1)], dim=1)
            lstms_state0 = new_lstms_state0[1:]
            lstms_state1 = new_lstms_state1[1:]

        logits = self.pred_fc(total_concat_out[:, 1:, :])
        logprobs = torch.gather(
            F.log_softmax(logits, dim=-1), dim=2, index=prev_char[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # backward las decoder
        lstms_state0 = torch.zeros(
            self.decoder_lstm_layer_num, 1, bsz, self.decoder_lstm_hidden_size, device='cuda'
        )
        lstms_state1 = torch.zeros(
            self.decoder_lstm_layer_num, 1, bsz, self.decoder_lstm_hidden_size, device='cuda'
        )

        total_concat_out = torch.zeros(
            bsz,
            1,
            prev_emb_bw.shape[-1] + encoder_out.shape[-1] + self.decoder_lstm_hidden_size,
            device='cuda',
        )

        for token_idx in range(prev_tgt_bw.shape[1]):
            att_ctx = self.attention_bw(
                encoder_out, encoder_proj_bw, prev_emb_bw[:, token_idx, :], lstms_state0[-1][0]
            )
            lstm_input = torch.cat([att_ctx, prev_emb_bw[:, token_idx, :]], dim=1).unsqueeze(1)
            new_lstms_state0 = torch.zeros(1, 1, bsz, self.decoder_lstm_hidden_size, device='cuda')
            new_lstms_state1 = torch.zeros(1, 1, bsz, self.decoder_lstm_hidden_size, device='cuda')
            for index, layer in enumerate(self.lstm_bw):
                lstm_out, (lstm_state0, lstm_state1) = layer(
                    lstm_input, (lstms_state0[index], lstms_state1[index])
                )
                lstm_input = lstm_out
                new_lstms_state0 = torch.cat([new_lstms_state0, lstm_state0.unsqueeze(0)])
                new_lstms_state1 = torch.cat([new_lstms_state1, lstm_state1.unsqueeze(0)])
            concat_out = torch.cat(
                [prev_emb_bw[:, token_idx, :], att_ctx, lstm_out[:, 0, :]], dim=1
            )
            total_concat_out = torch.cat([total_concat_out, concat_out.unsqueeze(1)], dim=1)
            lstms_state0 = new_lstms_state0[1:]
            lstms_state1 = new_lstms_state1[1:]

        logits = self.pred_fc_bw(total_concat_out[:, 1:, :])
        logprobs_bw = torch.gather(
            F.log_softmax(logits, dim=-1), dim=2, index=prev_char_bw[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        return logprobs, logprobs_bw

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        encoder_out = self._generate_input_data(0, dynamic_axis=[1, 512, -1])
        prev_char = self._generate_input_data(1, method='ones', dynamic_axis=[1, 16])
        prev_char_bw = self._generate_input_data(2, method='ones', dynamic_axis=[1, 16])
        datas = [encoder_out, prev_char, prev_char_bw]
        return tuple(datas)


class RnntLasTransformerDecoderExporter(BaseInfer):
    '''export transformer decoder onnx'''

    NAME = 'las_decoder'
    INPUTS = [
        FalconDict(name='las_encoder', type=torch.float32, shape=['B', 'T', -1]),
        FalconDict(name='target_rnnt', type=torch.long, shape=['B', 'T']),
        FalconDict(name='target_rnnt_bw', type=torch.long, shape=['B', 'T']),
    ]
    OUTPUTS = []
    ORIGIN_TEST_ATOL = 1e0
    OPTIMIZED_TEST_ATOL = 1e0
    ORIGIN_TEST_MSE = 0.1
    OPTIMIZED_TEST_MSE = 0.1

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.decoder_module = model.decoder_module
        self.decoder_module.prepare_for_onnx_export_()
        self.decoder_bw_module = model.decoder_bw_module
        self.decoder_bw_module.prepare_for_onnx_export_()
        self.backbone_memory_size = kwargs.get("backbone_memory_size", -1)
        assert self.backbone_memory_size > 0
        self._inputs[0].shape[2] = self.backbone_memory_size
        self.las_rescore_with_mask = model.args.get("las_rescore_with_mask", False)
        if self.las_rescore_with_mask:
            self._inputs.append(FalconDict(name='encoder_mask', type=torch.float, shape=['B', 'T']))
            self._inputs.append(FalconDict(name='target_mask', type=torch.float, shape=['B', 'T']))
            self._outputs.append(
                FalconDict(name='output', type=torch.float32, shape=['B', 'N', 'T'])
            )
            self._outputs.append(
                FalconDict(name='output_bw', type=torch.float32, shape=['B', 'N', 'T'])
            )
        else:
            self._outputs.append(FalconDict(name='output', type=torch.float32, shape=['B', 'T']))
            self._outputs.append(FalconDict(name='output_bw', type=torch.float32, shape=['B', 'T']))
        self._convert_stream_flag = False
        self._conv_w_list = None
        self.eval()

    def forward(
        self, las_encoder, target_rnnt, target_rnnt_bw, encoder_mask=None, target_mask=None
    ):
        '''
        Forward for RNN-T Las decoder module
        '''
        # repeat encoder relatedbeams times for efficiency
        if not self.las_rescore_with_mask:
            prev_tgt = target_rnnt[:, :-1]
            prev_tgt_bw = target_rnnt_bw[:, :-1]
            target_mask = torch.ones_like(prev_tgt).float()
            nbest = prev_tgt.shape[0]
            encoder_out = las_encoder.repeat(nbest, 1, 1)
            encoder_mask = torch.ones_like(encoder_out[:, :, 0])
        else:
            bsz, _, encoder_dim = las_encoder.size()
            prev_tgt = target_rnnt[:, :-1]
            prev_tgt_bw = target_rnnt_bw[:, :-1]
            nbest = prev_tgt.shape[0] // bsz
            target_mask = target_mask[:, 1:]
            encoder_mask = encoder_mask.unsqueeze(1).repeat(1, nbest, 1)  # [B, nbest, T]
            encoder_mask = encoder_mask.view(bsz * nbest, -1).contiguous()
            encoder_out = las_encoder.unsqueeze(1).repeat(1, nbest, 1, 1)  # [B, nbest, T, N]
            encoder_out = encoder_out.view(bsz * nbest, -1, encoder_dim).contiguous()

        # repeat encoder beams times for efficiency
        logits = self.decoder_module(encoder_out, encoder_mask, prev_tgt, prev_tgt_mask=target_mask)
        logprobs = torch.gather(
            F.log_softmax(logits, dim=-1), dim=2, index=target_rnnt[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # backward las decoder
        logits = self.decoder_bw_module(
            encoder_out, encoder_mask, prev_tgt_bw, prev_tgt_mask=target_mask
        )
        logprobs_bw = torch.gather(
            F.log_softmax(logits, dim=-1), dim=2, index=target_rnnt_bw[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        if self.las_rescore_with_mask:
            # [B, Nbest, U]
            logprobs = logprobs.view(bsz, nbest, -1).contiguous()
            logprobs_bw = logprobs_bw.view(bsz, nbest, -1).contiguous()
        return logprobs, logprobs_bw

    def sample_inputs(self):
        '''
        sample some inputs as example.
        '''
        las_encoder = self._generate_input_data(0, dynamic_axis=[1, 128, -1])
        if self.las_rescore_with_mask:
            target_rnnt = self._generate_input_data(1, method='ones', dynamic_axis=[10, 128])
            target_rnnt_bw = self._generate_input_data(2, method='ones', dynamic_axis=[10, 128])
            encoder_mask = self._generate_input_data(3, method='ones', dynamic_axis=[1, 128])
            target_mask = self._generate_input_data(4, method='ones', dynamic_axis=[10, 128])
            datas = [las_encoder, target_rnnt, target_rnnt_bw, encoder_mask, target_mask]
        else:
            target_rnnt = self._generate_input_data(1, method='ones', dynamic_axis=[1, 128])
            target_rnnt_bw = self._generate_input_data(2, method='ones', dynamic_axis=[1, 128])
            datas = [las_encoder, target_rnnt, target_rnnt_bw]
        return tuple(datas)


@register_solution("RnntLasRescoreModel")
class RnntLasRescoreModel(BaseRnntModel):
    '''Rnnt Two Pass model with LAS rescore'''

    def __init__(self, args):
        super().__init__(args)
        self.update_steps = 0
        self.init_acoustic_encoder(args)
        self.init_decoder(args)
        self._register_load_state_dict_pre_hook(self.compatible_load_hook)

    @staticmethod
    def compatible_load_hook(
        state_dict, _prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, error_msgs
    ):
        '''
        compatible for loading LAS Rescore encoder
        '''
        for key in list(state_dict.keys()):
            new_key = key.replace('las_added_backbone_module', 'las_acoustic_backbone_module')
            val = state_dict.pop(key, None)
            if state_dict.get(new_key, None) is not None:
                error_msgs.append(
                    'Both {0} and {1} exist, {1} will be overrided'.format(key, new_key)
                )
            state_dict[new_key] = val

    def init_acoustic_encoder(self, args):
        '''init acoustic encoder'''
        acoustic_args = copy.deepcopy(args)
        acoustic_args.update(args.twopass_acoustic_args)
        self.las_acoustic_backbone_type = acoustic_args.get('backbone_type', None)
        if self.las_acoustic_backbone_type is not None:
            self.las_backbone_final_norm = None
            if 'TransformerBackbone' in self.las_acoustic_backbone_type:
                acoustic_args.backbone_layer_num = len(eval(acoustic_args.backbone_topology))
                if (
                    acoustic_args.self_attn_layer_norm_before
                    and acoustic_args.backbone_layer_num > 0
                    and 'Transformer' in args.twopass_decoder_args.get('las_decoder_type', None)
                ):
                    self.las_backbone_final_norm = nn.LayerNorm(acoustic_args.backbone_memory_size)
            elif self.las_acoustic_backbone_type == 'LSTMPBackbone':
                if acoustic_args.backbone_bilstm:
                    assert acoustic_args.backbone_mask, "Using bilstm must set las_hyp_mask = True"
            else:
                raise RuntimeError("las_acoustic_backbone_type does not supported")
            self.acoustic_args = acoustic_args
            self.las_acoustic_backbone_module = eval(self.las_acoustic_backbone_type)(
                self.acoustic_args
            )
        else:
            self.las_acoustic_backbone_module = None

    def init_decoder(self, args):
        '''init decoder'''
        args = copy.deepcopy(args)
        args.update(args.twopass_decoder_args)
        if args.reorder_tgt_dict:
            self.bos = args.reorder_tgt_dict.bos()
            self.eos = args.reorder_tgt_dict.eos()
        else:
            self.bos = args.tgt_dict.bos()
            self.eos = args.tgt_dict.eos()

        self.fw_decoder_weight = args.get('las_forward_decoder_weight', 0.5)
        self.las_decoder_type = args.las_decoder_type

        if args.get('las_forward_decoder', True):
            self.decoder_module = eval(args.las_decoder_type)(args)
        else:
            self.decoder_module = None
            self.fw_decoder_weight = 0.0
        if args.get('las_backward_decoder', False):
            self.decoder_bw_module = eval(args.las_decoder_type)(args)
        else:
            self.decoder_bw_module = None
            self.fw_decoder_weight = 1.0
        assert (
            self.decoder_module is not None or self.decoder_bw_module is not None
        ), "at least one las decoder is needed"
        self.las_criterion_module = eval(args.las_criterion_type)(args)

    def las_encoder(self, batch_data, **_kwargs):
        '''additional las encoder layers above the rnnt encoder layers'''
        if 'las_acoustic_encoder_out' in batch_data:
            return batch_data['las_acoustic_encoder_out'], batch_data['backbone_mask']
        if self.las_acoustic_backbone_module is None:
            return batch_data['encoder_backbone_out'], batch_data['backbone_mask']
        encoder_backbone_out = batch_data['encoder_backbone_out']
        backbone_mask = batch_data['backbone_mask']

        encoder_out = self.las_acoustic_backbone_module(
            encoder_backbone_out, backbone_mask, frontend_shape="BTN"
        )
        if self.las_backbone_final_norm is not None:
            encoder_out = self.las_backbone_final_norm(encoder_out)
        return encoder_out, backbone_mask

    def las_decoder(self, encoder_out, backbone_mask, prev_char, prev_char_mask=None):
        '''
        Decoder Module in LAS
        '''
        if prev_char_mask is not None:
            fw_logits = self.decoder_module(encoder_out, backbone_mask, prev_char)
        else:
            # transformer decoder
            fw_logits = self.decoder_module(
                encoder_out, backbone_mask, prev_char, prev_tgt_mask=prev_char_mask
            )

        return fw_logits

    def las_forward(self, batch_data):
        '''forward'''
        encoder_out, backbone_mask = self.las_encoder(batch_data)
        src_mask = batch_data['src_mask']
        target_mask = batch_data['char_mask']
        fw_forward_out = None
        bw_forward_out = None
        if self.decoder_module is not None:
            target = batch_data['char']
            if 'Transformer' in self.las_decoder_type:
                fw_logits = self.decoder_module(
                    encoder_out, backbone_mask, batch_data['prev_char'], prev_tgt_mask=target_mask
                )
            else:
                fw_logits = self.decoder_module(encoder_out, backbone_mask, batch_data['prev_char'])
            fw_forward_out = self.las_criterion_module(fw_logits, src_mask, target, target_mask)
        if self.decoder_bw_module is not None:
            target_rev = batch_data['char_rev']
            if 'Transformer' in self.las_decoder_type:
                bw_logits = self.decoder_bw_module(
                    encoder_out, backbone_mask, batch_data['prev_char_rev'], target_mask
                )
            else:
                bw_logits = self.decoder_bw_module(
                    encoder_out, backbone_mask, batch_data['prev_char_rev']
                )
            bw_forward_out = self.las_criterion_module(bw_logits, src_mask, target_rev, target_mask)

        if fw_forward_out is not None and bw_forward_out is not None:
            forward_out = OrderedDict()
            forward_out['utt_num'] = fw_forward_out['utt_num']
            forward_out['backward_loss'] = (
                self.fw_decoder_weight * fw_forward_out['backward_loss']
                + (1 - self.fw_decoder_weight) * bw_forward_out['backward_loss']
            )
            forward_out['loss'] = (
                self.fw_decoder_weight * fw_forward_out['loss']
                + (1 - self.fw_decoder_weight) * bw_forward_out['loss']
            )
            forward_out['nll_loss'] = (
                self.fw_decoder_weight * fw_forward_out['nll_loss']
                + (1 - self.fw_decoder_weight) * bw_forward_out['nll_loss']
            )
            forward_out['acc'] = (
                self.fw_decoder_weight * fw_forward_out['acc']
                + (1 - self.fw_decoder_weight) * bw_forward_out['acc']
            )
            forward_out['frame_size'] = fw_forward_out['frame_size']
            forward_out['tgt_size'] = fw_forward_out['tgt_size']
            forward_out['fw_logits'] = fw_logits
            forward_out['bw_logits'] = bw_logits
        elif fw_forward_out is not None:
            forward_out = fw_forward_out
            forward_out['fw_logits'] = fw_logits
        elif bw_forward_out is not None:
            forward_out = bw_forward_out
            forward_out['bw_logits'] = bw_logits
        else:
            raise RuntimeError("neither forward decoder nor backward decoder")
        return forward_out

    def prepare_rnnt_encoder_out(self, batch_data):
        '''get rnnt encoder out'''
        acoustic_out, backbone_mask, _, mtl_logits, encoder_backbone_out = self.encoder(batch_data)
        batch_data['encoder_out'] = acoustic_out
        batch_data['backbone_mask'] = backbone_mask
        batch_data['mtl_logits'] = mtl_logits
        batch_data['encoder_backbone_out'] = encoder_backbone_out

    @torch.no_grad()
    def generate_rnnt_nbest_data(self, batch_data, nbest=1, **kwargs):
        '''generate rnnt nbest output'''
        output, _, _ = super().beam_inference(batch_data, nbest=nbest, **kwargs)
        rlt_list_nbest = output["nbest"]
        bsz = len(rlt_list_nbest)
        max_char_length = 0
        rnnt_scores = []
        hotword_fst_scores = []
        out_rlt_list_nbest = []
        len_frames = (batch_data['backbone_mask'].sum(dim=1)).int().tolist()
        enable_prefetch = kwargs.get("prefetch", False)
        for bid, rlt_nbest in enumerate(rlt_list_nbest):
            nbest_score = ['-1e8|0.0'] * nbest  # default 'rnnt_score|confidence_seq'
            nbest_list = [''] * nbest
            nbest_hotword_fst_scores = [''] * nbest
            for beam_idx, x in enumerate(rlt_nbest):
                max_char_length = max(len(x[0]) + 2, max_char_length)
                if enable_prefetch and beam_idx == 0:
                    prefetch_frames = x[3]
                    save_frames = 0
                    if prefetch_frames != -1:
                        save_frames = len_frames[bid] - prefetch_frames - 1
                        len_frames[bid] = prefetch_frames + 1
                        batch_data['backbone_mask'][bid, prefetch_frames + 1 :] = 0
                    logging.info(
                        "rank %d, uttid %s, prefetching saves %d frames.",
                        get_local_rank(),
                        batch_data["uttid"][bid],
                        save_frames,
                    )
                nbest_rnnt_score = float(x[2].split('|', 1)[0]) / len_frames[bid]
                nbest_confidence_score = ''.join(x[2].split('|', 1)[1:])
                nbest_score[beam_idx] = '{}|{}'.format(
                    str(nbest_rnnt_score), nbest_confidence_score
                )
                nbest_hotword_fst_scores[beam_idx] = x[1]
                nbest_list[beam_idx] = x[0]
            rnnt_scores.append(nbest_score)
            hotword_fst_scores.append(nbest_hotword_fst_scores)
            out_rlt_list_nbest.append(nbest_list)
        rnnt_hyp = np.ones((bsz, nbest, max_char_length), dtype='int') * self.eos
        rnnt_bw_hyp = np.ones((bsz, nbest, max_char_length), dtype='int') * self.bos
        rnnt_hyp_mask = np.zeros((bsz, nbest, max_char_length), dtype='float')
        for bid, rlt_nbest in enumerate(rlt_list_nbest):
            for beam_idx, x in enumerate(rlt_nbest):
                char_num = len(x[0]) + 2
                rnnt_hyp_mask[bid, beam_idx][:char_num] = 1
                rnnt_hyp[bid, beam_idx][:char_num] = [self.bos] + x[0] + [self.eos]
                rnnt_bw_hyp[bid, beam_idx][:char_num] = [self.eos] + x[0][::-1] + [self.bos]
        rnnt_hyp = torch.from_numpy(rnnt_hyp).cuda().long()
        rnnt_bw_hyp = torch.from_numpy(rnnt_bw_hyp).cuda().long()
        rnnt_hyp_mask = torch.from_numpy(rnnt_hyp_mask).cuda().float()
        batch_data['rnnt_hyp'] = rnnt_hyp
        batch_data['rnnt_bw_hyp'] = rnnt_bw_hyp
        batch_data['rnnt_hyp_mask'] = rnnt_hyp_mask

        return rnnt_scores, hotword_fst_scores, out_rlt_list_nbest

    @classmethod
    def compute_las_score(cls, decoder_module, acoustic_out, backbone_mask, target, target_mask):
        '''
        Compute Las Score
        '''
        if decoder_module is None:
            return [0.0] * target.size(0)
        logits = decoder_module(acoustic_out, backbone_mask, target[:, :-1])
        logprobs = torch.gather(
            F.log_softmax(logits, dim=-1), dim=2, index=target[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        las_scores = logprobs.masked_fill(target_mask[:, 1:], 0).sum(dim=1).tolist()
        hyp_lens = target_mask[:, 1:].eq(0).sum(dim=1).tolist()
        las_scores = [score / hyp_lens[i] for i, score in enumerate(las_scores)]
        return las_scores

    @classmethod
    def expand_acoustic_out(cls, acoustic_outs, acoustic_out_masks, beam_size):
        '''expand acoustic out by beam_size'''
        if isinstance(acoustic_outs, list):
            expand_acoustic_outs = []
            expand_acoustic_out_masks = []
            for item, item_mask in zip(acoustic_outs, acoustic_out_masks):
                expand_item, expand_item_mask = cls.expand_acoustic_out(item, item_mask, beam_size)
                expand_acoustic_outs.append(expand_item)
                expand_acoustic_out_masks.append(expand_item_mask)
        else:
            bsz, max_seq_len, _ = acoustic_outs.size()
            expand_acoustic_outs = (
                acoustic_outs.unsqueeze(1)
                .expand(-1, beam_size, -1, -1)
                .contiguous()
                .view(bsz * beam_size, max_seq_len, -1)
            )
            expand_acoustic_out_masks = (
                acoustic_out_masks.unsqueeze(1)
                .expand(-1, beam_size, -1)
                .contiguous()
                .view(bsz * beam_size, max_seq_len)
            )
        return expand_acoustic_outs, expand_acoustic_out_masks

    def twopass_rescore(self, batch_data, nbest=1, **kwargs):
        '''rescore mode for beam_infer'''
        self.prepare_rnnt_encoder_out(batch_data)
        rnnt_scores, hotword_fst_scores, out_rlt_list_nbest = self.generate_rnnt_nbest_data(
            batch_data, nbest=nbest, **kwargs
        )
        bsz, _, max_seq_len = batch_data['rnnt_hyp'].size()
        decoder_target = batch_data['rnnt_hyp'].view(bsz * nbest, max_seq_len)
        decoder_bw_target = batch_data['rnnt_bw_hyp'].view(bsz * nbest, max_seq_len)
        decoder_mask = batch_data['rnnt_hyp_mask'].view(bsz * nbest, max_seq_len).eq(0)
        acoustic_out, backbone_mask = self.las_encoder(batch_data, **kwargs)
        acoustic_out_expand, backbone_mask_expand = self.expand_acoustic_out(
            acoustic_out, backbone_mask, nbest
        )

        # compute las score
        las_score_list = self.compute_las_score(
            self.decoder_module,
            acoustic_out_expand,
            backbone_mask_expand,
            decoder_target,
            decoder_mask,
        )
        las_scores = [las_score_list[i * nbest : (i + 1) * nbest] for i in range(bsz)]
        las_bw_score_list = self.compute_las_score(
            self.decoder_bw_module,
            acoustic_out_expand,
            backbone_mask_expand,
            decoder_bw_target,
            decoder_mask,
        )
        las_bw_scores = [las_bw_score_list[i * nbest : (i + 1) * nbest] for i in range(bsz)]
        return (
            out_rlt_list_nbest,
            rnnt_scores,
            hotword_fst_scores,
            las_scores,
            las_bw_scores,
        )

    @staticmethod
    def max_decoder_positions():
        '''used for decoding'''
        return 10000

    @staticmethod
    def reorder_encoder_out(encoder_out, new_order):
        '''encoder_out shape (B,T,N) new_order shape (B*beam)'''
        for name, tensor in encoder_out.items():
            encoder_out[name] = tensor.index_select(0, new_order)
        return encoder_out

    @torch.no_grad()
    def twopass_beam_search(self, batch_data, **kwargs):
        '''search mode for beam_infer'''
        twopass_infer_cfg = kwargs.get('twopass_infer_cfg')
        lm_solution = twopass_infer_cfg.get('lm_solution', None)
        nbest_out = twopass_infer_cfg.get('nbest_out', False)

        self.prepare_rnnt_encoder_out(batch_data)
        acoustic_out, backbone_mask = self.las_encoder(batch_data, **kwargs)
        if lm_solution is None:
            beam_infer_rlt = self.twopass_beam_searcher(acoustic_out, backbone_mask)
        else:
            ## TODO (houjunfeng) add beam search with lm
            raise RuntimeError("las beam_search with lm does not supported")

        out_rlt_list = []
        nbest_rlt_list = []
        for hyp_token_list in beam_infer_rlt:
            out_rlt_list.append(hyp_token_list[0]['tokens'])  # top1
            nbest_rlt_list.append([t['tokens'] for t in hyp_token_list])  # topK

        if nbest_out:
            return out_rlt_list, nbest_rlt_list
        return out_rlt_list

    @torch.no_grad()
    def beam_inference(self, batch_data, nbest=1, **kwargs):
        twopass_infer_cfg = kwargs.get('twopass_infer_cfg')
        self.init_twopass_beam_search(twopass_infer_cfg)
        if twopass_infer_cfg.get('enable_twopass_rescore'):
            return self.twopass_rescore(batch_data, nbest=nbest, **kwargs)
        return self.twopass_beam_search(batch_data, **kwargs)

    def init_twopass_beam_search(self, inference_cfg):
        '''beam search init'''
        self.twopass_beam_searcher = BaseBeamSearch(
            self.args,
            inference_cfg,
            self.decoder_module,
        )

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        self.args.setdefault("return_backbone", True)
        super().register_infers()
        infers = [LasEncoderExporter(self, **self.args)]
        if 'Transformer' in self.args.twopass_decoder_args.las_decoder_type:
            infers.append(RnntLasTransformerDecoderExporter(self, **self.args))
        else:
            args = copy.deepcopy(self.args)
            args.update(self.args.twopass_decoder_args)
            infers.append(
                RnntLasLstmDecoderExporter(
                    args, self.decoder_module, self.decoder_bw_module, **self.args
                )
            )
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)


@register_solution("RnntDeliberationModel")
class RnntDeliberationModel(RnntLasRescoreModel):
    '''LAS deliberation model
    https://arxiv.org/pdf/2003.07962.pdf
    '''

    def __init__(self, args):
        '''init function for RNNT & LAS Deliberation model'''
        super().__init__(args)
        self.init_hyp_encoder(args)

    def init_hyp_encoder(self, args):
        '''init hyp encoder'''
        hyp_args = copy.deepcopy(args)
        hyp_args.update(args.twopass_hyp_args)
        self.las_hyp_encoder_type = hyp_args.hyp_encoder_type
        if 'Transformer' in hyp_args.backbone_type:
            hyp_args.backbone_layer_num = len(eval(hyp_args.backbone_topology))
        elif hyp_args.backbone_type == 'LSTMPBackbone':
            if hyp_args.backbone_bilstm:
                assert hyp_args.backbone_mask, "Using bilstm must set las_hyp_mask = True"
        else:
            raise RuntimeError("las_hyp_encoder_type does not supported")
        self.hyp_args = hyp_args
        if self.hyp_args.backbone_layer_num > 0:
            self.las_hyp_encoder_module = eval(self.las_hyp_encoder_type)(self.hyp_args)
        else:
            self.las_hyp_encoder_module = None

    @staticmethod
    def prepare_hyp_input(batch_data, nbest_input_size):
        '''prepare nbest input for deliberation'''
        bsz = batch_data['rnnt_hyp'].shape[0]
        batch_data['rnnt_hyp_input'] = (
            batch_data['rnnt_hyp'][:, :nbest_input_size]
            .contiguous()
            .view(bsz * nbest_input_size, -1)
        )
        batch_data['rnnt_hyp_input_mask'] = (
            batch_data['rnnt_hyp_mask'][:, :nbest_input_size]
            .contiguous()
            .view(bsz * nbest_input_size, -1)
        )

    def las_hyp_encoder(self, batch_data):
        '''
        Encoding for RNN-T output
        :param rnnt_hyp: [B, T_enc] if greedy or using 1 best result,
                         [B*nbest_size, T_enc] if using n best
        :param rnnt_out: [B, T_enc] the output for rnnt decoder,
        :param rnnt_mask: [B, T_enc] the mask for rnnt output, 1 for value,
                          0 for padding [B*nbest_size, T_enc]
        for n best input
        :return: las_encoder_out: [B, T_enc, embed_dim]
        '''
        if 'las_hyp_encoder_out' in batch_data:
            return
        rnnt_hyp = batch_data['rnnt_hyp_input']
        rnnt_hyp_mask = batch_data['rnnt_hyp_input_mask']
        bsz = batch_data['encoder_backbone_out'].size(0)
        las_hyp_encoder_out = self.las_hyp_encoder_module(rnnt_hyp, rnnt_hyp_mask)

        if rnnt_hyp.size(0) != bsz:  # Meaning it is an nbest out
            embed_dim = las_hyp_encoder_out.size(-1)
            las_hyp_encoder_out = las_hyp_encoder_out.view(bsz, -1, embed_dim)
            flattened_rnnt_hyp_mask = rnnt_hyp_mask.view(bsz, -1)
            batch_data['rnnt_hyp_input_mask'] = flattened_rnnt_hyp_mask
        batch_data['las_hyp_encoder_out'] = las_hyp_encoder_out

    def prepare_rnnt_nbest_data(self, batch_data_bulk, twopass_infer_cfg=None, **kwargs):
        '''prepare rnnt nbest data for twopass deliberation model'''
        if twopass_infer_cfg is not None:
            nbest = twopass_infer_cfg.get('nbest')
        else:
            nbest = self.args.get('nbest')
        self.generate_rnnt_nbest_data(batch_data_bulk, nbest=nbest, **kwargs)

    def las_encoder(self, batch_data, twopass_infer_cfg=None, **kwargs):
        '''las encoder of twopass deliberation model'''
        if twopass_infer_cfg is not None:
            nbest_input_size = twopass_infer_cfg.get('nbest_input_size')
        else:
            nbest_input_size = self.args.get('nbest_input_size')
        acoustic_encoder_out, backbone_mask = super().las_encoder(batch_data)
        if 'rnnt_hyp' not in batch_data:
            self.prepare_rnnt_nbest_data(batch_data, twopass_infer_cfg, **kwargs)
        self.prepare_hyp_input(batch_data, nbest_input_size)
        self.las_hyp_encoder(batch_data)
        encoder_outs = [acoustic_encoder_out, batch_data['las_hyp_encoder_out']]
        encoder_out_masks = [backbone_mask, batch_data['rnnt_hyp_input_mask']]
        return encoder_outs, encoder_out_masks


@register_solution("RnntAttnRescoreModel")
class RnntAttnRescoreModel(BaseRnntModel):
    '''For multi-loss RNN-T based model doing attention rescore.
    The difference with RnntLasRescoreModel is its training process:
    LAS decoder is trained together with RNN-T model.
    Attention rescore(2pass) is almost same with RnntLasRescoreModel
    '''

    def init_beam_search(self, inference_cfg, lm_solution):
        '''beam search init'''
        self.attn_rescore_weight = inference_cfg.setdefault('attn_rescore_weight', 0.0)
        assert self.attn_rescore_weight >= 0.0 and self.attn_rescore_weight < 1.0
        super().init_beam_search(inference_cfg, lm_solution)
        self.transducer_weight = 1.0 - self.attn_rescore_weight
        if self.attn_rescore_weight > 0.0:
            self.decoder = self.las_decoder_module

    @torch.no_grad()
    def attn_rescore(
        self, rnnt_bm_rlt, encoder_out, encoder_masks, attn_weight=0.5, transducer_weight=0.5
    ):
        '''
        attention rescore
        '''
        if isinstance(rnnt_bm_rlt, torch.Tensor):
            beam_size = rnnt_bm_rlt.size(0)
        else:
            beam_size = len(rnnt_bm_rlt)
        device = encoder_out.device
        hyps = [s[0][1:] for s in rnnt_bm_rlt]
        beam_score = [s[1] for s in rnnt_bm_rlt]
        padding_id = self.eos
        hyps_pad = pad_sequence(
            [torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps], True, padding_id
        )  # (beam_size, max_hyps_len)
        # add bos and eos in target for attention decoder input
        hyps_pad = torch.where(hyps_pad == padding_id, self.eos, hyps_pad)
        hyps_pad = F.pad(hyps_pad, (1, 0), value=self.bos)
        # hyps_lens = hyps_lens + 1  # Add <bos> at begining
        hyps_lens = [len(hyp) + 1 for hyp in hyps]
        max_hyp_len = max(hyp_len for hyp_len in hyps_lens)
        hyps_mask = torch.zeros(beam_size, max_hyp_len, device=device)
        for bid, hyp_len in enumerate(hyps_lens):
            hyps_mask[bid, 0:hyp_len] = 1
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_masks = encoder_masks.repeat(beam_size, 1)
        decoder_out = self.decoder(
            encoder_out,
            encoder_masks,
            hyps_pad,
            hyps_mask,
        )
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp)][self.eos]
            # add rnn-t score
            score = score * attn_weight + beam_score[i] * transducer_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index]

    @torch.no_grad()
    def beam_inference(
        self,
        batch_data,
        nbest=1,
        nbest_align_info=False,
        output_timestamp=False,
        prefetch=False,
        fixed_prefix=False,
        output_rnnt_confidence=False,
        endpoint=False,
        stable_metric_list=None,
        **_kwargs,
    ):
        '''
        Beam inference with attn rescore
        '''
        ############ Acoustic ############
        if 'encoder_out' in batch_data:
            # no more repetitive computation for efficiency
            acoustic_out = batch_data['encoder_out']
            backbone_mask = batch_data['backbone_mask']
            mtl_logits = batch_data['mtl_logits']
            encoder_backbone_out = batch_data['encoder_backbone_out']
        else:
            acoustic_out, backbone_mask, mtl_logits, encoder_backbone_out = self.forward(
                batch_data, inference=True
            )

        beam_infer_rlt, beam_infer_rlt_nbest = self.beam_searcher(
            acoustic_out,
            backbone_mask,
            stable_metric_list=stable_metric_list,
        )
        if self.attn_rescore_weight > 0.0:
            fbank = batch_data['src']  # (B, T, ndim)
            bsz = fbank.shape[0]
            for bid in range(bsz):
                beam_infer_rlt[bid] = self.attn_rescore(
                    beam_infer_rlt_nbest['nbest'][bid],
                    encoder_backbone_out[bid],
                    backbone_mask[bid],
                    self.attn_rescore_weight,
                    self.transducer_weight,
                )

        out_rlt_list, encoder_frames = self.process_beam_infer_rlt(
            batch_data,
            beam_infer_rlt,
            beam_infer_rlt_nbest,
            backbone_mask,
            nbest,
            nbest_align_info,
            output_timestamp,
            prefetch,
            fixed_prefix,
            output_rnnt_confidence,
            endpoint,
        )
        return out_rlt_list, mtl_logits, encoder_frames


@register_solution("RnntCaseModel")
class RnntCaseModel(RnntAttnRescoreModel):
    '''RNNT Cascaded Encoders(CASE) model'''

    def __init__(self, args):
        super().__init__(args)
        self.causal_loss_weight = args.get('causal_loss_weight', 0.5)
        self.update_steps = 0
        self.init_nc_acoustic_encoder(args)
        if self.mtl_type and self.training:
            self.nc_mtl_module = eval(args.mtl_head)(args)
        else:
            self.nc_mtl_module = None
        if self.joint_las_weight > 0.0:
            self.init_nc_las_decoder(args)
        self.causal_mode = False
        self.case_mini_batch_sampling = args.get('case_mini_batch_sampling', False)

        self.nc_backbone_pool_module = None
        if args.twopass_acoustic_args.get('backbone_pool_type', None):
            if args.twopass_acoustic_args.backbone_pool_type in (
                'AvgPoolTimeReduce',
                'MaxPoolTimeReduce',
            ):
                self.nc_backbone_pool_module = eval(args.twopass_acoustic_args.backbone_pool_type)(
                    ceil_mode=True
                )
            elif args.twopass_acoustic_args.backbone_pool_type == 'LinearTimeReduce':
                self.nc_backbone_pool_module = eval(args.twopass_acoustic_args.backbone_pool_type)(
                    2, args.jointer_hidden_size
                )
            elif args.twopass_acoustic_args.backbone_pool_type == 'Conv1dTimeReduce':
                self.nc_backbone_pool_module = eval(args.twopass_acoustic_args.backbone_pool_type)(
                    args.jointer_hidden_size
                )

    def init_nc_acoustic_encoder(self, args):
        '''non-causal acoustic encoder initiation'''
        nc_acoustic_args = args.twopass_acoustic_args
        if 'export_fused_conformer' in args:
            nc_acoustic_args.setdefault('export_fused_conformer', args['export_fused_conformer'])
        self.nc_acoustic_backbone_type = nc_acoustic_args.get('acoustic_backbone_type', None)
        self.nc_acoustic_backbone_module = eval(self.nc_acoustic_backbone_type)(nc_acoustic_args)

    def init_nc_las_decoder(self, args):
        '''init las decoder for causal encoder and non-causal encoder'''
        args = copy.deepcopy(args)
        args.update(args.las_args)
        self.nc_las_decoder_module = eval(args.las_decoder_type)(args)

    def forward(self, batch_data, inference=False):
        '''
        forward for RNNT-CASE model
        '''
        if self.training:
            self.update_steps += 1

        causal_mode = (
            (np.random.uniform() < self.causal_loss_weight)
            if (not inference and self.case_mini_batch_sampling)
            else self.causal_mode
        )
        # causal path
        encoder_out, backbone_mask, trainable, mtl_logits, encoder_backbone_out = self.encoder(
            batch_data
        )
        batch_data['backbone_mask'] = backbone_mask
        batch_data['encoder_backbone_out'] = encoder_backbone_out
        if inference and causal_mode:
            return encoder_out, backbone_mask, mtl_logits, encoder_backbone_out
        # non-causal path
        if not causal_mode:
            nc_encoder_out, nc_trainable, nc_mtl_logits, nc_encoder_backbone_out = self.nc_encoder(
                batch_data
            )
            if inference:
                nc_backbone_mask = backbone_mask
                if self.nc_backbone_pool_module is not None:
                    nc_backbone_mask = batch_data['nc_backbone_mask']
                return nc_encoder_out, nc_backbone_mask, nc_mtl_logits, nc_encoder_backbone_out
            if nc_trainable:
                trainable = nc_trainable

        # predictor
        predictor_out = self.predictor(batch_data, trainable)

        assert self.ilmt_weight == 0.0, "ilmt not supported in CASE yet"
        if self.joint_las_weight > 0.0:
            las_logits = self.las_decoder(batch_data, encoder_backbone_out, backbone_mask)
            nc_las_logits = self.nc_las_decoder(batch_data, nc_encoder_backbone_out, backbone_mask)

        # in training, if case_mini_batch_sampling False, causal_mode has no effect
        if causal_mode or not self.case_mini_batch_sampling:
            if self.joint_las_weight > 0.0:
                batch_data["las_logits"] = las_logits
            c_forward_out = self.rnnt_decoder(encoder_out, predictor_out, batch_data, mtl_logits)
        if not causal_mode or not self.case_mini_batch_sampling:
            if self.joint_las_weight > 0.0:
                batch_data["las_logits"] = nc_las_logits
            nc_forward_out = self.rnnt_decoder(
                nc_encoder_out,
                predictor_out,
                batch_data,
                nc_mtl_logits,
                causal=False,
            )

        forward_out = OrderedDict()
        if self.case_mini_batch_sampling:
            if causal_mode:
                forward_out['utt_num'] = c_forward_out['utt_num']
                forward_out['frame_size'] = c_forward_out['frame_size']
                forward_out['tgt_size'] = c_forward_out['tgt_size']
                forward_out['backward_loss'] = c_forward_out['backward_loss']
                forward_out['loss'] = c_forward_out['loss']
                forward_out['nll_loss'] = c_forward_out['nll_loss']
                forward_out['c_dist'] = c_forward_out['dist']
                forward_out['c_tgt_size'] = c_forward_out['tgt_size']
                forward_out['c_loss'] = c_forward_out['loss']
                forward_out['c_nll_loss'] = c_forward_out['nll_loss']
                forward_out['c_acc'] = c_forward_out['acc']
                forward_out['c_ctc_loss'] = c_forward_out['ctc_loss']
                forward_out['c_las_loss'] = c_forward_out['las_loss']
            else:
                forward_out['utt_num'] = nc_forward_out['utt_num']
                forward_out['frame_size'] = nc_forward_out['frame_size']
                forward_out['tgt_size'] = nc_forward_out['tgt_size']
                forward_out['backward_loss'] = nc_forward_out['backward_loss']
                forward_out['loss'] = nc_forward_out['loss']
                forward_out['nll_loss'] = nc_forward_out['nll_loss']
                forward_out['nc_dist'] = nc_forward_out['dist']
                forward_out['nc_tgt_size'] = nc_forward_out['tgt_size']
                forward_out['nc_loss'] = nc_forward_out['loss']
                forward_out['nc_nll_loss'] = nc_forward_out['nll_loss']
                forward_out['nc_acc'] = nc_forward_out['acc']
                forward_out['nc_cer'] = nc_forward_out['cer']
                forward_out['nc_ctc_loss'] = nc_forward_out['ctc_loss']
                forward_out['nc_las_loss'] = nc_forward_out['las_loss']
        else:
            forward_out['utt_num'] = c_forward_out['utt_num']
            forward_out['frame_size'] = c_forward_out['frame_size']
            forward_out['tgt_size'] = c_forward_out['tgt_size']
            forward_out['backward_loss'] = (
                self.causal_loss_weight * c_forward_out['backward_loss']
                + (1 - self.causal_loss_weight) * nc_forward_out['backward_loss']
            )
            forward_out['loss'] = (
                self.causal_loss_weight * c_forward_out['loss']
                + (1 - self.causal_loss_weight) * nc_forward_out['loss']
            )
            forward_out['nll_loss'] = (
                self.causal_loss_weight * c_forward_out['nll_loss']
                + (1 - self.causal_loss_weight) * nc_forward_out['nll_loss']
            )

            forward_out['c_dist'] = c_forward_out['dist']
            forward_out['c_tgt_size'] = c_forward_out['tgt_size']
            forward_out['c_loss'] = c_forward_out['loss']
            forward_out['c_nll_loss'] = c_forward_out['nll_loss']
            forward_out['c_ctc_loss'] = c_forward_out['ctc_loss']
            forward_out['c_las_loss'] = c_forward_out['las_loss']
            forward_out['c_acc'] = c_forward_out['acc']
            forward_out['c_cer'] = c_forward_out['cer']
            forward_out['nc_dist'] = nc_forward_out['dist']
            forward_out['nc_tgt_size'] = nc_forward_out['tgt_size']
            forward_out['nc_loss'] = nc_forward_out['loss']
            forward_out['nc_nll_loss'] = nc_forward_out['nll_loss']
            forward_out['nc_acc'] = nc_forward_out['acc']
            forward_out['nc_cer'] = nc_forward_out['cer']
            forward_out['nc_ctc_loss'] = nc_forward_out['ctc_loss']
            forward_out['nc_las_loss'] = nc_forward_out['las_loss']

        return forward_out

    def nc_encoder(self, batch_data):
        '''non-causal encoder in RNNT-CASE'''
        if not self.training or self.update_steps <= self.encoder_fix_steps:
            with torch.no_grad():
                # output (B, T, N)
                nc_encoder_backbone_out = self.nc_encoder_backbone(
                    batch_data['encoder_backbone_out'], batch_data['backbone_mask'], 'BTN'
                )
        else:
            with torch.enable_grad():
                nc_encoder_backbone_out = self.nc_encoder_backbone(
                    batch_data['encoder_backbone_out'], batch_data['backbone_mask'], 'BTN'
                )
        # ctc mtl branch
        if self.nc_mtl_module is not None:
            nc_mtl_logits = self.nc_mtl_module(nc_encoder_backbone_out)
        else:
            nc_mtl_logits = None
        # head
        nc_encoder_out = self.acoustic_head_module(nc_encoder_backbone_out)
        trainable = nc_encoder_out.requires_grad
        # nc_backbone_pool_module
        if self.nc_backbone_pool_module is not None:
            if isinstance(self.nc_backbone_pool_module, MaxPoolTimeReduce):
                nc_encoder_out, batch_data['nc_backbone_mask'] = self.nc_backbone_pool_module(
                    nc_encoder_out, batch_data['backbone_mask']
                )
            else:
                nc_encoder_out = self.nc_backbone_pool_module(nc_encoder_out)
                batch_data['nc_backbone_mask'] = batch_data['backbone_mask'][:, :-1][:, ::2]
        return nc_encoder_out, trainable, nc_mtl_logits, nc_encoder_backbone_out

    def nc_encoder_backbone(self, inputs, mask=None, input_shape='BTN'):
        '''non-causal encoder backbone'''
        attn_mask = False
        if (
            self.nc_acoustic_backbone_type
            in ('OfflineTransformerBackbone', 'RelTransformerBackbone', 'EmformerBackbone')
            and self.causal_transformer
        ):
            attn_mask = True
        backbone_out = self.nc_acoustic_backbone_module(
            inputs, mask, attn_mask=attn_mask, frontend_shape=input_shape
        )
        return backbone_out

    def nc_las_decoder(self, batch_data, nc_encoder_backbone_out, backbone_mask):
        '''
        Las Decoder Module
        '''
        # TODO: modify target input
        target = batch_data['char']
        # add bos and eos in target for attention decoder input
        target_att = torch.where(target == 0, self.eos, target)
        target_att = F.pad(target_att, (1, 0), value=self.bos)
        target_mask = batch_data['char_mask']
        if 'mtl_mask' in batch_data:
            backbone_mask = batch_data['mtl_mask']
        if 'Transformer' in self.las_decoder_type:
            las_logits = self.nc_las_decoder_module(
                nc_encoder_backbone_out,
                backbone_mask,
                target_att,
                prev_tgt_mask=F.pad(target_mask, (1, 0), value=1),
            )
        else:
            las_logits = self.nc_las_decoder_module(
                nc_encoder_backbone_out, backbone_mask, target_att
            )
        return las_logits

    def rnnt_decoder(self, encoder_out, predictor_out, batch_data, mtl_logits, causal=True):
        '''shared decoder in RNNT-CASE for causal&non-causal mode'''
        jointer_out = self.jointer(encoder_out, predictor_out, None, batch_data['target_lengths'])

        # criterion for loss computation
        if not causal and self.nc_backbone_pool_module is not None:
            batch_data['backbone_mask'] = batch_data['nc_backbone_mask']
        forward_out = self.criterion_module(
            jointer_out,
            batch_data,
            mtl_logits=mtl_logits,
            mtl_type=self.mtl_type,
        )
        if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out.requires_grad:
            target = batch_data['char']
            forward_out['cer'], forward_out['dist'] = self.greedy_searcher(
                encoder_out, target, batch_data
            )
        else:
            forward_out['cer'], forward_out['dist'] = 0, 0
        return forward_out

    def init_beam_search(self, inference_cfg, lm_solution):
        '''init beam search for causal or non-causal mode'''
        self.causal_mode = inference_cfg.get("causal_mode", True)
        super().init_beam_search(inference_cfg, lm_solution)
        if not self.causal_mode and self.attn_rescore_weight > 0.0:
            self.decoder = self.nc_las_decoder_module

    def register_infers(self):
        '''register infer object for export and beamsearch.'''
        self.args.setdefault("return_backbone", True)
        super().register_infers()
        assert RnntNCEncoderExporter.NAME not in INFERS
        INFERS[RnntNCEncoderExporter.NAME] = RnntNCEncoderExporter(self, **self.args)
        self._infer_names.append(RnntNCEncoderExporter.NAME)


class RnntNCEncoderExporter(BaseInfer):
    '''export rnnt non-causal encoder to onnx'''

    NAME = 'nc_encoder'
    INPUTS = [
        FalconDict(name='backbone_out', type=torch.float32, shape=['B', 'T', -1]),
    ]
    OUTPUTS = [
        FalconDict(name='output', type=torch.float32, shape=['B', 'T', -1]),
    ]

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)

        self.nc_acoustic_backbone_module = model.nc_acoustic_backbone_module
        self.nc_encoder_backbone = model.nc_encoder_backbone  # nc_encoder_backbone is function
        self.acoustic_head_module = model.acoustic_head_module
        self.nc_backbone_pool_module = model.nc_backbone_pool_module

        # fill out inputs and outputs info
        assert model.nc_acoustic_backbone_type == 'ConformerBackbone'
        self.input_dim = model.nc_acoustic_backbone_module.encoders[0].size
        self._inputs[0].shape[2] = self.input_dim  # eg: [B, T, 512]
        self._convert_stream_flag = kwargs.get('encoder_convert_stream', True)
        self._stream_trial_t = [2, 4, 6]

        if self._convert_stream_flag:
            self.global_state_size = getattr(self.nc_acoustic_backbone_module, 'state_size', 0)
            self._inputs.append(
                FalconDict(
                    name='global_state_in',
                    type=torch.float32,
                    shape=['Batch', self.global_state_size],
                )
            )
            self._inputs.append(
                FalconDict(
                    name='x_sign',
                    type=torch.int32,
                    shape=[1],
                )
            )

    def forward(self, backbone_out, backbone_mask=None, **_kwargs):
        '''
        forward for rnnt non-causal encoder
        '''
        nc_backbone_out = self.nc_encoder_backbone(backbone_out, backbone_mask, 'BTN')
        if self.acoustic_head_module is not None:
            nc_encoder_out = self.acoustic_head_module(nc_backbone_out)
        # nc_backbone_pool_module
        if self.nc_backbone_pool_module is not None:
            nc_encoder_out = self.nc_backbone_pool_module(nc_encoder_out)
            if backbone_mask is not None:
                backbone_mask = backbone_mask[:, :-1][:, ::2]
        if backbone_mask is not None:
            return nc_encoder_out, backbone_mask
        return nc_encoder_out

    def sample_inputs(self):
        '''
        sample some inputs as example.
        may be used for jit trace or test.
        '''
        backbone_out = self._generate_input_data(0, dynamic_axis=[1, 512, -1])
        datas = [backbone_out]
        return tuple(datas)
