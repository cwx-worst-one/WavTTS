''' RNN-T solution for dual-mode ASR '''
# pylint: disable=unused-import
# pylint: disable=too-many-branches
# pylint: disable=line-too-long
import random
import torch
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.models.layers.time_reduce_layer import *
from core.solutions.base_solution import register_solution
from .base_rnnt_model import BaseRnntModel


@register_solution("RnntUniversalModel")
class RnntUniversalModel(BaseRnntModel):
    '''RNN-T Dual-mode Model'''

    def __init__(self, args):
        '''init function for RNN-T skeleton model.'''
        super().__init__(args)
        self.streaming_mode = False
        self.distillation_scheduler = args.get('distillation_scheduler', None)
        self.frame_shift = args.get('frame_shift', None)
        self.symmetric = args.get('symmetric', None)
        self.peak = args.get('peak', None)
        assert self.args.acoustic_backbone_type in (
            'OfflineTransformerBackbone',
            'RelTransformerBackbone',
            'EmformerBackbone',
            'MaskedConformerBackbone',
            'ConformerBackbone',
        )

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''
        Backbone model for RNN-T encoder
        '''
        backbone_out = self.acoustic_backbone_module(
            inputs,
            mask,
            attn_mask=self.streaming_mode,
            frontend_shape=frontend_shape,
        )
        return backbone_out

    def encoder(self, batch_data):
        # 1: offline; 2: streaming
        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']
        ############ Encoder ############
        self.streaming_mode = False
        if hasattr(self.acoustic_backbone_module, 'set_stream_mode'):
            self.acoustic_backbone_module.set_stream_mode(False)
        front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
        encoder_out1 = self.encoder_backbone(front_end_out, backbone_mask, frontend_shape)
        self.streaming_mode = True
        if hasattr(self.acoustic_backbone_module, 'set_stream_mode'):
            self.acoustic_backbone_module.set_stream_mode(True)
        encoder_out2 = self.encoder_backbone(front_end_out, backbone_mask, frontend_shape)
        # ctc mtl branch
        if self.mtl_module is not None:
            mtl_logits1 = self.mtl_module(encoder_out1)
            mtl_logits2 = self.mtl_module(encoder_out2)
        else:
            mtl_logits1 = None
            mtl_logits2 = None
        # head
        encoder_out1 = self.acoustic_head_module(encoder_out1)
        encoder_out2 = self.acoustic_head_module(encoder_out2)
        trainable = encoder_out1.requires_grad
        return encoder_out1, encoder_out2, backbone_mask, trainable, mtl_logits1, mtl_logits2

    def forward(self, batch_data, inference=False):
        '''
        Forward for RNN-T base module
        '''
        if self.training:
            self.update_steps += 1
        ############ Encoder #############
        (
            encoder_out1,
            encoder_out2,
            backbone_mask,
            trainable,
            mtl_logits1,
            mtl_logits2,
        ) = self.encoder(batch_data)
        # when inference, return difference mode's encoder
        if inference:
            streaming_mode = self.args.get('streaming_mode', True)
            if streaming_mode:  # for streaming case
                return encoder_out2, backbone_mask, mtl_logits2, None
            return encoder_out1, backbone_mask, mtl_logits1, None
        ############ Predictor ############
        predictor_out = self.predictor(batch_data, trainable)
        target_lengths = batch_data['target_lengths']
        ############ Jointer ############
        jointer_out1 = self.jointer(encoder_out1, predictor_out, None, target_lengths)
        jointer_out2 = self.jointer(encoder_out2, predictor_out, None, target_lengths)
        batch_data['backbone_mask'] = backbone_mask
        # criterion for loss computation
        forward_out = self.criterion_module(
            jointer_out1,
            jointer_out2,
            batch_data,
            mtl_logits1=mtl_logits1,
            mtl_logits2=mtl_logits2,
            mtl_type=self.mtl_type,
            distillation_scheduler=self.distillation_scheduler,
            frame_shift=self.frame_shift,
            symmetric=self.symmetric,
            peak=self.peak,
        )
        if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out1.requires_grad:
            target = batch_data['char']
            error_dist1, total_dist = self.greedy_searcher(encoder_out1, target, batch_data)
            error_dist2, _ = self.greedy_searcher(encoder_out2, target, batch_data)
            forward_out['cer'] = error_dist1 + error_dist2
            forward_out['cer_offline'] = error_dist1
            forward_out['cer_stream'] = error_dist2
            forward_out['dist'] = total_dist
        return forward_out


@register_solution("RnntMultiLatencyModel")
class RnntMultiLatencyModel(RnntUniversalModel):
    '''RNN-T Multi-Latency Model'''

    def __init__(self, args):
        '''init function for RNN-T skeleton model.'''
        super().__init__(args)
        self.distillation_start_steps = args.get("distillation_start_steps", 0)
        self.distillation_only_last = args.get("distillation_only_last", False)
        conformer_mask_topology = eval(args.stream_conformer_mask_topology)
        self.mode_indexs = list(range(len(conformer_mask_topology[-1]) + 1))
        assert len(self.mode_indexs) >= 2
        assert self.args.acoustic_backbone_type == 'ConformerBackbone'

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN", mask_index=0):
        '''
        Backbone model for RNN-T encoder
        '''
        backbone_out = self.acoustic_backbone_module(
            inputs,
            mask,
            frontend_shape=frontend_shape,
            mask_index=mask_index,
        )
        return backbone_out

    def encoder(self, batch_data, mode_index):
        # 1: high latency; 2: low latency
        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']
        assert mode_index[0] < mode_index[1] and len(mode_index) == 2
        ############ Encoder ############
        front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
        if mode_index[1] == self.mode_indexs[-1]:
            self.acoustic_backbone_module.set_stream_mode(False)
        else:
            self.acoustic_backbone_module.set_stream_mode(True)
            self.acoustic_backbone_module.set_stream_index(mode_index[1])
        encoder_out1 = self.encoder_backbone(front_end_out, backbone_mask, frontend_shape)
        self.acoustic_backbone_module.set_stream_mode(True)
        self.acoustic_backbone_module.set_stream_index(mode_index[0])
        encoder_out2 = self.encoder_backbone(front_end_out, backbone_mask, frontend_shape)
        # ctc mtl branch
        mtl_logits1, mtl_logits2 = None, None
        if self.mtl_module is not None:
            mtl_logits1 = self.mtl_module(encoder_out1)
            mtl_logits2 = self.mtl_module(encoder_out2)
        # head
        encoder_out1 = self.acoustic_head_module(encoder_out1)
        encoder_out2 = self.acoustic_head_module(encoder_out2)
        trainable = encoder_out1.requires_grad
        return encoder_out1, encoder_out2, backbone_mask, trainable, mtl_logits1, mtl_logits2

    def forward(self, batch_data, inference=False):
        '''
        Forward for RNN-T base module
        '''
        if self.training:
            self.update_steps += 1
            mode_index = random.sample(self.mode_indexs, 2)
            mode_index.sort()
            mode_indexs = [mode_index]
        elif inference:
            streaming_index = self.args.get('streaming_index', 0)
            mode_indexs = [[streaming_index, self.mode_indexs[-1]]]
        else:
            tmp_index = (
                self.mode_indexs
                if len(self.mode_indexs) % 2 == 0
                else self.mode_indexs + [self.mode_indexs[-1]]
            )
            mode_indexs = [[a, b] for a, b in zip(tmp_index[::2], tmp_index[1::2])]
        distillation_scheduler, peak = None, None
        if self.training and self.update_steps >= self.distillation_start_steps:
            distillation_scheduler = self.distillation_scheduler
            peak = self.peak
        ############ Encoder #############
        forward_out = None
        for mode_index in mode_indexs:
            (
                encoder_out1,
                encoder_out2,
                backbone_mask,
                trainable,
                mtl_logits1,
                mtl_logits2,
            ) = self.encoder(batch_data, mode_index)
            # when inference, return difference mode's encoder
            if inference:
                streaming_mode = self.args.get('streaming_mode', True)
                if streaming_mode:  # for streaming case
                    return encoder_out2, backbone_mask, mtl_logits2, None
                return encoder_out1, backbone_mask, mtl_logits1, None
            ############ Predictor ############
            predictor_out = self.predictor(batch_data, trainable)
            target_lengths = batch_data['target_lengths']
            ############ Jointer ############
            jointer_out1 = self.jointer(encoder_out1, predictor_out, None, target_lengths)
            jointer_out2 = self.jointer(encoder_out2, predictor_out, None, target_lengths)
            batch_data['backbone_mask'] = backbone_mask
            # criterion for loss computation
            if self.distillation_only_last and mode_index[1] != self.mode_indexs[-1]:
                distillation_scheduler, peak = None, None
            tmp_out = self.criterion_module(
                jointer_out1,
                jointer_out2,
                batch_data,
                mtl_logits1=mtl_logits1,
                mtl_logits2=mtl_logits2,
                mtl_type=self.mtl_type,
                distillation_scheduler=distillation_scheduler,
                frame_shift=self.frame_shift,
                symmetric=self.symmetric,
                peak=peak,
            )
            if self.update_steps % self.args.cer_update_freq == 0 or not jointer_out1.requires_grad:
                target = batch_data['char']
                error_dist1, total_dist = self.greedy_searcher(encoder_out1, target, batch_data)
                error_dist2, _ = self.greedy_searcher(encoder_out2, target, batch_data)
                tmp_out['cer'] = error_dist1 + error_dist2
                tmp_out['cer_offline'] = error_dist1
                tmp_out['cer_stream'] = error_dist2
                tmp_out['dist'] = total_dist

            if forward_out is None:
                forward_out = tmp_out
            else:
                for key in tmp_out:
                    # pylint: disable=unsupported-assignment-operation
                    forward_out[key] += tmp_out[key]
        for key in forward_out:
            # pylint: disable=unsupported-assignment-operation
            forward_out[key] /= len(mode_indexs)
        return forward_out
