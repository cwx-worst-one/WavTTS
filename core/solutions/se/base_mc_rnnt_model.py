''' BaseRnntModel '''
# pylint: disable=unused-argument
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, unused-import
from collections import OrderedDict
import os.path as osp
import numpy as np
from packaging import version
import torch
from torch import nn

try:
    # pylint: disable=no-name-in-module,import-error
    from panther import slim
    from panther.slim.core.dynamic_graph.patch_pytorch import patch_torch_operators
    from panther.slim.quantize import QuantizeConfig
except Exception:
    pass
from core.solutions.base_solution import BaseSolution, register_solution
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.se.multi_ch_frontend import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.models.layers.time_reduce_layer import *
from core.solutions.inference.rnnt_beam_search import (
    BatchBeamSearch,
    NonBatchBeamSearch,
)
from ..asr.base_rnnt_model import BaseRnntModel


@register_solution("BaseMCRnntModel")
class BaseMCRnntModel(BaseRnntModel):
    '''
    Base multi-channel model for RNN-T.
    - Multichannel Frontend
    - Encoder
        - frontend: VGGFrontEnd
        - backbone: TransformerBackbone, DFSMNBackbone
    - Predictor
    - Jointer
    - criterion
    '''

    def __init__(self, args):
        '''
        init function for RNN-T skeleton model.
        '''
        super().__init__(args)
        self.args = args
        self.mc_frontend = eval(args.mc_front_end_type)(args)

    def forward(self, batch_data, inference=False):
        '''
        Forward for RNN-T base module
        '''
        if self.training:
            self.update_steps += 1
        ############ Multi-channel Front-end #############
        batch_data = self.mc_frontend(batch_data)
        ############ Encoder #############
        if 'encoder_out' in batch_data:
            encoder_out = batch_data['encoder_out']
            backbone_mask = batch_data['backbone_mask']
            trainable = batch_data['trainable']
            mtl_logits = batch_data['mtl_logits']
            encoder_backbone_out = batch_data['encoder_backbone_out']
        else:
            encoder_out, backbone_mask, trainable, mtl_logits, encoder_backbone_out = self.encoder(
                batch_data
            )
            batch_data['backbone_mask'] = backbone_mask
        if inference:
            return encoder_out, backbone_mask, mtl_logits, encoder_backbone_out
        ############ Predictor ############
        predictor_out = self.predictor(batch_data, trainable)
        target_lengths = batch_data['target_lengths']

        ############ Jointer ############
        # special jointer for large batch support.
        if self.args.get('jointer_split_num', 1) > 1:
            return self.splitting_jointer_forward(
                batch_data, encoder_out, predictor_out, mtl_logits
            )

        jointer_out = self.jointer(encoder_out, predictor_out, None, target_lengths)
        if self.ilmt_weight > 0.0:
            ilmt_encoder_out = torch.zeros_like(encoder_out[:, 0:1, :])
            ilmt_jointer_out = self.jointer(ilmt_encoder_out, predictor_out, None, target_lengths)
            batch_data["ilmt_jointer_out"] = ilmt_jointer_out

        # criterion for loss computation
        forward_out = self.criterion_module(
            jointer_out,
            batch_data,
            encoder_out,
            self.predictor_module,
            self.jointer_module,
            mtl_logits=mtl_logits,
            mtl_type=self.mtl_type,
        )
        return forward_out
