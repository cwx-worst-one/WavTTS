''' BaseRnntModel '''
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.solutions.base_solution import register_solution
from .base_rnnt_model import BaseRnntModel


@register_solution("BaseRnntModelEncoder")
class BaseRnntModelEncoder(BaseRnntModel):
    '''
    Base encoder model for DFSMN RNN-T.
    - Encoder
        - frontend: VGGFrontEnd
        - backbone: TransformerBackbone, DFSMNBackbone
    '''

    def encoder(self, batch_data):
        '''
        Encoder Module in RNN-T
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']
        ############ Encoder ############
        # front-end
        front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
        # output (B, N, T)
        encoder_out = self.encoder_backbone(front_end_out, backbone_mask, frontend_shape)
        encoder_out = self.acoustic_head_module(encoder_out)
        trainable = encoder_out.requires_grad
        return encoder_out, backbone_mask, trainable

    def forward(self, batch_data):
        '''
        Forward for RNN-T base module
        '''
        ############ Encoder #############
        encoder_out, backbone_mask, _ = self.encoder(batch_data)
        batch_data['backbone_mask'] = backbone_mask

        # criterion for loss computation
        src_mask = batch_data['src_mask']
        bsz = src_mask.shape[0]
        target = batch_data['ce_label']
        target = target.view(bsz, -1, self.args.downsampling_size)[:, :, 0]
        target_mask = batch_data['backbone_mask']
        forward_out = self.criterion_module(encoder_out, src_mask, target, target_mask)
        return forward_out
