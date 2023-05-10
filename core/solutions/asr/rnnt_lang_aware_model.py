''' LangAwareRnntModel '''
# pylint: disable=unused-argument
# pylint: disable=too-many-lines
import torch
from torch import nn

from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.asr.las_decoder import *
from core.criterions import *
from core.models.asr.rnnt_predictor import *
from core.models.asr.rnnt_jointer import *
from core.models.layers.time_reduce_layer import *
from core.solutions.base_solution import register_solution
from .base_rnnt_model import BaseRnntModel


@register_solution("LangAwareRnntModel")
class LangAwareRnntModel(BaseRnntModel):
    '''
    language awre model for RNN-T.
    - Encoder
        - frontend
        - backbone
        - language aware backbone
    - Predictor
    - Jointer
    - criterion
    '''

    def __init__(self, args):
        '''
        init function for RNN-T skeleton model.
        '''
        super().__init__(args)
        # add lat branch
        self.lat_weight = args.get('lat_weight', 0.0)
        self.lang_list = args.get('lang_list', ['zh', 'en'])
        self.lang_aware_encoder_config = args.get('lang_aware_encoder_config', None)

        self.lang_aware_encoders = nn.ModuleDict({})
        self.lang_aware_encoders_out = {}
        if self.lat_weight > 0.0:
            self.lang_aware_mtl_module = nn.ModuleDict({})
            self.lang_aware_encoders_logits = {}

        for lang in self.lang_list:
            self.lang_aware_encoders[lang] = eval(
                args.lang_aware_encoder_config.acoustic_backbone_type
            )(args.lang_aware_encoder_config)
            if self.lat_weight > 0.0:
                self.lang_aware_mtl_module[lang] = eval(args.mtl_head)(args)

    def lang_aware_encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''
        Backbone model for RNN-T encoder
        '''
        attn_mask = False
        if (
            self.args.acoustic_backbone_type
            in ('OfflineTransformerBackbone', 'RelTransformerBackbone', 'EmformerBackbone')
            and self.causal_transformer
        ):
            attn_mask = True

        for lang in self.lang_list:
            self.lang_aware_encoders_out[lang] = self.lang_aware_encoders[lang](
                inputs, mask, attn_mask=attn_mask, frontend_shape=frontend_shape
            )
            if self.lat_weight > 0.0:
                self.lang_aware_encoders_logits[lang] = self.lang_aware_mtl_module[lang](
                    self.lang_aware_encoders_out[lang]
                )
        # sum all language aware output
        lang_aware_output = sum(output for lang, output in self.lang_aware_encoders_out.items())
        return lang_aware_output

    def encoder(self, batch_data):
        '''
        Encoder Module in RNN-T
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']
        if 'domain' in batch_data:
            domain_emb = F.one_hot(batch_data['domain'], num_classes=self.args.domain_num).cuda()
            fbank = torch.cat(
                (fbank, domain_emb.unsqueeze(1).to(fbank.dtype).repeat(1, fbank.shape[1], 1)), dim=2
            )
        ############ Encoder ############
        if not self.training or self.update_steps <= self.encoder_fix_steps:
            with torch.no_grad():
                # front-end
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                # output (B, T, N)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )
                # language aware encoders
                encoder_backbone_out = self.lang_aware_encoder_backbone(
                    encoder_backbone_out, backbone_mask, frontend_shape
                )
        else:
            with torch.enable_grad():
                front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )
                # language aware encoders
                encoder_backbone_out = self.lang_aware_encoder_backbone(
                    encoder_backbone_out, backbone_mask, frontend_shape
                )

        if self.lat_weight > 0.0:
            batch_data["lat_logits"] = self.lang_aware_encoders_logits

        # ctc mtl branch
        if self.mtl_module is not None:
            mtl_logits = self.mtl_module(encoder_backbone_out)
        else:
            mtl_logits = None
        # head
        encoder_out = self.acoustic_head_module(encoder_backbone_out)
        # backbone_pool_module
        if self.backbone_pool_module is not None:
            encoder_out = self.backbone_pool_module(encoder_out)
            batch_data["mtl_mask"] = backbone_mask.clone() if backbone_mask is not None else None
            backbone_mask = backbone_mask[:, :-1][:, ::2] if backbone_mask is not None else None
        if self.do_inter_subsample:
            backbone_mask = backbone_mask[:, :-1][:, ::2] if backbone_mask is not None else None
            batch_data["mtl_mask"] = backbone_mask.clone() if backbone_mask is not None else None
        if 'eos' in batch_data:
            batch_data['eos'] = batch_data['eos'] // self.args.downsampling_size
        trainable = encoder_out.requires_grad
        return encoder_out, backbone_mask, trainable, mtl_logits, encoder_backbone_out
