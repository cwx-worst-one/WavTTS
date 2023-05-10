"""best-rq pretrain"""
import logging
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.asr.acoustic_head import *
from core.models.pretrained.acoustic_tokenizer import *
from core.criterions.criterion import Xentropy
from core.solutions.base_solution import BaseSolution, register_solution


@register_solution("AcousticTokenizationModel")
class AcousticTokenizationModel(BaseSolution):
    """AcousticTokenizationModel"""

    def __init__(self, args):
        """init"""
        super().__init__()
        self.args = args

        args.front_end_type = getattr(args, "front_end_type", "")
        args.acoustic_backbone_type = getattr(args, "acoustic_backbone_type", "")

        self.acoustic_front_end_module = None
        self.acoustic_backbone_module = None
        if args.front_end_type:
            self.acoustic_front_end_module = eval(args.front_end_type)(args)
        if args.acoustic_backbone_type:
            self.acoustic_backbone_module = eval(args.acoustic_backbone_type)(args)
        self.acoustic_tokenizer = eval(args.acoustic_tokenizer_type)(args)

    def frontend(self, fbank, mask):
        '''frontend'''
        if self.acoustic_front_end_module is not None:
            return self.acoustic_front_end_module(fbank, mask)
        # Nothing to do in frontend
        return fbank, mask, "BTN"

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''Backbone'''
        attn_mask = False
        if (
            self.args.acoustic_backbone_type
            in ('OfflineTransformerBackbone', 'RelTransformerBackbone', 'EmformerBackbone')
            and self.acoustic_backbone_module.causal_transformer
        ):
            attn_mask = True
        backbone_out = self.acoustic_backbone_module(
            inputs, mask, attn_mask=attn_mask, frontend_shape=frontend_shape
        )
        return backbone_out

    def forward(self, batch_data):
        """forward"""
        raise NotImplementedError

    def extract_encoder_backbone_out(self, batch_data):
        """Extract output of the backbone."""
        inputs = batch_data['src']  # (B, T, ndim)
        input_masks = batch_data['src_mask']

        if not self.training:
            with torch.no_grad():
                # front-end
                front_end_out, backbone_mask, frontend_shape = self.frontend(inputs, input_masks)
                # output (B, T, N)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )
        else:
            with torch.enable_grad():
                front_end_out, backbone_mask, frontend_shape = self.frontend(inputs, input_masks)
                encoder_backbone_out = self.encoder_backbone(
                    front_end_out, backbone_mask, frontend_shape
                )

        return encoder_backbone_out, backbone_mask

    def extract_codes(self, batch_data):
        """forward"""
        if 'waveform' in batch_data:
            inputs = batch_data['waveform']  # (B, T, ndim)
            input_masks = batch_data['wav_mask']
        else:
            inputs = batch_data['src']  # (B, T, ndim)
            input_masks = batch_data['src_mask']
        if self.acoustic_backbone_module is not None:
            if not self.training:
                with torch.no_grad():
                    # front-end
                    front_end_out, backbone_mask, frontend_shape = self.frontend(
                        inputs, input_masks
                    )
                    # output (B, T, N)
                    encoder_backbone_out = self.encoder_backbone(
                        front_end_out, backbone_mask, frontend_shape
                    )
            else:
                with torch.enable_grad():
                    front_end_out, backbone_mask, frontend_shape = self.frontend(
                        inputs, input_masks
                    )
                    encoder_backbone_out = self.encoder_backbone(
                        front_end_out, backbone_mask, frontend_shape
                    )

            batch_size, seq_len_enc, enc_dim = encoder_backbone_out.shape
            codes = self.acoustic_tokenizer(
                encoder_backbone_out.view(batch_size * seq_len_enc, enc_dim)
            )
            codes = codes.view(batch_size, seq_len_enc)
        else:
            codes, backbone_mask = self.acoustic_tokenizer(inputs, input_masks)
        return codes, backbone_mask

    def extract_hubert_codes(self, batch_data):
        """forward"""
        inputs = batch_data['waveform']  # (B, T, ndim)
        input_masks = batch_data['wav_mask']

        codes, backbone_mask = self.acoustic_tokenizer(inputs, input_masks)
        return codes, backbone_mask

    def extract_data(self, batch_data, extraction_mode):
        if extraction_mode == "extract_encoder_backbone_out":
            encoder_backbone_out, backbone_mask = self.extract_encoder_backbone_out(batch_data)
            data = encoder_backbone_out
            lengths = torch.sum(backbone_mask, dim=1).long()
        elif extraction_mode == "extract_codes":
            codes, backbone_mask = self.extract_codes(batch_data)
            data = codes
            lengths = torch.sum(backbone_mask, dim=1).long()
        elif extraction_mode == "extract_hubert_codes":
            codes, backbone_mask = self.extract_hubert_codes(batch_data)
            data = codes
            lengths = torch.sum(backbone_mask, dim=1).long()
        else:
            raise ValueError("Unknown extraction_mode: {}".format(extraction_mode))
        return data, lengths

    def set_num_updates(self, num_updates):
        '''set_num_updates'''
        self.num_updates = num_updates
