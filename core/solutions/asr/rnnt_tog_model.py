''' RnntTogModel, https://arxiv.org/pdf/2202.13155.pdf '''
# pylint: disable=unused-argument
import torch
import torch.nn.functional as F
from core.models.asr.tog_frontend import *
from core.models.asr.acoustic_backbone import rnnt_transpose
from core.solutions.base_solution import register_solution
from .base_rnnt_model import BaseRnntModel
from .rnnt_twopass_model import RnntCaseModel


@register_solution("RnntTogModel")
class RnntTogModel(BaseRnntModel):
    '''RnntTogModel'''

    def __init__(self, args):
        super().__init__(args)
        tog_args = args.tog_front_end_args
        self.token_frontend_type = tog_args.get("token_frontend_type", None)
        assert self.token_frontend_type is not None
        self.token_frontend = eval(tog_args.token_frontend_type)(tog_args)
        self.merge_by_concat = tog_args.get("merge_by_concat", True)
        self.concat_fc = None
        if self.merge_by_concat:
            self.concat_fc = nn.Linear(
                args.backbone_memory_size + tog_args.backbone_memory_size, args.backbone_memory_size
            )
        self.tog_fix = args.get('tog_fix', False)
        if self.tog_fix:
            self.token_frontend.requires_grad_(False)
            if self.concat_fc is not None:
                self.concat_fc.requires_grad_(False)

    def encoder(self, batch_data):
        '''
        Encoder Module in RNN-T
        '''
        fbank = batch_data['src']  # (B, T, ndim)
        fbank_mask = batch_data['src_mask']
        token = batch_data['phone_src']
        token_mask = batch_data['phone_mask']
        if 'domain' in batch_data:
            domain_emb = F.one_hot(batch_data['domain'], num_classes=self.args.domain_num).cuda()
            fbank = torch.cat(
                (fbank, domain_emb.unsqueeze(1).to(fbank.dtype).repeat(1, fbank.shape[1], 1)), dim=2
            )
        # acoustic front-end
        front_end_out, backbone_mask, frontend_shape = self.frontend(fbank, fbank_mask)
        # tog_frontend
        token_out, token_mask, token_shape = self.token_frontend(token.long(), token_mask)
        # set acoustic & tog front_end output to the same length
        front_end_out = rnnt_transpose(front_end_out, "BTN", frontend_shape)
        token_out = rnnt_transpose(token_out, "BTN", token_shape)
        frontend_shape = 'BTN'
        acoustic_len = backbone_mask.size(1)
        token_len = token_mask.size(1)
        if acoustic_len >= token_len:
            token_out = F.pad(token_out, (0, 0, 0, acoustic_len - token_len))
            token_mask = F.pad(token_mask, (0, acoustic_len - token_len))
        else:
            backbone_mask = F.pad(backbone_mask, (0, token_len - acoustic_len))
            front_end_out = F.pad(front_end_out, (0, 0, 0, token_len - acoustic_len))
        backbone_mask = (backbone_mask + token_mask).bool().float()
        if self.merge_by_concat:
            front_end_out = torch.cat((front_end_out, token_out), dim=-1)
            front_end_out = self.concat_fc(front_end_out)
        else:
            # just add acoustic & tog front_end output
            front_end_out = front_end_out + token_out
        # output (B, T, N)
        encoder_backbone_out = self.encoder_backbone(front_end_out, backbone_mask, frontend_shape)
        # ctc mtl branch
        mtl_logits = None
        if self.mtl_module is not None:
            mtl_logits = self.mtl_module(encoder_backbone_out)
        # head
        encoder_out = self.acoustic_head_module(encoder_backbone_out)
        # backbone_pool_module
        if self.backbone_pool_module is not None:
            encoder_out = self.backbone_pool_module(encoder_out)
            batch_data["mtl_mask"] = backbone_mask.clone() if backbone_mask is not None else None
            backbone_mask = backbone_mask[:, :-1][:, ::2] if backbone_mask is not None else None
        if 'eos' in batch_data:
            batch_data['eos'] = batch_data['eos'] // self.args.downsampling_size
        return encoder_out, backbone_mask, self.training, mtl_logits, encoder_backbone_out


@register_solution('RnntCaseTogModel')
class RnntCaseTogModel(RnntTogModel, RnntCaseModel):
    '''RnntCaseTogModel'''

    def __init__(self, args):
        '''init'''
        # pylint: disable=super-init-not-called
        RnntCaseModel.__init__(self, args)
        self.merge_by_concat = False
        tog_args = args.tog_front_end_args
        assert not tog_args.get("merge_by_concat", False)
        self.token_frontend = eval(tog_args.token_frontend_type)(tog_args)
        if args.get('tog_fix', False):
            self.token_frontend.requires_grad_(False)
