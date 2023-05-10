'''
rnnt_encoder.py.
'''
import torch
from torch import nn
from core.models.layers.unfold import PantherUnFold
from core.models.asr.acoustic_frontend import FSMNModule
from core.models.asr.acoustic_backbone import DFSMNBackboneLN


class ConvDFSMNEncoder(nn.Module):
    '''Conv DFSMN encoder'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_trans_fc = nn.Sequential(
            *[
                nn.Linear(
                    args.fbank_dim * args.input_concat_size,
                    args.backbone_hidden_size,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(args.backbone_hidden_size, args.backbone_memory_size, bias=False),
            ]
        )
        self.fsmn = FSMNModule(
            args.backbone_memory_size,
            args.fsmn_left_kernel_size,
            args.fsmn_right_kernel_size,
            dilation=args.fsmn_dilation,
        )
        self.dfsmn = DFSMNBackboneLN(args)
        self.pred_fc = torch.nn.Sequential(
            *[
                nn.Linear(args.backbone_memory_size, args.backbone_hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.backbone_hidden_size, args.rnnt_hidden_size),
            ]
        )
        self.unfold = PantherUnFold(
            (args.input_concat_size, 1),
            stride=(args.downsampling_size, 1),
            padding=(args.input_concat_size // 2, 0),
        )
        self.encoder_out_layer_norm = args.encoder_out_layer_norm
        if self.encoder_out_layer_norm:
            self.encoder_out_ln = nn.LayerNorm(args.rnnt_hidden_size)
        self.input_concat_size = args.input_concat_size
        self.fbank_dim = args.fbank_dim
        self.selected_dfsmn_layer = getattr(args, 'selected_dfsmn_layer', -1)
        self.selected_label_idx = getattr(args, 'selected_label_idx', args.input_concat_size // 2)
        self.do_cmvn = getattr(args, 'model_do_cmvn', False)
        if self.do_cmvn:
            self.mean = nn.Parameter(torch.zeros(args.fbank_dim))
            self.inv_std = nn.Parameter(torch.ones(args.fbank_dim))

    def forward(self, fbank, fbank_mask=None, ce_label=None):
        """forward"""
        if self.do_cmvn:
            fbank = (fbank - self.mean) * self.inv_std
        bsz = fbank.size()[0]
        unfold_fbank = (
            (self.unfold(fbank.unsqueeze(1)))
            .contiguous()
            .view(bsz, self.input_concat_size, -1, self.fbank_dim)
        )
        unfold_fbank = (
            unfold_fbank.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.input_concat_size * self.fbank_dim)
        )
        expand_dfsmn_mask = hard_mask = fbank_mask
        if hard_mask is not None:
            dfsmn_mask = self.unfold(fbank_mask.unsqueeze(1).unsqueeze(3))
            hard_mask = dfsmn_mask[:, self.input_concat_size - 1, :].contiguous()
            expand_dfsmn_mask = hard_mask.unsqueeze(1)
        input_trans = self.input_trans_fc(unfold_fbank)
        input_trans = input_trans.transpose(1, 2).contiguous()
        input_trans = self.fsmn(input_trans)
        if hard_mask is not None:
            input_trans = input_trans * expand_dfsmn_mask
            expand_dfsmn_mask = hard_mask.squeeze(1)
        selected_dfsmn_out = None
        if self.selected_dfsmn_layer != -1:
            dfsmn_out, selected_dfsmn_out = self.dfsmn(
                input_trans, expand_dfsmn_mask, "BNT", self.selected_dfsmn_layer
            )
        else:
            dfsmn_out = self.dfsmn(input_trans, expand_dfsmn_mask, "BNT", self.selected_dfsmn_layer)
        encoder_out = self.pred_fc(dfsmn_out)
        if self.encoder_out_layer_norm:
            encoder_out = self.encoder_out_ln(encoder_out)
        tgt = hard_tgt = ce_label
        if tgt is not None:
            unfold_tgt = self.unfold(tgt.type_as(fbank).unsqueeze(1).unsqueeze(3))
            hard_tgt = unfold_tgt[:, self.selected_label_idx, :].contiguous().long()
        return encoder_out, selected_dfsmn_out, hard_mask, hard_tgt


class ConvNolowrankDFSMNEncoder(ConvDFSMNEncoder):
    '''Conv Nolowrank DFSMN encoder'''

    def __init__(self, args):
        super().__init__(args)
        self.input_trans_fc = nn.Sequential(
            *[
                nn.Linear(args.fbank_dim * args.input_concat_size, args.input_lowrank_size, 1),
                nn.ReLU(inplace=True),
                nn.Linear(
                    args.input_lowrank_size,
                    args.backbone_hidden_size,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(args.backbone_hidden_size, args.backbone_memory_size, bias=False),
            ]
        )
