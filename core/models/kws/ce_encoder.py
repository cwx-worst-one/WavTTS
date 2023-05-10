'''
ce_encoder.py.
'''
import torch
from torch import nn
from core.models.layers.fsmn_layer import FSMNLayer
from core.models.layers.unfold import PantherUnFold
from core.models.asr.acoustic_backbone import DFSMNBackboneLN


class DFSMNEncoder(nn.Module):
    '''DFSMN encoder'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_trans_fc = nn.Sequential(
            *[
                nn.Linear(args.fbank_dim * args.input_concat_size, args.dfsmn_hidden_size),
                nn.ReLU(),
                nn.Linear(args.dfsmn_hidden_size, args.dfsmn_memory_size, bias=False),
            ]
        )
        self.fsmn = FSMNLayer(
            args.dfsmn_memory_size,
            args.fsmn_left_kernel_size,
            args.fsmn_right_kernel_size,
            dilation=args.fsmn_dilation,
        )
        self.dfsmn = DFSMNBackboneLN(args)
        self.pred_fc = torch.nn.Sequential(
            *[
                torch.nn.Linear(args.dfsmn_memory_size, args.dfsmn_hidden_size),
                torch.nn.ReLU(),
                nn.Dropout(args.dfsmn_dropout),
                torch.nn.Linear(args.dfsmn_hidden_size, args.tgt_vocab_size),
            ]
        )

        self.unfold = PantherUnFold(
            (args.input_concat_size, 1), stride=(args.downsampling_size, 1), padding=(0, 0)
        )
        self.input_concat_size = args.input_concat_size
        self.fbank_dim = args.fbank_dim
        self.selected_dfsmn_layer = args.get('selected_dfsmn_layer', -1)
        self.do_cmvn = args.get('model_do_cmvn', False)
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
        hard_mask = expand_hard_mask = fbank_mask
        if hard_mask is not None:
            dfsmn_mask = self.unfold(fbank_mask.unsqueeze(1).unsqueeze(3))
            hard_mask = dfsmn_mask[:, self.input_concat_size - 1, :].contiguous()
            expand_hard_mask = hard_mask.unsqueeze(2)
        input_trans = self.input_trans_fc(unfold_fbank)
        fsmn_out = self.fsmn(input_trans, expand_hard_mask)
        layer_out = self.dfsmn(
            fsmn_out,
            expand_hard_mask.squeeze(2),
            "BTN",
            selected_layer_idx=self.selected_dfsmn_layer,
        )
        if self.selected_dfsmn_layer != -1:
            dfsmn_out, selected_dfsmn_out = layer_out
        else:
            dfsmn_out = layer_out
            selected_dfsmn_out = None
        encoder_out = self.pred_fc(dfsmn_out)

        tgt = hard_tgt = ce_label
        if tgt is not None:
            tgt = tgt.type_as(fbank).unsqueeze(1).unsqueeze(3)
            unfold_tgt = self.unfold(tgt)  # (B,downsample_size,T)
            hard_tgt = unfold_tgt[:, self.args.downsampling_size // 2, :].contiguous().long()

        return encoder_out, selected_dfsmn_out, hard_mask, hard_tgt
