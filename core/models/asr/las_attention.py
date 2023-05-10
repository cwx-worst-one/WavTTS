''' MHAttention '''
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..utils import xavier_init


class MHAttention(nn.Module):
    '''MHAttention'''

    def __init__(self, args):
        super().__init__()
        self.prev_hid_proj_fc = nn.Linear(
            args.embedding_size + args.decoder_lstm_hidden_size, args.atten_hidden_size
        )
        self.args = args
        xavier_init(self.prev_hid_proj_fc)

    def init_states(self, bsz, enc_len):
        '''init_states'''
        param = next(self.parameters())
        return param.data.new(bsz, 2, enc_len).zero_()

    def forward(self, encoder, encoder_proj, yt_1, ht_1, encoder_mask=None, prev_att_state=None):
        """forward"""
        head_dim = self.args.atten_hidden_size // self.args.multi_head_num
        head_num = self.args.multi_head_num
        (bsz, enc_len, _) = encoder.size()
        if encoder_mask is not None:
            expand_encoder_mask = (
                encoder_mask.unsqueeze(1)
                .repeat(1, head_num, 1)
                .contiguous()
                .view(bsz * head_num, enc_len)
            )
        else:
            expand_encoder_mask = 1
        q = self.prev_hid_proj_fc(torch.cat([yt_1, ht_1], dim=-1)).view(bsz * head_num, 1, head_dim)
        q = q / math.sqrt(head_dim)
        k = (
            encoder_proj.view(bsz, enc_len, head_num, head_dim)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(bsz * head_num, head_dim, enc_len)
        )
        v = encoder.view(bsz, enc_len, head_num, -1).permute(0, 2, 1, 3).contiguous()
        v = v.view(bsz * head_num, enc_len, -1)
        multi_head_att_weights = F.softmax(
            torch.bmm(q, k).squeeze(1) - 1000 * (1 - expand_encoder_mask), dim=1
        )  # (B, enc_len)
        multi_head_att_weights = F.dropout(
            multi_head_att_weights, p=self.args.multi_att_weight_drop, training=self.training
        )
        multi_head_att_ctx = (v * (multi_head_att_weights.unsqueeze(2))).sum(dim=1)
        att_ctx = multi_head_att_ctx.view(bsz, -1)
        if prev_att_state is None:
            return att_ctx
        return att_ctx, prev_att_state
