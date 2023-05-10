''' TransformerXL modules '''
from torch import nn

from core.models.layers.multi_head_attn import RelPartialLearnableMultiHeadAttn
from core.models.layers.feed_forward import PositionwiseFeedForward


class RelPartialLearnableDecoderLayer(nn.Module):
    '''RelPartialLearnableDecoderLayer'''

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, activation_fn, **kwargs):
        '''constructor'''
        super(__class__, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt=kwargs.get('dropatt'),
            pre_lnorm=kwargs.get('pre_lnorm'),
        )
        self.pos_ff = PositionwiseFeedForward(
            d_model,
            d_inner,
            dropout,
            activation_fn,
            extra_dropout=True,
            use_layer_norm=True,
            pre_layer_norm=kwargs.get('pre_lnorm'),
            swish_beta=kwargs.get('swish_beta'),
            relu_inplace=kwargs.get('relu_inplace'),
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        '''forward'''
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output
