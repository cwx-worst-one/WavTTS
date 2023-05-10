''' LAS Hypotheses encoder '''
# pylint: disable=unused-wildcard-import
from torch import nn
from core.models.layers.embedding import AbsPositionalEncoding
from core.models.asr.acoustic_backbone import *


class TextEncoder(nn.Module):
    """
    Encoder with Embedding, transformer/lstm layers, residual connections and optional
    dropout.
    """

    def __init__(self, args):
        '''init'''
        super().__init__()
        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.embedding_size)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=args.embedding_size**-0.5)
        self.pos_embed = None
        if 'Transformer' in args.backbone_type:
            self.pos_embed = AbsPositionalEncoding(
                d_model=args.embedding_size, dropout_rate=args.dropout
            )
        self.encoder_layers = eval(args.backbone_type)(args)

    def forward(self, inputs, enc_mask):
        """
        Execute the encoder.
        :param inputs: tensor with word piece numbers, must be a long tensor [B, T_enc]
        :param enc_mask: mask tensor indicating rnnt_out tensor true lengths, must be a bool tensor
        returns: tensor with encoded sequences, shape: (B, T_enc, args.embedding_size)
        output: shape `(batch, seq_len, args.embedding_size)`
        """
        x = self.embed_tokens(inputs)
        if self.pos_embed is not None:
            x = self.pos_embed(x)  # output (B, T, ndim)
        output = self.encoder_layers(x, enc_mask, 'BTN')
        return output
