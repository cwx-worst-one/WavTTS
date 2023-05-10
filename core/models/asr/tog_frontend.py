'''
text frontend
'''

from torch import nn

from core.models.layers.embedding import AbsPositionalEncoding
from core.models.layers.transformer import BiTransformerLayer


class TextEmbeddingTransformer(nn.Module):
    '''TextEmbeddingTransformer'''

    def __init__(self, args):
        '''init'''
        super().__init__()
        self.token_embedding = nn.Embedding(args.input_token_size, args.backbone_memory_size)
        self.pos_en = AbsPositionalEncoding(args.backbone_memory_size, args.dropout)
        self.transformer_layer = BiTransformerLayer(
            args.backbone_memory_size,
            args.self_attn_heads,
            args.backbone_hidden_size,
            args.self_attn_dropout,
            args.dropout,
            activation_dropout=args.self_attn_activation_dropout,
            activation=args.self_attn_activation_fn,
            normalize_before=args.self_attn_layer_norm_before,
            clamp_inf=True,
        )
        self.out_fc = nn.Linear(args.backbone_memory_size, args.backbone_memory_size)

    def forward(self, token, token_mask=None):
        '''forward
        token and token_mask: [B, T]
        output: [B, T, D]
        '''
        output = self.token_embedding(token)
        output = self.pos_en(output)
        output = output.transpose(0, 1).contiguous()
        token_mask_tmp = (1 - token_mask).bool()
        # set first two positions to False to avoid nan for all-zero tog
        token_mask_tmp[:, :2] = False
        output = self.transformer_layer(output, encoder_padding_mask=token_mask_tmp)
        output = output.transpose(0, 1).contiguous()
        output = self.out_fc(output)
        if token_mask is not None:
            output = output * token_mask.unsqueeze(2)
        return output, token_mask, "BTN"
