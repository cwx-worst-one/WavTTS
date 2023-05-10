'''test wav2vec bitransformer'''

import torch
from torch import nn
from core.models.layers.transformer import BiTransformerLayer
from core.models.layers.multi_head_attn import MultiheadAttention
from core.models.layers.active_function import get_activation_fn


def _reset_parameters(module):
    '''reset parameters'''
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        if module.merge_qkv:
            module.in_proj.weight.data.normal_(mean=0.0, std=0.02)
        else:
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            enable_torch_version=False,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = torch.nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ):
        """
        torch.nn.LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                need_head_weights=need_head_weights,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                need_head_weights=need_head_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


def test_wav2vec_bitransformer():
    '''test wav2vec bitransformer'''
    bsz = 2
    seq_len = 512
    embed_dim = 768
    attention_heads = 12
    ffn_dim = 3072
    dropout = 0.0
    attention_dropout = 0.0
    activation_dropout = 0.0
    layer_norm_first = False
    activation = 'gelu'

    bitransformer = BiTransformerLayer(
        embed_dim=embed_dim,
        attention_heads=attention_heads,
        ffn_embed_dim=ffn_dim,
        attention_dropout=attention_dropout,
        hidden_dropout=dropout,
        activation_dropout=activation_dropout,
        activation=activation,
        normalize_before=layer_norm_first,
        clamp_inf=False,
    )

    w2v_transformer = TransformerSentenceEncoderLayer(
        embedding_dim=embed_dim,
        ffn_embedding_dim=ffn_dim,
        num_attention_heads=attention_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        activation_fn=activation,
        layer_norm_first=layer_norm_first,
    )

    bitransformer.cuda()
    w2v_transformer.cuda()

    w2v_transformer.apply(_reset_parameters)

    bitransformer.fc1.load_state_dict(w2v_transformer.fc1.state_dict())
    bitransformer.fc2.load_state_dict(w2v_transformer.fc2.state_dict())
    bitransformer.self_attn.in_proj.weight.data.copy_(w2v_transformer.self_attn.in_proj.weight.data)
    bitransformer.self_attn.in_proj.bias.data.copy_(w2v_transformer.self_attn.in_proj.bias.data)
    bitransformer.self_attn.out_proj.weight.data.copy_(
        w2v_transformer.self_attn.out_proj.weight.data
    )
    bitransformer.self_attn.out_proj.bias.data.copy_(w2v_transformer.self_attn.out_proj.bias.data)

    x = torch.randn(seq_len, bsz, embed_dim, device='cuda').requires_grad_()
    padding_mask = torch.randn(bsz, seq_len, device='cuda')
    padding_mask = padding_mask.int().bool()

    mx = x.clone().detach_().requires_grad_()
    mpadding_mask = padding_mask.clone().detach_()

    std_out, _ = w2v_transformer(x, self_attn_padding_mask=padding_mask, need_weights=False)
    my_out = bitransformer(mx, encoder_padding_mask=mpadding_mask)

    x.retain_grad()
    mx.retain_grad()

    std_out.sum().backward()
    my_out.sum().backward()

    assert torch.allclose(std_out, my_out, atol=3e-4)
    assert torch.allclose(x.grad.data, mx.grad.data, atol=1e-6)
