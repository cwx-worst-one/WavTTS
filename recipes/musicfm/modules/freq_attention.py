import torch
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_flash_attn=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        self.use_flash_attn = use_flash_attn

    def forward(self, x, rotary_emb=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if rotary_emb is not None:
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)

        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout_p,
                is_causal=False,
            )

            out = rearrange(out, "b h n d -> b n (h d)")
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class FreqAttention(nn.Module):
    def __init__(self, dim=128, n_head=4, is_flash=False):
        super(FreqAttention, self).__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.attention = Attention(
            dim,
            heads=n_head,
            dim_head=int(dim / n_head),
            dropout=0.1,
            use_flash_attn=is_flash,
        )
        self.rotary_emb = RotaryEmbedding(dim // n_head)

    def forward(self, emb):
        """
        Input:
            emb (torch.Tensor): input embedding (batch, channel, frequency, time)
        Output:
            emb (torch.Tensor): output embedding(batch, time, channel')
        """
        b, c, f, t = emb.shape
        emb = rearrange(emb, "b c f t -> (b t) f c")
        emb = self.input_norm(emb)
        emb = self.attention(emb, self.rotary_emb)
        emb = rearrange(emb, "(b t) f c -> b c f t", b=b)
        return emb
