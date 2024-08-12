import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from .util_layer import RMSNorm
from samantha.models.based_ctiga_llama import ModelArgs as LLamaArgs
from samantha.models.speechdit import TimeEmbedding
from samantha.models.based_ctiga_llama import LLaMa   

class MultiTransformer(nn.Module):
    def __init__(
            self,
            n_blocks,
            input_dim,
            hidden_dim,
            n_layers,
            n_heads
            ):
        super().__init__()
        self.n_blocks = n_blocks
        self.discs = nn.ModuleList([
            TransformerDisc(input_dim, hidden_dim, n_layers, n_heads) for _ in range(n_blocks)
            ])

    def forward(self, feats, t, attention_mask=None):
        assert len(feats) == self.n_blocks
        if attention_mask.shape[1] == feats[0].shape[1] - 2:
            attention_mask = F.pad(attention_mask, (2,0), "constant", 1)
        out = []
        for feat, disc in zip(feats, self.discs):
            out.append(disc(feat, t, attention_mask=attention_mask))
        return out

class TransformerDisc(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            n_layers, 
            n_heads
            ):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False)
        config = LLamaArgs(
            dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            causal=False,
            use_qk_norm="head",
            flashattn_version="2"
            )
        self.net = LLaMa(config, "ctiga")
        self.t_embed = TimeEmbedding(hidden_dim, bias=False)

    def forward(self, x, t, attention_mask=None):
        time_emb = self.t_embed(t).unsqueeze(1)

        x = self.in_layer(x) + time_emb
        x = self.net(x, x.shape[1], attention_mask=attention_mask)
        x = self.out_layer(x)
        return x


class MultiResNet1d(nn.Module):
    def __init__(
            self,
            n_blocks,
            input_dim,
            n_conv_layers,
            hidden_dim,
            kernel_size,
            ):
        super().__init__()
        self.n_blocks = n_blocks
        self.discs = nn.ModuleList([ResNet1d(input_dim, 1, hidden_dim, n_conv_layers, kernel_size) for _ in range(n_blocks)])

    def forward(self, feats):
        assert len(feats) == self.n_blocks
        out = []
        for feat, disc in zip(feats, self.discs):
            out.append(disc(feat))
        return out
        

class ResNet1d(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
            n_conv_layers,
            kernel_size
            ):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2

        self.prenet = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, padding=0, bias=False)

        self.resblock = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False),
                        nn.GELU(),
                        RMSNorm(hidden_dim, feat_dim=1),
                        ) for _ in range(n_conv_layers)
                    ]
                )

        self.postnet = nn.Conv1d(hidden_dim, output_dim, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.prenet(x)
        for layer in self.resblock:
            x += layer(x)
        x = self.postnet(x)
        return x

