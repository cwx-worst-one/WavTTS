import torch
from torch import nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from torch.nn import init

from samantha.components.ctiga.ops.rms_norm import rms_norm


class ConditionRMSNorm(nn.Module):
    def __init__(self, n_emb, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.cond_embed = nn.Embedding(n_emb, hidden_size),
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.cond_embed.weight)
    

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class DropoutV1(nn.Module):
    def __init__(self, p, activated):
        super().__init__()
        self.p = p
        self.activated = activated

    def forward(self, x):
        if self.activated:
            return F.dropout(x, self.p, training=True)
        else:
            return F.dropout(x, self.p, training=self.training)
            

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class UnitVector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert x.shape[-1] == self.dim
        norm = torch.norm(x, dim=-1, keepdim=True) * self.dim **-0.5
        return x / norm.clamp(min=1e-8)


class FrontendEmbedding(nn.Module):
    def __init__(self,
                 phone_embed_dim,
                 tone_embed_dim,
                 out_dim,
                 padding_idx=0,
                 n_phone=200,
                 n_tone=20,
                 #n_phone=148,
                 #n_tone=13,
                 #n_wordcateg=6,
                 #n_prosody=7
                 ):
        super().__init__()
        self.phone_embedding = nn.Embedding(n_phone,
                phone_embed_dim, padding_idx=padding_idx)
        self.tone_embeddig = nn.Embedding(n_tone,
                tone_embed_dim, padding_idx=padding_idx)
        input_dim = phone_embed_dim + tone_embed_dim

        self.out_linear = nn.Linear(input_dim, out_dim, bias=False)

    def forward(self, inputs):
        phone_emb = self.phone_embedding(inputs["phone"])
        tone_emb = self.tone_embeddig(inputs["tone"])
        emb = torch.cat([phone_emb, tone_emb], dim=-1)

        return self.out_linear(emb)
