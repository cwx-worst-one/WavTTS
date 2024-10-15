import torch
from torch import nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, decay_factor=10000):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        #self.encoding = torch.zeros(max_len, d_model)
        self.register_buffer('encoding', torch.zeros(max_len, d_model))
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (decay_factor ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (decay_factor ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, T):
        return self.encoding[:T, :]
    

class FrontendEmbedding(nn.Module):
    def __init__(self,
                 phone_embed_dim,
                 tone_embed_dim,
                 wordseg_embed_dim,
                 out_dim,
                 padding_idx=0,
                 n_phone=200,
                 n_tone=20,
                 n_wordseg=8,
                 lang_embed_dim=0,
                 n_lang=0,
                 ):
        super().__init__()
        self.phone_embedding = nn.Embedding(n_phone,
                phone_embed_dim, padding_idx=padding_idx)
        self.tone_embeddig = nn.Embedding(n_tone,
                tone_embed_dim, padding_idx=padding_idx)
        self.wordseg_embeddig = nn.Embedding(n_wordseg,
                wordseg_embed_dim, padding_idx=padding_idx)

        if lang_embed_dim > 0:
            self.lang_embedding = nn.Embedding(n_lang,
                    lang_embed_dim, padding_idx=padding_idx)
            input_dim = phone_embed_dim + tone_embed_dim + wordseg_embed_dim + lang_embed_dim
        else:
            self.lang_embedding = None
            input_dim = phone_embed_dim + tone_embed_dim + wordseg_embed_dim
        self.out_linear = nn.Linear(input_dim, out_dim, bias=False)

    def forward(self, inputs):
        phone_emb = self.phone_embedding(inputs["phone"])
        tone_emb = self.tone_embeddig(inputs["tone"])
        wordseg_emb = self.wordseg_embeddig(inputs["word_seg"])
        if self.lang_embedding is not None:
            lang_embed = self.lang_embedding(inputs["lang"])
            emb = torch.cat([phone_emb, tone_emb, wordseg_emb, lang_embed], dim=-1)
        else:
            emb = torch.cat([phone_emb, tone_emb, wordseg_emb], dim=-1)

        return self.out_linear(emb)


class DurationEmbedding(nn.Module):
    def __init__(self, 
                duration_embed_dim,
                out_dim,
                padding_idx=0,
                n_duration=300,):
        super().__init__()
        self.duration_embedding = nn.Embedding(n_duration, 
                duration_embed_dim, padding_idx=padding_idx)
        input_dim = duration_embed_dim
        self.out_linear = nn.Linear(input_dim, out_dim, bias=False)

    def forward(self, inputs):
        duration_emb = self.duration_embedding(inputs)
        return self.out_linear(duration_emb)

