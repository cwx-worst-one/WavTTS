import torch
from torch import nn

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

