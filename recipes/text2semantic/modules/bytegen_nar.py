import os 
import math
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import TransformerEncoderLayer


class NewGELUActivation(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class NAR(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        n_embd=1024 # 隐藏层dim
        n_layer=12 # 多少层
        n_head=16 # 多少头

        self.hp = hp
        self.wpe = nn.Embedding(2048, n_embd) # [max_pos, dim]
        self.embeddings = nn.ModuleList(
            # pad 1 + text 1000 + wav 1024 + eos 2
            [nn.Embedding(1 + self.hp.num_units + hp.quant_token_num + 2, embedding_dim=n_embd, padding_idx=0)] * hp.num_res
        )
        self.dense_layer = nn.Linear(n_embd * hp.num_res, n_embd, bias=False)
        self.res_embedding = nn.Embedding(hp.num_res - 1, n_embd)

        activation = NewGELUActivation()
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                TransformerEncoderLayer(
                    d_model=n_embd,
                    nhead=n_head,
                    dim_feedforward=n_embd * 4,
                    dropout=0.1,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=True,
                )
            )
        self.mlp_layer = nn.Linear(n_embd, hp.quant_token_num + 2, bias=False) # quant_token_num + eos

    def forward(self, x, seq_sen_ids, position_ids):
        # x: [B, n_codebook, t]
        # training, random mask
        device = x.device
        b, n_codebook, t = x.size()

        text_len = (seq_sen_ids == 1).sum(dim=1)
        wav_len = (seq_sen_ids == 2).sum(dim=1)
        # random mask n_codebook
        layer_index = torch.randint(low=1, high=n_codebook, size=[b,], device=device)
        mask1 = layer_index.unsqueeze(1) > torch.arange(n_codebook, device=device).unsqueeze(0) # [b, 1] > [1, n_codebook] = [b, n_codebook]
        # random mask length
        unmask_len = torch.randint(low=1, high=320, size=[b,], device=device) # 320 frames means 4 seconds
        rand_len = torch.randint(low=1, high=10000, size=[b,], device=device)
        unmask_len = torch.where(wav_len < unmask_len, rand_len % wav_len, unmask_len).clamp(1) # [b,]
        mask2 = (unmask_len + text_len).unsqueeze(1) > torch.arange(t, device=device).unsqueeze(0) # [b, 1] > [1, t] = [b, t]
        # sequence_mask
        seq_mask = (text_len + wav_len).unsqueeze(1) > torch.arange(t, device=device).unsqueeze(0) # [b, 1] > [1, t] = [b, t]
        mask = (mask1.unsqueeze(2) + mask2.unsqueeze(1)) * seq_mask.unsqueeze(1)
        mask_x = torch.where(mask, x, torch.zeros_like(x)) # [b, n_codebook, t]
        embeddings = []
        for i in range(n_codebook):
            embeddings.append(self.embeddings[i](mask_x[:, i, :]))
        embeddings = torch.cat(embeddings, dim=-1) # [b, t, d*num_res]
        # dense to fit transformer
        embeddings = self.dense_layer(embeddings)

        ### transformers
        res_embeddings = self.res_embedding(layer_index - 1).unsqueeze(1) # [b, 1, d]
        # position embeddings
        outputs = embeddings + self.wpe(position_ids) + res_embeddings

        for layer in self.layers:
            outputs = layer(outputs, src_key_padding_mask=~seq_mask)
        logits = self.mlp_layer(outputs) # [b, t, n_logits]

        # get target token
        target_x = []
        for i, ind in enumerate(layer_index):
            target_x.append(x[i, ind, :]) # [b, n_codebook, t] -> n * [t]
        target_x = torch.stack(target_x, dim=0)
        target_mask = seq_mask * (~mask2)
        target_x = torch.where(target_mask, target_x - 1 - self.hp.num_units, torch.zeros_like(target_x)) # remove padding & text id
        return logits, target_x, target_mask

    def predict(self, x, unmask_len, seq_sen_ids, position_ids, layer_index):
        device = x.device
        b, n_codebook, t = x.size()

        text_len = (seq_sen_ids == 1).sum(dim=1)
        wav_len = (seq_sen_ids == 2).sum(dim=1)
        # codebook mask
        if isinstance(layer_index, int):
            layer_index = torch.LongTensor([layer_index,]).repeat(b,).to(device)
        mask1 = layer_index.unsqueeze(1) > torch.arange(n_codebook, device=device).unsqueeze(0) # [b, 1] > [1, n_codebook] = [b, n_codebook]
        # unmask wav len
        mask2 = (unmask_len + text_len).unsqueeze(1) > torch.arange(t, device=device).unsqueeze(0) # [b, 1] > [1, t] = [b, t]
        # sequence_mask
        seq_mask = (text_len + wav_len).unsqueeze(1) > torch.arange(t, device=device).unsqueeze(0) # [b, 1] > [1, t] = [b, t]
        # total mask for x
        mask = (mask1.unsqueeze(2) + mask2.unsqueeze(1)) * seq_mask.unsqueeze(1)
        mask_x = torch.where(mask, x, torch.zeros_like(x)) # [b, n_codebook, t]
        embeddings = []
        for i in range(n_codebook):
            embeddings.append(self.embeddings[i](mask_x[:, i, :]))
        embeddings = torch.cat(embeddings, dim=-1) # [b, t, d*num_res]
        # dense to fit transformer
        embeddings = self.dense_layer(embeddings)

        ### transformers
        res_embeddings = self.res_embedding(layer_index - 1).unsqueeze(1) # [b, 1, d]
        # position embeddings
        outputs = embeddings + self.wpe(position_ids) + res_embeddings

        for layer in self.layers:
            outputs = layer(outputs, src_key_padding_mask=~seq_mask)
        logits = self.mlp_layer(outputs) # [b, t, n_logits]
        return logits
