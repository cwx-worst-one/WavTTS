''' rare_word
'''
import torch
from torch import nn
from core.models.layers.normalization import LayerNorm
from core.models.layers.multi_head_attn import MultiheadAttention


class LSTMSentenceEmbedding(nn.Module):
    '''LSTMSentenceEmbedding'''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.embedding_size)
        self.lstm = nn.LSTM(
            input_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_layers=args.layer_num,
            dropout=args.dropout,
            batch_first=True,
            proj_size=args.get("proj_size", 0),
        )
        self.embed_using_states = args.get('embed_using_states', False)
        # (h_n)
        lstm_out_size = args.proj_size if args.get("proj_size", 0) else args.hidden_size
        if self.embed_using_states:
            lstm_out_size += args.hidden_size  # (c_n)
        self.proj_fc = nn.Sequential(
            *[
                nn.Linear(lstm_out_size, args.jointer_hidden_size),
                LayerNorm(args.jointer_hidden_size),
            ]
        )

    def forward(self, token):
        '''token: [N, L]
        output: [N, D]
        '''
        embed = self.embed_tokens(token)
        lstm_out, (state_h, state_c) = self.lstm(embed)
        if self.embed_using_states:
            sentence_embed = self.proj_fc(
                torch.cat(
                    (state_h[-1, :, :], state_c[-1, :, :]),
                    dim=-1,
                )
            )
        else:
            sentence_embed = self.proj_fc(lstm_out[:, -1, :])
        return sentence_embed


class RareWordAttnBias(nn.Module):
    '''RareWordAttnBias'''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.attn = MultiheadAttention(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            kdim=args.embed_dim,
            vdim=args.embed_dim,
            dropout=args.dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
        )
        self.proj_fc = nn.Linear(args.embed_dim, args.embed_dim)
        if hasattr(args, "predictor_emb_size"):
            self.query_cat_fc = nn.Linear(args.embed_dim + args.predictor_emb_size, args.embed_dim)

    def forward(self, query1, value, query2=None):
        '''query1: [B, U, D]
        query2: [B, U, D']
        value: [N, D] or [B, N, D]
        output: [B, U, D]
        '''
        if query2 is None:
            query = query1
        else:
            query = self.query_cat_fc(torch.cat((query1, query2), dim=-1))
        bsz = query.size(0)
        if len(value.size()) == 2:
            value = value.unsqueeze(0).repeat(bsz, 1, 1)
        attn_out, attn_weights = self.attn(
            query=query.permute(1, 0, 2),
            key=value.permute(1, 0, 2),
            need_weights=True,
        )
        return self.proj_fc(attn_out.permute(1, 0, 2)), attn_weights

    def forward_step(self, query1, value, query2=None):
        '''forward_step'''
        if len(query1.size()) == 2:
            query1 = query1.unsqueeze(1)
        if query2 is not None and len(query2.size()) == 2:
            query2 = query2.unsqueeze(1)
        out, _ = self.forward(query1, value, query2)
        if len(out.size()) == 3 and out.size(1) == 1:
            out = out.squeeze(1)
        return out
