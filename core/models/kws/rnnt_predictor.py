'''
rnnt_predictor.py.
'''
import torch
from torch import nn


class DNNPredictor(nn.Module):
    '''DNN predictor'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.predictor_out_layer_norm = bool(args.predictor_out_layer_norm)
        self.tgt_vocab_size = args.tgt_vocab_size
        self.embed_tokens = nn.Embedding(self.tgt_vocab_size, args.predictor_embedding_dim)
        self.dnn_layers = [
            nn.ReLU(),
            nn.Linear(args.predictor_embedding_dim, args.predictor_hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(args.predictor_layer_num):
            self.dnn_layers.append(nn.Linear(args.predictor_hidden_dim, args.predictor_hidden_dim))
            self.dnn_layers.append(nn.ReLU())
        self.dnn_layers.append(nn.Linear(args.predictor_hidden_dim, args.rnnt_hidden_size))
        self.dnn = nn.Sequential(*self.dnn_layers)
        if self.predictor_out_layer_norm:
            self.predictor_out_ln = nn.LayerNorm(args.rnnt_hidden_size)

    def forward(self, prev_tgt):
        '''forward'''
        prev_emb = self.embed_tokens(prev_tgt)
        predictor_out = self.dnn(prev_emb)
        if self.predictor_out_layer_norm:
            predictor_out = self.predictor_out_ln(predictor_out)
        return predictor_out

    def step(self, prev_tgt, states):
        '''for decoding step'''
        prev_emb = self.embed_tokens(prev_tgt)
        predictor_out = self.dnn(prev_emb)
        if self.predictor_out_layer_norm:
            predictor_out = self.predictor_out_ln(predictor_out)
        if states is None:
            states = []
            for _ in range(self.args.predictor_layer_num):
                zeros = torch.zeros(
                    prev_tgt.size(0),
                    self.args.predictor_hidden_dim,
                    dtype=predictor_out.dtype,
                    device=predictor_out.device,
                )
                state_i = (zeros, zeros)
                states.append(state_i)
        return predictor_out, states
