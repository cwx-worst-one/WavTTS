"""lstm lm."""
import torch
from torch import nn

from .base_model import NNLM
from ..layers.lstmp_layer import IncrementalLSTM


class LSTMLM(NNLM):
    '''
    LM lstm
    '''

    def __init__(self, args):
        '''init
        Args:
            args: solution config, dict type.
        '''
        super().__init__()
        self.embed_tokens = nn.Embedding(args['tgt_vocab_size'], args['embedding_size'])
        self.lstm = IncrementalLSTM(
            args['embedding_size'],
            args['lstm_cell_size'],
            num_layers=args['lstm_num_layers'],
            batch_first=True,
            bidirectional=False,
        )
        self.pred_fc = torch.nn.Sequential(
            *[
                torch.nn.Linear(args['lstm_cell_size'], args['pred_fc_hidden_size']),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(args['pred_fc_hidden_size'], args['tgt_vocab_size']),
            ]
        )

    @property
    def state_size(self):
        '''state size.'''
        return self.lstm.num_layers * 2 * self.lstm.hidden_size

    def forward(self, prev_tgt):
        '''forward
        Args:
            prev_tgt: [B, U], previous target
        Return:
            predictor_out: [B, U, H], predictor output
        '''

        prev_embedding = self.embed_tokens(prev_tgt)
        lstm_out, _ = self.lstm(prev_embedding)
        logits = self.pred_fc(lstm_out)
        return logits

    def forward_step(self, prev_tgt, states=None):
        '''froward by a token step'''
        if isinstance(states, torch.Tensor):
            assert states.shape[1] == self.state_size
            offset, hidden_size = 0, self.lstm.hidden_size
            new_states = []
            for _ in range(self.lstm.num_layers):
                h0 = states[:, offset : offset + hidden_size]
                offset += hidden_size
                c0 = states[:, offset : offset + hidden_size]
                offset += hidden_size
                new_states.append((h0, c0))
            states = new_states

        prev_emb = self.embed_tokens(prev_tgt)
        lstm_out, states = self.lstm.forward_step(prev_emb, states)
        logits = self.pred_fc(lstm_out)
        lprobs = logits.log_softmax(dim=1)
        return lprobs, list(states)
