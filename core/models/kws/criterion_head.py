'''
criterion_head.py.
'''
import torch
from torch import nn
from core.models.asr.acoustic_backbone import DFSMNBackboneLN


class DNNHead(nn.Module):
    '''DNN head'''

    def __init__(self, args):
        super().__init__()
        self.hidden_layer_num = args.head_hidden_layer_num
        self.layer_norm = args.get('head_layer_norm', 0)
        self.dropout = args.get('head_dropout', 0.0)
        self.layers = []
        assert self.hidden_layer_num >= 0
        self.relu_before_input = args.get('head_relu_before_input', True)
        if self.relu_before_input:
            self.layers.append(nn.ReLU())
        for i in range(self.hidden_layer_num):
            if i == 0:
                input_size = args.head_input_size
            else:
                input_size = args.head_hidden_size
            output_size = args.head_hidden_size
            if self.layer_norm:
                self.layers.append(nn.LayerNorm(input_size))
            self.layers.append(nn.Linear(input_size, output_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout))
        if self.layer_norm:
            self.layers.append(nn.LayerNorm(args.head_hidden_size))
        self.layers.append(nn.Linear(args.head_hidden_size, args.tgt_vocab_size))
        self.stack_layers = nn.Sequential(*(self.layers))

    def forward(self, inputs):
        """forward"""
        head_out = self.stack_layers(inputs)
        return head_out


class DFSMNHead(nn.Module):
    '''DFSMN head'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dfsmn = DFSMNBackboneLN(args)
        self.pred_fc = torch.nn.Sequential(
            *[torch.nn.Linear(args.head_dfsmn_memory_size, args.tgt_vocab_size)]
        )

    def forward(self, inputs, input_mask):
        """forward"""
        dfsmn_out = self.dfsmn(inputs, input_mask)
        encoder_out = self.pred_fc(dfsmn_out)
        return encoder_out
