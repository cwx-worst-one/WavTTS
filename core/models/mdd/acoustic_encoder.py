'''
The conv, multihead-attention, LSTM and self-attention pooling network for mdd.
'''
import torch
from torch import nn

from core.models.layers.cnn import ConvNorm
from core.models.layers.time_reduce_layer import Conv1dTimeReduce
from core.models.layers.pooling import SelfAttentionPooling
from core.models.layers.multi_head_attn import MultiheadAttention


class UttCMLPEncoder(nn.Module):
    '''
    Utterance level Acoustic encoder
    CNN + MultiheadAttention + LSTM + TimeReduceCNN + Pooling
    '''

    def __init__(self, args):
        super().__init__()

        self.args = args

        if args.cnn_layers:
            convolutions = []
            padding1 = args.kernel_size1 // 2 - (1 - args.kernel_size1 % 2)
            for i in range(3):
                conv_layer = nn.Sequential(
                    ConvNorm(
                        args.input_dim if i == 0 else args.hidden_dim,
                        args.hidden_dim,
                        kernel_size=args.kernel_size1,
                        stride=args.stride1,
                        padding=padding1,
                        dilation=1,
                        w_init_gain='relu',
                    )
                )
                convolutions.append(conv_layer)
            self.convolutions = nn.ModuleList(convolutions)

        if args.multiheadaAttention_layer:
            self.multihead_attantion = MultiheadAttention(args.hidden_dim, args.num_heads)

        if args.lstm_layer:
            self.lstm = nn.LSTM(
                args.hidden_dim,
                args.hidden_dim,
                args.num_lstm_hidden,
                batch_first=True,
                bidirectional=False,
            )

        if args.time_reduce_cnn:
            time_reduce_layers = []
            padding2 = args.kernel_size2 // 2 - (1 - args.kernel_size2 % 2)
            for i in range(3):
                time_reduce_layer = nn.Sequential(
                    Conv1dTimeReduce(
                        args.hidden_dim,
                        kernel_size=args.kernel_size2,
                        stride=args.stride2,
                        padding=padding2,
                    )
                )
                time_reduce_layers.append(time_reduce_layer)

        if args.selfattentionpooling:
            self.selfattention_pooling = SelfAttentionPooling(args.hidden_dim)

        if args.utt_feat_dim > 0:
            self.utt_linear = nn.Sequential(
                nn.Linear(args.hidden_dim + args.utt_feat_dim, args.hidden_dim), nn.ReLU()
            )

        self.linear_out = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.Sigmoid())

    def forward(self, x, utt_feat=None, mask=None):
        """forward"""
        if self.args.cnn_layers:
            for conv in self.convolutions:
                x = conv(x)

        if self.args.multiheadaAttention_layer:
            pre_cnn_out = x.transpose(0, 1)
            mha_put = self.multihead_attantion(pre_cnn_out, key_padding_mask=mask)
            x = mha_put.transpose(0, 1)

        if self.args.lstm_layer:
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)

        if self.args.time_reduce_cnn:
            for time_reduce_layer in self.time_reduce_layers:
                x = time_reduce_layer(x)

        if self.args.selfattentionpooling:
            pooling_out = self.selfattention_pooling(x)
        else:
            pooling_out = torch.mean(x, 1)

        pooling_out = pooling_out.squeeze(1)

        if self.args.utt_feat_dim > 0 and utt_feat is not None:
            pooling_out = torch.cat((pooling_out, utt_feat), dim=-1)
            pooling_out = self.utt_linear(pooling_out)

        out = self.linear_out(pooling_out)

        return out


class FrameCMLEncoder(nn.Module):
    '''
    Frame level Acoustic encoder
    CNN + MultiheadAttention + LSTM
    '''

    def __init__(self, args):
        super().__init__()

        self.args = args
        convolutions = []
        padding = args.kernel_size // 2 - (1 - args.kernel_size % 2)
        for i in range(args.num_cnn):
            conv_layer = nn.Sequential(
                ConvNorm(
                    args.input_dim if i == 0 else args.hidden_dim,
                    args.hidden_dim,
                    kernel_size=args.kernel_size,
                    stride=args.stride,
                    padding=padding,
                    dilation=1,
                    w_init_gain='relu',
                )
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        if args.multiheadaAttention_layer:
            self.multihead_attantion = MultiheadAttention(args.hidden_dim, args.num_heads)

        self.lstm = nn.LSTM(
            args.hidden_dim,
            args.hidden_dim,
            args.num_lstm_hidden,
            batch_first=True,
            bidirectional=args.bidirectional,
        )

        if args.bidirectional:
            self.linear = nn.Sequential(nn.Linear(args.hidden_dim * 2, args.hidden_dim), nn.ReLU())
        else:
            self.linear = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU())

    def forward(self, x, mask=None):
        """forward"""
        for conv in self.convolutions:
            x = conv(x)

        if self.args.multiheadaAttention_layer:
            pre_cnn_out = x.transpose(0, 1)

            mha_put = self.multihead_attantion(pre_cnn_out, key_padding_mask=mask)
            x = mha_put.transpose(0, 1)

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)

        out = self.linear(lstm_out)

        return out
