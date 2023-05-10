''' pooling layer '''
import torch
from torch import nn


class MaxPoolLayer(nn.Module):
    '''MaxPoolLayer'''

    # pylint: disable=no-self-use
    def forward(self, input_data, input_mask=None):
        """forward"""
        # input (B, T, ndim), output (B, ndim)
        if input_mask is not None:
            fill_mask = (1 - input_mask).bool().unsqueeze(2)
            input_data = input_data.masked_fill(fill_mask, float("-inf"))
        output = torch.max(input_data, dim=1)
        return output[0]


class AvgPoolLayer(nn.Module):
    '''AvgPoolLayer'''

    # pylint: disable=no-self-use
    def forward(self, input_data, input_mask=None):
        """forward"""
        # input (B, T, ndim), output (B, ndim)
        if input_mask is not None:
            output = (input_data * input_mask.unsqueeze(2)).sum(dim=1)
            output = output / input_mask.sum(dim=1).unsqueeze(1)
        else:
            output = output.mean(dim=1)
        return output


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        input:
            x : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (B, T, 1)

        return:
            utter_rep: size (B, H)
        """
        softmax = nn.functional.softmax
        # att_w = softmax(self.W(x).squeeze(-1)).unsqueeze(-1)
        att_w = softmax(self.linear(x), dim=1)

        utter_rep = torch.sum(x * att_w, dim=1)

        return utter_rep
