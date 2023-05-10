'''Squeeze and Excitation'''
from torch import nn


class SE1d(nn.Module):
    '''Squeeze and excitation block for 1-D feature'''

    def __init__(self, channel, reduction_dim):
        '''initialize a SE block given the reduction dim (rather than reduction factor).
        The shapes of the input and the output are [B, C, T].'''
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channel, reduction_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduction_dim, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, input_feat, mask=None):
        '''forward'''
        if mask is not None:
            total = mask.sum(dim=2, keepdims=True)
            s = (input_feat * mask).sum(dim=2, keepdims=True) / total
        else:
            s = input_feat.mean(dim=2, keepdims=True)
        out = self.layers(s) * input_feat
        return out
