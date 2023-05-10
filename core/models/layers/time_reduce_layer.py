''' time_reduce_layer '''
from torch import nn


class TimeReduce(nn.Module):
    '''TimeReduce Layer'''

    def __init__(self, n_concat, input_size, n_downsample=0, padding=-1):
        super().__init__()
        self.n_concat = n_concat
        self.input_size = input_size
        if n_downsample == 0:
            self.n_downsample = self.n_concat
        else:
            self.n_downsample = n_downsample
        if padding == -1:
            padding = self.n_concat // 2
        self.unfold = nn.Unfold(
            (self.n_concat, 1), stride=(self.n_downsample, 1), padding=(padding, 0)
        )

    def forward(self, input_data):
        """forward"""
        # input (B, T, ndim)
        bsz = input_data.size(0)
        unfold_data = (
            (self.unfold(input_data.unsqueeze(1)))
            .contiguous()
            .view(bsz, self.n_concat, -1, self.input_size)
        )
        unfold_data = (
            unfold_data.transpose(1, 2).contiguous().view(bsz, -1, self.n_concat * self.input_size)
        )
        # output (B, T/x, x*ndim)
        return unfold_data


class LinearTimeReduce(TimeReduce):
    '''LinearTimeReduce'''

    def __init__(self, n_concat, input_size, n_downsample=0, padding=-1):
        super().__init__(n_concat, input_size, n_downsample, padding)
        self.linear = nn.Linear(2 * input_size, input_size)

    def forward(self, input_data):
        # input (B, T, ndim)
        bsz = input_data.size(0)
        unfold_data = (
            (self.unfold(input_data.unsqueeze(1)))
            .contiguous()
            .view(bsz, self.n_concat, -1, self.input_size)
        )
        unfold_data = (
            unfold_data.transpose(1, 2).contiguous().view(bsz, -1, self.n_concat * self.input_size)
        )
        return self.linear(unfold_data)


class MaxPoolTimeReduce(nn.Module):
    '''MaxPoolTimeReduce'''

    def __init__(self, kernel_size=2, stride=2, padding=0, ceil_mode=False):
        super().__init__()
        if padding == -1:
            padding = self.stride // 2
        self.pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    def forward(self, input_data, input_mask=None):
        """forward"""
        # input (B, T, ndim)
        if input_mask is not None:
            input_data.masked_fill_(~input_mask.bool().unsqueeze(-1), float('-inf'))
            output_mask = self.pool(input_mask.unsqueeze(1)).squeeze(1)
        output = self.pool(input_data.transpose(1, 2).contiguous())
        if input_mask is not None:
            return output.transpose(1, 2).contiguous(), output_mask
        return output.transpose(1, 2).contiguous()


class AvgPoolTimeReduce(MaxPoolTimeReduce):
    '''AvgPoolTimeReduce'''

    def __init__(self, kernel_size=2, stride=2, padding=0, ceil_mode=False):
        super().__init__()
        if padding == -1:
            padding = self.stride // 2
        self.pool = nn.AvgPool1d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)


class Conv1dTimeReduce(nn.Module):
    '''Conv1dTimeReduce'''

    def __init__(self, input_size, kernel_size=2, stride=2, padding=0):
        """__init__"""
        super().__init__()
        self.conv = nn.Conv1d(input_size, input_size, kernel_size, stride=stride, padding=padding)

    def forward(self, input_data):
        """forward"""
        output = self.conv(input_data.transpose(1, 2).contiguous())
        return output.transpose(1, 2).contiguous()
