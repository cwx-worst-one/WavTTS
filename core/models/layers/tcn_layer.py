''' tcn layer
'''
import math
import torch
from torch import nn
import torch.nn.functional as F

# pylint: disable='unused-argument'
class TcnAttBlock(nn.Module):
    """Tcn block"""

    def __init__(
        self,
        in_ch,
        mid_ch,
        out_ch,
        kernel_size=3,
        padding=1,
        dilation=0,
        dropout=0.1,
        residual=False,
        causal=True,
        faam=False,
    ):
        """
        TCN block
        reference paper: https://arxiv.org/pdf/1809.07454.pdf
        """
        super().__init__()
        self.casual = causal
        self.faame = faam
        if self.casual:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = (kernel_size - 1) // 2 * dilation
        self.dilation = dilation
        self.conv1 = nn.Sequential(nn.Conv1d(in_ch, mid_ch, 1), nn.BatchNorm1d(mid_ch), nn.PReLU())
        self.conv2 = nn.Conv1d(
            mid_ch, mid_ch, kernel_size=kernel_size, dilation=self.dilation, groups=mid_ch
        )
        self.norm2 = nn.BatchNorm1d(mid_ch)
        self.act2 = nn.PReLU()

        # self.att = MHAttentionEM(featdim=mid_ch, num_heads=16,left_size=4,win_len=kernel_size)
        self.conv3 = nn.Conv1d(mid_ch, out_ch, 1)
        self.norm3 = nn.BatchNorm1d(out_ch)
        self.act3 = nn.PReLU()
        if self.faame:
            self.faam = FAAM(in_dim=out_ch, ratio=4, causal=self.casual)

        if in_ch == out_ch:
            self.dropout = True
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False
        self.residual = residual

    def forward(self, x, res=None):
        """
        Args:
            x [B,C,T]
        Returns:
            x [B,C,T]
        """
        if res is not None:
            residual = res
        else:
            residual = x

        x = self.conv1(x)
        # x_att = self.att(x)
        if self.casual:
            x1 = F.pad(x, [self.padding, 0], 'constant', 0.0)
        else:
            x1 = F.pad(x, [self.padding, self.padding], 'constant', 0.0)
        x1 = self.conv2(x1)  # B F T
        # x_cat = torch.cat((x1,x_att),dim=1)
        x_cat = x1
        x_cat = self.norm2(x_cat)
        x_cat = self.act2(x_cat)
        x = self.conv3(x_cat)
        x = self.act3(self.norm3(x))
        if self.faame:
            x = self.faam(x)
        if self.dropout:
            x = self.dropout(x)
        if self.residual:
            x += residual
        return x


# pylint: disable='redefined-builtin'
class FAAM(nn.Module):
    """
    frequency dimension adaptive attention module
    refernece: https://drive.google.com/file/d/18kFNW2oPFe5hhpdQ2AwNWnBMgPl47jCF/view
    modified by zhangpeng
    """

    def __init__(self, in_dim, ratio, causal=True):
        super().__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.causal = causal
        self.hidden_dim = self.in_dim // self.ratio
        self.fc1 = torch.nn.Conv1d(self.in_dim, self.hidden_dim, 1)
        self.fc2 = torch.nn.Conv1d(self.hidden_dim, self.in_dim, 1)

    def reset_parameters(self):
        """reset paramters"""
        nn.init.kaiming_uniform_(self.atten_matrix, a=math.sqrt(5))

    def forward(self, x):
        """forward"""
        input = x
        # if not self.causal:
        #     mean = torch.mean(input, dim=-1, keepdim=True)
        # else:
        time = input.shape[-1]
        mean = torch.cumsum(input, dim=-1) / (torch.arange(1, time + 1, 1.0).to(input.device))
        w1 = torch.relu(self.fc1(mean))
        w2 = torch.sigmoid(self.fc2(w1))
        x = w2 * x
        return x
