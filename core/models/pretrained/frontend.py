""" frontend """

from torch import nn
from core.utils import get_rank, get_communicator
from core.extensions import fused_conv_feature_extraction


class TransposeLast(nn.Module):
    """TransposeLast"""

    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        """forward"""
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


def make_conv(conv_type, n_in, n_out, k, stride, conv_bias):
    '''make conv'''
    if conv_type == 'default':
        conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
        nn.init.kaiming_normal_(conv.weight)
    elif conv_type == 'SeparableConv1D':
        conv = nn.Sequential(
            nn.Conv1d(n_in, n_out, k, stride=stride, groups=n_in, bias=False),
            nn.Conv1d(n_out, n_out, 1, bias=conv_bias),
        )
    else:
        raise NotImplementedError
    return conv


def nnblock(
    n_in,
    dim,
    k,
    stride,
    norm_type,
    conv_bias=False,
    conv_type="default",
    dropout=0.0,
):
    '''nn block'''
    conv = make_conv(conv_type, n_in, dim, k, stride, conv_bias)
    norm = None
    dropout = nn.Dropout(p=dropout)
    if norm_type == 'layer_norm':
        norm = nn.Sequential(
            TransposeLast(), nn.LayerNorm(dim, elementwise_affine=True), TransposeLast()
        )
    if norm_type == 'group_norm':
        norm = nn.GroupNorm(dim, dim, affine=True)
    gelu = nn.GELU()
    if norm is None:
        return nn.Sequential(conv, dropout, gelu)
    return nn.Sequential(conv, dropout, norm, gelu)


class ConvFeatureExtractionModel(nn.Module):
    '''ConvFeatureExtractionModel'''

    def __init__(
        self,
        conv_layers,
        dropout=0.0,
        mode="default",
        conv_bias=False,
        conv_type="default",
        recompute=False,
        dp_parallel_size=0,
    ):
        super().__init__()
        assert mode in {"none", "default", "layer_norm"}
        in_d = 1
        self.conv_layers = nn.ModuleList()
        strides = []
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl
            strides.append(stride)
            norm_type = None
            if mode == 'layer_norm':
                norm_type = 'layer_norm'
            if mode == 'default' and i == 0:
                norm_type = 'group_norm'
            self.conv_layers.append(
                nnblock(
                    in_d,
                    dim,
                    k,
                    stride,
                    norm_type,
                    conv_bias=conv_bias,
                    conv_type=conv_type,
                    dropout=dropout,
                )
            )
            in_d = dim
        self._cfg = dict(
            norm_type=mode,
            conv_type=conv_type,
            dropout=dropout,
            conv_bias=conv_bias,
            recompute=recompute,
            dp_parallel=dp_parallel_size > 1,
            local_size=dp_parallel_size,
            communicator=get_communicator(dp_parallel_size),
            eps=1e-5,
            strides=strides,
        )
        if dp_parallel_size > 1:
            self._cfg['local_rank'] = get_rank() % dp_parallel_size
        self.fusable = conv_type == 'default' and mode == 'layer_norm'

    def forward(self, x, fused=True):
        """forward"""
        if self.fusable and fused:
            ws = list(self.parameters())
            return fused_conv_feature_extraction(ws, x, training=self.training, **self._cfg)
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x
