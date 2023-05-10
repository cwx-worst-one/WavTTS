''' TDNN/FTDNN layer
'''
import torch
from torch import nn
import torch.nn.functional as F
from .se_layer import SE1d


class TDNN(nn.Module):
    '''
    TDNN layer consists of Conv1D+Norm+ReLU.
    It is implemented by a single Conv1D layer plus the normalization
    and nonlinear layers.
    The input is expected to be [B, D, T], in which:
      B is the batch size,
      D is the dimension,
      T is the time.
    This is due to the convolution in pytorch, which requires the channel first format.
    '''

    def __init__(
        self,
        input_dim,
        output_dim,
        left_kernel_size=None,
        right_kernel_size=None,
        kernel_size=None,
        dilation=1,
        dropout=0.0,
        activation_fn='relu',
        normalization_fn='batch_norm',
        normalization_after=False,
        batchnorm_momentum=0.1,
        batchnorm_eps=1e-5,
        batchnorm_affine=True,
        padding_mode='constant',
    ):
        super().__init__()
        if activation_fn != "relu":
            raise ValueError("Unsupported activation function {}".format(activation_fn))

        if kernel_size is None:
            self.left_kernel_size = left_kernel_size
            self.right_kernel_size = right_kernel_size
        else:
            if left_kernel_size is not None or right_kernel_size is not None:
                raise ValueError("When kernel_size is given, left/right_kernel_size should be None")
            self.left_kernel_size = kernel_size // 2
            self.right_kernel_size = kernel_size // 2
        self.dilation = dilation
        self.dropout = dropout
        self.normalization_after = normalization_after
        self.padding_mode = padding_mode
        self.internal_layers = nn.ModuleList()

        if self.left_kernel_size == self.right_kernel_size:
            # set the padding in the conv op
            conv_padding_mode = {"constant": "zeros", "reflect": "reflect"}
            self.internal_layers.add_module(
                'conv',
                nn.Conv1d(
                    input_dim,
                    output_dim,
                    kernel_size=self.left_kernel_size + self.right_kernel_size + 1,
                    stride=1,
                    padding=self.left_kernel_size * self.dilation,
                    padding_mode=conv_padding_mode[padding_mode],
                    dilation=dilation,
                ),
            )
        else:
            # For unbalanced TDNN (left_kernel_size != right_kernel_size),
            # padding is applied in the runtime.
            self.internal_layers.add_module(
                'conv',
                nn.Conv1d(
                    input_dim,
                    output_dim,
                    kernel_size=self.left_kernel_size + self.right_kernel_size + 1,
                    stride=1,
                    padding=0,
                    dilation=dilation,
                ),
            )

        if normalization_fn == "batch_norm":
            norm_layer = nn.BatchNorm1d(
                output_dim, momentum=batchnorm_momentum, eps=batchnorm_eps, affine=batchnorm_affine
            )
            self.normalization_type = "batch_norm"
        elif normalization_fn == "layer_norm":
            norm_layer = nn.LayerNorm(output_dim)
            self.normalization_type = "layer_norm"
        else:
            raise NotImplementedError(
                "Cannot find the normalization type {}".format(normalization_fn)
            )
        if self.normalization_after:
            self.internal_layers.add_module('nonlinear', nn.ReLU(inplace=True))
            self.internal_layers.add_module('norm', norm_layer)
        else:
            self.internal_layers.add_module('norm', norm_layer)
            self.internal_layers.add_module('nonlinear', nn.ReLU(inplace=True))
        self.dropout = None
        if dropout > 0:
            self.internal_layers.add_module('dropout', nn.Dropout(dropout, inplace=True))

    def forward(self, input_feat, mask=None):
        '''
        Args:
            input_feat: the input feature with shape [B, D, T]
        '''
        if self.left_kernel_size != self.right_kernel_size:
            # Do the padding separately only when the left and right context are unbalanced.
            input_feat = F.pad(
                input_feat,
                (self.left_kernel_size * self.dilation, self.right_kernel_size * self.dilation),
                mode=self.padding_mode,
            )
        for layer in self.internal_layers:
            if isinstance(layer, nn.LayerNorm):
                input_feat = layer(input_feat.transpose(1, 2)).transpose(1, 2)
            else:
                input_feat = layer(input_feat)
        if mask is not None:
            input_feat = input_feat * mask
        return input_feat


class Res2TDNN(nn.Module):
    '''Bottleneck1d for Res2Net'''

    def __init__(
        self,
        inplanes,
        outplanes,
        kernel_size,
        dilation,
        scale,
        activation_fn='relu',
        normalization_fn='batch_norm',
        normalization_after=False,
        batchnorm_momentum=0.1,
        batchnorm_eps=1e-5,
        batchnorm_affine=True,
        padding_mode='constant',
        block_type='default',
    ):
        super().__init__()
        assert inplanes % scale == 0
        assert outplanes % scale == 0
        in_channel = inplanes // scale
        hidden_channel = outplanes // scale
        self.scale = scale
        self.block_type = block_type
        assert self.block_type in ("default", "full"), "block_type should be default or full."

        self.blocks = nn.ModuleList(
            [
                TDNN(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                    normalization_after=normalization_after,
                    batchnorm_momentum=batchnorm_momentum,
                    batchnorm_eps=batchnorm_eps,
                    batchnorm_affine=batchnorm_affine,
                    padding_mode=padding_mode,
                )
                for i in range(scale - 1 if block_type == "default" else scale)
            ]
        )

    def forward(self, input_feat, mask=None):
        '''forward'''
        out = []
        for i, x_i in enumerate(torch.chunk(input_feat, self.scale, dim=1)):
            if self.block_type == "default":
                if i == 0:
                    y_i = x_i
                elif i == 1:
                    y_i = self.blocks[i - 1](x_i)
                else:
                    y_i = self.blocks[i - 1](x_i + y_i)
            else:
                if i == 0:
                    y_i = self.blocks[i](x_i)
                else:
                    y_i = self.blocks[i](x_i + y_i)
            out.append(y_i)
        out = torch.cat(out, dim=1)
        if mask is not None:
            out = out * mask
        return out


class SERes2TDNNBlock(nn.Module):
    '''The SE-Res2Block'''

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        dilation,
        scale,
        se_channels,
        activation_fn='relu',
        normalization_fn='batch_norm',
        normalization_after=False,
        batchnorm_momentum=0.1,
        batchnorm_eps=1e-5,
        batchnorm_affine=True,
        padding_mode='constant',
        block_type="default",
    ):
        '''initialize a SE-Res2Block-1D'''
        super().__init__()
        self.conv1 = TDNN(
            input_dim,
            output_dim,
            kernel_size=1,
            dilation=1,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
            normalization_after=normalization_after,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_eps=batchnorm_eps,
            batchnorm_affine=batchnorm_affine,
            padding_mode=padding_mode,
        )
        self.res2block = Res2TDNN(
            output_dim,
            output_dim,
            kernel_size,
            dilation,
            scale,
            normalization_fn=normalization_fn,
            normalization_after=normalization_after,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_eps=batchnorm_eps,
            batchnorm_affine=batchnorm_affine,
            activation_fn=activation_fn,
            padding_mode=padding_mode,
            block_type=block_type,
        )
        self.conv2 = TDNN(
            output_dim,
            output_dim,
            kernel_size=1,
            dilation=1,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
            normalization_after=normalization_after,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_eps=batchnorm_eps,
            batchnorm_affine=batchnorm_affine,
            padding_mode=padding_mode,
        )
        self.se1d = SE1d(output_dim, se_channels)
        self.shortcut = None
        if input_dim != output_dim:
            self.shortcut = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self._output_dim = output_dim

    def forward(self, input_feat, mask):
        '''forward'''
        residual = input_feat
        if self.shortcut:
            residual = self.shortcut(input_feat)
        out = self.conv1(input_feat)
        out = self.res2block(out)
        out = self.conv2(out)
        out = self.se1d(out, mask)
        return out + residual

    @property
    def output_dim(self):
        '''The output dimension of this module'''
        return self._output_dim
