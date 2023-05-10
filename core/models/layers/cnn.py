''' CNN '''
import torch
from torch import nn
import torch.nn.functional as F
from core.models.layers.active_function import get_activation_fn
from core.models.layers.normalization import LayerNormNCHW, DualModeNorm
from core.extensions import conv1d


class Conv1d(nn.Conv1d):
    '''conv1d with fused conv mma.'''

    def forward(self, x):
        y = conv1d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride[0],
            padding=self.padding[0],
            dilation=self.dilation[0],
            groups=self.groups,
        )
        return y


class ConvNorm(nn.Module):
    '''1D convolution + BatchNorm + dropout'''

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain='linear',
        batch_norm=True,
        activation='relu',
        dropout=0,
    ):
        super().__init__()
        if padding is None:
            assert (kernel_size % 2) == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.conv_bn = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation_fn = get_activation_fn(activation=activation)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

        self.batch_norm = batch_norm

    def forward(self, x):
        """forward"""
        x = x.transpose(1, 2).contiguous()

        out = self.conv(x)
        if self.batch_norm:
            out = self.conv_bn(out)

        out = self.dropout(self.activation_fn(out))

        return out.transpose(1, 2).contiguous()


class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed
    to replace positionwise feed-forward network
    in Transformer block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate, activation_fn):
        """Initialize MultiLayeredConv1d module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(__class__, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans,
            in_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = self.activation_fn(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class Conv1dLinear(torch.nn.Module):
    """Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate, activation_fn):
        """Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(__class__, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., in_chans).

        Returns:
            Tensor: Batch of output tensors (B, ..., hidden_chans).

        """
        x = self.activation_fn(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x))


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    :param int channels: channels of cnn
    :param int kernel_size: kernerl size of cnn

    """

    def __init__(
        self,
        channels,
        kernel_size,
        activation_fn,
        norm_type='batch_norm',
        dual_mode=False,
        bias=True,
        dropout_rate=0.0,
        use_layer_norm_before=False,
        layer_norm_eps=1e-6,
        use_residual=False,
    ):
        """Construct an ConvolutionModule object."""
        super(__class__, self).__init__()
        # kernel_size should be a odd number for 'SAME' padding
        kernel_size = eval(kernel_size)
        if isinstance(kernel_size, int):
            assert (kernel_size - 1) % 2 == 0
            self.left_kernel_size = self.right_kernel_size = (kernel_size - 1) // 2
        else:
            self.left_kernel_size, self.right_kernel_size = kernel_size

        self.dropout_rate = dropout_rate
        if self.dropout_rate != 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.use_layer_norm_before = use_layer_norm_before
        if self.use_layer_norm_before:
            self.layer_norm_before = nn.LayerNorm(channels, eps=layer_norm_eps)
        self.use_residual = use_residual

        # change Pointwise Conv1D to Linear to support slim compression (SVD)
        self.pointwise_conv1 = nn.Linear(channels, 2 * channels, bias=bias)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            self.left_kernel_size + self.right_kernel_size + 1,
            stride=1,
            padding=0,
            groups=channels,
            bias=bias,
        )
        self.norm_type = norm_type
        if norm_type == "batch_norm":
            self.norm = nn.BatchNorm1d(channels)
        elif norm_type == "layer_norm":
            self.norm = nn.LayerNorm(channels)
        elif norm_type == "group_norm":
            # set num_groups to 1 for GroupNorm
            self.norm = nn.GroupNorm(1, channels)
        elif norm_type == 'none':
            self.norm = None
        else:
            raise ValueError("unknow norm_type for ConvolutionModule " + norm_type)
        self.dual_mode = dual_mode
        if self.norm is not None:
            self.norm = DualModeNorm(self.norm, self.dual_mode)
        self.stream_mode = False
        # change Pointwise Conv1D to Linear to support slim compression (SVD)
        self.pointwise_conv2 = nn.Linear(channels, channels, bias=bias)
        self.channels = channels
        self.activation = get_activation_fn(activation_fn)
        self._register_load_state_dict_pre_hook(self.compatible_load_hook)

    def compatible_load_hook(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        error_msgs,
    ):
        '''
        compatible for load ConvolutionModule.
        '''
        conv1_weight_name = prefix + 'pointwise_conv1.weight'
        conv2_weight_name = prefix + 'pointwise_conv2.weight'

        if conv1_weight_name in state_dict and conv2_weight_name in state_dict:
            conv1_weight = state_dict[conv1_weight_name]
            conv2_weight = state_dict[conv2_weight_name]
            assert conv1_weight.shape[0] == 2 * self.channels
            assert conv1_weight.shape[1] == self.channels
            assert conv2_weight.shape[0] == self.channels
            assert conv2_weight.shape[1] == self.channels
            if len(conv1_weight.shape) == 3:
                if conv1_weight.shape[2] != 1:
                    error_msgs.append(
                        'last dim of pointwise_conv1.weight must be 1, but get: {}'.format(
                            conv1_weight.shape[2]
                        )
                    )
                else:
                    state_dict[conv1_weight_name] = conv1_weight.squeeze(2)

            if len(conv2_weight.shape) == 3:
                if conv2_weight.shape[2] != 1:
                    error_msgs.append(
                        'last dim of pointwise_conv2.weight must be 1, but get: {}'.format(
                            conv2_weight.shape[2]
                        )
                    )
                else:
                    state_dict[conv2_weight_name] = conv2_weight.squeeze(2)

    def set_stream_mode(self, stream_mode=False):
        '''set stream mode for dual mode'''
        if not self.dual_mode:
            return
        self.stream_mode = stream_mode
        if self.norm is not None:
            self.norm.set_stream_mode(stream_mode)

    def apply_norm(self, x, mask=None):
        '''apply_norm
        x: (batch, channel, dim) or (batch, channel, seq_len)
        '''
        if mask is not None:
            x = x * mask
        if self.norm_type in ("batch_norm", "group_norm"):
            x = self.norm(x)
        elif self.norm_type == 'layer_norm':
            x = self.norm(x.transpose(1, 2))
            x = x.transpose(1, 2)
        if mask is not None:
            x = x * mask
        return x

    def apply_depthwise_conv(self, x):
        '''apply_depthwise_conv'''
        if self.dual_mode and self.stream_mode:
            weight = self.depthwise_conv.weight[:, :, : self.left_kernel_size + 1]
            # (input, weight, bias, stride, padding, dilation, groups)
            x = F.pad(x, (self.left_kernel_size, 0))
            x = F.conv1d(x, weight, self.depthwise_conv.bias, 1, 0, 1, self.channels)
        else:
            x = F.pad(x, (self.left_kernel_size, self.right_kernel_size))
            x = self.depthwise_conv(x)
        return x

    def forward(self, x, conv_memory=None, mask=None):
        """Compute convolution module.

        :param torch.Tensor x: (batch, time, size)
        :return torch.Tensor: convoluted `value` (batch, time, d_model)
        """
        if conv_memory is not None:
            x = torch.cat([conv_memory, x], dim=1)

        x_org = x
        if self.use_layer_norm_before:
            x = self.layer_norm_before(x)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, time, 2*channel)
        x = F.glu(x, dim=2)  # (batch, time, channel)

        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (batch, channel, time)
        # 1D Depthwise Conv
        if mask is not None:
            x = x * mask
        x = self.apply_depthwise_conv(x)
        x = self.activation(self.apply_norm(x, mask=mask))
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (batch, time, channel)
        x = self.pointwise_conv2(x)

        if self.dropout_rate != 0.0:
            x = self.dropout(x)
        if self.use_residual:
            x = x + x_org

        if conv_memory is not None:
            updated_conv_memory = x_org[:, -self.left_kernel_size :]
            x = x[:, self.left_kernel_size :]
            return x, updated_conv_memory

        return x

    def forward_step(self, x, cache, total_right_context, mask=None):
        """Compute convolution module.

        :param torch.Tensor x: (batch, time, size)
        :return torch.Tensor: convoluted `value` (batch, time, d_model)
        """
        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, time, 2*channel)
        x = nn.functional.glu(x, dim=2)  # (batch, time, channel)

        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)
        if mask is not None:
            x = x * mask
        # 1D Depthwise Conv
        assert self.right_kernel_size == 0  # don't support non causal conv
        # if cache is None:
        #     x = F.pad(x, (self.left_kernel_size, self.right_kernel_size))
        # else:
        assert cache.size(0) == x.size(0)
        assert cache.size(1) == self.left_kernel_size
        assert cache.size(2) == x.size(1)
        cache = cache.transpose(1, 2)
        x = torch.cat((cache, x), dim=2)
        # assert x.size(2) > self.left_kernel_size

        if total_right_context > 0:
            cache = x[:, :, -self.left_kernel_size - total_right_context : -total_right_context]
        else:
            cache = x[:, :, -self.left_kernel_size :]
        x = self.depthwise_conv(x)

        x = self.activation(self.apply_norm(x))

        if mask is not None:
            x = x * mask

        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        cache = cache.transpose(1, 2)
        return x, cache

    @property
    def depthwise_conv_weight(self):
        """get depthwise conv weight"""
        if self.dual_mode and self.stream_mode:
            return self.depthwise_conv.weight[:, :, : self.left_kernel_size + 1]
        return self.depthwise_conv.weight

    @property
    def depthwise_conv_bias(self):
        """get depthwise conv bias"""
        return self.depthwise_conv.bias

    @property
    def depthwise_conv_left_kernel_size(self):
        """get depthwise conv left kernel size"""
        return self.left_kernel_size

    # pylint: disable='invalid-name'
    @property
    def depthwise_conv_right_kernel_size(self):
        """get depthwise conv right kernel size"""
        if self.dual_mode and self.stream_mode:
            return 0
        return self.right_kernel_size


class Conv1DBlockLD(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self, in_channels=256, conv_channels=512, kernel_size=3, dilation=1, causal=True):
        super().__init__()
        # 1x1 conv
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = nn.BatchNorm1d(conv_channels)
        self.causal = causal

        dconv_pad = (
            (dilation * (kernel_size - 1)) // 2
            if not self.causal
            else (dilation * (kernel_size - 1))
        )

        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.lnorm2 = nn.BatchNorm1d(conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.dconv_pad = dconv_pad

    def forward(self, x):
        '''Compute 1D convolutional block module.
        Input: (B, F, T)
        Output: (B, F, T)
        '''
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, : -self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class Conv1DBlockLDP(nn.Module):
    """
    1D convolutional projection block:
        DConv - PReLU - Norm - SConv - PReLU - Norm
    """

    def __init__(self, in_channels=256, out_channels=512, kernel_size=3, dilation=1, causal=True):
        super().__init__()
        # 1x1 conv
        self.causal = causal
        dconv_pad = (
            (dilation * (kernel_size - 1)) // 2
            if not self.causal
            else (dilation * (kernel_size - 1))
        )

        # depthwise conv
        self.dconv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True,
        )
        self.prelu1 = nn.PReLU()
        self.lnorm1 = nn.BatchNorm1d(in_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(in_channels, out_channels, 1, bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = nn.BatchNorm1d(out_channels)
        # different padding way
        self.dconv_pad = dconv_pad

    def forward(self, x):
        '''Compute 1D convolutional projection block module.
        Input: (B, F, T)
        Output: (B, H, T)
        '''
        y = self.dconv(x)
        if self.causal:
            y = y[:, :, : -self.dconv_pad]
        y = self.lnorm1(self.prelu1(y))
        y = self.sconv(y)
        y = self.lnorm2(self.prelu2(y))
        return y


class SamePaddingConv2DFunc(torch.autograd.Function):
    '''onnx conv2d with SAME_UPPER auto_pad'''

    @staticmethod
    def _padding_for_same(in_size, kernel, stride, dilation):
        '''get padding'''
        out_size = (in_size + stride - 1) // stride
        pad_size = (out_size - 1) * stride + (kernel - 1) * dilation + 1 - in_size
        pad_before = pad_size // 2
        pad_after = pad_size - pad_before
        return pad_before, pad_after

    @staticmethod
    def pad_input(x, kernels, strides, dilations):
        '''pad conv input'''
        in_shape = x.shape
        pad_t, pad_b = SamePaddingConv2DFunc._padding_for_same(
            in_shape[2], kernels[0], strides[0], dilations[0]
        )
        pad_l, pad_r = SamePaddingConv2DFunc._padding_for_same(
            in_shape[3], kernels[1], strides[1], dilations[1]
        )
        return F.pad(x, [pad_l, pad_r, pad_t, pad_b])

    @staticmethod
    def forward(ctx, x, w, b, kernels, strides, dilations):
        '''forward'''
        padded = SamePaddingConv2DFunc.pad_input(x, kernels, strides, dilations)
        return F.conv2d(padded, w, b, stride=strides, dilation=dilations)

    @staticmethod
    def backward(ctx):
        '''backward'''
        raise RuntimeError('backward of SamePaddingConv2DFunc is not supported')

    @staticmethod
    def symbolic(g, x, w, b, kernels, strides, dilations):
        '''symbolic'''
        inputs = [x, w] if b is None else [x, w, b]
        return g.op(
            'Conv',
            *inputs,
            kernel_shape_i=kernels,
            strides_i=strides,
            dilations_i=dilations,
            auto_pad_s='SAME_UPPER',
        )


class SamePaddingConv2D(nn.Module):
    '''Conv2d with SAME padding mode'''

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel,
        stride,
        dilation=1,
        norm_type=None,
        norm_eps=1e-6,
        act_type=None,
        dropout_rate=0.0,
        norm_momentum=0.1,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel,
                stride,
                dilation=dilation,
                bias=(norm_type != 'batch_norm'),
            )
        ]
        # norm layer
        if norm_type is None or norm_type == 'none':
            pass
        elif norm_type == 'batch_norm':
            layers.append(nn.BatchNorm2d(out_channel, norm_eps, norm_momentum))
        elif norm_type == "layer_correct":
            layers.append(LayerNormNCHW(out_channel, norm_eps))
        else:
            raise NotImplementedError("unsupport norm type: {}".format(norm_type))
        # activation layer
        if act_type is None or act_type == 'none':
            pass
        elif act_type == 'relu':
            layers.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError("unsupport act type: {}".format(act_type))
        # dropout layer
        if dropout_rate != 0.0:
            layers.append(nn.Dropout(dropout_rate))

        self.fused_conv = nn.Sequential(*layers)
        # sizes of tuple
        self.kernel = kernel if not isinstance(kernel, int) else (kernel, kernel)
        self.stride = stride if not isinstance(stride, int) else (stride, stride)
        self.dilation = dilation if not isinstance(dilation, int) else (dilation, dilation)

    def forward(self, inputs):
        '''forward for SamePaddingConv2D'''
        if (torch.jit.is_scripting() or torch.jit.is_tracing()) and not self.training:
            out = SamePaddingConv2DFunc.apply(
                inputs,
                self.fused_conv[0].weight,
                self.fused_conv[0].bias,
                self.kernel,
                self.stride,
                self.dilation,
            )
            for m in self.fused_conv[1:]:
                out = m(out)
            return out

        padded = SamePaddingConv2DFunc.pad_input(inputs, self.kernel, self.stride, self.dilation)
        return self.fused_conv(padded)


class MultiplicativeUnit(nn.Module):
    '''Multiplicative Unit (Mu) layer'''

    def __init__(self, in_channel, out_channel):
        '''init'''
        super().__init__()
        self.mu_conv = SamePaddingConv2D(
            in_channel, 4 * out_channel, 3, 1, norm_type='layer_correct'
        )
        self.channel = out_channel

    def forward(self, inputs):
        '''forward for MultiplicativeUnit'''
        gates = self.mu_conv(inputs)
        g = gates.split(self.channel, dim=1)
        new_cell = g[1].sigmoid() * inputs + g[2].sigmoid() * g[3].tanh()
        return g[0].sigmoid() * new_cell.tanh()
