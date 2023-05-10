'''
normalization modules
'''
import copy
import torch
from torch import nn
from core.extensions import fused_layernorm as layernorm


class LayerNormNCHW(nn.Module):
    '''LayerNorm for NCHW'''

    def __init__(self, channels, eps):
        super().__init__()
        self.eps = eps
        self.gamma = nn.parameter.Parameter(torch.empty(1, channels, 1, 1))
        self.beta = nn.parameter.Parameter(torch.empty(1, channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        '''reset parameters'''
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, inputs):
        '''forward for LayerNormNCHW'''
        mean = inputs.mean(dim=1, keepdim=True)
        diff = inputs - mean
        variance = diff.square().mean(dim=1, keepdim=True)
        norm = diff * torch.rsqrt(variance + self.eps)
        return norm * self.gamma + self.beta


class LayerNorm(nn.LayerNorm):
    '''LayerNorm support fused kernel'''

    def __init__(self, normalized_shape, eps=1e-05, fused=True):
        super().__init__(normalized_shape, eps)
        self.eps = eps
        self.fused = fused

    def forward(self, inputs):
        '''forward for LayerNorm'''
        if self.fused:
            weight = self.weight.view(-1)
            bias = self.bias.view(-1)
            norm_inp = inputs.view(-1, weight.numel())
            norm_out = layernorm(norm_inp, weight, bias, self.eps)
            return torch.reshape(norm_out, inputs.shape)
        return super().forward(inputs)


class DualModeNorm(nn.Module):
    '''DualModeNorm'''

    def __init__(self, norm, dual_mode=False):
        '''init'''
        super().__init__()
        self.norm = norm
        self.dual_mode = dual_mode
        self.stream_mode = False
        if dual_mode:
            self.norm_stream = copy.deepcopy(norm)
        self._register_load_state_dict_pre_hook(self.compatible_load_hook)

    def set_stream_mode(self, stream_mode=False):
        '''set stream mode for dual mode'''
        if not self.dual_mode:
            return
        self.stream_mode = stream_mode

    def compatible_load_hook(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        '''
        compatible for load weight form normal norm.
        '''
        if '{}norm.weight'.format(prefix) in state_dict:
            # the norm in checkpoint is already DualModeNorm
            return
        old_streaming_norm_flag = False
        old_prefix = '{}_streaming'.format(prefix[:-1])
        if self.dual_mode and '{}.weight'.format(old_prefix) in state_dict:
            # load norm_stream's weight & bias from old style code
            # TODO(huanglu.thu19) delete this part
            old_streaming_norm_flag = True
            weight = state_dict.pop(old_prefix + '.weight', None)
            bias = state_dict.pop(old_prefix + '.bias', None)
            state_dict[prefix + 'norm_stream.weight'] = weight
            state_dict[prefix + 'norm_stream.bias'] = bias
            if '{}.running_var'.format(old_prefix) in state_dict:
                running_mean = state_dict.pop(old_prefix + '.running_mean', None)
                running_var = state_dict.pop(old_prefix + '.running_var', None)
                num_batches_tracked = state_dict.pop(old_prefix + '.num_batches_tracked', None)
                state_dict[prefix + 'norm_stream.running_mean'] = running_mean
                state_dict[prefix + 'norm_stream.running_var'] = running_var
                state_dict[prefix + 'norm_stream.num_batches_tracked'] = num_batches_tracked
        if '{}weight'.format(prefix) in state_dict:
            weight = state_dict.pop(prefix + 'weight')
            bias = state_dict.pop(prefix + 'bias')
            running_mean = state_dict.pop(prefix + 'running_mean', None)
            running_var = state_dict.pop(prefix + 'running_var', None)
            num_batches_tracked = state_dict.pop(prefix + 'num_batches_tracked', None)
            state_dict[prefix + 'norm.weight'] = weight
            state_dict[prefix + 'norm.bias'] = bias
            if running_mean is not None:
                state_dict[prefix + 'norm.running_mean'] = running_mean
                state_dict[prefix + 'norm.running_var'] = running_var
                state_dict[prefix + 'norm.num_batches_tracked'] = num_batches_tracked
            if self.dual_mode and not old_streaming_norm_flag:
                state_dict[prefix + 'norm_stream.weight'] = weight
                state_dict[prefix + 'norm_stream.bias'] = bias
                if running_mean is not None:
                    state_dict[prefix + 'norm_stream.running_mean'] = running_mean
                    state_dict[prefix + 'norm_stream.running_var'] = running_var
                    state_dict[prefix + 'norm_stream.num_batches_tracked'] = num_batches_tracked

    @property
    def running_mean(self):
        '''running_mean'''
        if self.dual_mode and self.stream_mode:
            return self.norm_stream.running_mean
        return self.norm.running_mean

    @property
    def running_var(self):
        '''running_var'''
        if self.dual_mode and self.stream_mode:
            return self.norm_stream.running_var
        return self.norm.running_var

    @property
    def weight(self):
        '''weight'''
        if self.dual_mode and self.stream_mode:
            return self.norm_stream.weight
        return self.norm.weight

    @property
    def bias(self):
        '''bias'''
        if self.dual_mode and self.stream_mode:
            return self.norm_stream.bias
        return self.norm.bias

    @property
    def eps(self):
        '''eps'''
        if self.dual_mode and self.stream_mode:
            return self.norm_stream.eps
        return self.norm.eps

    @property
    def momentum(self):
        '''momentum'''
        norm = self.norm_stream if self.dual_mode and self.stream_mode else self.norm
        return getattr(norm, 'momentum', 0)

    def forward(self, x):
        '''forward'''
        if self.dual_mode and self.stream_mode:
            return self.norm_stream(x)
        return self.norm(x)


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""

    def __init__(self, channel_size, lr_flag=True):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.lr_flag = lr_flag
        self.reset_parameters()
        self.eps = 1e-08

    def reset_parameters(self):
        """reset to initial mode"""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            norm_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        if self.lr_flag:
            norm_y = self.gamma * (y - mean) / torch.pow(var + self.eps, 0.5) + self.beta
        else:
            norm_y = (y - mean) / torch.pow(var + self.eps, 0.5)

        return norm_y
