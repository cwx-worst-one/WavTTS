''' Statistics pooling
'''
import torch
from torch import nn
import torch.nn.functional as F
from core.models.layers.tdnn_layer import TDNN

VAR2STD_EPSILON = 1e-12


class StatPoolingLayer(nn.Module):
    '''implementation of statistics pooling'''

    def __init__(self, pooling_orders, input_dim, pooling_frames=None):
        '''This is also competible with average(mean) and stdddev pooling when
        different orders of statistics are given.
        When the shape of the input feature is [B, D, T], the input_dim should be D,
        while the shape is [B, C, D, T], the input_dim should be C*D.'''
        super().__init__()
        self.pooling_orders = pooling_orders
        self.input_dim = input_dim
        self.pooling_frames = pooling_frames
        self._output_dim = input_dim * len(self.pooling_orders)

    def forward(self, input_feat, mask=None):
        '''forward

        Args:
            input_feat: the input feature with shape [B, D, T] or [B, C, D, T]
            mask: the mask with shape [B, T]
        Return:
            The pooling result, while the shape is determined by pooling_frames
            If pooling_frames is None: [B, D]
            If pooling_frames is not None (a int): [B, T, D]
        '''
        embedding = {}
        if self.pooling_frames is None:
            if mask is not None:
                if input_feat.dim() == 4:
                    mask = mask.unsqueeze(1).unsqueeze(1)
                else:
                    mask = mask.unsqueeze(1)
                input_feat = input_feat * mask
                cnt = mask.sum(-1).float()
                mean = input_feat.sum(-1)
                mean = mean / cnt
                var = (((input_feat - mean.unsqueeze(-1)) * mask) ** 2).sum(-1)
                var = var / cnt
            else:
                mean = torch.mean(input_feat, dim=-1)
                meansq = torch.mean(input_feat * input_feat, dim=-1)
                # pooling_std = torch.sqrt(meansq - mean ** 2 + 1e-10)
                var = meansq - mean**2
            stddev = var.clamp(min=VAR2STD_EPSILON).sqrt()
            if input_feat.dim() == 4:
                mean = torch.flatten(mean, start_dim=1)
                stddev = torch.flatten(stddev, start_dim=1)
        else:
            bs, _, _, ts = input_feat.shape
            if input_feat.dim() == 4:
                input_feat = input_feat.view(bs, -1, ts)
            # Padding before averaging
            input_feat = F.pad(
                input_feat,
                ((self.pooling_frames - 1) // 2, self.pooling_frames // 2),
                mode="reflect",
            )

            mean = F.avg_pool1d(input_feat, kernel_size=self.pooling_frames, stride=1)
            var = (
                F.avg_pool1d(input_feat**2, kernel_size=self.pooling_frames, stride=1) - mean**2
            )
            stddev = var.clamp(min=VAR2STD_EPSILON).sqrt()

            if mask is not None:
                mean = mask.unsqueeze(1) * mean
                stddev = mask.unsqueeze(1) * stddev

        output = []
        if 1 in self.pooling_orders:
            output.append(mean)
            embedding['pooling_mean'] = mean
        if 2 in self.pooling_orders:
            output.append(stddev)
            embedding['pooling_stddev'] = stddev
        embedding['pooling'] = torch.cat(output, 1)
        return embedding['pooling'], embedding

    @property
    def output_dim(self):
        '''The output dimension of this module'''
        return self._output_dim


class AttentiveStatPoolingLayer(nn.Module):
    '''implementation of channel-dependent statistics pooling'''

    def __init__(
        self,
        input_dim,
        bottleneck_dim,
        use_global_context,
        activation_fn='relu',
        normalization_fn='batch_norm',
        normalization_after=False,
        batchnorm_momentum=0.1,
        batchnorm_eps=1e-5,
        batchnorm_affine=True,
    ):

        '''initialization'''
        super().__init__()
        expand_input_dim = input_dim * 3 if use_global_context else input_dim
        self.tdnn = TDNN(
            expand_input_dim,
            bottleneck_dim,
            kernel_size=1,
            dilation=1,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
            normalization_after=normalization_after,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_eps=batchnorm_eps,
            batchnorm_affine=batchnorm_affine,
        )
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(bottleneck_dim, input_dim, kernel_size=1)
        self.use_global_context = use_global_context

        if self.use_global_context:
            self.stat_pool = StatPoolingLayer([1, 2], input_dim)

    def forward(self, input_feat, mask=None):
        '''forward'''
        embedding = {}
        if input_feat.dim() == 4:
            raise NotImplementedError("Not implemented for input_shape=4.")
        length = input_feat.shape[-1]

        if self.use_global_context:
            pool = self.stat_pool(input_feat, mask)[0]
            attn = torch.cat([input_feat, pool.unsqueeze_(2).repeat(1, 1, length)], dim=1)
        else:
            attn = input_feat

        attn = self.conv(self.tanh(self.tdnn(attn)))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn.masked_fill_(mask < 0.5, float('-inf'))
        alpha = F.softmax(attn, dim=2)
        mean = torch.sum(alpha * input_feat, dim=2)
        stddev = torch.sqrt(
            (alpha * (input_feat - mean.unsqueeze(2)).pow(2)).sum(2).clamp(VAR2STD_EPSILON)
        )
        output = []
        output.append(mean)
        embedding['pooling_mean'] = mean
        output.append(stddev)
        embedding['pooling_stddev'] = stddev
        embedding['pooling'] = torch.cat(output, 1)
        return embedding['pooling'], embedding

    @property
    def output_dim(self):
        '''The output dimension of this module'''
        return self._output_dim


class StreamStatPoolingLayer(StatPoolingLayer):
    '''implementation of stream statistics pooling'''

    def __init__(self, pooling_orders, input_dim):
        '''This is also competible with average(mean) and stdddev pooling when
        different orders of statistics are given.
        When the shape of the input feature is [B, D, T], the input_dim should be D,
        while the shape is [B, C, D, T], the input_dim should be C*D.'''
        super().__init__(pooling_orders, input_dim)

    def forward(self, input_feat, mask=None):
        '''forward

        Args:
            input_feat: the input feature with shape [B, D, T]
            mask: the mask with shape [B, T]
        Return:
            The pooling result, B, D, T
        '''
        embedding = {}
        if mask is not None:
            mask = mask.unsqueeze(1)
            input_feat = input_feat * mask
            cnt = torch.cumsum(mask, -1).float()
            mean = torch.cumsum(input_feat, -1)
            mean = mean / cnt
            var = torch.cumsum((input_feat - mean * mask) ** 2, -1)
            var = var / cnt
        else:
            bs, _, step = input_feat.shape
            mask = torch.ones(bs, 1, step)
            cnt = torch.cumsum(mask, -1).float()
            mean = torch.cumsum(input_feat, -1)
            mean = mean / cnt
            var = torch.cumsum((input_feat - mean) ** 2, -1)
            var = var / cnt
        stddev = var.clamp(min=VAR2STD_EPSILON).sqrt()

        output = []
        if 1 in self.pooling_orders:
            output.append(mean)
            embedding['pooling_mean'] = mean
        if 2 in self.pooling_orders:
            output.append(stddev)
            embedding['pooling_stddev'] = stddev
        embedding['pooling'] = torch.cat(output, 1)
        return embedding['pooling'], embedding
