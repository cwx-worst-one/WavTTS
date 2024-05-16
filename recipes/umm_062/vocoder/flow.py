import torch

from torch import nn
from torch.nn.utils import weight_norm


@torch.jit.script
def fused_tanh_sigmoid(in_act, n_channels):
    n_channels_int = n_channels[0]
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(nn.Module):

    def __init__(self,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0):
        super().__init__()
        assert (kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(hidden_channels,
                                 2 * hidden_channels,
                                 kernel_size,
                                 dilation=dilation,
                                 padding=padding)
            in_layer = weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x):
        output = 0
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            # fused kernel
            n_channels = torch.IntTensor([self.hidden_channels])
            acts = fused_tanh_sigmoid(x_in, n_channels)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output


class Flip(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ResidualCouplingLayer(nn.Module):

    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0,
                 mean_only=True):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels,
                      kernel_size,
                      dilation_rate,
                      n_layers,
                      p_dropout=p_dropout)
        self.post = nn.Conv1d(hidden_channels,
                              self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.normal_(0, 0.001)
        self.post.bias.data.normal_(0, 0.001)

    def forward(self, x, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.enc(h)
        stats = self.post(h)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs)
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlock(nn.Module):

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size=5,
        dilation_rate=1,
        n_layers=4,
        n_flows=4,
        affine=False,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.affine = affine

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels,
                                      hidden_channels,
                                      kernel_size,
                                      dilation_rate,
                                      n_layers,
                                      mean_only=not affine))
            self.flows.append(Flip())

    def forward(self, x, reverse=False):
        logdet_tot = 0.0
        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, reverse=reverse)
                logdet_tot += logdet
        else:
            for flow in reversed(self.flows):
                x = flow(x, reverse=reverse)
        return x, logdet_tot
