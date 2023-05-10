'''CAM layer.
'''
from torch import nn
from .stat_pooling import StatPoolingLayer


class CAM(nn.Module):
    '''Yu Y Q, Zheng S, Suo H, et al. Cam: Context-Aware Masking for Robust
       Speaker Verification. ICASSP 2021.
    An in-place speech enhanced layer for speaker verificaiton.
    '''

    def __init__(
        self,
        input_dim,
        hidden_size,
        batchnorm_momentum=0.1,
        batchnorm_eps=1e-5,
        batchnorm_affine=True,
        activation_fn='relu',
    ):
        '''input_dim is D, or [C, D]'''
        super().__init__()
        if activation_fn != "relu":
            raise ValueError("Unsupported activation function {}".format(activation_fn))
        self.input_dim = input_dim

        if isinstance(input_dim, (tuple, list)):
            # 4-d input [B, C, D, T]
            self.main_fc = nn.Sequential(
                *[
                    nn.Conv2d(input_dim[0], input_dim[0], kernel_size=1),
                    nn.BatchNorm2d(
                        input_dim[0],
                        momentum=batchnorm_momentum,
                        eps=batchnorm_eps,
                        affine=batchnorm_affine,
                    ),
                    nn.ReLU(),
                ]
            )

            self.content_pre_fc = nn.Conv2d(input_dim[0], hidden_size, kernel_size=1, bias=False)
            pool_dim = input_dim[0] * input_dim[1]
            self.aux_pool = StatPoolingLayer([1, 2], pool_dim)
            self.aux_fc = nn.Conv2d(2 * input_dim[0], hidden_size, kernel_size=1)
            self.content_net = nn.Sequential(
                *[
                    nn.BatchNorm2d(
                        hidden_size,
                        momentum=batchnorm_momentum,
                        eps=batchnorm_eps,
                        affine=batchnorm_affine,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(hidden_size, input_dim[0], kernel_size=1),
                    nn.Sigmoid(),
                ]
            )
        else:
            # 3-d input [B, D, T]
            self.main_fc = nn.Sequential(
                *[
                    nn.Conv1d(input_dim, input_dim, kernel_size=1),
                    nn.BatchNorm1d(
                        input_dim,
                        momentum=batchnorm_momentum,
                        eps=batchnorm_eps,
                        affine=batchnorm_affine,
                    ),
                    nn.ReLU(),
                ]
            )

            self.content_pre_fc = nn.Conv1d(input_dim, hidden_size, kernel_size=1, bias=False)
            self.aux_pool = StatPoolingLayer([1, 2], input_dim)
            self.aux_fc = nn.Conv1d(2 * input_dim, hidden_size, kernel_size=1)
            self.content_net = nn.Sequential(
                *[
                    nn.BatchNorm1d(
                        hidden_size,
                        momentum=batchnorm_momentum,
                        eps=batchnorm_eps,
                        affine=batchnorm_affine,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(hidden_size, input_dim, kernel_size=1),
                    nn.Sigmoid(),
                ]
            )

    def forward(self, input_feat, mask=None):
        '''forward'''
        out = self.main_fc(input_feat)
        content_pre_out = self.content_pre_fc(input_feat)

        if len(input_feat.shape) == 4:
            pool_out, _ = self.aux_pool(
                input_feat, mask.squeeze(1).squeeze(1) if mask is not None else None
            )
            pool_out = pool_out.view(-1, 2 * self.input_dim[0], self.input_dim[1], 1)
        else:
            pool_out, _ = self.aux_pool(input_feat, mask.squeeze(1) if mask is not None else None)
            pool_out = pool_out.unsqueeze(-1)
        aux_out = self.aux_fc(pool_out)

        content_out = self.content_net(content_pre_out + aux_out)
        out = out * content_out

        if mask is not None:
            out = out * mask
        return out, mask
