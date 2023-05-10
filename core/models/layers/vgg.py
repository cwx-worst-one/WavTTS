"""
vgg models
"""

import torch
from torch import nn
import torch.nn.functional as F
from core.models.layers.cnn import SamePaddingConv2D


class PantherVarMeanFunc(torch.autograd.Function):
    '''symbolic for panther custom VarMean op'''

    @staticmethod
    def forward(ctx, x, dim, keepdim):
        '''forward'''
        return torch.var_mean(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def backward(ctx):
        '''backward'''
        raise RuntimeError('backward of VarMeanPantherFunc is not supported')

    @staticmethod
    def symbolic(g, x, dim, keepdim):
        '''symbolic'''
        return g.op('PantherVarMean', x, dim_i=dim, keepdim_i=keepdim, outputs=2)


class Vggish10(nn.Module):
    '''vggish_10 classification model'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        norm_type = 'batch_norm' if self.args.use_batch_norm else 'none'
        norm_momentum = 1.0 - self.args.batch_norm_decay
        norm_eps = self.args.batch_norm_eps
        # conv block
        conv_layers = [
            SamePaddingConv2D(
                1,
                64,
                3,
                1,
                norm_type=norm_type,
                norm_eps=norm_eps,
                norm_momentum=norm_momentum,
                act_type='relu',
            ),
            nn.MaxPool2d((2, 2)),
            SamePaddingConv2D(
                64,
                128,
                3,
                1,
                norm_type=norm_type,
                norm_eps=norm_eps,
                norm_momentum=norm_momentum,
                act_type='relu',
            ),
            nn.MaxPool2d((2, 2)),
            SamePaddingConv2D(
                128,
                256,
                3,
                1,
                norm_type=norm_type,
                norm_eps=norm_eps,
                norm_momentum=norm_momentum,
                act_type='relu',
            ),
            SamePaddingConv2D(
                256,
                256,
                3,
                1,
                norm_type=norm_type,
                norm_eps=norm_eps,
                norm_momentum=norm_momentum,
                act_type='relu',
            ),
            nn.MaxPool2d((2, 2)),
            SamePaddingConv2D(
                256,
                512,
                3,
                1,
                norm_type=norm_type,
                norm_eps=norm_eps,
                norm_momentum=norm_momentum,
                act_type='relu',
            ),
            SamePaddingConv2D(
                512,
                512,
                3,
                1,
                norm_type=norm_type,
                norm_eps=norm_eps,
                norm_momentum=norm_momentum,
                act_type='relu',
            ),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(512, 1500, (1, self.args.fbank_dim // 16)),
        ]
        if self.args.use_batch_norm:
            conv_layers.append(nn.BatchNorm2d(1500, norm_eps, norm_momentum))
        conv_layers.append(nn.ReLU())
        self.convs = nn.Sequential(*conv_layers)
        # fc block
        if self.args.use_batch_norm:
            self.fcs = nn.Sequential(
                nn.Linear(3000, self.args.feature_dim, bias=False),
                nn.BatchNorm1d(self.args.feature_dim, norm_eps, norm_momentum),
                nn.ReLU(),
                nn.Dropout(self.args.dropout_rate),
                nn.Linear(self.args.feature_dim, self.args.feature_dim, bias=False),
                nn.BatchNorm1d(self.args.feature_dim, norm_eps, norm_momentum),
                nn.ReLU(),
                nn.Dropout(self.args.dropout_rate),
                nn.Linear(self.args.feature_dim, self.args.num_classes, bias=True),
            )
        else:
            self.fcs = nn.Sequential(
                nn.Linear(3000, self.args.feature_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.args.dropout_rate),
                nn.Linear(self.args.feature_dim, self.args.feature_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(self.args.dropout_rate),
                nn.Linear(self.args.feature_dim, self.args.num_classes, bias=True),
            )
        self.reset_parameters()

    def reset_parameters(self):
        '''reset_parameters'''
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _verify_batch_size(self, inputs):
        '''verify_batch_size'''
        if not self.training or not self.args.use_batch_norm:
            return
        if inputs.shape[0] <= 1:
            raise RuntimeError(
                'Expected more than 1 samples when training with batch_norm'
                ', get input size {}'.format(inputs.shape)
            )

    def forward(self, inputs):
        '''forward for vggish_10 model'''
        self._verify_batch_size(inputs)
        conv_feat = self.convs(inputs)
        if (torch.jit.is_scripting() or torch.jit.is_tracing()) and not self.training:
            feat_var, feat_mean = PantherVarMeanFunc.apply(conv_feat, 2, False)
        else:
            feat_var, feat_mean = torch.var_mean(conv_feat, dim=2, keepdim=False)
        feat_mean = torch.flatten(feat_mean, 1).contiguous()
        feat_var = torch.flatten(feat_var, 1).contiguous()
        feat = torch.cat([feat_mean, feat_var], dim=1)
        logits = self.fcs(feat)
        predicts = F.sigmoid(logits) if self.args.multi_labels else F.softmax(logits, -1)
        return logits, predicts
