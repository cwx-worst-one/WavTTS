''' Pooling network
'''
from torch import nn
from core.models.layers.stat_pooling import StatPoolingLayer, AttentiveStatPoolingLayer


class StatPooling(nn.Module):
    '''Statistics pooling'''

    def __init__(self, args, input_dim):
        super().__init__()
        self.pooling_orders = args.pooling_orders
        self.input_dim = input_dim

        self.pooling_frames = args.get("pooling_frames", None)
        if not isinstance(self.pooling_orders, list):
            raise ValueError("The orders in stat pooling should be a list.")
        self._output_dim = self.input_dim * len(self.pooling_orders)
        self.pooling = StatPoolingLayer(
            self.pooling_orders, input_dim, pooling_frames=self.pooling_frames
        )
        self.norm = None
        if args.get('norm_after_pooling', False):
            norm_momentum = args.get("norm_momentum", 0.1)
            norm_eps = args.get("norm_eps", 1e-5)
            batchnorm_momentum = args.get("batchnorm_momentum", norm_momentum)
            batchnorm_eps = args.get("batchnorm_eps", norm_eps)
            batchnorm_affine = args.get("batchnorm_affine", True)
            if args.normalization_fn == "batch_norm":
                self.norm = nn.BatchNorm1d(
                    self.pooling.output_dim,
                    momentum=batchnorm_momentum,
                    eps=batchnorm_eps,
                    affine=batchnorm_affine,
                )
            elif args.normalization_fn == "layer_norm":
                self.norm = nn.LayerNorm(self.pooling.output_dim)
            else:
                raise NotImplementedError(
                    "Cannot find the normalization type {}".format(args.normalization_fn)
                )

    def forward(self, input_feat, mask=None):
        '''forward
        Args:
            input_feat: the input feature with shape [B, D, T] (TDNN) or [B, C, D, T] (ResNet)
            mask: the mask with shape [B, T]
        Return:
            an output embedding and a dict containing useful information
        '''
        out, embedding = self.pooling(input_feat, mask)
        if self.norm is not None:
            if self.pooling_frames is None or isinstance(self.norm, nn.BatchNorm1d):
                out = self.norm(out)
            else:
                out = self.norm(out.transpose(1, 2)).transpose(1, 2)

        return out, embedding

    @property
    def output_dim(self):
        '''Get the output dimension'''
        return self._output_dim


class AttentivePooling(nn.Module):
    '''Attentive statistics pooling'''

    def __init__(self, args, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self._output_dim = self.input_dim * 2

        activation_fn = args.get("activation_fn", "relu")
        norm_momentum = args.get("norm_momentum", 0.1)
        norm_eps = args.get("norm_eps", 1e-5)
        batchnorm_momentum = args.get("batchnorm_momentum", norm_momentum)
        batchnorm_eps = args.get("batchnorm_eps", norm_eps)
        batchnorm_affine = args.get("batchnorm_affine", True)
        normalization_fn = args.get("normalization_fn", "batch_norm")
        normalization_after = args.get("normalization_after", False)
        if args.normalization_fn == "batch_norm":
            self.norm = nn.BatchNorm1d(
                self._output_dim,
                momentum=batchnorm_momentum,
                eps=batchnorm_eps,
                affine=batchnorm_affine,
            )
        else:
            raise NotImplementedError(
                "Cannot find the normalization type {}".format(args.normalization_fn)
            )
        attentive_bottleneck_dim = args.get("attentive_bottleneck_dim", 128)
        use_global_context = args.get("use_global_context", False)
        self.pooling = AttentiveStatPoolingLayer(
            input_dim,
            attentive_bottleneck_dim,
            use_global_context,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
            normalization_after=normalization_after,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_eps=batchnorm_eps,
            batchnorm_affine=batchnorm_affine,
        )

    def forward(self, input_feat, mask=None):
        '''forward'''
        out, embedding = self.pooling(input_feat, mask)
        out = self.norm(out)
        return out, embedding

    @property
    def output_dim(self):
        '''Get the output dimension'''
        return self._output_dim
