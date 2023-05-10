'''
The frame-level network for SID/LID.
'''
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from core.models.layers.resnet_layer import BasicBlock, Bottleneck, ResBlock
from core.models.layers.res2net_layer import Bottle2neck, Res2Block
from core.models.layers.tdnn_layer import TDNN, SERes2TDNNBlock
from core.models.layers.cam import CAM


class TransposeLayer(nn.Module):
    '''Transpose the input Tensor to the desired shape.
    In Linear+BN pipeline, the output of Linear would be [B, T, D], while the BN
    needs [B, D, T] as the input shape.
    '''

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input_feat):
        '''forward'''
        return input_feat.transpose(*self.shape)


class TDNNBackbone(nn.Module):
    '''
    TDNN backbone

    This frame-level network consists of tdnn and aggregation.
    The topology for TDNN is [num_nodes, left_kernel_size, right_kernel_size[,
    dilation[, dropout]]]. The last two values are optional.

    The input of the backbone is expected to be [B, D, T], in which
      B: the batch size, (num_speakers * num_egs_per_speaker)
      D: the dimension of the feature
      T: the length of the features (time)
    The 1D-conv only supports 3-dim tensor as the input.
    '''

    def __init__(self, args):
        super().__init__()
        self.backbone_topology = eval(args.backbone_topology)
        self.backbone_layers = nn.ModuleList()
        # The names of the frame-level layers are maintained since we may need to
        # use the outputs from multiple layers.
        self.backbone_layer_num = len(self.backbone_topology)
        input_dim = args.fbank_dim
        self.norm_momentum = args.get("norm_momentum", 0.1)
        self.norm_eps = args.get("norm_eps", 1e-5)
        self.batchnorm_momentum = args.get("batchnorm_momentum", self.norm_momentum)
        self.batchnorm_eps = args.get("batchnorm_eps", self.norm_eps)
        self.batchnorm_affine = args.get("batchnorm_affine", True)
        for layer_idx in range(self.backbone_layer_num):
            topology = self.backbone_topology[layer_idx]
            dilation = 1
            dropout = 0.0
            if len(topology) == 5:
                num_nodes, left_kernel_size, right_kernel_size, dilation, dropout = topology
            elif len(topology) == 4:
                num_nodes, left_kernel_size, right_kernel_size, dilation = topology
            else:
                if len(topology) != 3:
                    raise ValueError(
                        "The topology for each layer should be: "
                        "[num_nodes, left_kernel_size, right_kernel_size"
                        "[, dilation=1[, dropout=0]]]"
                    )
                num_nodes, left_kernel_size, right_kernel_size = topology
            output_dim = num_nodes
            # TDNN + Normalization + ReLU
            self.backbone_layers.add_module(
                'frm{}'.format(layer_idx),
                TDNN(
                    input_dim,
                    output_dim,
                    left_kernel_size=left_kernel_size,
                    right_kernel_size=right_kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation_fn=args.activation_fn,
                    normalization_fn=args.normalization_fn,
                    normalization_after=args.normalization_after,
                    batchnorm_momentum=self.batchnorm_momentum,
                    batchnorm_eps=self.batchnorm_eps,
                    batchnorm_affine=self.batchnorm_affine,
                ),
            )
            input_dim = output_dim
        self.backbone_layer_names = list(self.backbone_layers._modules.keys())
        self._output_dim = output_dim

    def forward(self, input_feat, backbone_mask=None):
        '''forward'''
        # Mask should be [B, 1, T]
        if backbone_mask is not None:
            backbone_mask = backbone_mask.unsqueeze(1)
        for layer in self.backbone_layers:
            input_feat = layer(input_feat, backbone_mask)
        output = input_feat
        if backbone_mask is not None:
            backbone_mask = backbone_mask.squeeze(1)
        return output, backbone_mask

    @property
    def output_dim(self):
        '''The output dimension of this module'''
        return self._output_dim


class ECAPABackbone(nn.Module):
    '''
    ECAPA-TDNN backbone

    This frame-level network consists of tdnn, se-Res2Block and aggregation.
    The first block is a TDNN which kernel_size=5 and dilation=1.
    The option for the se-res2bock is:
    [[num_nodes], [num_nodes, kernel_size, dilation, scale] * N, [num_nodes]]
    The ECAPA-TDNN also needs:
        se_reduction_dim: the redunction dimension in the SE block.
    '''

    def __init__(self, args):
        super().__init__()
        input_dim = args.fbank_dim
        channels = eval(args.channels)
        kernel_sizes = eval(args.kernel_sizes)
        dilations = eval(args.dilations)
        activation_fn = args.get("activation_fn", "relu")
        res2net_scale = args.get("res2net_scale", 8)
        se_channels = args.get("se_channels", 128)
        norm_momentum = args.get("norm_momentum", 0.1)
        norm_eps = args.get("norm_eps", 1e-5)
        batchnorm_momentum = args.get("batchnorm_momentum", norm_momentum)
        batchnorm_eps = args.get("batchnorm_eps", norm_eps)
        batchnorm_affine = args.get("batchnorm_affine", True)
        normalization_fn = args.get("normalization_fn", "batch_norm")
        normalization_after = args.get("normalization_after", False)
        padding_mode = args.get("padding_mode", 'reflect')
        assert len(channels) == len(kernel_sizes) == len(dilations)
        self.backbone_layers = nn.ModuleList()

        self.backbone_layers.append(
            TDNN(
                input_dim,
                channels[0],
                kernel_size=kernel_sizes[0],
                dilation=dilations[0],
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
                normalization_after=normalization_after,
                batchnorm_momentum=batchnorm_momentum,
                batchnorm_eps=batchnorm_eps,
                batchnorm_affine=batchnorm_affine,
                padding_mode=padding_mode,
            ),
        )

        for i in range(1, len(channels) - 1):
            self.backbone_layers.append(
                SERes2TDNNBlock(
                    channels[i - 1],
                    channels[i],
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    scale=res2net_scale,
                    se_channels=se_channels,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                    normalization_after=normalization_after,
                    batchnorm_momentum=batchnorm_momentum,
                    batchnorm_eps=batchnorm_eps,
                    batchnorm_affine=batchnorm_affine,
                    padding_mode=padding_mode,
                    block_type=args.get("block_type", "default"),
                ),
            )

        self.mfa = TDNN(
            channels[-1],
            channels[-1],
            kernel_size=kernel_sizes[-1],
            dilation=dilations[-1],
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
            normalization_after=normalization_after,
            batchnorm_momentum=batchnorm_momentum,
            batchnorm_eps=batchnorm_eps,
            batchnorm_affine=batchnorm_affine,
            padding_mode=padding_mode,
        )
        self._output_dim = channels[-1]

    def forward(self, input_feat, backbone_mask=None):
        '''forward'''
        # Mask should be [B, 1, T]
        if backbone_mask is not None:
            backbone_mask = backbone_mask.unsqueeze(1)

        input_feat = self.backbone_layers[0](input_feat)
        xl = []
        for layer in self.backbone_layers[1:]:
            input_feat = layer(input_feat, backbone_mask)
            xl.append(input_feat)

        input_feat = torch.cat(xl, dim=1)
        output = self.mfa(input_feat)

        if backbone_mask is not None:
            backbone_mask = backbone_mask.squeeze(1)
        return output, backbone_mask

    @property
    def output_dim(self):
        '''The output dimension of this module'''
        return self._output_dim


class ResnetBackbone(nn.Module):
    '''ResNet backbone
    The ResNet is composed here because it is more convenient to use the internal
    outputs of the ResBlock, which is required in speaker recognition.
    If you want to a standard ResNet, just use the blocks in layers/resnet_layer.py
    to build your own model.

    The options of the topology for each ResNet block are:
        1. ['conv', num_planes, kernel_size, stride]
        2. ['basic_block', num_conv, num_planes, kernel_size, stride]
        3. ['bottleneck', num_conv, num_planes, kernel_size, stride]
    kernel_size and stride are a int or a list with 2 elements, for
    *feature* and *time*, respectively.
    For example, the topology could be:
    [['conv', 64, [3, 3], [1, 1]], ['basic_block', 3, 64, 3, 1]]

    Only batch_norm and ReLU are supported.
    '''

    def __init__(self, args):
        # pylint:disable=too-many-branches
        super().__init__()
        fbank_dim = args.fbank_dim
        self.norm_momentum = args.get("norm_momentum", 0.1)
        self.norm_eps = args.get("norm_eps", 1e-5)
        self.batchnorm_momentum = args.get("batchnorm_momentum", self.norm_momentum)
        self.batchnorm_eps = args.get("batchnorm_eps", self.norm_eps)
        self.batchnorm_affine = args.get("batchnorm_affine", True)
        self.groups = args.get('resnet_groups', 1)
        self.base_width = args.get('resnet_base_width', 64)
        self.resnet_zero_init_residual = args.get("resnet_zero_init_residual", False)
        if args.normalization_fn == "batch_norm":
            self._norm_layer_1d = nn.BatchNorm1d
            self._norm_layer_2d = nn.BatchNorm2d
        else:
            raise NotImplementedError("ResNet only supports BatchNorm.")
        if args.activation_fn != "relu":
            raise NotImplementedError("ResNet only supports ReLU.")

        self.backbone_topology = eval(args.backbone_topology)
        self.backbone_block_num = len(self.backbone_topology)
        self.backbone_conv_blocks = nn.ModuleList()

        # Use CAM
        self.cam_block = eval(args.get('cam_block', '[]'))
        self.cam_size = eval(args.get('cam_size', '[]'))
        if len(self.cam_block) > 0:
            self.cam_block = [(self.backbone_block_num + b if b < 0 else b) for b in self.cam_block]
        assert len(self.cam_block) == len(self.cam_size)

        self.in_planes = 1
        # save the total stride in the freq axis
        freq_stride = 1
        for block_index in range(self.backbone_block_num):
            topology = self.backbone_topology[block_index]
            if topology[0] == 'conv':
                topology = self.parse_topology(topology)
                self.backbone_conv_blocks.add_module(
                    'frm{}'.format(block_index),
                    self._make_conv(
                        topology['num_planes'], topology['kernel_size'], topology['stride']
                    ),
                )
                self.in_planes = topology['num_planes']
                freq_stride *= topology['stride'][0]
            elif topology[0] == 'basic_block':
                topology = self.parse_topology(topology)
                self.backbone_conv_blocks.add_module(
                    'frm{}'.format(block_index),
                    ResBlock(
                        BasicBlock,
                        topology['num_conv'],
                        self.in_planes,
                        topology['num_planes'],
                        topology['kernel_size'],
                        stride=topology['stride'],
                        norm_layer=self._norm_layer_2d,
                        batchnorm_momentum=self.batchnorm_momentum,
                        batchnorm_eps=self.batchnorm_eps,
                        batchnorm_affine=self.batchnorm_affine,
                        groups=self.groups,
                        base_width=self.base_width,
                    ),
                )
                self.in_planes = topology['num_planes'] * BasicBlock.expansion
                freq_stride *= topology['stride'][0]
            elif topology[0] == 'bottleneck':
                topology = self.parse_topology(topology)
                self.backbone_conv_blocks.add_module(
                    'frm{}'.format(block_index),
                    ResBlock(
                        Bottleneck,
                        topology['num_conv'],
                        self.in_planes,
                        topology['num_planes'],
                        topology['kernel_size'],
                        stride=topology['stride'],
                        norm_layer=self._norm_layer_2d,
                        batchnorm_momentum=self.batchnorm_momentum,
                        batchnorm_eps=self.batchnorm_eps,
                        batchnorm_affine=self.batchnorm_affine,
                        groups=self.groups,
                        base_width=self.base_width,
                    ),
                )
                self.in_planes = topology['num_planes'] * Bottleneck.expansion
                freq_stride *= topology['stride'][0]
            else:
                raise NotImplementedError("Unknown block name {}".format(topology[0]))

            if block_index in self.cam_block:
                cam_size = self.cam_size[self.cam_block.index(block_index)]
                self.backbone_conv_blocks.add_module(
                    f'cam{block_index}',
                    CAM(
                        [self.in_planes, int(fbank_dim // freq_stride)],
                        (self.in_planes if cam_size == -1 else cam_size),
                    ),
                )

        self.backbone_conv_names = list(self.backbone_conv_blocks._modules.keys())

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if self.resnet_zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        # The dimension is reduce by factor total_stride
        # TODO: the dimension may be wrong if the feature dim cannot be
        # divided by 8.
        self._output_dim = int(fbank_dim // freq_stride * self.in_planes)

    @staticmethod
    def expand_param(x):
        '''expand parameters when it is a single value'''
        if isinstance(x, list):
            return x
        return [x, x]

    def parse_topology(self, topology):
        '''parse topology'''
        if topology[0] == 'conv':
            _, num_planes, kernel_size, stride = topology
            attrib = {
                'num_planes': num_planes,
                'kernel_size': self.expand_param(kernel_size),
                'stride': self.expand_param(stride),
            }
        elif topology[0] == 'basic_block':
            _, num_conv, num_planes, kernel_size, stride = topology
            attrib = {
                'num_conv': num_conv,
                'num_planes': num_planes,
                'kernel_size': self.expand_param(kernel_size),
                'stride': self.expand_param(stride),
            }
        elif topology[0] == 'bottleneck':
            _, num_conv, num_planes, kernel_size, stride = topology
            attrib = {
                'num_conv': num_conv,
                'num_planes': num_planes,
                'kernel_size': self.expand_param(kernel_size),
                'stride': self.expand_param(stride),
            }
        else:
            raise NotImplementedError("Unknown block name {}".format(topology[0]))
        return attrib

    def _make_conv(self, planes, kernel_size, stride):
        '''Make conv layer'''
        models = nn.Sequential(
            OrderedDict(
                [
                    (
                        'conv',
                        nn.Conv2d(
                            self.in_planes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=[i // 2 for i in kernel_size],
                            bias=False,
                        ),
                    ),
                    (
                        'norm',
                        self._norm_layer_2d(
                            planes,
                            momentum=self.batchnorm_momentum,
                            eps=self.batchnorm_eps,
                            affine=self.batchnorm_affine,
                        ),
                    ),
                    ('nonlinear', nn.ReLU(inplace=True)),
                ]
            )
        )
        return models

    def forward(self, input_feat, backbone_mask=None):
        '''forward
        Args:
            input_feat: The feature is with [B, D, T]
            mask: The mask is [B, T]
        '''
        input_feat = input_feat.unsqueeze(1)
        if backbone_mask is not None:
            # and the mask should be [B, 1, 1, T]
            backbone_mask = backbone_mask.unsqueeze(1).unsqueeze(1)
        for block in self.backbone_conv_blocks:
            if isinstance(block, (ResBlock, CAM)):
                input_feat, backbone_mask = block(input_feat, backbone_mask)
            else:
                stride = block[0].stride[1]
                if stride > 1:
                    dilation = block[0].dilation[1]
                    kernel_size = block[0].kernel_size[1]
                    if backbone_mask is not None:
                        backbone_mask = F.pad(
                            backbone_mask,
                            (0, 2 * dilation * (kernel_size // 2), 0, 0),
                            mode='replicate',
                        )
                        backbone_mask = F.unfold(
                            backbone_mask, (1, kernel_size), dilation, 0, stride=(1, stride)
                        )
                        backbone_mask = backbone_mask[:, :1, :].unsqueeze(2)
                input_feat = block(input_feat)
                if backbone_mask is not None:
                    input_feat = input_feat * backbone_mask
        if backbone_mask is not None:
            # mask is tranposed to [B, 1, T]
            backbone_mask = backbone_mask.squeeze(1).squeeze(1)
        output = input_feat
        return output, backbone_mask

    @property
    def output_dim(self):
        '''The output dimension of this module'''
        return self._output_dim


class Res2netBackbone(nn.Module):
    '''Res2Net backbone
    The Res2Net shares almost the same configurations with conventional ResNet,
    while it add an additional hyper-parameters 'scale'.

    The options of the topology for each Res2Net block are:
        1. ['conv', num_planes, kernel_size, stride]
        2. ['bottle2neck', num_conv, num_planes, kernel_size, stride]
    kernel_size and stride are a int or a list with 2 elements, for
    *feature* and *time*, respectively.
    For example, the topology could be:
    [['conv', 64, [3, 3], [1, 1]], ['basic_block', 3, 64, 3, 1]]
    '''

    def __init__(self, args):
        # pylint: disable=too-many-branches
        super().__init__()
        fbank_dim = args.fbank_dim
        self.norm_momentum = args.get("norm_momentum", 0.1)
        self.norm_eps = args.get("norm_eps", 1e-5)
        self.batchnorm_momentum = args.get("batchnorm_momentum", self.norm_momentum)
        self.batchnorm_eps = args.get("batchnorm_eps", self.norm_eps)
        self.batchnorm_affine = args.get("batchnorm_affine", True)
        self.base_width = args.get('resnet_base_width', 26)
        self.scale = args.get('resnet_scale', 4)
        self.resnet_zero_init_residual = args.get("resnet_zero_init_residual", False)
        if args.normalization_fn == "batch_norm":
            self._norm_layer_1d = nn.BatchNorm1d
            self._norm_layer_2d = nn.BatchNorm2d
        else:
            raise NotImplementedError("ResNet only supports BatchNorm.")
        if args.activation_fn != "relu":
            raise NotImplementedError("ResNet only supports ReLU.")

        self.backbone_topology = eval(args.backbone_topology)
        self.backbone_block_num = len(self.backbone_topology)
        self.backbone_conv_blocks = nn.ModuleList()

        self.in_planes = 1
        freq_stride = 1
        for block_index in range(self.backbone_block_num):
            topology = self.backbone_topology[block_index]
            if topology[0] == 'conv':
                topology = self.parse_topology(topology)
                self.backbone_conv_blocks.add_module(
                    'frm{}'.format(block_index),
                    self._make_conv(
                        topology['num_planes'], topology['kernel_size'], topology['stride']
                    ),
                )
                self.in_planes = topology['num_planes']
                freq_stride *= topology['stride'][0]
            elif topology[0] == 'bottle2neck':
                topology = self.parse_topology(topology)
                self.backbone_conv_blocks.add_module(
                    'frm{}'.format(block_index),
                    Res2Block(
                        topology['num_conv'],
                        self.in_planes,
                        topology['num_planes'],
                        topology['kernel_size'],
                        stride=topology['stride'],
                        batchnorm_momentum=self.batchnorm_momentum,
                        batchnorm_eps=self.batchnorm_eps,
                        batchnorm_affine=self.batchnorm_affine,
                        base_width=self.base_width,
                        scale=self.scale,
                    ),
                )
                self.in_planes = topology['num_planes'] * Bottle2neck.expansion
                freq_stride *= topology['stride'][0]
            else:
                raise NotImplementedError("Unknown block name {}".format(topology[0]))

        self.backbone_conv_names = list(self.backbone_conv_blocks._modules.keys())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.resnet_zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottle2neck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

        # The dimension is reduce by factor total_stride
        # TODO: the dimension may be wrong if the feature dim cannot be
        # divided by 8.
        self._output_dim = int(fbank_dim // freq_stride * self.in_planes)

    @staticmethod
    def expand_param(x):
        '''expand parameters when it is a single value'''
        if isinstance(x, list):
            return x
        return [x, x]

    def parse_topology(self, topology):
        '''parse topology'''
        if topology[0] == 'conv':
            _, num_planes, kernel_size, stride = topology
            attrib = {
                'num_planes': num_planes,
                'kernel_size': self.expand_param(kernel_size),
                'stride': self.expand_param(stride),
            }
        elif topology[0] == 'bottle2neck':
            _, num_conv, num_planes, kernel_size, stride = topology
            attrib = {
                'num_conv': num_conv,
                'num_planes': num_planes,
                'kernel_size': self.expand_param(kernel_size),
                'stride': self.expand_param(stride),
            }
        else:
            raise NotImplementedError("Unknown block name {}".format(topology[0]))
        return attrib

    def _make_conv(self, planes, kernel_size, stride):
        '''Make conv layer'''
        models = nn.Sequential(
            OrderedDict(
                [
                    (
                        'conv',
                        nn.Conv2d(
                            self.in_planes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=[i // 2 for i in kernel_size],
                            bias=False,
                        ),
                    ),
                    (
                        'norm',
                        nn.BatchNorm2d(
                            planes,
                            momentum=self.batchnorm_momentum,
                            eps=self.batchnorm_eps,
                            affine=self.batchnorm_affine,
                        ),
                    ),
                    ('nonlinear', nn.ReLU(inplace=True)),
                ]
            )
        )
        return models

    def forward(self, input_feat, backbone_mask=None):
        '''forward'''
        # pylint: disable=bare-except,too-many-branches
        # Transpose the input feature from [B, T, D] to [B, 1, D, T].
        input_feat = input_feat.unsqueeze(1)
        if backbone_mask is not None:
            # and the mask should be [B, 1, 1, T]
            backbone_mask = backbone_mask.unsqueeze(1).unsqueeze(1)
        for block in self.backbone_conv_blocks:
            if isinstance(block, Res2Block):
                input_feat, backbone_mask = block(input_feat, backbone_mask)
            else:
                stride = block[0].stride[1]
                if stride > 1:
                    dilation = block[0].dilation[1]
                    kernel_size = block[0].kernel_size[1]
                    if backbone_mask is not None:
                        backbone_mask = F.pad(
                            backbone_mask,
                            (0, 2 * dilation * (kernel_size // 2), 0, 0),
                            mode='replicate',
                        )
                        backbone_mask = F.unfold(
                            backbone_mask, (1, kernel_size), dilation, 0, stride=(1, stride)
                        )
                        backbone_mask = backbone_mask[:, :1, :].unsqueeze(2)
                input_feat = block(input_feat)
                if backbone_mask is not None:
                    input_feat = input_feat * backbone_mask
        if backbone_mask is not None:
            # mask is tranposed to [B, 1, T]
            backbone_mask = backbone_mask.squeeze(1).squeeze(1)
        output = input_feat
        return output, backbone_mask

    @property
    def output_dim(self):
        '''The output dimension of this module'''
        return self._output_dim
