'''
The segment-level network for SID/LID.
'''
import torch
from torch import nn
from core.models.layers.large_margin_softmax import (
    LargeMarginSoftmax,
    CurricularFace,
    ClassificationCircleLoss,
)


class SegmentMLP(nn.Module):
    '''The segment-level network simply consists of several MLPs.'''

    def __init__(self, args, input_dim):
        # pylint:disable=too-many-branches
        super().__init__()
        # The segment topology only contains the num of nodes in each layer.
        self.norm_momentum = args.get("norm_momentum", 0.1)
        self.norm_eps = args.get("norm_eps", 1e-5)
        self.batchnorm_momentum = args.get("batchnorm_momentum", self.norm_momentum)
        self.batchnorm_eps = args.get("batchnorm_eps", self.norm_eps)
        self.batchnorm_affine = args.get("batchnorm_affine", True)
        self.layernorm_eps = args.get("layernorm_eps", self.norm_eps)
        self.use_conv = args.get("segment_use_conv", False)
        self.segment_topology = eval(args.segment_topology)
        self.segment_layers = nn.ModuleList()
        for layer_idx, _ in enumerate(self.segment_topology):
            # The topology consists of num_nodes.
            output_dim = self.segment_topology[layer_idx]
            if self.use_conv:
                self.segment_layers.add_module(
                    "utt{}_affine".format(layer_idx),
                    nn.Conv1d(input_dim, output_dim, kernel_size=1),
                )
            else:
                self.segment_layers.add_module(
                    "utt{}_affine".format(layer_idx), nn.Linear(input_dim, output_dim)
                )

            norm_layer = None
            if not args.get('is_export_onnx', False):
                # When exporting onnx, the batchnorm is fused with the linear layer (if exists).
                #
                # The last fully-connected in the segment-level network should be
                # handled more carefully. We may not use normalization and nonlinear
                # layers after the linear layer.
                # Reference:
                #   https://arxiv.org/abs/1704.08063
                if layer_idx < len(self.segment_topology) - 1 or args.segment_last_layer_use_norm:
                    if args.normalization_fn == "batch_norm":
                        norm_layer = nn.BatchNorm1d(
                            output_dim,
                            momentum=self.batchnorm_momentum,
                            eps=self.batchnorm_eps,
                            affine=self.batchnorm_affine,
                        )
                    elif args.normalization_fn == "layer_norm":
                        # norm_layer = nn.LayerNorm(output_dim, eps=self.layernorm_eps)
                        raise NotImplementedError("Layer_norm is not supported.")
                    else:
                        raise NotImplementedError(
                            "Cannot find the normalization type {}".format(args.normalization_fn)
                        )
                    norm_name = "utt{}_norm".format(layer_idx)

            if args.normalization_after:
                if (
                    layer_idx < len(self.segment_topology) - 1
                    or args.segment_last_layer_use_nonlinear
                ):
                    self.segment_layers.add_module(
                        "utt{}_nonlinear".format(layer_idx), nn.ReLU(inplace=True)
                    )

                if norm_layer is not None:
                    self.segment_layers.add_module(norm_name, norm_layer)
            else:
                if norm_layer is not None:
                    self.segment_layers.add_module(norm_name, norm_layer)
                if (
                    layer_idx < len(self.segment_topology) - 1
                    or args.segment_last_layer_use_nonlinear
                ):
                    self.segment_layers.add_module(
                        "utt{}_nonlinear".format(layer_idx), nn.ReLU(inplace=True)
                    )
            input_dim = output_dim
        self.segment_layer_names = list(self.segment_layers._modules.keys())
        self._output_dim = output_dim

        if args.get('is_export_onnx', False):
            # Fuse linear and batchnorm when exporting onnx.
            self._register_load_state_dict_pre_hook(self.fused_load_hook)

    def forward(self, input_feat):
        '''forward
        The input feature can be two types: 1. 2D feature ([B, D]) or 2. 3D feature ([B, D, T])
        '''
        embedding = {}
        for layer_idx, layer in enumerate(self.segment_layers):
            if input_feat.dim() == 2:
                # When the rank of the tensor is 2([B, D]), the batch_norm and layer_norm share
                # the same arg list.
                if isinstance(layer, nn.Conv1d):
                    input_feat = layer(input_feat.unsqueeze(2)).squeeze(2)
                else:
                    input_feat = layer(input_feat)
            else:
                if isinstance(layer, (nn.Linear, nn.LayerNorm)):
                    input_feat = layer(input_feat.transpose(1, 2)).transpose(1, 2)
                else:
                    input_feat = layer(input_feat)

            embedding[self.segment_layer_names[layer_idx]] = input_feat
        return input_feat, embedding

    @property
    def output_dim(self):
        '''Get the output dimension'''
        return self._output_dim

    def fused_load_hook(
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
        When exporting the onnx, the linear weights are fused with the batchnorm stat.
        '''
        index = 0
        while True:
            linear_weight_name = prefix + f'segment_layers.utt{index}_affine.weight'
            linear_bias_name = prefix + f'segment_layers.utt{index}_affine.bias'
            if linear_weight_name not in state_dict:
                if index == 0:
                    error_msgs.append("No linear layer found in the segment-level network.")
                break

            linear_weight = state_dict[linear_weight_name]
            linear_bias = state_dict[linear_bias_name] if linear_bias_name in state_dict else 0

            norm_mean_name = prefix + f'segment_layers.utt{index}_norm.running_mean'
            norm_mean = state_dict.pop(norm_mean_name, None)
            if norm_mean is not None:
                norm_var = state_dict.pop(prefix + f'segment_layers.utt{index}_norm.running_var')
                norm_weight = state_dict.pop(prefix + f'segment_layers.utt{index}_norm.weight', 1)
                norm_bias = state_dict.pop(prefix + f'segment_layers.utt{index}_norm.bias', 0)
                state_dict.pop(prefix + f'segment_layers.utt{index}_norm.num_batches_tracked')

                # The batchnorm_eps is set in the configuration.
                weight_adj = norm_weight / torch.sqrt(norm_var + self.batchnorm_eps)
                weight_new = linear_weight * weight_adj.unsqueeze(1)
                bias_new = linear_bias * weight_adj - norm_mean * weight_adj + norm_bias

                state_dict[linear_weight_name] = weight_new
                state_dict[linear_bias_name] = bias_new
            index += 1


class MLPSoftmax(nn.Module):
    '''
    MLP (segment-level network) + Softmax
    '''

    def __init__(self, args, input_dim):
        super().__init__()
        self.softmax_type = args.softmax_type
        self.mlp = SegmentMLP(args, input_dim)

        # In some cases, the tgt_vocab_size is not given. So use a default value
        # as a placeholder.
        tgt_vocab_size = args.get('tgt_vocab_size', 1)
        if args.softmax_type in [
            "angular_softmax",
            "additive_margin_softmax",
            "additive_angular_margin_softmax",
        ]:
            self.affine_softmax = LargeMarginSoftmax(args, self.mlp.output_dim, tgt_vocab_size)
        elif args.softmax_type == "curricular_face":
            self.affine_softmax = CurricularFace(args, self.mlp.output_dim, tgt_vocab_size)
        elif args.softmax_type == "class_circle_loss":
            self.affine_softmax = ClassificationCircleLoss(
                args, self.mlp.output_dim, tgt_vocab_size
            )
        elif args.softmax_type == "softmax":
            # When vanilla softmax is applied, a simple affine layer is used.
            self.affine_softmax = nn.Linear(self.mlp.output_dim, tgt_vocab_size)
        else:
            raise NotImplementedError("Unknown softmax type {}".format(args.softmax_type))

    def forward(self, input_feat, label=None, step=0, export_onnx=False):
        '''forward
        Args:
            input_feat: the input feature with shape [B, D]
        This returns the logits of the softmax layer. It also returns the embedding
        we need.
        '''
        input_feat, embedding = self.mlp(input_feat)
        embedding['utt_output'] = input_feat

        # Compute the logits
        if self.softmax_type in [
            "angular_softmax",
            "additive_margin_softmax",
            "additive_angular_margin_softmax",
            "curricular_face",
            "class_circle_loss",
        ]:
            if label is None and self.training:
                raise ValueError(
                    "The label cannot be None in the training mode "
                    "if the large margin softmax is applied"
                )
            logits, meta = self.affine_softmax(input_feat, label, step)
        else:
            logits = self.affine_softmax(input_feat)
            # Make a dummy meta if no margin is applied.
            meta = {"margin": 0, "margin_lambda": 0}
        embedding['logit'] = logits
        if export_onnx:
            return embedding
        return logits, embedding, meta
