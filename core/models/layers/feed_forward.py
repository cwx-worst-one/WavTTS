''' Feed Forward Network '''
import math
import torch
from torch import nn
from janus.layer import MoE

from core.models.layers.active_function import get_activation_module, get_activation_fn
from core.extensions import fused_ffn


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.

    :param int input_dim: input dimension
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate

    """

    def __init__(
        self,
        input_dim,
        hidden_units,
        dropout_rate,
        activation_fn,
        use_layer_norm=False,
        pre_layer_norm=False,
        layer_norm_eps=1e-5,
        extra_dropout=False,
        residual_out_scaler=1.0,
        fused=True,
        **kwargs,
    ):
        """Construct an PositionwiseFeedForward object."""
        super().__init__()
        activation_module = get_activation_module(activation_fn)
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        net = [nn.Linear(input_dim, hidden_units), activation_module(**kwargs)]
        # TODO(mst) re-verify whether Dropout(0.0) == Identity
        if dropout_rate != 0.0:
            net.append(nn.Dropout(dropout_rate))
        net.append(nn.Linear(hidden_units, input_dim))

        if isinstance(extra_dropout, float):
            extra_dropout_rate = extra_dropout
        elif extra_dropout:
            extra_dropout_rate = dropout_rate
        else:
            extra_dropout_rate = 0.0
        if extra_dropout_rate != 0.0:
            net.append(nn.Dropout(extra_dropout_rate))

        if fused:
            self.fused_cfg = {
                'embed_dim': input_dim,
                'ffn_hidden': hidden_units,
                'hidden_drop': extra_dropout_rate,
                'act_drop': dropout_rate,
                'eps': layer_norm_eps,
                'res_weight': residual_out_scaler,
                'do_residual': use_layer_norm,
                'clamp_inf': False,
                'squeeze_mem': False,
                'activation': activation_fn,
                'norm_mode': '',
            }
            self._fc2_idx = -2 if extra_dropout_rate != 0.0 else -1
            if use_layer_norm:
                self.fused_cfg['norm_mode'] = 'before' if pre_layer_norm else 'after'

        self.core_net = nn.Sequential(*net)
        self.use_layer_norm = use_layer_norm
        self.pre_layer_norm = pre_layer_norm
        self.residual_out_scaler = residual_out_scaler
        self.fused = fused
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)

    def _forward_fused(self, x, batch_first):
        '''forward fused'''
        params = [
            self.core_net[0].weight,
            self.core_net[0].bias,
            self.core_net[self._fc2_idx].weight,
            self.core_net[self._fc2_idx].bias,
        ]
        if self.use_layer_norm:
            params.extend([self.layer_norm.weight, self.layer_norm.bias])
        else:
            params.extend([None, None])
        # pylint: disable=no-member
        return fused_ffn(params, x, batch_first, **self.fused_cfg)

    def forward(self, x, batch_first=True):
        """Forward function."""
        if self.training and self.fused:
            return self._forward_fused(x, batch_first)

        if self.use_layer_norm:
            if self.pre_layer_norm:
                ##### layer normalization + positionwise feed-forward
                core_out = self.core_net(self.layer_norm(x))

                ##### residual connection
                output = self.residual_out_scaler * core_out + x
            else:
                ##### positionwise feed-forward
                core_out = self.core_net(x)

                ##### residual connection + layer normalization
                output = self.layer_norm(x + self.residual_out_scaler * core_out)
        else:
            output = self.core_net(x)

        return output


class FFN(nn.Module):
    """feed forward layer.

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation_fn):
        """Construct an Positionwise FeedForward object."""
        super(__class__, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = get_activation_fn(activation_fn)
        self.fused_cfg = {
            'embed_dim': idim,
            'ffn_hidden': hidden_units,
            'hidden_drop': 0.0,
            'act_drop': dropout_rate,
            'activation': activation_fn,
            'batch_first': True,
        }

    def forward(self, x, fused=True):
        """Forward funciton."""
        if self.training and fused:
            return fused_ffn(
                [
                    self.w_1.weight,
                    self.w_1.bias,
                    self.w_2.weight,
                    self.w_2.bias,
                ],
                x,
                **self.fused_cfg,
            )
        return self.w_2(self.dropout(self.activation_fn(self.w_1(x))))


class MoEFFN(nn.Module):
    '''MoE FFN layer'''

    def __init__(
        self,
        idim,
        hidden_units,
        dropout_rate=0.0,
        activation_fn='gelu',
        num_expert=8,
        topk=2,
        output_dropout_prob=0.0,
        moe_loss_scale=0.0,
        z_loss_scale=0.0,
        noisy_gate_policy=None,
        use_lego=False,
    ):
        '''init using FFN'''
        super().__init__()
        self.use_lego = use_lego
        is_dropout = output_dropout_prob > 0.0
        if self.use_lego:
            self.pwff = MoE(
                hidden_size=idim,
                num_experts=num_expert,
                k=topk,
                use_lego=True,
                expert_type='ffn',
                intermediate_size=hidden_units,
                activation=activation_fn,
                lego_dropout_rate=dropout_rate,
                is_dropout=is_dropout,
                output_dropout_prob=output_dropout_prob,
            )
        else:
            pwff = FFN(idim, hidden_units, dropout_rate, activation_fn)
            self.pwff = MoE(
                hidden_size=idim,
                num_experts=num_expert,
                k=topk,
                expert=pwff,
                output_dropout_prob=output_dropout_prob,
                is_dropout=is_dropout,
                noisy_gate_policy=noisy_gate_policy,
            )
        self.moe_loss_scale = moe_loss_scale
        self.z_loss_scale = z_loss_scale
        self.num_expert = num_expert
        self.aux_loss = 0.0
        self._register_load_state_dict_pre_hook(self.compatible_load_hook)
        self.reset_parameters()

    def reset_parameters(self):
        '''reset_parameters: the same init as torch Linear'''
        if self.use_lego:
            nn.init.kaiming_uniform_(self.pwff.moe.wg.weight, a=math.sqrt(5))
            for i in range(self.pwff.moe.num_experts):
                nn.init.kaiming_uniform_(
                    self.pwff.moe.experts.expert1.weight[i, :, :],
                    a=math.sqrt(5),
                )
                nn.init.kaiming_uniform_(
                    self.pwff.moe.experts.expert2.weight[i, :, :],
                    a=math.sqrt(5),
                )
        else:
            nn.init.kaiming_uniform_(self.pwff.moe.gate.wg.weight, a=math.sqrt(5))

    def forward(self, x, **_kwargs):
        """Forward funciton."""
        output, aux_loss, _ = self.pwff(x)
        if self.moe_loss_scale > 0.0:
            self.aux_loss = self.moe_loss_scale * aux_loss
        if self.z_loss_scale > 0.0 and not self.use_lego:
            self.aux_loss += self.z_loss_scale * self.pwff.get_z_loss()
        return output

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
        compatible for load MOE FFN w/wo lego.
        '''
        gate_weight_wo_lego = prefix + 'pwff.moe.gate.wg.weight'
        fc1_weight_wo_lego = [
            prefix + 'pwff.moe.experts.experts.{}.w_1.weight'.format(i)
            for i in range(self.num_expert)
        ]
        fc1_bias_wo_lego = [
            prefix + 'pwff.moe.experts.experts.{}.w_1.bias'.format(i)
            for i in range(self.num_expert)
        ]
        fc2_weight_wo_lego = [
            prefix + 'pwff.moe.experts.experts.{}.w_2.weight'.format(i)
            for i in range(self.num_expert)
        ]
        fc2_bias_wo_lego = [
            prefix + 'pwff.moe.experts.experts.{}.w_2.bias'.format(i)
            for i in range(self.num_expert)
        ]

        gate_weight_with_lego = prefix + 'pwff.moe.wg.weight'
        fc1_weight_with_lego = prefix + 'pwff.moe.experts.expert1.weight'
        fc1_bias_with_lego = prefix + 'pwff.moe.experts.expert1.bias'
        fc2_weight_with_lego = prefix + 'pwff.moe.experts.expert2.weight'
        fc2_bias_with_lego = prefix + 'pwff.moe.experts.expert2.bias'

        if self.use_lego and gate_weight_wo_lego in state_dict:
            state_dict[gate_weight_with_lego] = state_dict[gate_weight_wo_lego]
            state_dict[fc1_weight_with_lego] = torch.stack(
                [state_dict[key] for key in fc1_weight_wo_lego], dim=0
            )
            state_dict[fc1_bias_with_lego] = torch.stack(
                [state_dict[key] for key in fc1_bias_wo_lego], dim=0
            )
            state_dict[fc2_weight_with_lego] = torch.stack(
                [state_dict[key] for key in fc2_weight_wo_lego], dim=0
            )
            state_dict[fc2_bias_with_lego] = torch.stack(
                [state_dict[key] for key in fc2_bias_wo_lego], dim=0
            )
            del state_dict[gate_weight_wo_lego]
            for key in (
                fc1_weight_wo_lego + fc1_bias_wo_lego + fc2_weight_wo_lego + fc2_bias_wo_lego
            ):
                del state_dict[key]


class LinearClamp(nn.Linear):
    '''clamp after linear.'''

    def __init__(self, in_features, out_features, bias=True, clamp=0):
        '''init.'''
        super().__init__(in_features, out_features, bias)
        self.clamp = clamp

    def forward(self, x):
        y = super().forward(x)
        if self.training and self.clamp:
            y = torch.clamp(y, -4e4, 4e4)
        return y
