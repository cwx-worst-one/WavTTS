'''
cif calculator
'''
import torch
from torch import nn
from core.models.layers.cnn import SamePaddingConv2D
from core.utils.dict import FalconDict
from core.extensions import cif_calculator


class CifWeightEstimator(nn.Module):
    '''CifWeightEstimator'''

    def __init__(self, args):
        super().__init__()
        self.args = FalconDict(args)

        if args.produce_weights_type == 'conv':
            conv_cif_widths_list = args.conv_cif_width_string.split(',')
            assert len(conv_cif_widths_list) == args.conv_cif_num_layers
            layers = []
            in_channel = args.hidden_size
            for w in conv_cif_widths_list:
                # conv layer
                layers.append(
                    SamePaddingConv2D(
                        in_channel,
                        args.conv_cif_num_filters,
                        (int(w), 1),
                        (1, 1),
                        norm_type=args.conv_norm_type,
                        act_type='relu',
                        dropout_rate=args.conv_cif_dropout,
                    )
                )
                in_channel = args.conv_cif_num_filters
            self.middle_projs = nn.Sequential(*layers)
            self.final_proj = nn.Sequential(nn.Linear(in_channel, 1), nn.Sigmoid())
        else:
            raise NotImplementedError('{} not support yet'.format(args.produce_weights_type))

    def forward(self, encoder_outputs, not_padding):
        '''
        Args:
          encoder_outputs: a Tensor of shape [batch_size, encoder_output_length, hidden_size].
          not_padding: shows the effective and non-effective, with shape
                       [batch_size, encoder_output_length].

        Returns:
          a: the estimated cif weight for each step of encoder outputs, with shape
             [batch_size, encoder_output_length]
        '''
        if self.args.produce_weights_type == 'conv':
            outs = encoder_outputs.transpose(1, 2).unsqueeze(3)
            outs = self.middle_projs(outs)
            outs = outs.squeeze(3).transpose(1, 2).contiguous()
            outs = self.final_proj(outs)
        else:
            raise NotImplementedError('{} not support yet'.format(self.args.produce_weights_type))

        if not_padding is None:
            return outs.squeeze(-1)
        return outs.squeeze(-1) * not_padding.float()


class CifCalculator(nn.Module):
    '''CifCalculator'''

    def __init__(self, args):
        super().__init__()
        self.args = FalconDict(args)

    def forward(self, cif_inputs, not_padding, a, targets=None, is_training=False):
        """The CIF part in between the encoder and decoder,
           firing the representation of one acoustic unit
           when the information is accumulated over the given threshold.
        Args:
          cif_inputs: a Tensor of shape [batch_size, input_length, hidden_size].
          not_padding: shows the effective and non-effective, with shape
                       [batch_size, encoder_output_length].
          a: the estimated cif weight for each step of encoder outputs, with shape
             [batch_size, encoder_output_length]
          hidden_size: the dimension of the last dimension of cif_inputs.
          targets: the batch of targets used for score scaling,
                   with shape [batch_size, target_length]
        Returns:
          cif_outputs: the output of the CIF part,
                       with shape [batch_size, cif_output_length, hidden_size]
          not_padding_after_cif: shows the effective and non-effective.
          sum_a: the sum of predicted score of each high-level frame, used for loss calculation.
          integrated_logits_on_encoder: the integrated logits by using the same weight,
                                        which is estimated by CIF,
                                        with shape [batch_size, cif_output_length, vocab_size]
        """
        # pylint:disable=too-many-locals
        hparams = self.args
        # is_training = self.training
        # cif_inputs = cif_inputs * not_padding.unsqueeze(-1)
        if is_training and hparams.use_scaling_strategy:
            targets_mask = (targets != 0.0).float()
            targets_length = targets_mask.sum(-1)
            a_sum = a.sum(-1)
            normalize_scalar = torch.unsqueeze(targets_length / a_sum, -1)
            a_org = a
            a = a * normalize_scalar
        # used for the handling of tail
        first_padding_pos = not_padding.sum(1, keepdim=True)
        threshold = hparams.cif_weight_threshold
        use_tail_handling = (not is_training) and hparams.use_tail_handling
        if self.training and self.args.get('use_fused_kernel', True):
            [
                accumulated_weights,
                accumulated_states,
                fired_marks,
                cif_outputs,
                not_padding_after_cif,
            ] = cif_calculator(cif_inputs, a, first_padding_pos, threshold, use_tail_handling)
        else:
            (
                batch_size,
                cif_input_length,
                hidden_size,  # pylint: disable=unused-variable
            ) = cif_inputs.shape
            # init loop states
            accumulated_weights = []
            accumulated_states = []
            fired_states = []
            prev_accumulated_weight = 0
            prev_accumulated_state = 0
            zeros_state = torch.zeros_like(cif_inputs[:, 0, :])
            for i in range(cif_input_length):
                # update the accumulated weights by considering whether positioning a boundary
                cur_weight = torch.unsqueeze(a[:, i], -1)
                remained_weight = 1.0 - prev_accumulated_weight
                # decide whether positioning a boundary
                cur_is_fired = ((prev_accumulated_weight + cur_weight) > threshold).int()
                cur_accumulated_weight = torch.where(
                    cur_is_fired == 1,
                    cur_weight - remained_weight,
                    cur_weight + prev_accumulated_weight,
                )
                cur_accumulated_state = torch.where(
                    cur_is_fired == 1,
                    (cur_weight - remained_weight) * cif_inputs[:, i, :],
                    prev_accumulated_state + cur_weight * cif_inputs[:, i, :],
                )
                cur_fired_state = torch.where(
                    cur_is_fired == 1,
                    prev_accumulated_state + remained_weight * cif_inputs[:, i, :],
                    zeros_state,
                )
                # handling the speech tail by rounding up and down
                if use_tail_handling:
                    cur_fired_state = torch.where(
                        i == first_padding_pos,
                        torch.where(
                            cur_accumulated_weight <= 0.5,
                            zeros_state,
                            cur_accumulated_state / (cur_accumulated_weight + 1e-10),
                        ),
                        cur_fired_state,
                    )
                cur_fired_state = torch.where(i > first_padding_pos, zeros_state, cur_fired_state)
                prev_accumulated_weight = cur_accumulated_weight
                prev_accumulated_state = cur_accumulated_state
                accumulated_weights.append(cur_accumulated_weight)
                accumulated_states.append(cur_accumulated_state.unsqueeze(1))
                fired_states.append(cur_fired_state.unsqueeze(1))
            accumulated_weights = torch.cat(accumulated_weights, dim=1)
            accumulated_states = torch.cat(accumulated_states, dim=1)
            fired_states = torch.cat(fired_states, dim=1)
            fired_marks = fired_states.abs().sum(-1) != 0.0
            # fired_utt_length = fired_marks.count_nonzero(-1)
            fired_utt_length = fired_marks.sum(-1)
            fired_max_length = fired_utt_length.max().int()
            cif_outputs = []
            for j in range(batch_size):
                cur_utt_fired_mark = fired_marks[j, :]
                cur_utt_fired_state = fired_states[j, :, :]
                cur_utt_outputs = cur_utt_fired_state[cur_utt_fired_mark, :]
                cur_utt_outputs = nn.functional.pad(
                    cur_utt_outputs, (0, 0, 0, fired_max_length - fired_utt_length[j])
                )
                cif_outputs.append(cur_utt_outputs.unsqueeze(0))
            cif_outputs = torch.cat(cif_outputs, dim=0)
            # calculate the not_padding according to the cif_outputs
            not_padding_after_cif = (cif_outputs.abs().sum(-1) != 0.0).int()
        # for the calculation of num char
        if is_training and hparams.use_scaling_strategy:
            sum_a = a_org.sum(1)
        else:
            sum_a = a.sum(1)
        internal_dict = dict()
        internal_dict['boundary_marks'] = fired_marks
        internal_dict['weights'] = a
        internal_dict['accumulated_weights'] = accumulated_weights
        return cif_outputs, not_padding_after_cif, sum_a, None, internal_dict
