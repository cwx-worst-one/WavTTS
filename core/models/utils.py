''' utils
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''

from collections import defaultdict
from typing import Dict, Optional

import uuid
import torch
import torch.nn.functional as F
from torch import nn, Tensor


def xavier_init(module):
    '''xavier uniform'''
    for p in module.parameters():
        if len(p.size()) > 1:
            torch.nn.init.xavier_uniform_(p)


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    '''_get_full_incremental_state_key
    assign a unique ID to each module instance, so that incremental state is
    not shared across module instances
    '''
    # pylint: disable=protected-access
    module_name = module_instance.__class__.__name__
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


# pylint: disable=unused-argument
def make_positions(tensor, padding_idx, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    The series of casts and type-conversions here are carefully
    balanced to both work with ONNX export and XLA. In particular XLA
    prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    how to handle the dtype kwarg in cumsum.
    """
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def softmax(x, dim, onnx_trace=False):
    '''softmax with onnx support'''
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    return F.softmax(x, dim=dim, dtype=torch.float32)


########################################################################
#               NOTE: This Incremental Module is used for streaming
########################################################################


class ByteSpeechIncrementalState:
    '''ByteSpeechIncrementalState'''

    def __init__(self, *args, **kwargs):
        '''init.'''
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        '''init_incremental_state'''
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        '''_get_full_incremental_state_key'''
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    '''with_incremental_state'''
    cls.__bases__ = (ByteSpeechIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != ByteSpeechIncrementalState
    )
    return cls


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            if isinstance(m, nn.LayerNorm):
                if isinstance(args[0], tuple):
                    x, pos_emb = args[0]
                    return (m(x), pos_emb), args[1]
                return m(args[0]), args[1]
            if isinstance(m, nn.Conv1d):
                if isinstance(args[0], tuple):
                    x, pos_emb = args[0]
                    return (
                        m(x.transpose(1, 2)).transpose(1, 2).contiguous(),
                        pos_emb[:, ::2],
                    ), args[1][:, :, ::2]
                return (m(args[0].transpose(1, 2)).transpose(1, 2).contiguous(), args[1][:, :, ::2])
            args = m(*args)
        return args


def repeat(count, fn, layernorm_interval=0, layernorm_dim=None, pooling=False):
    """Repeat module times.

    :param int count: repeat time
    :param function fn: function to generate module
    :return: repeated modules
    :rtype: MultiSequential
    """
    if layernorm_interval > 0:
        # layernom_num = (N - 1) // layernorm_interval
        l, r = 0, 0
        module_list = []
        while l < count:
            r = min(r + layernorm_interval, count)
            module_list.extend([fn(n) for n in range(l, r)])
            l = r
            if r != count:
                module_list.append(nn.LayerNorm(layernorm_dim))
    else:
        module_list = [fn(n) for n in range(count)]
    if pooling:
        pos_half = len(module_list) // 2
        module_list.insert(pos_half, nn.Conv1d(layernorm_dim, layernorm_dim, 3, 2, 1))
    return MultiSequential(*module_list)


def streaming_conv2d_input(x, state_in, x_sign, state_num=1, padding=0, required_right_context=0):
    '''
    processs streamed conv2d inputs.
    Args:
      x(Tensor): [B, C, T, H]
      state_in(Tensor): [B, C, state_num, H]
      x_sign(Tensor): [1]

    Return:
      (new_x, state_out)
      new_x(Tensor): [B, C, x, H]
      state_out(Tensor): [B, C, state_num, H]
    '''
    # assert state_num >= 0
    b, c, _, h = x.size()
    if x_sign == 0:  # middle frame
        new_x = torch.cat([state_in, x], dim=2)
    elif x_sign == 1:  # first frame
        if padding > 0:
            pad = torch.zeros([b, c, padding, h], dtype=x.dtype, device=x.device)
            new_x = torch.cat([pad, x], dim=2)
        else:
            new_x = x
    elif x_sign == 2:  # last frame
        if padding > 0:
            pad = torch.zeros([b, c, padding, h], dtype=x.dtype, device=x.device)
            new_x = torch.cat([state_in, x, pad], dim=2)
        else:
            new_x = x
    elif x_sign == 3:  # fist and last frame
        if padding > 0:
            pad = torch.zeros([b, c, padding, h], dtype=x.dtype, device=x.device)
            new_x = torch.cat([pad, x, pad], dim=2)
        else:
            new_x = x

    if x_sign in (0, 1):
        state_out = new_x[
            :, :, -state_num - required_right_context : new_x.size(2) - required_right_context, :
        ]
    else:
        state_out = new_x[
            :,
            :,
            -padding
            - state_num
            - required_right_context : new_x.size(2)
            - required_right_context
            - padding,
            :,
        ]
    return new_x, state_out


def kl_divloss_rnnt(pred, target):
    '''kl_divloss of rnn-t
    pred: log probs of predicted/student model [B, T, U, 2], last dim accounts for blank/token
    target: log probs of target/teacher model, the same size as pred
    '''
    bsz = pred.size(0)
    target_exp = target.exp()
    target_exp_other = (1 - target_exp.sum(dim=-1)).clamp(1e-8, 1.0)
    target_other = torch.log(target_exp_other)
    pred_other = torch.log((1 - pred.exp().sum(dim=-1)).clamp(1e-8, 1.0))
    loss1 = (target_exp * (target - pred)).sum()
    # other token's loss
    loss2 = (target_exp_other * (target_other - pred_other)).sum()
    return (loss1 + loss2) / bsz


def unfold(tensor_input, num_neighbors):
    """
    Along the frequency axis, this function is used for splitting overlapped sub-band units.

    Args:
        input: four-dimension input.
        num_neighbors: number of neighbors in each side.

    Returns:
        Overlapped sub-band units.

    Shapes:
        input: [B, C, F, T]
        return: [B, N, C, F_s, T]. F_s represents the frequency axis of the sub-band unit,
                e.g. [2, 161, 1, 19, 200]
    """
    assert (
        tensor_input.dim() == 4
    ), f"The dim of the input is {tensor_input.dim()}. It should be four dim."
    batch_size, num_channels, num_freqs, num_frames = tensor_input.size()

    if num_neighbors < 1:  # No change on the input
        return tensor_input.permute(0, 2, 1, 3).reshape(
            batch_size, num_freqs, num_channels, 1, num_frames
        )

    output = tensor_input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)
    sub_band_unit_size = num_neighbors * 2 + 1

    # Pad the top and bottom of the original spectrogram
    output = F.pad(output, [0, 0, num_neighbors, num_neighbors], mode="reflect")  # [B * C, 1, F, T]

    output = F.unfold(output, (sub_band_unit_size, num_frames))  # move on the F and T axes.
    assert (
        output.shape[-1] == num_freqs
    ), f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

    # Split the dim of the unfolded feature
    output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
    output = output.permute(0, 4, 1, 2, 3).contiguous()  # [B, N, C, F_s, T]

    return output
