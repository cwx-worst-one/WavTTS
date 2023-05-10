''' DFSMN layers '''
import torch
import torch.nn.functional as F
from torch import nn
from core.extensions import fused_dfsmn


class FSMNLayer(nn.Module):
    '''FSMNLayer'''

    def __init__(self, memory_size, left_kernel_size, right_kernel_size, dilation=1, bias=False):
        super().__init__()
        self.memory = nn.Conv1d(
            memory_size,
            memory_size,
            kernel_size=left_kernel_size + right_kernel_size + 1,
            padding=0,
            stride=1,
            dilation=dilation,
            groups=memory_size,
            bias=bias,
        )
        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.memory_size = memory_size
        self._register_load_state_dict_pre_hook(self.compatible_load_hook)

    @staticmethod
    def compatible_load_hook(
        state_dict, prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, _error_msgs
    ):
        '''
        compatible for load FSMN layer.
        '''
        name = prefix + 'memory_coffe'
        if name in state_dict:
            new_key = name.replace('.memory_coffe', '.memory.weight')
            val = state_dict.pop(name, None)  # (1, kernel_size, 1, memory_size)
            val = val.squeeze(0).permute(2, 1, 0).contiguous()  # (memory_size, 1, kernel_size)
            state_dict[new_key] = val

    def forward(self, input_feat, input_mask=None, input_res=None):
        """forward"""
        residual = input_feat  # (B, T, N)
        if input_res is not None:
            residual = input_res
        pad_input_fea = F.pad(
            input_feat.transpose(1, 2).contiguous(),
            (self.left_kernel_size * self.dilation, self.right_kernel_size * self.dilation, 0, 0),
        )  # (B,N,T+(l+r)*d)
        memory_out = (self.memory(pad_input_fea) + residual.transpose(1, 2)).transpose(
            1, 2
        )  # (B, T, N)
        if input_mask is not None:
            memory_out = memory_out * input_mask
        return memory_out


class DFSMNLayer(FSMNLayer):
    # pylint: disable=protected-access
    """DFSMNLayer"""

    def __init__(
        self,
        hidden_size,
        memory_size,
        left_kernel_size,
        right_kernel_size,
        dilation=1,
        dropout=0.0,
        weight_scale=1.0,
        bias=True,
        fc_layernorm=True,
        memory_first=True,
    ):
        super().__init__(
            memory_size,
            left_kernel_size,
            right_kernel_size,
            dilation,
            bias=bias,
        )
        module_list = []
        if fc_layernorm:
            module_list.append(nn.LayerNorm(memory_size))
        module_list += [
            nn.Linear(memory_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(hidden_size, memory_size, bias=bias),
        ]
        if memory_first:
            module_list.append(nn.Dropout(dropout, inplace=True))
        self.fc_trans = nn.Sequential(*module_list)

        self.memory_ln = nn.LayerNorm(memory_size) if memory_first else None
        self.memory_first = memory_first
        self.fc_layernorm = fc_layernorm
        self.dropout = dropout

        if weight_scale < 1.0:
            for _, param in self.named_parameters():
                param.data.mul_(weight_scale)

        self._register_load_state_dict_pre_hook(self.compatible_load_hook)

    @staticmethod
    def compatible_load_hook(
        state_dict, prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, _error_msgs
    ):
        '''
        compatible for load DFSMN layer.
        '''
        name = prefix + 'memory_coffe'
        if name in state_dict:
            new_key = name.replace('.memory_coffe', '.memory.weight')
            val = state_dict.pop(name, None)  # (1, kernel_size, 1, memory_size)
            val = val.squeeze(0).permute(2, 1, 0).contiguous()  # (memory_size, 1, kernel_size)
            state_dict[new_key] = val

        for idx in range(5):
            for param in ['.weight', '.bias']:
                name = prefix + 'input_trans_fc.' + str(idx) + param
                if name in state_dict:
                    new_key = name.replace('.input_trans_fc', '.fc_trans')
                    state_dict[new_key] = state_dict.pop(name, None)

    @property
    def state_size(self):
        '''streamed state size.'''
        # remain redundant fsmn_right_kernel_size * fsmn_dilation space
        # align to panther impl
        fsmn_state_size = (
            (self.left_kernel_size + self.right_kernel_size * 2) * self.dilation * self.memory_size
        )
        return fsmn_state_size

    def forward(self, input_feat, *_args, fused=False, **_kwargs):
        """forward"""
        if self.training and fused and self.memory_first and self.fc_layernorm:
            return fused_dfsmn(
                [
                    self.memory.weight,
                    self.memory.bias,
                    self.fc_trans[1].weight,
                    self.fc_trans[1].bias,
                    self.fc_trans[4].weight,
                    self.fc_trans[4].bias,
                    self.memory_ln.weight,
                    self.memory_ln.bias,
                    self.fc_trans[0].weight,
                    self.fc_trans[0].bias,
                ],
                input_feat,
                left_kernel_size=self.left_kernel_size,
                right_kernel_size=self.right_kernel_size,
                dilation=self.dilation,
                dropout=self.dropout,
                training=self.training,
            )
        fc_input = input_feat
        if self.memory_first:
            # dfsmn-memory
            memory_input = self.memory_ln(input_feat)  # (B, T, N)
            memory_out = super().forward(memory_input, input_res=input_feat)  # (B, T, N)
            fc_input = memory_out
        # fc-transform
        fc_out = self.fc_trans(fc_input)  # (B, T, N)
        if self.memory_first:
            if (torch.jit.is_scripting() or torch.jit.is_tracing()) and not self.training:
                return (fc_out.transpose(1, 2) + memory_out.transpose(1, 2)).transpose(1, 2)
            return fc_out + memory_out
        return super().forward(fc_out) + input_feat  # (B, T, N)

    def forward_step(
        self,
        x,
        x_mask=None,
        x_sign=None,
        states=None,
    ):
        """forward_step"""
        bsz, _, _ = x.size()
        residual = x
        x = self.memory_ln(x.transpose(1, 2)).transpose(1, 2).contiguous()  # (B, N, T)

        if self.right_kernel_size > 0:
            residual_states = states[
                :,
                (self.left_kernel_size + self.right_kernel_size)
                * self.dilation
                * self.memory_size :,
            ].reshape(bsz, self.memory_size, self.right_kernel_size * self.dilation)
            new_residual_states = residual[:, :, -self.right_kernel_size * self.dilation :]

        states = states[
            :,
            : (self.left_kernel_size + self.right_kernel_size) * self.dilation * self.memory_size,
        ].reshape(
            bsz,
            self.memory_size,
            (self.left_kernel_size + self.right_kernel_size) * self.dilation,
        )

        if x_sign == 0:
            # mid
            x = torch.cat([states, x], dim=2)  # (B, N, cache_length + chunk_lenth)
        elif x_sign == 1:
            # first
            states = states[:, :, self.right_kernel_size * self.dilation :]
            x = torch.cat([states, x], dim=2)  # (B, N, cache_length + chunk_lenth)
        elif x_sign == 2:
            # last
            # (B, N, cache_length + chunk_lenth + right padding)
            x = torch.cat([states, x], dim=2)
            x = F.pad(x, (0, self.right_kernel_size * self.dilation, 0, 0))
        elif x_sign == 3:
            # first&last == nonstream
            x = F.pad(
                x,
                (
                    self.left_kernel_size * self.dilation,
                    self.right_kernel_size * self.dilation,
                    0,
                    0,
                ),
            )
        else:
            raise NotImplementedError('x_sign only support 0, 1, 2, 3 !')

        new_states = x[:, :, -((self.left_kernel_size + self.right_kernel_size) * self.dilation) :]

        memory_input = x

        if self.right_kernel_size > 0:
            if x_sign == 2:
                residual = torch.cat([residual_states, residual], dim=2)
            elif x_sign == 1:
                residual = residual[:, :, : -self.right_kernel_size * self.dilation]
            elif x_sign == 0:
                residual = torch.cat([residual_states, residual], dim=2)
                residual = residual[:, :, : -self.right_kernel_size * self.dilation]
            elif x_sign == 3:
                pass

        memory_out = self.memory(memory_input) + residual

        residual = memory_out

        fc_output = self.fc_trans(memory_out.transpose(1, 2))
        x = fc_output.transpose(1, 2) + residual
        new_states = new_states.reshape(bsz, -1)
        states = torch.zeros([bsz, self.state_size]).type_as(new_states)
        states[:, : new_states.size(1)] = new_states
        if self.right_kernel_size > 0:
            new_residual_states = new_residual_states.reshape(bsz, -1)
            states[:, new_states.size(1) :] = new_residual_states
        return x, states, x_mask
