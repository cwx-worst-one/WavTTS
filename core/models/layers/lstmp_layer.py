''' LSTMP '''
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch.nn import _VF
except Exception:
    from torch import _VF
from core.extensions import LSTMFunction, get_amp_level


class LSTM(nn.LSTM):
    '''
    LSTM Module.
    move from falconpai to dolphin.
    '''

    def __init__(self, *args, **kwargs):
        '''init.'''
        super().__init__(*args, **kwargs)
        if self.bidirectional and LSTMFunction is not None:
            raise RuntimeError('{} not support bidirectional'.format(self.__class__.__name__))

    def forward(self, x, hx=None):
        '''forward.'''
        if not self.training or LSTMFunction is None:
            return super().forward(x, hx=hx)
        if self.batch_first:
            x = x.permute(1, 0, 2)  # steps, batch, input_size

        ih_weight, ih_bias = self.weight_ih_l0, self.bias_ih_l0
        hh_weight, hh_bias = self.weight_hh_l0, self.bias_hh_l0
        if get_amp_level() in ('O1', 'O2', 'O3'):
            x = x.half()
            ih_weight = ih_weight.half()
            hh_weight = hh_weight.half()

        zeros = torch.zeros(x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)
        bsz = x.size(1)
        h_n = torch.empty([self.num_layers, bsz, self.hidden_size], dtype=x.dtype, device='cuda')
        c_n = torch.empty([self.num_layers, bsz, self.hidden_size], dtype=x.dtype, device='cuda')
        h0 = hx[0][0] if hx is not None else zeros
        c0 = hx[1][0] if hx is not None else zeros
        h, c = LSTMFunction.apply(self.training, x, h0, c0, ih_weight, ih_bias, hh_weight, hh_bias)
        if self.num_layers > 1:
            h_n[0] = h[-1]
            c_n[0] = c[-1]
            for l in range(1, self.num_layers):
                ih_weight = getattr(self, 'weight_ih_l{}'.format(l))
                ih_bias = getattr(self, 'bias_ih_l{}'.format(l))
                hh_weight = getattr(self, 'weight_hh_l{}'.format(l))
                hh_bias = getattr(self, 'bias_hh_l{}'.format(l))
                if get_amp_level() in ('O1', 'O2', 'O3'):
                    ih_weight = ih_weight.half()
                    hh_weight = hh_weight.half()
                if self.dropout == 0.0:
                    h_drop = h[1:]
                else:
                    h_drop = F.dropout(h[1:], self.dropout)
                h0 = hx[0][l] if hx is not None else zeros
                c0 = hx[1][l] if hx is not None else zeros
                h, c = LSTMFunction.apply(
                    self.training, h_drop, h0, c0, ih_weight, ih_bias, hh_weight, hh_bias
                )
                h_n[l] = h[-1]
                c_n[l] = c[-1]
        out = h[1:]
        if self.batch_first:
            out = out.permute(1, 0, 2)
        if self.num_layers == 1:
            return out, (h[-1], c[-1])
        return out, (h_n, c_n)


class LSTMP(nn.Module):
    '''LSTMP'''

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        bidirectional=False,
        output_size=0,
        dropout=0.0,
        residual=False,
    ):
        super().__init__()
        if bidirectional:
            # TODO(liyong): support bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size, hidden_size, batch_first=batch_first, bidirectional=bidirectional
            )
        else:
            self.lstm = LSTM(
                input_size, hidden_size, batch_first=batch_first, bidirectional=bidirectional
            )
        if output_size > 0:
            if bidirectional:
                self.linear_proj = nn.Linear(2 * hidden_size, output_size)
            else:
                self.linear_proj = nn.Linear(hidden_size, output_size)
        else:
            if bidirectional:
                self.linear_proj = nn.Linear(2 * hidden_size, input_size)
            else:
                self.linear_proj = nn.Linear(hidden_size, input_size)
        if residual:
            if output_size in (0, input_size):
                self.residual_linear = None
            else:
                self.residual_linear = nn.Linear(input_size, output_size)
        self.residual = residual
        self.dropout = dropout

    def forward(self, lstm_input):
        """forward"""
        lstm_out, _ = self.lstm(lstm_input)
        proj_out = self.linear_proj(lstm_out)
        if self.dropout > 0.0:
            proj_out = F.dropout(proj_out, self.dropout, training=self.training)
        if self.residual:
            if self.residual_linear is None:
                proj_out = proj_out + lstm_input
            else:
                proj_out = proj_out + self.residual_linear(lstm_input)
        return proj_out

    def forward_step(self, lstm_input, lstm_states=None):
        """
        forward step.

        Args:
          lstm_input[Tensor]:
          lstm_states: [h0, c0]
            h0[Tensor]: shape is [bsz, hidden_size]
            c0[Tensor]: shape is [bsz, hidden_size]
        """
        lstm_out, new_states = self.lstm(lstm_input, lstm_states)
        proj_out = self.linear_proj(lstm_out)
        if self.dropout > 0.0:
            proj_out = F.dropout(proj_out, self.dropout, training=self.training)
        if self.residual:
            if self.residual_linear is None:
                proj_out = proj_out + lstm_input
            else:
                proj_out = proj_out + self.residual_linear(lstm_input)
        # new_states is an instance of (hn, cn)
        return proj_out, new_states


class IncrementalLSTM(LSTM):
    """IncrementalLSTM"""

    def forward_step(self, inp, states):
        """forward_step"""
        assert not self.bidirectional
        out_states = []
        for i in range(self.num_layers):
            if states is None:
                zeros = torch.zeros(
                    inp.size(0), self.hidden_size, dtype=inp.dtype, device=inp.device
                )
                state_i = (zeros, zeros)
            else:
                state_i = states[i]
            weight_ih = eval('self.weight_ih_l{}'.format(i))
            weight_hh = eval('self.weight_hh_l{}'.format(i))
            bias_ih = eval('self.bias_ih_l{}'.format(i))
            bias_hh = eval('self.bias_hh_l{}'.format(i))

            new_h, new_c = _VF.lstm_cell(inp, state_i, weight_ih, weight_hh, bias_ih, bias_hh)
            out_states.append((new_h, new_c))
            inp = new_h
        return new_h, out_states


class IncrementalLimitedContextLSTM(nn.Module):
    '''
    prev_emb: [B, N, E]
    '''

    def __init__(
        self,
        predictor_emb_size,
        predictor_lstm_hidden_size,
        predictor_lstm_layer_num,
        jointer_hidden_size,
        limited_context,
        limited_context_padding,
        batch_first,
        bidirectional,
    ):
        super().__init__()
        self.predictor_emb_size = predictor_emb_size
        self.predictor_lstm_hidden_size = predictor_lstm_hidden_size
        self.num_layers = predictor_lstm_layer_num
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.limited_context = limited_context
        self.limited_context_padding = limited_context_padding
        assert self.limited_context is not None

        self.lstm = IncrementalLSTM(
            predictor_emb_size,
            predictor_lstm_hidden_size,
            num_layers=predictor_lstm_layer_num,
            batch_first=True,
            bidirectional=False,
        )

        self.lstm_proj_fc = nn.Sequential(
            *[
                nn.Linear(predictor_lstm_hidden_size, jointer_hidden_size),
                nn.LayerNorm(jointer_hidden_size),
            ]
        )

    def forward(self, prev_emb, prev0_emb):
        """forward"""
        assert self.limited_context_padding is not True
        out_states = prev_emb
        assert out_states.size(1) == self.limited_context
        if self.limited_context_padding is not True:
            num = (
                (((out_states != prev0_emb).sum(dim=2) > 0).sum(dim=1))
                .expand(self.limited_context - 1, out_states.size(2), -1)
                .permute(2, 0, 1)
                .contiguous()
            )
            index = (
                torch.arange(0, self.limited_context - 1, device=out_states.device)
                .expand(num.size(0), num.size(2), -1)
                .permute(0, 2, 1)
                .contiguous()
            )
            index = (index - num + self.limited_context).clamp(min=0)
            zero = torch.zeros_like(index)
            index = torch.where(index < self.limited_context, index, zero)
            index = F.pad(index, (0, 0, 1, 0, 0, 0), "constant", 0)
            lstm_states = torch.gather(out_states, 1, index)
            num = ((out_states != prev0_emb).sum(dim=2) > 0).sum(dim=1)
        else:
            lstm_states = out_states

        limited_context_lstm_out, _ = self.lstm(lstm_states)
        return limited_context_lstm_out, out_states, num

    def forward_step(self, prev_emb, states, prev0_emb):
        """forward_step"""
        assert self.limited_context
        if states is None:
            out_states = prev_emb.unsqueeze(dim=1)
        else:
            out_states = torch.cat((states, prev_emb.unsqueeze(dim=1)), 1)[
                :, -(self.limited_context - 1) :, :
            ]
        if out_states.size(1) < self.limited_context:  # pad
            pad = out_states
            pad = torch.cat(
                (
                    prev0_emb.unsqueeze(0).expand(
                        pad.size(0), self.limited_context - out_states.size(1), prev0_emb.size(1)
                    ),
                    pad,
                ),
                1,
            )
            out_states = pad
        assert out_states.size(1) == self.limited_context
        if self.limited_context_padding is not True:
            num = (
                (((out_states != prev0_emb).sum(dim=2) > 0).sum(dim=1))
                .expand(self.limited_context - 1, out_states.size(2), -1)
                .permute(2, 0, 1)
                .contiguous()
            )
            index = (
                torch.arange(0, self.limited_context - 1, device=out_states.device)
                .expand(num.size(0), num.size(2), -1)
                .permute(0, 2, 1)
                .contiguous()
            )
            index = (index - num + self.limited_context).clamp(min=0)
            zero = torch.zeros_like(index)
            index = torch.where(index < self.limited_context, index, zero)
            index = F.pad(index, (0, 0, 1, 0, 0, 0), "constant", 0)
            lstm_states = torch.gather(out_states, 1, index)
            num = ((out_states != prev0_emb).sum(dim=2) > 0).sum(dim=1)
        else:
            lstm_states = out_states

        limited_context_lstm_out, _ = self.lstm(lstm_states)
        return limited_context_lstm_out, out_states, num
