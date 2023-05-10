''' adaptive_softmax '''
# pylint: disable=invalid-name
# TODO(huanglu) fix this
import operator
import functools
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from core.utils import ceil, TC_ALIGN, logging
from core.extensions import (
    fused_adaptive_softmax,
)
from core.extensions.panther_symbol import PantherAdaptiveSoftmaxFunc

try:
    from byteslim.core.torch.base_classes import SlimModule
except ImportError:
    SlimModule = None
try:
    from panther_utilities.torch_symbolic.linear import PantherLinear
except ImportError:
    PantherLinear = None


class TiedLinear(nn.Module):
    '''TiedLinear'''

    def __init__(self, weight, transpose):
        '''init.'''
        super().__init__()
        self.weight = weight
        self.transpose = transpose

    def forward(self, input_data):
        """forward"""
        return F.linear(input_data, self.weight.t() if self.transpose else self.weight)


class TiedHeadModule(nn.Module):
    '''TiedHeadModule'''

    def __init__(self, weights, input_dim, num_classes):
        '''init.'''
        super().__init__()
        tied_emb, _ = weights
        self.num_words, emb_dim = tied_emb.size()

        self.word_proj = TiedLinear(tied_emb, transpose=False)
        if input_dim != emb_dim:
            self.word_proj = nn.Sequential(
                nn.Linear(input_dim, emb_dim, bias=False),
                self.word_proj,
            )

        self.class_proj = nn.Linear(input_dim, num_classes, bias=False)
        self.out_dim = self.num_words + num_classes

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def forward(self, input_data):
        """forward"""
        inp_sz = functools.reduce(operator.mul, input_data.shape[:-1], 1)
        out = self._float_tensor.new(inp_sz, self.out_dim)
        out[:, : self.num_words] = self.word_proj(input_data.view(inp_sz, -1))
        out[:, self.num_words :] = self.class_proj(input_data.view(inp_sz, -1))
        return out


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """

    def __init__(
        self,
        vocab_size,
        input_dim,
        cutoff,
        dropout,
        factor=4.0,
        adaptive_inputs=None,
        tie_proj=False,
        use_proj=True,
        adaptive_tail_small=False,
    ):
        super().__init__()

        if vocab_size != cutoff[-1]:
            raise ValueError('cutoff is not equal to vocab size')

        output_dim = cutoff[0] + len(cutoff) - 1

        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout = dropout
        self.input_dim = input_dim
        self.factor = factor
        self.adaptive_tail_small = adaptive_tail_small

        self.lsm = nn.LogSoftmax(dim=1)

        if adaptive_inputs is not None:
            self.head = TiedHeadModule(
                adaptive_inputs.weights_for_band(0), input_dim, len(cutoff) - 1
            )
        else:
            self.head = nn.Linear(input_dim, output_dim, bias=False)

        self._make_tail(adaptive_inputs, tie_proj, use_proj)

        def init_weights(m):
            if (
                hasattr(m, 'weight')
                and not isinstance(m, TiedLinear)
                and not isinstance(m, TiedHeadModule)
            ):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer('version', torch.LongTensor([1]))

    def clear_combined_weight(self):
        '''remove combined weight'''

    def combine_weight(self):
        '''combine_weight for inference'''

    def _make_tail(self, adaptive_inputs=None, tie_proj=False, use_proj=True):
        '''make tail'''
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            if use_proj:
                dim = int(self.input_dim // self.factor ** (i + 1))

                tied_emb, tied_proj = (
                    adaptive_inputs.weights_for_band(i + 1)
                    if adaptive_inputs is not None
                    else (None, None)
                )

                if tied_proj is not None:
                    if tie_proj:
                        proj = TiedLinear(tied_proj, transpose=True)
                    else:
                        proj = nn.Linear(tied_proj.size(0), tied_proj.size(1), bias=False)
                else:
                    proj = nn.Linear(self.input_dim, dim, bias=False)

                m = nn.Sequential(
                    proj,
                    nn.Dropout(self.dropout),
                    nn.Linear(
                        dim,
                        self.cutoff[i + 1] - self.cutoff[i],
                        bias=False,
                    )
                    if tied_emb is None
                    else TiedLinear(tied_emb, transpose=False),
                )
            else:
                m = nn.Linear(self.input_dim, self.cutoff[i + 1] - self.cutoff[i], bias=False)
            self.tail.append(m)

    @staticmethod
    def upgrade_state_dict_named(state_dict, name):
        '''upgrade_state_dict_named'''
        version_name = name + '.version'
        if version_name not in state_dict:
            raise Exception('This version of the model is no longer supported')

    def adapt_target(self, target):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """

        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.any():
                target_idxs.append(mask.nonzero().squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)

        return new_target, target_idxs

    def forward(self, input_data, target):
        """
        Args:
            input_data: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """

        input_data = input_data.contiguous().view(-1, input_data.size(-1))
        input_data = F.dropout(input_data, p=self.dropout, training=self.training)

        new_target, target_idxs = self.adapt_target(target)
        # new_target [cutoff target] [group1 target] [group2 target] [...]
        # target_idxs [group1 idx] [group2 idx] [group3 idx] [...]
        output = [self.head(input_data)]

        for i, t in enumerate(target_idxs):
            if t is not None:
                output.append(self.tail[i](input_data.index_select(0, t)))
            else:
                output.append(None)

        return output, new_target

    def get_log_prob(self, input_data, target):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """

        bsz, length, dim = input_data.size()
        input_data = input_data.contiguous().view(-1, dim)

        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None

        head_y = self.head(input_data)
        log_probs = head_y.new_zeros(input_data.size(0), self.vocab_size)

        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y)
        tail_priors = log_probs[:, self.cutoff[0] : head_sz].clone()

        for i, tail in enumerate(self.tail):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]

            if target_idxs is None:
                tail_out = log_probs[:, start:end]
                tail_out.copy_(tail(input_data))
                log_probs[:, start:end] = self.lsm(tail_out).add_(tail_priors[:, i, None])
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = log_probs[idxs, start:end]
                tail_out.copy_(tail(input_data[idxs]))
                log_probs[idxs, start:end] = self.lsm(tail_out).add_(tail_priors[idxs, i, None])

        log_probs = log_probs.view(bsz, length, -1)
        return log_probs


class RNNTAdaptiveSoftmax(nn.Module):
    '''RNNTAdaptiveSoftmax'''

    def __init__(
        self,
        vocab_size,
        input_dim,
        cutoff,
        dropout,
        factor=4.0,
        concate_U=1,
        use_proj=True,
        tc_align=True,
        adaptive_tail_small=False,
        squeeze_mem=False,
        **_kwargs
    ):
        super().__init__()
        if vocab_size != cutoff[-1]:
            raise ValueError(
                'cutoff is not equal to vocab size: {} vs. {}'.format(vocab_size, cutoff)
            )
        self.head_dim = cutoff[0] + len(cutoff) - 1
        if tc_align:
            self.head_dim_aligned = ceil(self.head_dim, TC_ALIGN)
        else:
            self.head_dim_aligned = self.head_dim
        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout = dropout
        self.input_dim = input_dim
        self.factor = factor
        self.adaptive_tail_small = adaptive_tail_small
        self.concate_U = concate_U
        # TODO: wrap head linear and log softmax with recompute to save more memory.
        self.squeeze_mem = squeeze_mem

        self.head = nn.Linear(input_dim, self.head_dim_aligned, bias=False)
        self._make_tail(use_proj=use_proj)
        self.register_buffer('version', torch.LongTensor([1]))

        self._register_load_state_dict_pre_hook(self.compatible_load_hook)
        self._register_state_dict_hook(self.compatible_save_hook)

        self.w_head = None
        self.w_tail = None
        # for PantherAdaptiveSoftmax v2
        self.w_tail_left = None
        self.w_tail_right = None

        self.reset_parameters()

    def reset_parameters(self):
        '''reset params.'''
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def _make_tail(self, use_proj=True):
        '''make tail'''
        self.tail = nn.ModuleList()
        self.total_internal_dim = 0
        for i in range(len(self.cutoff) - 1):
            if use_proj:
                # get internal dim
                if self.adaptive_tail_small:
                    dim = len(self.cutoff) + 10 - i
                else:
                    dim = (
                        len(self.cutoff)
                        + int((self.cutoff[i + 1] - self.cutoff[i]) / self.factor)
                        - i
                    )
                if i == 0:
                    self.tail_hidden_dim = dim
                    self.tail_hidden_dim_decrease = 1
                self.total_internal_dim += dim

                m = nn.Sequential(
                    nn.Linear(self.input_dim, dim, bias=False),
                    nn.Dropout(self.dropout),
                    nn.Linear(
                        dim,
                        self.cutoff[i + 1] - self.cutoff[i],
                        bias=False,
                    ),
                )
            else:
                m = nn.Linear(self.input_dim, self.cutoff[i + 1] - self.cutoff[i], bias=False)
            self.tail.append(m)

    def clear_combined_weight(self):
        '''remove combined weight'''
        self.w_head = None
        self.w_tail = None
        self.w_tail_left = None
        self.w_tail_right = None

    def combine_weight(self, tail_svd=True, need_qat=False):
        '''combine_weight for inference v2'''
        w_tails = []
        tail_left_list = []
        tail_right_list = []
        # pylint:disable=consider-using-enumerate
        for i in range(len(self.tail)):
            if isinstance(self.tail[i], nn.Sequential):
                assert (not self.training) or self.dropout == 0
                w_tails.append(torch.matmul(self.tail[i][2].weight, self.tail[i][0].weight))
                if tail_svd:
                    tail_left_list.append(self.tail[i][0].weight.t())
                    tail_right_list.append(self.tail[i][2].weight.t())
            else:
                w_tails.append(self.tail[i].weight)
        self.w_head = self.head.weight.data.clone()  # [head_dim_aligned, H]
        self.w_tail = torch.cat(w_tails, dim=0)  # [TG_NUM * VT, H]
        if tail_svd and isinstance(self.tail[i], nn.Sequential):
            self.w_tail_left = torch.cat(tail_left_list, dim=1)
            self.w_tail_right = torch.cat(tail_right_list, dim=0)
        if need_qat:
            w_tails = []
            for i in range(len(self.tail)):
                w_tails.append(
                    torch.matmul(
                        torch.fake_quantize_per_tensor_affine(
                            self.tail[i][2].weight.contiguous(),
                            self.w_tail_right.data.abs().max().item() / 127.0,
                            0,
                            -127,
                            127,
                        ),
                        torch.fake_quantize_per_tensor_affine(
                            self.tail[i][0].weight.contiguous(),
                            self.w_tail_left.data.abs().max().item() / 127.0,
                            0,
                            -127,
                            127,
                        ),
                    )
                )
            self.w_tail_q = torch.cat(w_tails, dim=0)

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
        compatible for load previous checkpoint.
        '''
        if not hasattr(self.head, 'weight'):
            return
        saved_weight = state_dict.get(prefix + 'head.weight', None)
        in_dim = self.head.weight.shape[1]
        if saved_weight is not None and saved_weight.shape[0] != self.head_dim_aligned:
            new_weight = saved_weight.new_empty(self.head_dim_aligned, in_dim)
            new_weight[: self.head_dim] = saved_weight
            new_weight[self.head_dim :] = 0
            state_dict[prefix + 'head.weight'] = new_weight

    def compatible_save_hook(self, _model, destination, prefix, _local_metadata):
        '''
        compatible for save previous checkout
        '''
        if PantherLinear is not None and isinstance(self.head, PantherLinear):
            h_w = destination.get(prefix + 'head._torch_module.weight')
            h_w = h_w[: self.head_dim, :]
            destination[prefix + 'head._torch_module.weight'] = h_w
        else:
            h_w = destination.get(prefix + 'head.weight')
            h_w = h_w[: self.head_dim, :]
            destination[prefix + 'head.weight'] = h_w
        return destination

    def cal_final_prob(self, head_lprob, select_group0_lprob, target_lengths):
        '''adaptive softmax cal final prob py impl'''
        t_num = head_lprob.size(1)
        blank_lprob = head_lprob[:, :, 0:1]  # [sumU, T, 1]
        concat_lprob = torch.cat([blank_lprob, select_group0_lprob], dim=2)  # [sumU, T, 2]
        concat_lprob = concat_lprob.transpose(0, 1).contiguous()  # [T, sumU, 2]
        # convert concat lattice to padding, [B, T, U, 2]
        final_lprob = torch.zeros(
            [target_lengths.shape[0], t_num, target_lengths.max().item(), 2],
            dtype=head_lprob.dtype,
            device=head_lprob.device,
        )
        suml = 0
        for i, tl in enumerate(target_lengths):
            final_lprob[i, :, :tl] = concat_lprob[:, suml : suml + tl]
            suml += tl
        # final_lprob[:, :, -1] = 0
        return final_lprob

    def forward(self, input_data, target, _input_lengths, target_lengths, **kwargs):
        '''
        Args:
            input_data: [sum(U+1), T, H] or [B, T, U+1, H]
                U+1 is rnnt format
            target: [B, U], int64
            target_lengths: [B], int64
                each value is tgt_len

        Return:
            Tensor, [B, T, U+1, 2] or [B, T, U+1, V]
        '''
        # pylint:disable=too-many-branches
        # pylint:disable=too-many-locals
        if hasattr(self, 'w_tail_q'):
            delattr(self, 'w_tail_q')

        if not kwargs.get('rnnt_gather', True):
            jointer_hidden = input_data.size(-1)
            lprob = self.get_log_prob(input_data.view(-1, jointer_hidden))
            return lprob.view(*input_data.shape[:-1], -1)  # as input shape

        # rnnt gather case, input data shape [sum(U+1), T, H] or [B, T, U+1, H]
        assert input_data.dim() in (3, 4)  # concat_u or not
        adaptive_tgt_indices = kwargs['adaptive_tgt_indices']

        need_qat = (
            SlimModule is not None
            and isinstance(self.head, SlimModule)
            and 'QATQuantize' in self.head.pre_ops
        )
        if fused_adaptive_softmax is not None and not need_qat:
            tail_weights = list(self.tail.parameters())
            tail_svd = len(tail_weights) != len(self.tail)
            return fused_adaptive_softmax(
                input_data,
                target,
                target_lengths,
                adaptive_tgt_indices,
                self.head.weight,
                tail_weights,
                head_dim=self.head_dim,
                tail_svd=tail_svd,
                blank_setto_zero=kwargs.get('blank_setto_zero', False),
            )

        head_input, tail_inputs = input_data, []
        for tgt_idx in adaptive_tgt_indices[1:]:
            if tgt_idx is None:  # not target in this tail group
                tail_inputs.append(torch.Tensor())
            elif input_data.dim() == 3:  # concat_u
                bu_idx = tgt_idx[:, 2]
                tail_inputs.append(input_data[bu_idx])
            else:  # not concat_u
                b_idx = tgt_idx[:, 0]
                u_idx = tgt_idx[:, 1]
                tail_inputs.append(input_data[b_idx, :, u_idx])

        # get head log prob
        head_logits = self.head(head_input).view(-1, self.head_dim_aligned)
        with torch.no_grad():
            head_logits[:, self.head_dim :] = float('-inf')
            if kwargs.get('blank_setto_zero', False):
                head_logits[:, 0] = float('-inf')
        if head_logits.size(0) * head_logits.size(1) >= 2**31:
            logging.warning('head_logits\'s size is too big for log_softmax')
        head_lprob = head_logits.log_softmax(-1, dtype=torch.float32)
        head_lprob = head_lprob.view(*input_data.shape[:-1], -1)

        # get tail log probs
        if need_qat:
            self.combine_weight()
        tail_lprobs = []
        for i, tail_input in enumerate(tail_inputs):
            if tail_input.numel() <= 0:
                tail_lprobs.append(torch.Tensor())  # fake tensor take a place.
            else:
                if need_qat:
                    tail_logit = F.linear(
                        torch.fake_quantize_per_tensor_affine(
                            tail_input.contiguous(),
                            self.head.pre_ops['QATQuantize']
                            .input_pre_process_modules.get_scale()
                            .item()
                            / 127.0,
                            0,
                            -127,
                            127,
                        ),
                        torch.matmul(
                            torch.fake_quantize_per_tensor_affine(
                                self.tail[i][2].weight.contiguous(),
                                self.w_tail_right.data.abs().max().item() / 127.0,
                                0,
                                -127,
                                127,
                            ),
                            torch.fake_quantize_per_tensor_affine(
                                self.tail[i][0].weight.contiguous(),
                                self.w_tail_left.data.abs().max().item() / 127.0,
                                0,
                                -127,
                                127,
                            ),
                        ),
                        bias=None,
                    )
                else:
                    tail_logit = self.tail[i](tail_input)
                tail_lprob = tail_logit.log_softmax(-1, dtype=torch.float32)
                tail_lprobs.append(tail_lprob)

        seq_len = input_data.size(1)
        target = torch.cat([target, torch.zeros([target.shape[0], 1]).to(target)], dim=1)
        if input_data.dim() == 3:  # concat_u
            target = torch.cat(
                [tgt[: tl + 1] for tgt, tl in zip(target, target_lengths.cpu())], dim=0
            )
        else:
            head_lprob = head_lprob.permute(0, 2, 1, 3).reshape(-1, seq_len, self.head_dim_aligned)
        new_target, target_idxs = self.adapt_target(target)
        select_group0_lprob = head_lprob.gather(
            2, new_target[0].view(-1, 1, 1).repeat(1, seq_len, 1)
        )  # [sum(U+1), T, 1] or [B*(U+1), T, 1]
        # cutoff group
        for tail_lprob, new_t, t_idx in zip(tail_lprobs, new_target[1:], target_idxs):
            if t_idx is None:
                continue
            select_tail_lprob = tail_lprob.gather(2, new_t.view(-1, 1, 1).repeat(1, seq_len, 1))
            select_group0_lprob.scatter_add_(
                0, t_idx.view(-1, 1, 1).repeat(1, seq_len, 1), select_tail_lprob
            )
        if input_data.dim() == 3:  # concat_u
            return self.cal_final_prob(
                head_lprob[:, :, : self.head_dim], select_group0_lprob, target_lengths + 1
            )
        bsz, _, u1, _ = input_data.shape
        blank_lprob = head_lprob[:, :, 0].view(bsz, u1, seq_len, 1)
        select_group0_lprob = select_group0_lprob.view(bsz, u1, seq_len, 1)
        return torch.cat([blank_lprob, select_group0_lprob], dim=3).permute(
            0, 2, 1, 3
        )  # [B, T, U+1, 2]

    def adapt_target(self, target):
        '''map target to each group.'''
        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.any():
                target_idxs.append(mask.nonzero().squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)

        return new_target, target_idxs

    def forward_one_step(self, input_data, target):
        '''
        calculate adaptive softmax for nbest alignment path forward prob
        Args:
            input_data: jointer output (B, dim)
            target: output (B,)
        return:
            prob of each target: (B,)
        '''
        new_target, target_idxs = self.adapt_target(target)  # [(B*T,)],

        head_logits = self.head(input_data)  # (B, headDim)
        with torch.no_grad():
            head_logits[:, self.head_dim :] = float('-inf')
        head_lprob = head_logits.log_softmax(dim=1, dtype=torch.float32)  # [B, V0]

        # final (sumU, T, 1)
        select_group0_lprob = head_lprob.gather(1, new_target[0].view(-1, 1))  # [B, 1]

        for i, t in enumerate(target_idxs):
            if t is None:
                continue
            tail_logits = self.tail[i](input_data.index_select(0, t))
            tail_lprob = tail_logits.log_softmax(dim=1, dtype=torch.float32)  # (n, V)
            select_tail_lprob = tail_lprob.gather(1, new_target[i + 1].view(-1, 1))
            select_group0_lprob.scatter_add_(0, t.view(-1, 1), select_tail_lprob)

        concat_lprob = select_group0_lprob  # [B, 1]

        return concat_lprob

    def get_log_prob(self, input_data, rnnt_temperature=1.0, **kwargs):
        '''
        fast version.

        Args:
            input_data: [B, H]
        '''
        need_qat = (
            SlimModule is not None
            and isinstance(self.head, SlimModule)
            and 'QATQuantize' in self.head.pre_ops
        )
        if self.w_head is None or (need_qat and not hasattr(self, 'w_tail_q')):
            self.combine_weight(
                tail_svd=False or need_qat,
                need_qat=need_qat,
            )
        batch_size, bins = input_data.shape[0], len(self.cutoff) - 1
        vocab_head = self.head_dim - bins
        if need_qat:
            pred_head = F.linear(
                torch.fake_quantize_per_tensor_affine(
                    input_data.contiguous(),
                    self.head.pre_ops['QATQuantize'].input_pre_process_modules.get_scale().item()
                    / 127.0,
                    0,
                    -127,
                    127,
                ),
                torch.fake_quantize_per_tensor_affine(
                    self.w_head.contiguous(),
                    self.w_head.data.abs().max().item() / 127.0,
                    0,
                    -127,
                    127,
                ),
                bias=None,
            )
        else:
            pred_head = F.linear(input_data, self.w_head, bias=None)
        if rnnt_temperature != 1:
            pred_head = pred_head * rnnt_temperature
        with torch.no_grad():
            if kwargs.get('blank_setto_zero', False):
                pred_head[:, 0] = float('-inf')
        pred_head = pred_head[:, : self.head_dim].log_softmax(-1, dtype=torch.float32)
        dims_per_bin = self.w_tail.shape[0] // bins
        if need_qat:
            pred_tail = F.linear(
                torch.fake_quantize_per_tensor_affine(
                    input_data.contiguous(),
                    self.head.pre_ops['QATQuantize'].input_pre_process_modules.get_scale().item()
                    / 127.0,
                    0,
                    -127,
                    127,
                ),
                self.w_tail_q,
                bias=None,
            )
        else:
            pred_tail = F.linear(input_data, self.w_tail, bias=None)
        pred_tail = pred_tail.view(batch_size, -1, dims_per_bin).log_softmax(
            -1, dtype=torch.float32
        )
        pred_tail = pred_tail + pred_head[:, vocab_head : self.head_dim].unsqueeze(2)
        return torch.cat((pred_head[:, :vocab_head], pred_tail.view(batch_size, -1)), dim=-1)

    @torch.no_grad()
    def get_log_prob2(
        self, input_data, adapt_softmax_thresh=1.0, only_head=False, rnnt_temperature=1.0, **_kwargs
    ):
        '''
        if only_head or head_accumulate_prob > adapt_softmax_thresh: average tail prob
        '''
        voc_size = self.cutoff[-1]
        bsz = input_data.size(0)

        head_y = self.head(input_data) * rnnt_temperature
        head_y = head_y[:, : self.head_dim]
        log_probs = head_y.new_zeros(bsz, voc_size).float()

        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = F.log_softmax(head_y.float(), dim=1)

        # if head prob >= threshold: everage tail prob
        head_probs = log_probs[:, : self.cutoff[0]]
        log_thresh = np.log(adapt_softmax_thresh)
        batch_mask = head_probs.logsumexp(-1) < log_thresh
        # + (head_probs.max(-1)[0] < np.log(0.8) + log_thresh)
        if batch_mask.sum().item() == 0:
            only_head = True
        else:
            input_data = input_data[batch_mask]

        tail_priors = log_probs[:, self.cutoff[0] : head_sz].clone()
        for i, tail in enumerate(self.tail):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]
            log_probs[:, start:end] = tail_priors[:, i : i + 1] + np.log(1.0 / (end - start))
            if not only_head:
                tail_out = tail(input_data)
                log_probs[batch_mask, start:end] = F.log_softmax(tail_out.float(), dim=1).add_(
                    tail_priors[batch_mask, i : i + 1]
                )

        return log_probs

    @torch.no_grad()
    def jit_forward(self, input_data, rnnt_temperature=1.0):
        '''to export onnx'''
        bins = len(self.cutoff) - 1
        return PantherAdaptiveSoftmaxFunc.apply(
            input_data, self.w_head, self.w_tail, bins, rnnt_temperature
        )

    @torch.no_grad()
    def jit_forward_non_combined(self, input_data, rnnt_temperature=1.0):
        '''to export onnx'''
        bins = len(self.cutoff) - 1
        return PantherAdaptiveSoftmaxFunc.apply(
            input_data,
            self.w_head,
            None,
            bins,
            rnnt_temperature,
            self.w_tail_left,
            self.w_tail_right,
            self.tail_hidden_dim,  # hidden dim of the first tail
            self.tail_hidden_dim_decrease,
        )  # hidden dim of each tail decrease by 1

    def jit_init(self):
        '''do init for export onnx'''
        self.w_head = (
            self.w_head[: self.head_dim, :].t().contiguous().detach().requires_grad_(False)
        )
        self.w_tail = self.w_tail.t().detach().requires_grad_(False)  # better to be contiguous
        if self.w_tail_left is not None:
            self.w_tail_left = self.w_tail_left.detach().requires_grad_(False)
        if self.w_tail_right is not None:
            self.w_tail_right = self.w_tail_right.detach().requires_grad_(False)
