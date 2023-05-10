''' rnnt criterions '''
# pylint:disable=too-many-lines,unused-import
from collections import OrderedDict, Counter
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from core.models.layers.adaptive_softmax import RNNTAdaptiveSoftmax
from core.models.utils import kl_divloss_rnnt
from core.extensions import (
    ext_ctc_force_alignment,
    rnnt_loss,
    ctc_loss as d_ctc_loss,
    rnnt_force_alignment,
)
from .criterion import Xentropy
from .las_criterion import LasCE, CTCLasCE


def log_sum(tensor_a, tensor_b):
    '''log_sum'''
    return torch.logsumexp(torch.tensor([tensor_a, tensor_b]), 0)


class RnntCE(nn.Module):
    """simple CE for rnnt, which has a layer linear layer and log softmax layer

    .. note::
        这是rnnt_criterion 的基类所有的rnnt_criterion 都会继承它
    """

    def __init__(self, args):
        """init

        Args:
            - args: solution config
        """

        super().__init__()
        self.args = args
        self.log_softmax_fc = nn.Linear(args.jointer_hidden_size, args.tgt_vocab_size)
        self.update_steps = 0
        self.target_lens = None
        self.eos_alignment = None
        self.do_non_pronounce_token_penalty = args.get('do_non_pronounce_token_penalty', False)
        if self.do_non_pronounce_token_penalty:
            self.init_non_pronounce_token_penalty_weights()
        self.eos_shift_frame = args.get('eos_shift_frame', 0)
        self.early_penalty = args.get('eos_early_penalty', 0.1)
        self.late_penalty = args.get('eos_late_penalty', 10.0)
        self.late_buffer = args.get('eos_late_buffer', 5)
        self.ilmt_weight = args.get('ilmt_weight', 0.0)
        self.skip_ctc_eos = args.get('skip_ctc_eos', False)
        self.training_without_eos = args.get('training_without_eos', False)
        self.use_rnnt_alignment = args.get('use_rnnt_alignment', False)
        self.rnnt_alignment_start_iters = args.get('rnnt_alignment_start_iters', 0)
        self.transducer_weight = args.get('transducer_weight', 1.0)
        self.ctc_weight = args.get('mtl_weight', 1.0)
        self.las_criterion = None
        self.use_distiller = args.get('use_distiller', False)
        self.joint_ctc_weight = args.get('joint_ctc_weight', 1.0)
        self.joint_las_weight = args.get('joint_las_weight', 0.0)
        self.lid_criterion = None
        self.lid_weight = args.get('lid_weight', 0.0)
        self.lat_weight = args.get('lat_weight', 0.0)  # language aware training weight
        if self.joint_las_weight > 0.0:
            args = copy.deepcopy(args)
            args.update(args.las_args)
            self.eos = self.args.tgt_dict.eos()
            self.pad = self.args.tgt_dict.pad()
            self.las_criterion = eval(args.get("las_criterion_type", "CTCLasCE"))(args)
        if self.lid_weight > 0.0:
            self.lid_criterion = Xentropy(args)

    def get_log_prob(self, jointer_hidden, rnnt_temperature=1.0, **kwargs):
        """get the log probs

        Args:

            - jointer_hidden: [B, T, U, N], the output of jointer

        Return:

            - the log prob with shape [B, T, U, V]
        """

        logits = self.log_softmax_fc(jointer_hidden)
        if kwargs.get('blank_setto_zero', False):
            logits[:, :, :, 0] = -1e8
        return F.log_softmax(logits.float() * rnnt_temperature, dim=-1)

    @staticmethod
    def get_log_prob_noblk(jointer_hidden):
        """get the log probs del blk score

        Args:

            - jointer_hidden: [B, T, U, N], the output of jointer

        Return:

            - the log prob with shape [B, T, U, V]

        """

        jointer_hidden[:, 0] = float('-inf')
        return jointer_hidden.log_softmax(dim=-1, dtype=torch.float32)

    # pylint: disable=invalid-name
    def init_non_pronounce_token_penalty_weights(self):
        '''
        init non pronounce token penalty weights
        '''
        self.non_pronounce_token_idxs = set(eval(self.args.non_pronounce_token_idxs))
        if self.args.use_constant_penalty:
            self.non_pronounce_token_penalty_weights = self.args.penalty_frame_num * [
                self.args.penalty_weight
            ]
        else:
            self.non_pronounce_token_penalty_weights = [
                (i + 1) * self.args.penalty_weight for i in range(self.args.penalty_frame_num)
            ]
        self.non_pronounce_token_penalty_weights = torch.tensor(
            self.non_pronounce_token_penalty_weights
        ).cuda()

    def non_pronounce_token_penalty(self, log_probs, target, input_lengths, target_lengths):
        '''
        do non pronounce token penalty
        '''
        bsz = log_probs.shape[0]
        target = target.tolist()
        input_lengths = input_lengths.tolist()
        target_lengths = target_lengths.tolist()
        for i in range(bsz):
            last_token_idx = target[i][target_lengths[i] - 1]
            if last_token_idx in self.non_pronounce_token_idxs:
                log_prob_beg = max(0, input_lengths[i] - self.args.penalty_frame_num)
                log_prob_end = input_lengths[i]
                penalty_weight_len = min(self.args.penalty_frame_num, input_lengths[i])
                token_emit_u_idx = target_lengths[i] - 1
                token_idx = last_token_idx
                if log_probs.size(-1) == 2:
                    token_idx = 1
                log_probs[
                    i, log_prob_beg:log_prob_end, token_emit_u_idx, token_idx
                ] -= self.non_pronounce_token_penalty_weights[0:penalty_weight_len]
        return log_probs

    def do_eos_penalty(self, log_probs, eos_alignment, max_input_lengths, target_lengths):
        '''
        do eos penalty
        '''
        bsz = log_probs.size(0)
        if self.eos_shift_frame != 0:
            eos_alignment[eos_alignment > 0] += self.eos_shift_frame
            eos_alignment[eos_alignment < 0] = 0
        time_range = (
            torch.arange(
                1,  # start
                1 + max_input_lengths,  # end (not included)
                device=log_probs.device,
                dtype=log_probs.dtype,
            )
            .unsqueeze(0)
            .repeat(bsz, 1)
        )
        eos_penalty = self.early_penalty * torch.clamp_min(
            (eos_alignment - time_range), 0
        ) + self.late_penalty * torch.clamp_min((time_range - eos_alignment - self.late_buffer), 0)
        eos_penalty = eos_penalty.masked_fill(eos_alignment.le(0), 0).float()
        for bid in range(bsz):
            log_probs[bid, :max_input_lengths, target_lengths[bid] - 1, 1] -= eos_penalty[bid]
        return log_probs

    def get_rnnt_loss(self, jointer_hidden, target, input_lengths, target_lengths, **_kwargs):
        """get the rnnt loss

        Args:
            - jointer_hidden: [B, T, U, N], the output of jointer
            - target: [B, U], the target
            - input_lengths: [B], the length of encoder output
            - target_lengths: [B], the length of target

        Return:

            - a Tensor: [B], the sum rnnt loss of each target
        """

        log_probs = self.get_log_prob(jointer_hidden)
        if self.do_non_pronounce_token_penalty:
            log_probs = self.non_pronounce_token_penalty(
                log_probs, target, input_lengths, target_lengths
            )
        if self.eos_alignment is not None and self.training:
            max_input_lengths = input_lengths.max().item()
            log_probs = self.do_eos_penalty(
                log_probs, self.eos_alignment.unsqueeze(1), max_input_lengths, target_lengths
            )
        return log_probs, rnnt_loss(
            log_probs,
            target.int(),
            input_lengths,
            target_lengths,
            average_frames=False,
            reduction=None,
            gather=True,
            blank=0,
        )

    def get_ilmt_loss(self, ilmt_jointer_out, target, target_mask, **_kwargs):
        '''ilmt_jointer_out: [B, 1, U, V]'''
        ilmt_prob = self.get_log_prob(ilmt_jointer_out, blank_setto_zero=True)
        index = target.clone().unsqueeze(2)
        log_probs = ilmt_prob[:, 0, :-1, :].gather(dim=2, index=index)[:, :, 0]  # [B, U]
        return (-log_probs.masked_fill(~target_mask.bool(), 0)).sum() / (target_mask.sum() + 1e-8)

    @staticmethod
    def scale_backward_loss(forward_loss, input_dict):
        '''scale the loss if the loss_scales of each data-item are provided'''
        item_loss_scales = input_dict.get('item_loss_scales')
        if item_loss_scales is None:
            backward_loss = forward_loss.mean(0)
        else:
            backward_loss = (forward_loss * item_loss_scales).mean(0)
        return backward_loss

    def forward(self, jointer_hidden, input_dict, mtl_logits=None, mtl_type=None):
        """forward

        Args:

            - jointer_hidden: [B, T, U, N], the output of jointer
            - input_dict: input dictionary
            - acoustic_out: [B, T, N], the output of encoder
            - predictor_module: the module of predictor
            - jointer_module: the module of jointer
            - mtl_logits: [B, T, V], the logits of multi-task learning
            - mtl_type: the type of multi-task learning, now only support ctc

        Return:

            - forward_out: a dictionary containing loss, cer and etc.
        """
        # pylint:disable=too-many-locals,too-many-branches
        if self.training:
            self.update_steps += 1
        adaptive_tgt_indices = input_dict['adaptive_tgt_indices']
        # mask and length
        src_mask = input_dict['src_mask']
        target = input_dict['char']
        target_mask = input_dict['char_mask']
        input_lengths = (input_dict['backbone_mask'].sum(dim=1)).int()
        target_lengths = (target_mask.sum(dim=1)).int()
        self.target_lens = input_dict['target_lengths']
        self.eos_alignment = input_dict.get('eos', None)
        # rnnt loss
        log_probs, nll_loss = self.get_rnnt_loss(
            jointer_hidden,
            target,
            input_lengths,
            target_lengths,
            adaptive_tgt_indices=adaptive_tgt_indices,
        )
        backward_nll_loss = self.scale_backward_loss(nll_loss, input_dict)
        nll_loss = nll_loss.sum(0) / target_lengths.sum(0)
        # mtl loss, e.g ctc
        if mtl_logits is not None and mtl_type == 'ctc' and self.ctc_weight > 0.0:
            ctc_lprobs = mtl_logits.log_softmax(dim=-1, dtype=torch.float32)
            if "mtl_mask" in input_dict:
                input_lengths = (input_dict['mtl_mask'].sum(dim=1)).int()
            ctc_target_lengths = target_lengths
            ctc_target = target.int()
            if self.args.use_eos and self.skip_ctc_eos:
                ctc_target[ctc_target == 2] = 0
                ctc_target_lengths = target_lengths - 1
            ctc_loss = d_ctc_loss(
                ctc_lprobs.transpose(0, 1).contiguous(),
                ctc_target,
                input_lengths,
                ctc_target_lengths,
                blank=0,
                reduction='none',
                zero_infinity=True,
            )
            backward_ctc_loss = (
                self.scale_backward_loss(ctc_loss / target_lengths, input_dict)
                * self.joint_ctc_weight
            )
            ctc_loss = ctc_loss.sum(0) / target_lengths.sum(0) * self.joint_ctc_weight
        else:
            ctc_loss = 0.0
            backward_ctc_loss = 0.0

        # language aware training
        if self.lat_weight > 0.0:
            lang_aware_ctc_loss = 0.0
            backward_lang_aware_ctc_loss = 0.0
            lang_ctc_lprobs = {}
            lang_target = {}
            lang_target_lengths = {}
            lang_ctc_loss = {}

            for lang in input_dict["lat_logits"]:
                lang_ctc_lprobs[lang] = input_dict["lat_logits"][lang].log_softmax(
                    dim=-1, dtype=torch.float32
                )
                lang_target[lang] = input_dict[lang + '_char']
                lang_target_lengths[lang] = (input_dict[lang + '_char_mask'].sum(dim=1)).int()
                lang_ctc_loss[lang] = d_ctc_loss(
                    lang_ctc_lprobs[lang].transpose(0, 1).contiguous(),
                    lang_target[lang].int(),
                    input_lengths,
                    lang_target_lengths[lang],
                    blank=0,
                    reduction='none',
                    zero_infinity=True,
                )
            backward_lang_aware_ctc_loss = sum(
                self.scale_backward_loss(loss / lang_target_lengths[lang], input_dict)
                for lang, loss in lang_ctc_loss.items()
            )
            lang_aware_ctc_loss = sum(
                loss.sum(0) / lang_target_lengths[lang].sum(0)
                for lang, loss in lang_ctc_loss.items()
            )

            backward_lang_aware_ctc_loss = self.lat_weight * backward_lang_aware_ctc_loss
            lang_aware_ctc_loss = self.lat_weight * lang_aware_ctc_loss

        # ILMT
        ilmt_loss = 0.0
        if self.ilmt_weight > 0.0 and "ilmt_jointer_out" in input_dict:
            ilmt_loss = self.ilmt_weight * self.get_ilmt_loss(
                input_dict["ilmt_jointer_out"],
                target,
                target_mask,
                adaptive_tgt_indices=adaptive_tgt_indices,
            )
        # Las loss
        las_loss = 0.0
        las_acc = 0.0
        if self.joint_las_weight > 0.0 and "las_logits" in input_dict:
            # target need padding eos
            _eos = torch.tensor(
                [self.eos], dtype=torch.long, requires_grad=False, device=target.device
            )
            ys = [y[y != 0] for y in target]
            ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
            target_out = pad_sequence(ys_out, True, self.pad)
            target_mask_out = F.pad(target_mask, (1, 0), value=1)
            las_loss = self.las_criterion(
                input_dict["las_logits"], src_mask, target_out, target_mask_out
            )
            las_acc = las_loss['acc']
            las_loss = las_loss['loss']
        # lid loss
        lid_loss = 0.0
        if self.lid_weight > 0.0 and 'lid_logits' in input_dict:
            lid_target = input_dict['lang']
            lid_logits = input_dict['lid_logits']
            lid_mask = torch.ones(lid_target.shape[0]).int().cuda()
            if len(lid_logits.shape) > 2:  # B*T*C
                lid_target = lid_target.unsqueeze(1).repeat(1, lid_logits.shape[1])
                lid_mask = src_mask
            lid_loss = self.lid_criterion(input_dict['lid_logits'], src_mask, lid_target, lid_mask)
            lid_loss = lid_loss['loss'] * self.lid_weight

        # stat related
        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()
        forward_out = OrderedDict()
        if self.use_distiller:
            distiller_config = self.args.get('distiller_config')
            if 'logits' in distiller_config:
                method_config = distiller_config['logits']
                method_name = method_config['loss_func']['name']
                if method_name == 'RnntKL':
                    forward_out['log_probs'] = log_probs
                elif method_name == 'LasKL' and "las_logits" in input_dict:
                    forward_out['las_logits'] = input_dict["las_logits"]
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = (
            self.transducer_weight * backward_nll_loss
            + self.ctc_weight * backward_ctc_loss
            + ilmt_loss
            + las_loss * self.joint_las_weight
        )
        forward_out['loss'] = (
            self.transducer_weight * nll_loss
            + self.ctc_weight * ctc_loss
            + ilmt_loss
            + las_loss * self.joint_las_weight
        )
        if self.args.get('moe_args', None) is not None and 'aux_loss' in input_dict:
            forward_out['backward_loss'] += input_dict['aux_loss']
            forward_out['loss'] += input_dict['aux_loss']
        if lid_loss > 0.0:
            forward_out['backward_loss'] += lid_loss
            forward_out['loss'] += lid_loss
        if self.lat_weight > 0.0:
            forward_out['backward_loss'] += backward_lang_aware_ctc_loss
            forward_out['loss'] += lang_aware_ctc_loss
        forward_out['nll_loss'] = nll_loss
        forward_out['ctc_loss'] = ctc_loss
        forward_out['las_loss'] = las_loss
        forward_out['acc'] = las_acc
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size
        return forward_out

    def combine_weight(self):
        '''combine weight for log_softmax_fc'''

    def clear_combined_weight(self):
        '''clear combined weight for log_softmax_fc'''

    @staticmethod
    @torch.no_grad()
    def edit_distance(ref, hyp):
        '''edit disttance between ref and hyp'''
        # pylint:disable=too-many-branches
        assert isinstance(ref, list) and isinstance(hyp, list)
        dist = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.uint32)
        for i in range(len(ref) + 1):
            for j in range(len(hyp) + 1):
                if i == 0:
                    dist[0][j] = j
                elif j == 0:
                    dist[i][0] = i
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dist[i][j] = dist[i - 1][j - 1]
                else:
                    substitute = dist[i - 1][j - 1] + 1
                    insert = dist[i][j - 1] + 1
                    delete = dist[i - 1][j] + 1
                    dist[i][j] = min(substitute, insert, delete)
        i = len(ref)
        j = len(hyp)
        steps = []
        while True:
            if i == 0 and j == 0:
                break
            if i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] and ref[i - 1] == hyp[j - 1]:
                steps.append('corr')
                i, j = i - 1, j - 1
            elif i >= 1 and j >= 1 and dist[i][j] == dist[i - 1][j - 1] + 1:
                assert ref[i - 1] != hyp[j - 1]
                steps.append('sub')
                i, j = i - 1, j - 1
            elif j >= 1 and dist[i][j] == dist[i][j - 1] + 1:
                steps.append('ins')
                j = j - 1
            else:
                assert i >= 1 and dist[i][j] == dist[i - 1][j] + 1
                steps.append('del')
                i = i - 1
        steps = steps[::-1]

        counter = Counter({'words': len(ref), 'corr': 0, 'sub': 0, 'ins': 0, 'del': 0})
        counter.update(steps)
        return dist, steps, counter


class RnntAdaptiveCE(RnntCE):
    """RNN-T Adatptive CE Loss

    .. note::
        mtl_head 决定了ctc_logits,ce_logits,teacher_ctc_logits的生成

    Example::

        criterion_type='RnntAdaptiveCE',
        cer_update_freq=200,
        label_smooth_factor=0.1,
        reorder_dict_by_freq=True,
        mtl_type='ctc',
        mtl_head='HybridCEHead',
        backbone_pool_type='Conv1dTimeReduce',
    """

    def __init__(self, args):
        super().__init__(args)
        cutoff = [args.adaptive_head_size] + [
            item * args.adaptive_tail_size + args.adaptive_head_size
            for item in range(1, args.adaptive_tail_groups + 1)
        ]
        dropout = 0.0
        self.fast_emit = args.get('fast_emit', None)
        self.self_align = args.get('self_align', None)
        self.self_align_frame_shift = args.get('self_align_frame_shift', 0)
        adaptive_tail_small = args.get('adaptive_tail_small', False)
        self.log_softmax_fc = RNNTAdaptiveSoftmax(
            args.tgt_vocab_size,
            args.jointer_hidden_size,
            cutoff,
            dropout,
            concate_U=args.concate_U,
            adaptive_tail_small=adaptive_tail_small,
        )

    def get_log_prob(self, jointer_hidden, **kwargs):
        return self.log_softmax_fc.get_log_prob(jointer_hidden, **kwargs)

    def combine_weight(self):
        '''combine weight for log_softmax_fc'''
        if self.log_softmax_fc is not None:
            self.log_softmax_fc.combine_weight()

    def clear_combined_weight(self):
        '''clear combined weight for log_softmax_fc'''
        if self.log_softmax_fc is not None:
            self.log_softmax_fc.clear_combined_weight()

    def get_self_align_loss(self, log_probs, input_lengths, target_lengths, fused=True):
        '''calculate self align loss'''
        if fused:
            outs = self.rnnt_force_alignment(
                log_probs,
                input_lengths,
                target_lengths + 1,
                u_path_mask=True,
                frame_shift=self.self_align_frame_shift,
            )
            shift_u_path_mask = outs[1]
        else:
            bsz, fl, tl, _ = log_probs.shape
            u_path_mask = torch.zeros(bsz, fl, tl).int().cuda()
            shift_u_path_mask = torch.zeros(bsz, fl, tl).int().cuda()
            u_path_index = rnnt_force_alignment(log_probs, input_lengths, target_lengths + 1)
            u_path_index = u_path_index[:, 1:]
            for b in range(bsz):
                for u in range(target_lengths[b]):
                    if u_path_index[b, u] > 0:
                        u_path_mask[b, u_path_index[b, u], u] = 1
            if self.self_align_frame_shift > 0:
                shift_u_path_mask[:, : fl - self.self_align_frame_shift, :] = u_path_mask[
                    :, self.self_align_frame_shift :, :
                ]
            elif self.self_align_frame_shift < 0:
                shift_u_path_mask[:, -self.self_align_frame_shift :, :] = u_path_mask[
                    :, : fl + self.self_align_frame_shift, :
                ]
            else:
                shift_u_path_mask = u_path_mask
        distill_loss = -(log_probs[:, :, :, 1] * shift_u_path_mask).sum() / shift_u_path_mask.sum()
        return distill_loss

    def get_rnnt_loss(self, jointer_hidden, target, input_lengths, target_lengths, **kwargs):
        '''calculate rnnt loss'''
        log_probs = self.log_softmax_fc(
            jointer_hidden, target, input_lengths, target_lengths, **kwargs
        )
        if self.do_non_pronounce_token_penalty:
            log_probs = self.non_pronounce_token_penalty(
                log_probs, target, input_lengths, target_lengths
            )
        if self.use_rnnt_alignment:
            self.get_eos_alignment_from_rnnt(log_probs, input_lengths, target_lengths)
        if self.eos_alignment is not None and self.training:
            max_input_lengths = input_lengths.max().item()
            log_probs = self.do_eos_penalty(
                log_probs, self.eos_alignment.unsqueeze(1), max_input_lengths, target_lengths
            )
        if self.fast_emit is not None and self.training and log_probs.requires_grad:
            log_probs.register_hook(
                lambda grad: torch.cat(
                    (grad[:, :, :, 0:1], grad[:, :, :, 1:2] * float(self.fast_emit)), -1
                )
            )
        nll_loss = rnnt_loss(
            log_probs,
            target.int(),
            input_lengths,
            target_lengths,
            average_frames=False,
            reduction=None,
            gather=False,
            blank=-1,
        )
        if self.eos_alignment is not None and self.training_without_eos:
            new_target = target.int().clone()
            new_target[new_target == 2] = 0
            nll_loss_without_eos = rnnt_loss(
                log_probs,
                new_target,
                input_lengths - 1,
                target_lengths - 1,
                average_frames=False,
                reduction=None,
                gather=False,
                blank=-1,
            )
            if self.training:
                nll_loss += nll_loss_without_eos
                nll_loss /= 2.0
            else:
                nll_loss = nll_loss_without_eos
        if self.self_align is not None:
            nll_loss += float(self.self_align) * self.get_self_align_loss(
                log_probs, input_lengths, target_lengths
            )
        return log_probs, nll_loss

    def get_ilmt_loss(self, ilmt_jointer_out, target, target_mask, **kwargs):
        input_lengths = torch.ones_like(target_mask[:, 0]).int()  # (B)
        ilmt_prob = self.log_softmax_fc(
            ilmt_jointer_out,
            target,
            input_lengths,
            target_mask.sum(1).int(),
            blank_setto_zero=True,
            **kwargs,
        )  # (B, 1, U, 2)
        log_probs = ilmt_prob[:, 0, :-1, 1].contiguous()
        return (-log_probs.masked_fill(~target_mask.bool(), 0)).sum() / (target_mask.sum() + 1e-8)

    @staticmethod
    def rnnt_force_alignment(
        rnnt_lprobs, input_lengths, target_lengths, u_path_mask=False, frame_shift=0
    ):
        """do rnnt force alignment.

        Args:
            - rnnt_lprobs: Tensor, [bsz, frame_len, tgt_len, 2], fp32
            - input_lengths: Tensor, [bsz], int64
            - target_lengths: Tensor, [bsz], int64
            - u_path_mask: bool, whether output u_path_mask
            - frame_shift: u_path_mask frame shift

        Return:
           List of Tensor if u_path_mask else Tensor(just u_path_index)
           List of Tensor:
               0: u_path_index: [bsz, tgt_len] int64
                   which frame launch the target.\
                   Non target position set to zero.
               1: u_path_mask: [bsz, frame_len, tgt_len], int32\
                   the frame index which launch target set to 1.
                   others are zeros.
        """
        return rnnt_force_alignment(
            rnnt_lprobs,
            input_lengths,
            target_lengths,
            u_path_mask=u_path_mask,
            frame_shift=frame_shift,
        )

    def get_eos_alignment_from_rnnt(self, rnnt_lprobs, input_lengths, target_lengths):
        '''get eos_alignment using rnnt_force_alignment'''
        if self.update_steps < self.rnnt_alignment_start_iters or not self.training:
            self.eos_alignment = None
            return
        if self.args.use_eos:
            target_lengths_align = target_lengths - 1
            rnn_alignment = self.rnnt_force_alignment(
                rnnt_lprobs, input_lengths - 1, target_lengths
            )
        else:
            target_lengths_align = target_lengths
            rnn_alignment = self.rnnt_force_alignment(
                rnnt_lprobs, input_lengths, target_lengths + 1
            )
        target_lengths_list = target_lengths_align.cpu().tolist()
        self.eos_alignment = torch.cat(
            [rnn_alignment[b : b + 1, tl] for b, tl in enumerate(target_lengths_list)],
            dim=0,
        )


class RnntPretrainAdaptiveCE(RnntAdaptiveCE):
    """RNN-T adaptive CE loss for pretrain

    .. note::
        mtl_head 决定了ctc_logits,ce_logits,teacher_ctc_logits的生成

    Example::

        # criterion
        criterion_type='RnntPretrainAdaptiveCE',
        cer_update_freq=50,
        label_smooth_factor=0.0,
        reorder_dict_by_freq=True,
        mtl_head='HybridCEHead',
    """

    def __init__(self, args):
        super().__init__(args)
        if '_rnnt' not in self.args.stage:
            self.log_softmax_fc = None

    def forward(
        self,
        jointer_hidden,
        input_dict,
        ctc_logits=None,
        ce_logits=None,
        teacher_ctc_logits=None,
    ):
        """forward

        Args:

            - jointer_hidden: [B, T, U, N], the output of jointer
            - input_dict: input dictionary
            - acoustic_out: [B, T, N], the output of encoder
            - predictor_module: the module of predictor
            - jointer_module: the module of jointer
            - ctc_logits: the logit of ctc module(mtl)
            - ce_logits: the logit of ce module(mtl)
            - teacher_ctc_logits: the logit of tearcher ctc module(mtl)

        Return:

            - forward_out: a dictionary containing loss, cer and etc.
        """
        adaptive_tgt_indices = input_dict['adaptive_tgt_indices']
        src_mask = input_dict['src_mask']
        target = input_dict['char']
        target_mask = input_dict['char_mask']
        input_lengths = (input_dict['backbone_mask'].sum(dim=1)).int()
        target_lengths = (target_mask.sum(dim=1)).int()
        self.target_lens = input_dict['target_lengths']
        self.eos_alignment = input_dict.get('eos', None)

        # RNN-T loss
        nll_loss = 0.0
        backward_nll_loss = 0.0
        if jointer_hidden is not None:
            _, nll_loss = self.get_rnnt_loss(
                jointer_hidden,
                target,
                input_lengths,
                target_lengths,
                adaptive_tgt_indices=adaptive_tgt_indices,
            )
            backward_nll_loss = nll_loss.mean(0)
            nll_loss = nll_loss.sum(0) / target_lengths.sum(0)
        # CTC loss
        ctc_loss = 0.0
        backward_ctc_loss = 0.0
        if ctc_logits is not None:
            ctc_lprobs = ctc_logits.log_softmax(dim=-1, dtype=torch.float32)
            ctc_loss = d_ctc_loss(
                ctc_lprobs.transpose(0, 1).contiguous(),
                target.int(),
                input_lengths,
                target_lengths,
                blank=0,
                reduction='none',
                zero_infinity=True,
            )
            backward_ctc_loss = (ctc_loss / target_lengths).mean(0)
            ctc_loss = ctc_loss.sum(0) / target_lengths.sum(0)

        # CE loss
        ce_loss = 0.0
        if ce_logits is not None and teacher_ctc_logits is not None:
            teacher_ctc_logits = teacher_ctc_logits.log_softmax(-1, dtype=torch.float32)
            ctc_alignments = ext_ctc_force_alignment(
                teacher_ctc_logits.transpose(1, 2),
                target.int(),
                input_lengths,
                self.target_lens,
                soft=False,
                left=self.args.left_factor,
                right=self.args.right_factor,
            )
            ce_logits = ce_logits.log_softmax(-1, dtype=torch.float32)
            ce_loss = F.cross_entropy(ce_logits.transpose(1, 2), ctc_alignments, reduction='mean')

        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()  # tokens
        forward_out = OrderedDict()
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = backward_nll_loss + backward_ctc_loss + ce_loss
        forward_out['loss'] = nll_loss + ctc_loss + ce_loss
        forward_out['nll_loss'] = nll_loss
        forward_out['acc'] = tgt_size * 0.0
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size
        return forward_out


class RnntDualChannelAdaptiveCE(RnntAdaptiveCE):
    """RNN-T adaptive CE loss for 2-ch asr

    Example::

        criterion_type='RnntDualChannelAdaptiveCE',
        cer_update_freq=100,
        label_smooth_factor=0.0,
        reorder_dict_by_freq=True,
        mask_loss=True,
        mtl_type='ctc',
        mtl_head='HybridCEHead'
    """

    def forward(
        self,
        jointer_hidden1,
        jointer_hidden2,
        input_dict,
        acoustic_out1,
        acoustic_out2,
        ctc_logits1=None,
        ctc_logits2=None,
    ):
        """forward

        Args:

            - jointer_hidden1: [B, T, U, N], the output1 of jointer
            - jointer_hidden2: [B, T, U, N], the output2 of jointer
            - input_dict: input dictionary
            - acoustic_out1: [B, T, N], the output1 of encoder
            - acoustic_out2: [B, T, N], the output2 of encoder
            - predictor_module: the module of predictor
            - jointer_module: the module of jointer
            - ctc_logits1: the output1 of ctc module(mtl)
            - ctc_logits2: the output2 of ctc module(mtl)

        Return:

            - forward_out: a dictionary containing loss, cer and etc.
        """
        # pylint:disable=too-many-locals

        src_mask = input_dict['src_mask']
        target_lengths1 = (input_dict['char1_mask'].sum(dim=1)).int()
        target_lengths2 = (input_dict['char2_mask'].sum(dim=1)).int()
        target1 = input_dict['char1']
        target2 = input_dict['char2']
        input_lengths = input_dict['backbone_mask'].sum(dim=1).int()
        adaptive_tgt_indices1 = input_dict['adaptive_tgt_indices1']
        adaptive_tgt_indices2 = input_dict['adaptive_tgt_indices2']

        # RNN-T loss
        _, nll_loss = self.get_rnnt_loss(
            jointer_hidden1,
            target1,
            input_lengths,
            target_lengths1,
            adaptive_tgt_indices=adaptive_tgt_indices1,
        )
        nll_loss += self.get_rnnt_loss(
            jointer_hidden2,
            target2,
            input_lengths,
            target_lengths2,
            adaptive_tgt_indices=adaptive_tgt_indices2,
        )[1]
        backward_nll_loss = nll_loss.mean(0)
        nll_loss = nll_loss.sum(0) / (target_lengths1.sum(0) + target_lengths2.sum(0))
        # CTC loss
        ctc_loss = 0.0
        backward_ctc_loss = 0.0
        if ctc_logits1 is not None:
            ctc_lprobs = ctc_logits1.log_softmax(-1, dtype=torch.float32)
            if "mtl_mask" in input_dict:
                input_lengths = input_dict['mtl_mask'].sum(dim=1).int()
            ctc_loss1 = F.ctc_loss(
                ctc_lprobs.transpose(0, 1).contiguous(),
                target1.int(),
                input_lengths,
                target_lengths1,
                blank=0,
                reduction='none',
                zero_infinity=True,
            )
            backward_ctc_loss = (ctc_loss1 / target_lengths1).mean(0)
            ctc_loss = ctc_loss1.sum(0) / target_lengths1.sum(0)
        if ctc_logits2 is not None:
            ctc_lprobs = ctc_logits2.log_softmax(-1, dtype=torch.float32)
            ctc_loss2 = F.ctc_loss(
                ctc_lprobs.transpose(0, 1).contiguous(),
                target2.int(),
                input_lengths,
                target_lengths2,
                blank=0,
                reduction='none',
                zero_infinity=True,
            )
            backward_ctc_loss += (ctc_loss2 / target_lengths2).mean(0)
            ctc_loss += ctc_loss2.sum(0) / target_lengths2.sum(0)

        # Mask loss
        mask_loss = 0.0
        if self.args.get('mask_loss', False):
            st = [a // self.args.downsampling_size for a in input_dict['overlap_st']]
            et = [a // self.args.downsampling_size for a in input_dict['overlap_et']]

            def generate_sent_masks(batch_size, max_seq_length, source_lengths, dtype, device):
                '''Generate sentence masks for encoder hidden states.'''
                enc_masks = torch.ones(batch_size, max_seq_length, dtype=dtype, device=device)
                for e_id, src_len in enumerate(source_lengths):
                    if src_len >= max_seq_length:
                        src_len = max_seq_length - 1
                    enc_masks[e_id, :src_len] = 0.0
                return enc_masks.unsqueeze(2)

            batch_size, max_len, _ = acoustic_out1.size()
            dtype = acoustic_out1.dtype
            device = acoustic_out1.device
            acoustic_mask1 = generate_sent_masks(batch_size, max_len, et, dtype, device)
            acoustic_mask2 = 1 - generate_sent_masks(batch_size, max_len, st, dtype, device)
            acoustic_out_tmp = acoustic_mask1 * acoustic_out1 + acoustic_mask2 * acoustic_out2
            mask_loss = (acoustic_out_tmp**2).mean()

        frame_size = src_mask.float().sum()
        tgt_size = target_lengths1.float().sum() + target_lengths2.float().sum()  # tokens
        forward_out = OrderedDict()
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = backward_nll_loss + backward_ctc_loss + mask_loss
        forward_out['loss'] = nll_loss + ctc_loss + mask_loss
        forward_out['nll_loss'] = nll_loss
        forward_out['acc'] = tgt_size * 0.0
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size
        return forward_out


class RnntUniversalCE(RnntAdaptiveCE):
    """RNN-T Universal CE loss for dual-mode asr

    Example::

        # criterion
        criterion_type='RnntUniversalCE',
        cer_update_freq=500,
        label_smooth_factor=0.0,
        reorder_dict_by_freq=True,
    """

    def forward(
        self,
        jointer_hidden1,
        jointer_hidden2,
        input_dict,
        mtl_logits1=None,
        mtl_logits2=None,
        mtl_type=None,
        distillation_scheduler=None,
        frame_shift=None,
        symmetric=None,
        peak=None,
    ):
        # 1 for non-streaming, 2 for streaming
        """forward

        Args:

            - jointer_hidden1: [B, T, U, N], the output1 of jointer_hidden
            - jointer_hidden2: [B, T, U, N], the output2 of jointer_hidden
            - input_dict: input dictionary
            - acoustic_out1: [B, T, N], the output1 of encoder
            - acoustic_out2: [B, T, N], the output2 of encoder
            - predictor_module: the module of predictor
            - jointer_module: the module of jointer
            - mtl_logits1: [B, T, V], the logits1 of multi-task learning
            - mtl_logits2: [B, T, V], the logits2 of multi-task learning
            - mtl_type: the type of multi-task learning, now only support ctc

        Return:

            - forward_out: a dictionary containing loss, cer and etc.
        """
        # pylint:disable=too-many-locals
        if self.training:
            self.update_steps += 1
        adaptive_tgt_indices = input_dict['adaptive_tgt_indices']
        # mask and length
        src_mask = input_dict['src_mask']
        target = input_dict['char']
        target_mask = input_dict['char_mask']
        input_lengths = (input_dict['backbone_mask'].sum(dim=1)).int()
        target_lengths = (target_mask.sum(dim=1)).int()
        self.target_lens = input_dict['target_lengths']
        # rnn-t loss
        log_prob1, nll_loss1 = self.get_rnnt_loss(
            jointer_hidden1,
            target,
            input_lengths,
            target_lengths,
            adaptive_tgt_indices=adaptive_tgt_indices,
        )
        log_prob2, nll_loss2 = self.get_rnnt_loss(
            jointer_hidden2,
            target,
            input_lengths,
            target_lengths,
            adaptive_tgt_indices=adaptive_tgt_indices,
        )
        backward_nll_loss1 = nll_loss1.mean(0)
        nll_loss1 = nll_loss1.sum(0) / target_lengths.sum(0)
        backward_nll_loss2 = nll_loss2.mean(0)
        nll_loss2 = nll_loss2.sum(0) / target_lengths.sum(0)
        # mtl loss, e.g ctc
        if mtl_logits1 is not None and mtl_type == 'ctc' and self.training:
            ctc_lprobs1 = mtl_logits1.log_softmax(-1, dtype=torch.float32)
            ctc_loss1 = F.ctc_loss(
                ctc_lprobs1.transpose(0, 1).contiguous(),
                target.int(),
                input_lengths,
                target_lengths,
                blank=0,
                reduction='none',
                zero_infinity=True,
            )
            backward_ctc_loss1 = (ctc_loss1 / target_lengths).mean(0)
            ctc_loss1 = ctc_loss1.sum(0) / target_lengths.sum(0)
            ctc_lprobs2 = mtl_logits2.log_softmax(-1, dtype=torch.float32)
            ctc_loss2 = F.ctc_loss(
                ctc_lprobs2.transpose(0, 1).contiguous(),
                target.int(),
                input_lengths,
                target_lengths,
                blank=0,
                reduction='none',
                zero_infinity=True,
            )
            backward_ctc_loss2 = (ctc_loss2 / target_lengths).mean(0)
            ctc_loss2 = ctc_loss2.sum(0) / target_lengths.sum(0)

        else:
            ctc_loss1 = 0.0
            ctc_loss2 = 0.0
            backward_ctc_loss1 = 0.0
            backward_ctc_loss2 = 0.0
        # distillation loss, reverse version
        distillation_loss = 0.0
        if distillation_scheduler is not None and self.training:
            if frame_shift is None or frame_shift == 0:
                distillation_loss = distillation_scheduler * kl_divloss_rnnt(
                    log_prob2, log_prob1.detach()
                )
                if symmetric:
                    distillation_loss += distillation_scheduler * kl_divloss_rnnt(
                        log_prob1, log_prob2.detach()
                    )
            else:
                frame_shift = abs(frame_shift)
                distillation_loss = distillation_scheduler * kl_divloss_rnnt(
                    log_prob2[:, frame_shift:, :, :],
                    log_prob1[:, : -1 * frame_shift, :, :].detach(),
                )
                if symmetric:
                    distillation_loss += distillation_scheduler * kl_divloss_rnnt(
                        log_prob1[:, : -1 * frame_shift, :, :],
                        log_prob2[:, frame_shift:, :, :].detach(),
                    )
        # spike loss
        spike_loss = 0.0
        if peak is not None and self.training:
            teacher_spikes = torch.argmax(log_prob1, dim=-1).detach()
            spike_loss = peak * F.cross_entropy(
                log_prob2.transpose(1, 3).transpose(2, 3), teacher_spikes, reduction='mean'
            )
        distillation_loss += spike_loss

        # stat related
        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()
        forward_out = OrderedDict()
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = (
            backward_nll_loss1
            + backward_ctc_loss1
            + backward_nll_loss2
            + backward_ctc_loss2
            + distillation_loss
        )
        forward_out['loss'] = nll_loss1 + ctc_loss1 + nll_loss2 + ctc_loss2 + distillation_loss
        forward_out['loss_offline'] = nll_loss1 + ctc_loss1
        forward_out['nll_loss_offline'] = nll_loss1
        forward_out['loss_stream'] = nll_loss2 + ctc_loss2
        forward_out['nll_loss_stream'] = nll_loss2
        forward_out['distillation_loss'] = distillation_loss
        forward_out['acc'] = tgt_size * 0.0
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size
        return forward_out


class RnntAdaptiveMBR(RnntAdaptiveCE):
    """RNN-T MBR loss

    note::

        nbest alignment path based MBR
        https://arxiv.org/pdf/1911.12487.pdf

    Example::

        # criterion
        criterion_type='RnntAdaptiveMBR',
        cer_update_freq=200,
        label_smooth_factor=0.0,
        reorder_dict_by_freq=True,
        mtl_type='ctc',
        mtl_head='HybridCEHead'
    """

    def forward(
        self,
        jointer_hidden,
        input_dict,
        mtl_logits=None,
        mtl_type=None,
    ):
        """forward

        Args:

            - jointer_hidden: [B, T, U, N], the output of jointer
            - input_dict: input dictionary
            - acoustic_out: [B, T, N], the output of encoder
            - predictor_module: the module of predictor
            - jointer_module: the module of jointer
            - mtl_logits: [B, T, V], the logits of multi-task learning
            - mtl_type: the type of multi-task learning, now only support ctc

        Return:
            - forward_out: a dictionary containing loss, cer and etc.
        """
        # pylint:disable=too-many-locals
        if self.training:
            self.update_steps += 1
        adaptive_tgt_indices = input_dict['adaptive_tgt_indices']

        src_mask = input_dict['src_mask']
        target = input_dict['char']
        target_mask = input_dict['char_mask']
        input_lengths = (input_dict['backbone_mask'].sum(dim=1)).int()
        target_lengths = (target_mask.sum(dim=1)).int()
        self.target_lens = input_dict['target_lengths']

        (bsz, _) = target.size()

        # B, N_sample, T+U
        nbest_path = input_dict['nbest_hyp_path_idx']
        nbest_t_u_path_idx_mask = input_dict['nbest_t_u_path_idx_mask']
        nbest_jointer_out = input_dict['nbest_jointer_out']
        (bsz, beam_size, u_t_steps, jointer_dim) = nbest_jointer_out.size()
        sample_lprobs = self.log_softmax_fc.forward_one_step(
            nbest_jointer_out.view((-1, jointer_dim)), nbest_path.view((-1, u_t_steps))
        )

        sample_nll_loss = -sample_lprobs.view((bsz, beam_size, u_t_steps))
        masked_sample_nll_loss = (sample_nll_loss * nbest_t_u_path_idx_mask.float()).sum(-1)

        hyp_sent_probs = F.softmax(-masked_sample_nll_loss, -1)
        word_errors = input_dict['word_errors']
        total_words = input_dict['total_words']

        wer = word_errors / total_words.unsqueeze(-1)
        avg_wer = wer.mean(1).unsqueeze(-1)
        if self.args.use_wer:
            mbr_loss = (hyp_sent_probs * wer).sum(-1)
        else:
            mbr_loss = (hyp_sent_probs * word_errors).sum(-1)
        mbr_loss = mbr_loss.mean().type_as(sample_lprobs)

        nll_loss = 0.0
        ctc_loss = 0.0
        if self.args.rnnt_regular_factor > 0:
            _, nll_loss = self.get_rnnt_loss(
                jointer_hidden,
                target,
                input_lengths,
                target_lengths,
                adaptive_tgt_indices=adaptive_tgt_indices,
            )
            if mtl_logits is not None and mtl_type == 'ctc':
                ctc_lprobs = F.log_softmax(mtl_logits.float(), dim=-1)
                ctc_loss = F.ctc_loss(
                    ctc_lprobs.transpose(0, 1).contiguous(),
                    target.int(),
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction='mean',
                    zero_infinity=True,
                )

        # stat related
        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()

        forward_out = OrderedDict()
        rnnt_regular_loss = nll_loss + ctc_loss
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = mbr_loss + self.args.rnnt_regular_factor * rnnt_regular_loss
        forward_out['loss'] = mbr_loss + self.args.rnnt_regular_factor * rnnt_regular_loss
        forward_out['mbr_loss'] = mbr_loss
        forward_out['nll_loss'] = rnnt_regular_loss
        forward_out['acc'] = tgt_size * 0.0
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size
        forward_out['avg_wer'] = avg_wer.mean()
        forward_out['avg_best_wer'] = (word_errors[:, 0] / total_words).mean()
        forward_out['avg_worst_wer'] = (word_errors[:, -1] / total_words).mean()

        return forward_out


class KwsRnnt(nn.Module):
    """RNN-T loss for KWS

    Example::

            rnnt_weight=1.0,
            criterion_type='KwsRnnt',
            label_smooth_weight=0.1,
            rnnt_exclude_data_tag='kws',
            cer_update_freq=100,
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.zero_infinity = getattr(args, 'zero_infinity', True)

    def forward(
        self,
        logits,
        target,
        src_mask,
        logit_mask,
        target_mask,
        utt_ids,
        ce_logits,
        ce_target,
        ctc_logits,
        ctc_target,
        rnnt_exclude_data_tag,
    ):
        """forward

        Return:

            - forward_out: a dictionary containing loss, cer and etc.
        """
        # pylint:disable=too-many-locals
        if logits.size(-1) > 2:  # for basic softmax
            blank_idx = 0
            lprobs = logits.log_softmax(-1, dtype=torch.float32)
        else:  # for adaptive softmax
            blank_idx = -1
            lprobs = logits
        bsz = lprobs.size(0)
        input_lengths = (logit_mask.sum(dim=1)).int()
        max_input_lengths = input_lengths.max().item()
        lprobs = lprobs[:, 0:max_input_lengths, :, :].contiguous()
        target_lengths = (target_mask.sum(dim=1)).int()

        # ce loss
        if ce_logits is not None:
            ce_weight = self.args.ce_weight
            epsilon = 1e-5
            ce_weight = self.args.ce_weight
            ce_lprobs = ce_logits.log_softmax(-1, dtype=torch.float32)
            ce_target_mask = logit_mask
            ce_loss = -ce_lprobs.gather(dim=-1, index=ce_target.unsqueeze(-1)).squeeze(-1)
            ce_loss = (ce_loss * ce_target_mask.float()).sum() / (
                ce_target_mask.float().sum() + epsilon
            )
        else:
            ce_loss = 0.0
            ce_weight = 0.0

        # ctc loss
        if ctc_logits is not None:
            ctc_weight = self.args.ctc_weight
            ctc_lprobs = ctc_logits.log_softmax(-1, dtype=torch.float32)
            ctc_target_lengths = target_lengths
            ctc_loss = F.ctc_loss(
                ctc_lprobs.transpose(0, 1).contiguous(),
                ctc_target.int(),
                input_lengths,
                ctc_target_lengths,
                blank=0,
                reduction='mean',
                zero_infinity=self.zero_infinity,
            )
        else:
            ctc_loss = 0.0
            ctc_weight = 0.0

        # rnnt loss
        rnnt_weight = self.args.rnnt_weight
        if rnnt_weight > 0:
            if rnnt_exclude_data_tag is not None:
                rnnt_idxs = []
                for i in range(bsz):
                    utt_id = utt_ids[i]
                    if rnnt_exclude_data_tag not in utt_id:
                        rnnt_idxs.append(i)
                if rnnt_idxs:
                    lprobs = lprobs[rnnt_idxs, :, :, :]
                    target = target[rnnt_idxs]
                    input_lengths = input_lengths[rnnt_idxs]
                    logit_mask = logit_mask[rnnt_idxs, :]
                    target_lengths = target_lengths[rnnt_idxs]
                    target_mask = target_mask[rnnt_idxs, :]
            nll_loss = rnnt_loss(
                lprobs,
                target.int(),
                input_lengths,
                target_lengths,
                average_frames=False,
                reduction='mean',
                blank=blank_idx,
            )

            # label smooth loss
            lattic_mask = logit_mask.unsqueeze(2) * target_mask.unsqueeze(1)
            lattic_mask = lattic_mask[:, 0:max_input_lengths, :]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=False)
            smooth_loss = smooth_loss[:, :, :-1]  # prev char length = char length + 1
            smooth_loss = (smooth_loss * lattic_mask).sum() / (lattic_mask.sum() + 1e-5)
            eps_i = self.args.label_smooth_weight / lprobs.size(-1)
            rnnt_smooth_loss = (
                1.0 - self.args.label_smooth_weight
            ) * nll_loss + eps_i * smooth_loss
        else:
            rnnt_smooth_loss = 0.0

        # total_loss
        total_loss = rnnt_weight * rnnt_smooth_loss + ctc_weight * ctc_loss + ce_weight * ce_loss

        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()
        forward_out = OrderedDict()
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = total_loss
        forward_out['loss'] = total_loss
        forward_out['rnnt_loss'] = rnnt_smooth_loss
        forward_out['ctc_loss'] = ctc_loss
        forward_out['ce_loss'] = ce_loss
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size

        return forward_out
