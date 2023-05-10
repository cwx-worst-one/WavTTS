# pylint: disable=cell-var-from-loop
''' CifLoss '''
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
import torch
from core.utils.label_smooth import uniform_label_smooth
from .criterion import Xentropy

EPSILON = 1e-5


class CifLoss(nn.Module):
    '''CIFLoss针对CIF的相关criterion

    Args:

        - ctc_loss_lambda: 一个参数与ctc_loss相乘,得到新的的ctc_loss
        - quantity_loss_lambda: 一个参数与num_char_loss.sum()相乘得到quantity_loss_lambda
        - ls_type: mode='uniform',计算ce_lprob, smoothed_ce_loss
        - label_smoothing: 设置平滑标签的值
        - loss_multiplier: 一个参数,与每一个loss相乘
        - vocab_size: 字典的大小
        - calculated_loss: 需要计算的loss
        - delete_ctc_eos: 是否删除结束符,default=false
        - eos_id: 结束符个数,default=2

    Example::

        calculated_loss='hybrid_ce_loss, quantity_loss',
        label_smoothing=0.1,
        ls_type='uniform',
        quantity_loss_lambda=1.0,
        criterion_type='CifLoss',
        eos_id=1
    '''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ctc_loss_lambda = self.args.ctc_loss_lambda
        self.quantity_loss_lambda = self.args.quantity_loss_lambda
        self.ls_type = self.args.ls_type
        self.label_smoothing = self.args.label_smoothing
        self.loss_multiplier = self.args.loss_multiplier
        self.vocab_size = self.args.vocab_size
        self.calculated_loss = self.args.calculated_loss
        self.delete_ctc_eos = self.args.get('delete_ctc_eos', False)
        self.eos_id = self.args.get('eos_id', 2)
        self.ilmt_weight = args.get('ilmt_weight', 0.0)
        if "hybrid_ce_loss" in self.calculated_loss:
            self.xentropy_processer = Xentropy(args=args)

    def forward(
        self,
        logits,
        targets,
        src_mask,
        not_padding=None,
        logits_on_encoder=None,
        not_padding_on_encoder=None,
        sum_a=None,
        hybrid_ce_logits=None,
        hybrid_ce_targets=None,
        hybrid_ce_targets_mask=None,
        ilmt_logits=None,
    ):
        '''
        Calculating different loss for the training of the model.

        Args:
          logits: with shape [batch_size, cif_output_length, vocab_size]
          targets: with shape [batch_size, target_length]
          src_mask: with shape [batch_size, feat_length]
          not_padding: with shape [batch_size, cif_output_length],
                       same as target_mask in rnnt
          logits_on_encoder: with shape [batch_size, down_sampled_length, vocab_size],
                             same as ctc_logits in rnnt
          not_padding_on_encoder: with shape [batch_size, down_sampled_length],
                                  same as encoder_mask in rnnt
          sum_a: with shape [batch_size]
          hybrid_ce_logits: with shape [batch_size, subsampled_length, vocab_size]
          hybrid_ce_targets: with shape [batch_size, subsampled_length]
          hybrid_ce_targets_mask: with shape [batch_size, subsampled_length]

        Return:

        - losses: a dict of variant loss functions
        - weight: the weight of all losses,\
        here we refer to the sum of effective labels in target batch.

        '''
        # pylint:disable=too-many-branches
        # pylint:disable=too-many-locals
        if logits is not None:
            min_logit_len = min(logits.shape[1], targets.shape[1])
            logits = logits[:, :min_logit_len, :]
        assert self.calculated_loss != ''
        losses = dict()

        if targets is not None:
            targets_mask = (targets != 0).int()
            targets_lengths = targets_mask.sum(-1)
            weights = targets_mask.sum().float()

        pointed_loss_list = self.calculated_loss.replace(' ', '').split(',')

        if 'ctc_loss_on_encoder' in pointed_loss_list:
            assert not_padding_on_encoder is not None, 'invalid not_padding_on_encoder'
            if self.delete_ctc_eos:
                targets_ctc = torch.where(targets == self.eos_id, 0, targets)
                targets_lengths_ctc = targets_lengths - 1
            else:
                targets_ctc = targets
                targets_lengths_ctc = targets_lengths
            seq_lengths = not_padding_on_encoder.sum(-1)
            ctc_lprobs = F.log_softmax(logits_on_encoder, dim=-1)
            ctc_loss = F.ctc_loss(
                ctc_lprobs.transpose(0, 1).contiguous(),
                targets_ctc,
                seq_lengths,
                targets_lengths_ctc,
                blank=0,
                reduction='sum',
                zero_infinity=True,
            )
            losses['ctc_loss_on_encoder'] = self.ctc_loss_lambda * ctc_loss / weights

        # number of char loss
        if 'quantity_loss' in pointed_loss_list:
            target_num_chars = targets_lengths.float()
            num_char_loss = (sum_a - target_num_chars).abs()
            losses['quantity_loss'] = self.quantity_loss_lambda * num_char_loss.sum() / weights

        # ce loss
        if 'ce_loss' in pointed_loss_list:
            real_length = not_padding.sum(-1).max()
            logits_for_ce_loss = logits[:, :real_length]
            targets_for_ce_loss = targets[:, :real_length]
            targets_mask_for_ce_loss = targets_mask[:, :real_length]
            confidence = 1 - self.label_smoothing

            if self.ls_type == 'uniform':
                ce_lprob, smoothed_ce_loss = uniform_label_smooth(
                    confidence, self.vocab_size, targets_for_ce_loss, logits_for_ce_loss
                )
            else:
                raise NotImplementedError()

            hyp_tgt = ce_lprob.max(dim=-1)[1]
            hyp_acc = (
                (hyp_tgt == targets_for_ce_loss).float() * (targets_mask_for_ce_loss.float())
            ).sum() / (targets_mask_for_ce_loss.float().sum() + 1e-5)

            ce_loss = (smoothed_ce_loss * targets_mask_for_ce_loss.float()).sum()
            losses['ce_loss'] = ce_loss / weights

        if 'hybrid_ce_loss' in pointed_loss_list:
            items = self.xentropy_processer(
                logits=hybrid_ce_logits,
                src_mask=not_padding_on_encoder,
                target=hybrid_ce_targets,
                target_mask=hybrid_ce_targets_mask,
            )
            losses['hybrid_ce_loss'] = items['loss'] / 1.0
            # /1.0 means loss is already a reduce_mean loss

        if self.ilmt_weight > 0:
            real_length = not_padding.sum(-1).max()
            ilmt_logits_for_ce_loss = ilmt_logits[:, :real_length]
            targets_for_ce_loss = targets[:, :real_length]
            targets_mask_for_ce_loss = targets_mask[:, :real_length]
            confidence = 1 - self.label_smoothing

            if self.ls_type == 'uniform':
                ce_lprob, smoothed_ce_loss = uniform_label_smooth(
                    confidence, self.vocab_size, targets_for_ce_loss, ilmt_logits_for_ce_loss
                )
            else:
                raise NotImplementedError()

            ilmt_hyp_tgt = ce_lprob.max(dim=-1)[1]
            ilmt_hyp_acc = (
                (ilmt_hyp_tgt == targets_for_ce_loss).float() * (targets_mask_for_ce_loss.float())
            ).sum() / (targets_mask_for_ce_loss.float().sum() + 1e-5)

            ilmt_ce_loss = (smoothed_ce_loss * targets_mask_for_ce_loss.float()).sum()
            losses['ilmt_ce_loss'] = self.ilmt_weight * ilmt_ce_loss / weights

        for k in losses:
            losses[k] = losses[k] * self.loss_multiplier

        forward_out = OrderedDict()
        forward_out['frame_size'] = src_mask.float().sum()
        total_loss = 0

        if losses.get('hybrid_ce_loss', False):
            # for pretrain task
            forward_out['acc'] = items['acc']
            forward_out['tgt_size'] = hybrid_ce_targets_mask.float().sum()
            forward_out['hybrid_ce_loss'] = losses['hybrid_ce_loss']
            total_loss = losses['hybrid_ce_loss'].clone()
            if losses.get('quantity_loss', False):
                forward_out['quantity_loss'] = losses['quantity_loss']
                total_loss += losses['quantity_loss']
        else:
            # for ASR task
            forward_out['utt_num'] = logits.shape[0]
            if losses.get('ce_loss', False):
                forward_out['ce_loss'] = losses['ce_loss']
                forward_out['acc'] = hyp_acc
                total_loss += forward_out['ce_loss']
            if losses.get('quantity_loss', False):
                forward_out['quantity_loss'] = losses['quantity_loss']
                total_loss += forward_out['quantity_loss']
            if losses.get('ctc_loss_on_encoder', False):
                forward_out['ctc_loss'] = losses['ctc_loss_on_encoder']
                total_loss += forward_out['ctc_loss']
            forward_out['tgt_size'] = targets_mask.float().sum()
        forward_out['loss'] = total_loss
        forward_out['backward_loss'] = total_loss
        if self.ilmt_weight > 0:
            forward_out['ilmt_acc'] = ilmt_hyp_acc
            forward_out['ilmt_ce_loss'] = losses['ilmt_ce_loss']
        return forward_out


class UniversalCifLoss(CifLoss):
    '''CIFLoss'''

    def __init__(self, args):
        super().__init__(args)
        self.distill_weight = self.args.distill_weight
        self.kl_divloss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        logits,
        nonstream_logits,
        targets,
        src_mask,
        not_padding=None,
        nonstream_not_padding=None,
        logits_on_encoder=None,
        nonstream_logits_on_encoder=None,
        not_padding_on_encoder=None,
        nonstream_not_padding_on_encoder=None,
        sum_a=None,
        nonstream_sum_a=None,
    ):
        # pylint:disable=too-many-locals,invalid-name
        min_logit_len = min(logits.shape[1], targets.shape[1])
        logits = logits[:, :min_logit_len, :]
        assert self.calculated_loss != ''
        losses = dict()

        # if targets is not None:
        targets_mask = (targets != 0).int()
        targets_lengths = targets_mask.sum(-1)
        weights = targets_mask.sum().float()

        pointed_loss_list = self.calculated_loss.replace(' ', '').split(',')

        if 'ctc_loss_on_encoder' in pointed_loss_list:
            assert not_padding_on_encoder is not None, 'invalid not_padding_on_encoder'
            seq_lengths = not_padding_on_encoder.sum(-1)
            ctc_lprobs = F.log_softmax(logits_on_encoder, dim=-1)
            ctc_loss = F.ctc_loss(
                ctc_lprobs.transpose(0, 1).contiguous(),
                targets,
                seq_lengths,
                targets_lengths,
                blank=0,
                reduction='sum',
                zero_infinity=True,
            )
            nonstream_seq_lengths = nonstream_not_padding_on_encoder.sum(-1)
            nonstream_ctc_lprobs = F.log_softmax(nonstream_logits_on_encoder, dim=-1)
            nonstream_ctc_loss = F.ctc_loss(
                nonstream_ctc_lprobs.transpose(0, 1).contiguous(),
                targets,
                nonstream_seq_lengths,
                targets_lengths,
                blank=0,
                reduction='sum',
                zero_infinity=True,
            )
            losses['ctc_loss_on_encoder'] = self.ctc_loss_lambda * ctc_loss / weights
            losses['nonstream_ctc_loss_on_encoder'] = (
                self.ctc_loss_lambda * nonstream_ctc_loss / weights
            )
            losses['kld_on_encoder_ctc_logits'] = self.distill_weight * self.kl_divloss(
                ctc_lprobs, torch.exp(nonstream_ctc_lprobs).detach()
            )

        # number of char loss
        if 'quantity_loss' in pointed_loss_list:
            target_num_chars = targets_lengths.float()
            num_char_loss = (sum_a - target_num_chars).abs()
            nonstream_num_char_loss = (nonstream_sum_a - target_num_chars).abs()
            losses['quantity_loss'] = self.quantity_loss_lambda * num_char_loss.sum() / weights
            losses['nonstream_quantity_loss'] = (
                self.quantity_loss_lambda * nonstream_num_char_loss.sum() / weights
            )

        # ce loss
        if 'ce_loss' in pointed_loss_list:
            real_length = not_padding.sum(-1).max()
            nonstream_real_length = nonstream_not_padding.sum(-1).max()
            # if real_length != nonstream_real_length:
            #     logging.warning('strem nonstream max ci length mismatch'))
            logits_for_ce_loss = logits[:, :real_length]
            nonstream_logits_for_ce_loss = nonstream_logits[:, :nonstream_real_length]

            targets_for_ce_loss = targets[:, :real_length]
            targets_mask_for_ce_loss = targets_mask[:, :real_length]

            nonstream_targets_for_ce_loss = targets[:, :nonstream_real_length]
            nonstream_targets_mask_for_ce_loss = targets_mask[:, :nonstream_real_length]

            confidence = 1 - self.label_smoothing

            if self.ls_type == 'uniform':
                ce_lprob, smoothed_ce_loss = uniform_label_smooth(
                    confidence, self.vocab_size, targets_for_ce_loss, logits_for_ce_loss
                )
                nonstream_ce_lprob, nonstream_smoothed_ce_loss = uniform_label_smooth(
                    confidence,
                    self.vocab_size,
                    nonstream_targets_for_ce_loss,
                    nonstream_logits_for_ce_loss,
                )
            else:
                raise NotImplementedError()

            hyp_tgt = ce_lprob.max(dim=-1)[1]
            hyp_acc = (
                (hyp_tgt == targets_for_ce_loss).float() * (targets_mask_for_ce_loss.float())
            ).sum() / (targets_mask_for_ce_loss.float().sum() + 1e-5)

            nonstream_hyp_tgt = nonstream_ce_lprob.max(dim=-1)[1]
            nonstream_hyp_acc = (
                (nonstream_hyp_tgt == nonstream_targets_for_ce_loss).float()
                * (nonstream_targets_mask_for_ce_loss.float())
            ).sum() / (nonstream_targets_mask_for_ce_loss.float().sum() + 1e-5)

            ce_loss = (smoothed_ce_loss * targets_mask_for_ce_loss.float()).sum()
            nonstream_ce_loss = (
                nonstream_smoothed_ce_loss * nonstream_targets_mask_for_ce_loss.float()
            ).sum()
            losses['ce_loss'] = ce_loss / weights
            losses['nonstream_ce_loss'] = nonstream_ce_loss / weights
            min_length = min(ce_lprob.size(1), nonstream_ce_lprob.size(1))
            # strem nonstream max length maybe mismatch, use min length to calculate ce loss
            losses['kld_on_decoder_logits'] = self.distill_weight * self.kl_divloss(
                ce_lprob[:, :min_length, :],
                torch.exp(nonstream_ce_lprob[:, :min_length, :]).detach(),
            )

        for k in losses:
            losses[k] = losses[k] * self.loss_multiplier

        forward_out = OrderedDict()
        forward_out['frame_size'] = src_mask.float().sum()
        # for ASR task
        total_loss = (
            losses['ce_loss']
            + losses['quantity_loss']
            + losses['ctc_loss_on_encoder']
            + losses['nonstream_ce_loss']
            + losses['nonstream_quantity_loss']
            + losses['nonstream_ctc_loss_on_encoder']
            + losses['kld_on_encoder_ctc_logits']
            + losses['kld_on_decoder_logits']
        )
        forward_out['utt_num'] = logits.shape[0]
        forward_out['loss'] = total_loss
        forward_out['stream_ce_loss'] = losses['ce_loss']
        forward_out['nonstream_ce_loss'] = losses['nonstream_ce_loss']
        forward_out['stream_ctc_loss'] = losses['ctc_loss_on_encoder']
        forward_out['nonstream_ctc_loss'] = losses['nonstream_ctc_loss_on_encoder']
        forward_out['stream_quantity_loss'] = losses['quantity_loss']
        forward_out['nonstream_quantity_loss'] = losses['nonstream_quantity_loss']
        forward_out['kld_on_ctc'] = losses['kld_on_encoder_ctc_logits']
        forward_out['kld_on_ce'] = losses['kld_on_decoder_logits']
        forward_out['stream_acc'] = hyp_acc
        forward_out['nonstream_acc'] = nonstream_hyp_acc
        forward_out['tgt_size'] = targets_mask.float().sum()
        forward_out['backward_loss'] = total_loss
        return forward_out
