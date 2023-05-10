''' hybrid criterions '''

import math
from collections import OrderedDict
from sys import prefix
import torch
from torch import nn
import torch.nn.functional as F
from core.utils import edit_distance
from core.extensions import ctc_loss

EPSILON = 1e-5


class Xentropy(nn.Module):
    """This criterion computes the cross entropy loss between input and target.

    .. note::
        获得NLLloss和Xentropy的相关介绍你可以查询
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss

    Args:

        - label_smooth_factor: 平滑标签的值

    Example::

        criterion_type='Xentropy',
        label_smooth_factor=0.1,
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update_steps = 0

    def forward(self, logits, src_mask, target, target_mask, *_args, **_kwargs):
        """
        Calculating different loss for the training of the model.

        Args:

            - logits: [B, T, N], for N classes
            - src_mask: [B, Feature_T]
            - target: [B, T, N]
            - target_mask: [B, T]

        Return:

            - losses: a dict of variant loss functions
            - weight: the weight of all losses,\
            here we refer to the sum of effective labels in target batch.

        """

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        hyp_tgt = lprobs.max(dim=-1)[1]
        hyp_acc = ((hyp_tgt == target).float() * (target_mask.float())).sum() / (
            target_mask.float().sum() + EPSILON
        )
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1))
        smooth_loss = -lprobs.sum(dim=-1, keepdim=False)

        nll_loss = nll_loss.squeeze(-1)
        eps_i = self.args.label_smooth_factor / lprobs.size(-1)
        loss = (1.0 - self.args.label_smooth_factor) * nll_loss + eps_i * smooth_loss

        masked_loss = (loss * target_mask.float()).sum() / (target_mask.float().sum() + EPSILON)
        masked_nll_loss = (nll_loss * target_mask.float()).sum() / (
            target_mask.float().sum() + EPSILON
        )
        # for PPL
        sent_lprobs = lprobs.gather(dim=-1, index=target.unsqueeze(-1))
        masked_lprobs = (sent_lprobs.squeeze(-1) * target_mask.float()).sum(dim=-1)  # sentence prob

        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()  # tokens
        forward_out = OrderedDict()
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = masked_loss
        forward_out['loss'] = masked_loss.type_as(logits)
        forward_out['lprobs'] = masked_lprobs.type_as(logits)  # for PPL for sentence
        forward_out['nll_loss'] = masked_nll_loss.type_as(logits)
        forward_out['acc'] = hyp_acc
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size
        # TODO(zhangjun) add CER
        # forward_out['cer'] = tgt_size * 0. + error_dist
        # forward_out['dist'] = tgt_size * 0. + total_dist

        return forward_out


class CTC(nn.Module):
    """The Connectionist Temporal Classification loss.

    .. note::
        获得CTC的相关介绍你可以查询
        https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss

    Args:

        - zero_infinity: 平滑标签的值
        - pad_token_id: Whether to zero infinite losses and the associated gradients,default=false.
        - ctc_loss_reduction:  Specifies the reduction to apply to the output,default=mean

    Example::

        # criterion
        criterion_type='CTC',
        ctc_use_lm=0,
        zero_infinity=1,
        sentence_avg=1,
        lid_mask_ratio=1.0,
        lid_loss_scale=1.0,
        ctc_loss_reduction='mean',
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.zero_infinity = args.zero_infinity
        self.sentence_avg = args.sentence_avg
        self.update_steps = 0

    def forward(self, net_output):
        """Calculating different loss for the training of the model.

        Args:

            - net_output: the output of encoder

        Return:

            - losses: ctc_loss

        """

        encoder_out = net_output["encoder_out"].transpose(0, 1)  # T, B, C
        labels = net_output["labels"]
        lprobs = F.log_softmax(encoder_out.float(), dim=-1).contiguous()

        non_padding_mask = ~net_output["padding_mask"]
        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = labels >= 0
        targets_flat = labels.masked_select(pad_mask)
        target_lengths = pad_mask.long().sum(-1)

        loss = ctc_loss(
            lprobs,
            targets_flat,
            input_lengths,
            target_lengths,
            blank=self.args.pad_token_id,
            reduction=self.args.ctc_loss_reduction,
            zero_infinity=bool(self.zero_infinity),
        )
        return loss


class Wav2vecDecoder(nn.Module):
    """wav2vec decoder

    .. note ::
        设置 w2l_decoder="greedy"，则这里的decoder为Args中所示。

    Example::

        # criterion
        decoder_type='Wav2vecDecoder',
        cer_update_freq=100
        inference = dict(
        test_sets='shard.yueyu_clean###shard.yueyu_noise###shard.yueyu_test',
        search_params=True,
        am_scale=1.1,
        blk_scale=0.3,)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update_steps = 0

    def forward(self, batch_data, decoder, ctc_out):
        """forward

        Args:

            - batch_data: 数据组
            - decoder: W2lGreedyDecoder(args, args.tgt_dict)
            - ctc_out: the output of Wav2vecCtc

        Return:

            - losses: ctc_loss
        """

        lprobs = ctc_out['lprobs']
        input_lengths = ctc_out['input_lengths']
        target_lengths = ctc_out['target_lengths']
        error_dist = 0
        total_dist = 0
        self.update_steps += 1
        if (self.update_steps % self.args.cer_update_freq == 0) or (not self.training):
            lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

            if not self.training and hasattr(self, 'wfst_decoder'):
                blk_scale = math.log(self.args.blk_scale)
                logits = lprobs_t
                ilen = input_lengths.cpu()
                logits[:, :, 0] += blk_scale
                for i in range(logits.size(0)):
                    self.ref_dict[batch_data['utt_id'][i]] = batch_data['text'][i]
                    self.wfst_decoder(batch_data['utt_id'][i], logits[i, : ilen[i]].numpy())

            hyps = decoder.decode(lprobs_t, input_lengths)
            for hyp, tgt, tlen in zip(hyps, batch_data['char'], target_lengths):
                hyp_list = hyp[0]['tokens'].tolist()
                tgt_list = tgt[:tlen].tolist()

                _, _, counter = edit_distance(tgt_list, hyp_list)
                error_dist += counter['sub'] + counter['ins'] + counter['del']
                total_dist += counter['words']

        forward_out = OrderedDict()
        forward_out['cer'] = error_dist
        forward_out['dist'] = total_dist
        return forward_out


class Wav2vecCtc(CTC):
    """Wav2vecCtc

    .. note::
        注意要在config的data中设置tgt_dict的位置
        tgt_dict_dir='dict_phonetone.txt'

    Example::

        # criterion
        ctc_type='Wav2vecCtc',
        ctc_loss_reduction='sum'
        ctc_use_lm=False,
        zero_infinity=True,
        sentence_avg=True,
        cer_update_freq=100,
        modeling_unit_type='bpe',
    """

    def __init__(self, args):
        super().__init__(args)
        self.blank_idx = args.tgt_dict.bos()
        self.pad_idx = args.tgt_dict.pad()
        self.eos_idx = args.tgt_dict.eos()

    def forward(self, net_output):
        """forward

        Args:

            - net_output: the output of encoder

        Return:

            - losses: ctc_loss and acc and so on

        """

        encoder_out = net_output["encoder_out"]  # T, B, C
        labels = net_output["labels"]
        bsz = encoder_out.size(1)
        lprobs = F.log_softmax(encoder_out.float(), dim=-1).contiguous()

        non_padding_mask = ~net_output["padding_mask"]
        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (labels != self.pad_idx) & (labels != self.eos_idx)
        targets_flat = labels.masked_select(pad_mask)
        target_lengths = pad_mask.long().sum(-1)

        loss = ctc_loss(
            lprobs,
            targets_flat,
            input_lengths,
            target_lengths,
            blank=self.blank_idx,
            reduction=self.args.ctc_loss_reduction,
            zero_infinity=bool(self.zero_infinity),
        )
        sample_size = bsz if self.sentence_avg else target_lengths.sum().type_as(loss)
        loss = loss / sample_size / math.log(2)

        forward_out = OrderedDict()
        forward_out['backward_loss'] = loss
        forward_out['loss'] = loss
        forward_out['frame_size'] = net_output['src_mask'].float().sum()
        forward_out['tgt_size'] = target_lengths.float().sum()
        forward_out['wer'] = 1.0
        forward_out['dist_words'] = 1.0
        forward_out['batch_size'] = bsz
        forward_out['lprobs'] = lprobs
        forward_out['input_lengths'] = input_lengths
        forward_out['target_lengths'] = target_lengths

        return forward_out


class Bientropy(nn.Module):
    """binary cross entropy (BCE)

    Args:

        - detection_threshold: hyp_tgt 检测阈值

    Example::

        # criterion
        criterion_type='Bientropy',
        xavier_init=False,
        detection_threshold=0.5,
        label_pooling=8,
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update_steps = 0
        self.pos_weight = (
            self.args.bce_pos_weight * torch.ones([self.args.num_classes])
            if self.args.get("bce_pos_weight", None) is not None
            else None
        )

    def forward(self, logits, src_mask, target, target_mask):
        """
        forward

        Args:
            logits: [B, T, N], for N classes
            src_mask: [B, Feature_T]
            target: [B, T, N]
            target_mask: [B, T]

        Return:
            loss and acc
        """

        bce_loss = F.binary_cross_entropy_with_logits(
            logits.float(), target.float(), reduction='sum', pos_weight=self.pos_weight
        )
        masked_loss = bce_loss / (self.args.num_classes * target_mask.float().sum() + EPSILON)

        hyp_tgt = torch.sigmoid(logits) >= self.args.detection_threshold
        pre_mask = hyp_tgt == 1
        rec_mask = target == 1
        hyp_score = (hyp_tgt.int() == target.int()).float()
        hyp_acc = (hyp_score * target_mask.float().unsqueeze(dim=-1)).sum() / (
            target_mask.float().sum() * self.args.num_classes + EPSILON
        )
        hyp_pre = (hyp_score * pre_mask.float()).sum() / (pre_mask.float().sum() + EPSILON)
        hyp_rec = (hyp_score * rec_mask.float()).sum() / (rec_mask.float().sum() + EPSILON)
        hyp_f1 = (2 * hyp_pre * hyp_rec) / (hyp_pre + hyp_rec + EPSILON)
        active_mask = ((target == 1).float().sum(dim=-2) > 0.0).unsqueeze(dim=-2).float()
        hyp_active_acc = (hyp_score * active_mask).sum() / (
            active_mask.sum() * target.shape[-2] + EPSILON
        )
        hyp_active_pre = (hyp_score * active_mask * pre_mask.float()).sum() / (
            (active_mask * pre_mask.float()).sum() + EPSILON
        )
        hyp_active_rec = (hyp_score * active_mask * rec_mask.float()).sum() / (
            (active_mask * rec_mask.float()).sum() + EPSILON
        )
        hyp_deactive_acc = (hyp_score * (1 - active_mask)).sum() / (
            (1 - active_mask).sum() * target.shape[-2] + EPSILON
        )

        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()
        forward_out = OrderedDict()
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = masked_loss
        forward_out['loss'] = masked_loss.type_as(logits)
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size
        forward_out['acc'] = hyp_acc
        forward_out['f1'] = hyp_f1
        forward_out['pre'] = hyp_pre
        forward_out['rec'] = hyp_rec
        forward_out['active_acc'] = hyp_active_acc
        forward_out['active_pre'] = hyp_active_pre
        forward_out['active_rec'] = hyp_active_rec
        forward_out['deactive_acc'] = hyp_deactive_acc
        return forward_out


class XentropyWithoutMask(nn.Module):
    '''cross entropy loss with logits

    Args:

        - label_smoothing(float): 平滑标签,default=0.0
        - multi_labels: 设置是否是多标签模式

    Example::

        # criterion
        multi_labels=False,
        criterion_type='XentropyWithoutMask',
        label_smoothing=0.0,
    '''

    def __init__(self, args):
        '''init'''
        super().__init__()
        self.args = args

    def forward(self, logits, labels, onehot_labels):
        '''
        forward

        Args:
            - logits: [B, T, N], for N classes
            - label: [B, Feature_T]
            - onehot_labels: [B, T, N]

        Return:
            loss

        '''

        label_smoothing = self.args.get('label_smoothing', 0.0)
        if self.args.multi_labels:
            assert onehot_labels is not None
            if label_smoothing > 0.0:
                onehot_labels = onehot_labels * (1 - label_smoothing) + 0.5 * label_smoothing
            return F.binary_cross_entropy_with_logits(logits, onehot_labels)

        if label_smoothing > 0.0:
            assert onehot_labels is not None
            onehot_labels = (
                onehot_labels * (1 - label_smoothing) + label_smoothing / self.args.num_classes
            )
            softmax_ce = -torch.sum(F.log_softmax(logits, dim=-1) * onehot_labels, dim=-1)
            return torch.mean(softmax_ce)

        assert labels is not None
        return F.cross_entropy(logits, labels)


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(
        self, size: int, padding_idx: int, smoothing: float, normalize_length: bool = False
    ):
        """Construct an LabelSmoothingLoss object."""
        super().__init__()
        self.criterion = torch.nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.reshape(-1, self.size)

        target = target.reshape(target.shape[0] * target.shape[1])
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


class SpokenLlmAffixXentropy(nn.Module):
    """SpokenLlmAffixXentropy."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.prefix_loss_weight = args.get('prefix_loss_weight', 1.0)
        self.affix_loss_weight = args.get('affix_loss_weight', 1.0)
        self.update_steps = 0

    def forward(self, logits, targets, prefix_masks, affix_masks, masks):
        """
        Calculating different loss for the training of the model.

        Args:

            - logits: [B, T, N], for N classes
            - targets: [B, T]
            - prefix_masks: [B, T]
            - affix_masks: [B, T]
            - masks: [B, T]

        Return:

            - losses: a dict of variant loss functions
            - weight: the weight of all losses,\
            here we refer to the sum of effective labels in target batch.

        """
        prefix_masks = prefix_masks * masks
        affix_masks = affix_masks * masks

        lprobs = F.log_softmax(logits.float(), dim=-1)

        nll_loss = -lprobs.gather(dim=-1, index=targets.unsqueeze(-1))
        smooth_loss = -lprobs.sum(dim=-1, keepdim=False)

        nll_loss = nll_loss.squeeze(-1)
        eps_i = self.args.label_smooth_factor / lprobs.size(-1)
        loss = (1.0 - self.args.label_smooth_factor) * nll_loss + eps_i * smooth_loss

        # masked_loss = (loss * tgt_token_masks.float()).sum() / \
        #     (tgt_token_masks.float().sum() + EPSILON)
        # masked_nll_loss = (nll_loss * tgt_token_masks.float()).sum() / (
        #     tgt_token_masks.float().sum() + EPSILON
        # )

        weighted_masks = (
            prefix_masks.float() * self.prefix_loss_weight
            + affix_masks.float() * self.affix_loss_weight
        )
        masked_loss = (loss * weighted_masks).sum() / (weighted_masks.sum() + EPSILON)
        masked_nll_loss = (nll_loss * weighted_masks).sum() / (weighted_masks.sum() + EPSILON)

        masked_prefix_nll_loss = (nll_loss * prefix_masks.float()).sum() / (
            prefix_masks.float().sum() + EPSILON
        )
        masked_affix_nll_loss = (nll_loss * affix_masks.float()).sum() / (
            affix_masks.float().sum() + EPSILON
        )

        hyp_tgts = lprobs.max(dim=-1)[1]
        hyp_token_acc = ((hyp_tgts == targets).float() * (masks.float())).sum() / (
            masks.float().sum() + EPSILON
        )
        hyp_prefix_token_acc = ((hyp_tgts == targets).float() * (prefix_masks.float())).sum() / (
            prefix_masks.float().sum() + EPSILON
        )
        hyp_affix_token_acc = ((hyp_tgts == targets).float() * (affix_masks.float())).sum() / (
            affix_masks.float().sum() + EPSILON
        )

        # for PPL
        sent_lprobs = lprobs.gather(dim=-1, index=targets.unsqueeze(-1))
        masked_lprobs = (sent_lprobs.squeeze(-1) * masks.float()).sum()  # sentence prob
        masked_prefix_lprobs = (
            sent_lprobs.squeeze(-1) * prefix_masks.float()
        ).sum()  # sentence prob
        masked_affix_lprobs = (sent_lprobs.squeeze(-1) * affix_masks.float()).sum()  # sentence prob

        frame_size = masks.float().sum()
        tgt_size = weighted_masks.float().sum()  # tokens

        forward_out = OrderedDict()
        forward_out['utt_num'] = targets.shape[0]
        forward_out['backward_loss'] = masked_loss
        forward_out['loss'] = masked_loss
        forward_out['nll_loss'] = masked_nll_loss
        forward_out['prefix_nll_loss'] = masked_prefix_nll_loss
        forward_out['affix_nll_loss'] = masked_affix_nll_loss

        forward_out['lprobs'] = masked_lprobs  # for PPL for sentence
        forward_out['prefix_lprobs'] = masked_prefix_lprobs  # for PPL for sentence
        forward_out['affix_lprobs'] = masked_affix_lprobs  # for PPL for sentence

        forward_out['acc'] = hyp_token_acc
        forward_out['prefix_acc'] = hyp_prefix_token_acc
        forward_out['affix_acc'] = hyp_affix_token_acc

        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size

        return forward_out
