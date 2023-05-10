# pylint: disable=cell-var-from-loop
''' LasCE '''
from collections import OrderedDict

import torch
import torch.nn.functional as F
from .criterion import Xentropy, LabelSmoothingLoss

EPSILON = 1e-5


class LasCE(Xentropy):
    """Xentropy including las beam search

    .. note:: 注意tgt_dict 的设置需要依靠mata参数,请正确设置。

    Args:

        - args: the config

    Example::

        las_criterion_type='LasCE',
        label_smooth_factor=0.1,
        meta_file="meta",

    """

    def __init__(self, args):
        super().__init__(args)

        self.bos = self.args.tgt_dict.bos()
        self.eos = self.args.tgt_dict.eos()
        self.blank = self.args.tgt_dict.blank()

    @torch.no_grad()
    def greedy_infer(
        self,
        encoder_out,
        encoder_out_mask,
        decoder_module,
        max_decoder_positions=1000,
        max_len_a=1.0,
        max_len_b=10,
    ):
        '''
        LAS Greedy Inference
            Args:
                encoder_out: tensor or list, encoder_out (for each stream)
                encoder_out_mask: tensor or list, encoder_out_mask (for each stream)
                input_dict: batch_data
            Return:
                return_word_list: a list of sentences: [B]

        '''
        # TODO(tuming): add streaming decoder inference
        # use rnnt nbest src length as length constrain seems better
        if isinstance(encoder_out, list):
            bsz = encoder_out[1].shape[0]
            src_len = encoder_out[1].shape[1]
            device = encoder_out[1].device
        else:
            bsz = encoder_out.shape[0]  # (B, T, ndim)
            src_len = encoder_out.shape[1]  # padded input length
            device = encoder_out.device
        # max output length
        max_len = min(
            int(max_len_a * src_len + max_len_b),
            # exclude the EOS marker
            max_decoder_positions - 1,
        )
        num_remaining_sent = bsz
        # Output tokens
        tokens = torch.zeros(bsz, max_len + 2, device=device).long()
        tokens[:, 0] = self.bos
        output_mask = torch.ones(bsz, max_len + 2, device=device).float()

        if hasattr(decoder_module, 'incremental_states'):
            decoder_module.incremental_states = None
        for step in range(0, max_len - 1):
            assert num_remaining_sent >= 0, "Error occured, num remaining sent < 0"
            lprobs, _ = decoder_module.step(
                tokens[:, : step + 1],
                encoder_out,
                encoder_out_mask,
            )
            if (
                step > 0 and decoder_module.incremental_states is None
            ):  # not using incremental_states and step > 0
                lprobs = lprobs[:, -1, :]
            tokens[:, step + 1] = torch.argmax(lprobs, dim=-1) * output_mask[:, step]
            finished_idxs = (tokens[:, step + 1] == self.eos).nonzero(as_tuple=True)[0]
            output_mask[finished_idxs, step + 1 :] = 0
            num_remaining_sent -= finished_idxs.size(0)
            if num_remaining_sent == 0:
                break
        # Remove bos at front
        output = tokens[:, 1:]
        output_lengths = output_mask[:, 1:].sum(dim=-1)
        return output, output_lengths


class LasMWER(LasCE):
    """LasMWER, n-best version impelement

    .. note::
    相关资料你可以查询
    https://arxiv.org/pdf/1712.01818.pdf

    """

    def forward(self, encoder_out, encoder_out_mask, input_dict, decoder_module, reverse=False):
        # pylint:disable=too-many-locals
        src_mask = input_dict['src_mask']
        target = input_dict['char']
        (bsz, _) = target.size()
        if not reverse:
            prev_tgt = input_dict['prev_char']
            sample_target = input_dict['nbest_sample']
        else:
            prev_tgt = input_dict['prev_char_rev']
            sample_target = input_dict['nbest_sample_rev']
        target_mask = input_dict['char_mask']

        sample_target_mask = input_dict['nbest_sample_mask']

        beam_size = sample_target.size(1)
        nbest_tgt_len = sample_target.size(2)
        ### compute ce loss
        logits = decoder_module.forward(encoder_out, encoder_out_mask, prev_tgt)
        lprobs = F.log_softmax(logits.float(), dim=-1)
        hyp_tgt = lprobs.max(dim=-1)[1]
        hyp_acc = ((hyp_tgt == target).float() * (target_mask.float())).sum() / (
            target_mask.float().sum() + EPSILON
        )
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1))
        smooth_loss = -lprobs.sum(dim=-1, keepdim=False)
        nll_loss = nll_loss.squeeze(-1)
        eps_i = self.args.label_smooth_factor / lprobs.size(-1)
        loss = (1.0 - self.args.label_smooth_factor) * nll_loss + eps_i * smooth_loss

        # average by sequence
        # per sequence loss
        masked_loss = (loss * target_mask.float()).sum(-1) / (target_mask.float().sum(-1) + EPSILON)

        ### compute mwer loss
        sample_logits = decoder_module.forward_mwer(encoder_out, encoder_out_mask, input_dict)
        # B, N_sample, tgt_len, tgt_dim
        sample_logits = sample_logits.view(bsz, beam_size, nbest_tgt_len, -1)
        sample_lprobs = F.log_softmax(sample_logits.float(), dim=-1)
        sample_nll_loss = sample_lprobs.gather(dim=-1, index=sample_target.unsqueeze(-1))
        sample_nll_loss = sample_nll_loss.squeeze(-1)
        masked_sample_nll_loss = (sample_nll_loss * sample_target_mask.float()).sum(-1)
        # mask where hyps_num < beam_size
        sample_mask = sample_target_mask.sum(dim=-1).eq(0)
        masked_sample_nll_loss = masked_sample_nll_loss.masked_fill(sample_mask, -1000.0)

        word_errors = input_dict['word_errors']
        total_words = input_dict['total_words']

        # word error nums as reward
        wer = word_errors / total_words.unsqueeze(-1) if self.args.use_wer else word_errors
        avg_wer = wer.sum(1, keepdim=True) / ((~sample_mask).sum(1, keepdim=True) + EPSILON)

        # original version
        # # normalize by softmax, average by sequence
        # mwer_loss = (F.softmax(masked_sample_nll_loss, -1) * ((wer - avg_wer).float())).sum(-1)

        # stable version
        hyp_sent_probs = F.softmax(masked_sample_nll_loss, -1).detach()
        seq_prob_mask = (
            torch.max(hyp_sent_probs, dim=1)[0] < self.args.seq_prob_threshold
        ).type_as(hyp_sent_probs)
        mwer_loss = (
            F.log_softmax(masked_sample_nll_loss, -1).masked_fill(sample_mask, 0)
            * hyp_sent_probs
            * (wer - avg_wer)
        ).sum(-1) * seq_prob_mask
        mwer_loss = mwer_loss.sum() / (seq_prob_mask.sum() + EPSILON)
        total_loss = mwer_loss + (masked_loss * self.args.lambda_ce_factor).mean()
        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()  # tokens
        best_wer = torch.gather(
            word_errors, 1, torch.argmax(masked_sample_nll_loss, dim=-1).unsqueeze(1)
        ).squeeze(1)
        worst_wer = torch.gather(
            word_errors, 1, torch.argmin(masked_sample_nll_loss, dim=-1).unsqueeze(1)
        ).squeeze(1)
        forward_out = OrderedDict()
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = total_loss
        forward_out['loss'] = total_loss.type_as(logits)
        forward_out['sdt_loss'] = mwer_loss.type_as(logits)
        forward_out['nll_loss'] = masked_loss.mean().type_as(logits)
        forward_out['acc'] = hyp_acc

        forward_out['avg_wer'] = (word_errors / total_words.unsqueeze(-1)).mean()
        forward_out['avg_best_wer'] = (best_wer / total_words).mean()
        forward_out['avg_worst_wer'] = (worst_wer / total_words).mean()
        forward_out['expected_word_errs'] = (
            (hyp_sent_probs * (word_errors.float())).sum(-1)
        ).mean()
        forward_out['avg_errs'] = (word_errors).mean()
        forward_out['avg_best_errs'] = best_wer.mean()
        forward_out['avg_worst_errs'] = worst_wer.mean()
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size
        return forward_out


class CTCLasCE(LasCE):
    '''CTC LasCE'''

    def __init__(self, args):
        super().__init__(args)
        self.pad = self.args.tgt_dict.pad()
        self.normalize_length = args.get("normalize_length_loss", False)
        self.decoder_loss = LabelSmoothingLoss(
            args.tgt_vocab_size,
            self.pad,
            smoothing=args.label_smooth_factor,
            normalize_length=self.normalize_length,
        )

    def forward(
        self,
        logits,
        src_mask,
        target,
        target_mask,
        encoder_out_mask=None,
        mtl_logits=None,
        mtl_type=None,
    ):

        """forward"""
        target = target * target_mask
        target = torch.where(
            target == 0,
            (torch.ones_like(target, dtype=torch.long) * self.pad).to(target.device),
            target.long(),
        )
        lprobs = F.log_softmax(logits.float(), dim=-1)
        hyp_tgt = lprobs.max(dim=-1)[1]
        hyp_acc = ((hyp_tgt == target).float() * (target_mask.float())).sum() / (
            target_mask.float().sum() + EPSILON
        )
        decoder_loss = self.decoder_loss(logits, target)

        # mtl loss, e.g ctc
        if mtl_logits is not None and mtl_type == 'ctc':
            ctc_lprobs = mtl_logits.log_softmax(dim=-1, dtype=torch.float32)
            input_lengths = (encoder_out_mask.sum(dim=1)).int()
            ctc_target_lengths = (target_mask.sum(dim=1)).int()
            ctc_target = target.int()
            if self.args.use_eos and self.args.skip_ctc_eos:
                ctc_target[ctc_target == self.eos] = self.pad
                ctc_target = ctc_target[:, :-2]
                ctc_target_lengths = ctc_target_lengths - 1

            ctc_loss = F.ctc_loss(
                ctc_lprobs.transpose(0, 1).contiguous(),
                ctc_target.int(),
                input_lengths,
                ctc_target_lengths,
                blank=self.blank,
                reduction='sum',
                zero_infinity=True,
            )
            ctc_loss = ctc_loss / ctc_target.shape[0]
            backward_ctc_loss = self.args.mtl_weight * ctc_loss
        else:
            ctc_loss = 0.0
            backward_ctc_loss = 0.0

        # for PPL
        sent_lprobs = lprobs.gather(dim=-1, index=target.unsqueeze(-1))
        masked_lprobs = (sent_lprobs.squeeze(-1) * target_mask.float()).sum(dim=1)  # sentence prob
        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()  # tokens
        forward_out = OrderedDict()
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = (1 - self.args.mtl_weight) * decoder_loss + backward_ctc_loss
        forward_out['loss'] = decoder_loss.type_as(logits) + backward_ctc_loss
        forward_out['ctc_loss'] = ctc_loss
        forward_out['lprobs'] = masked_lprobs.type_as(logits)  # for PPL for sentence
        forward_out['nll_loss'] = decoder_loss.type_as(logits)
        forward_out['acc'] = hyp_acc
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size
        # TODO(zhangjun) add CER
        # forward_out['cer'] = tgt_size * 0. + error_dist
        # forward_out['dist'] = tgt_size * 0. + total_dist

        return forward_out
