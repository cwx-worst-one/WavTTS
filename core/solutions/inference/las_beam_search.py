''' las beam search '''

import math
from itertools import groupby
from typing import Dict, List
import torch


# for las beamsearch decoding
class Search:
    """single step search"""

    def __init__(self, tgt_dict):
        '''init.'''
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.scores_buf = None
        self.indices_buf = None
        self.beams_buf = None
        self.src_lengths = None

    def _init_buffers(self, t):
        '''
        init buffer
        '''
        if self.scores_buf is None:
            self.scores_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)
            self.beams_buf = torch.LongTensor().to(device=t.device)

    def step(self, step, lprobs, scores):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    def set_src_lengths(self, src_lengths):
        '''
        set src length
        '''
        self.src_lengths = src_lengths


class BeamSearch(Search):
    """single step beam search"""

    def step(self, step, lprobs, scores):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        self.scores_buf.resize_(0)
        self.indices_buf.resize_(0)
        self.beams_buf.resize_(0)
        torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )
        torch.div(self.indices_buf, vocab_size, out=self.beams_buf, rounding_mode='floor')
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.indices_buf, self.beams_buf


class BaseBeamSearch:
    '''
    Beam inference
    Args:
        encoder_outs (tensor or list): encoder_out (for each stream)
        encoder_out_masks (tensor or list): encoder_out_mask (for each stream)
        max_len_a/b (int, optional): generate sequences of maximum length
            ax + b, where x is the source length
        normalize_scores (bool, optional): normalize scores by the length
            of the output (default: True)
        min_len (int, optional): the minimum length of the generated output
            (not including end-of-sentence)
        len_penalty (float, optional): length penalty, where <1.0 favors
            shorter, >1.0 favors longer sentences (default: 1.0)
        unk_penalty (float, optional): unknown word penalty, where <0
            produces more unks, >0 produces fewer (default: 0.0)
        temperature (float, optional): temperature, where values
            >1.0 produce more uniform samples and values <1.0 produce
            sharper samples (default: 1.0)
        coverage_weight (float, optional): attention coverage weight for beam scores
    '''

    def __init__(
        self,
        cfg,
        inference_cfg,
        decoder_module,
        lm_solution=None,
    ):
        '''init.'''
        self.decoder_module = decoder_module
        self.lm_solution = lm_solution

        self.pad = cfg.tgt_dict.pad()
        self.unk = cfg.tgt_dict.unk()
        self.bos = cfg.tgt_dict.bos()
        self.eos = cfg.tgt_dict.eos()
        self.blank = cfg.tgt_dict.blank()

        self.max_decoder_positions = 10000
        self.streaming_decoder = cfg.get(
            "streaming_decoder", False
        )  # indicate at which training step
        if self.streaming_decoder:
            self.decoder_chunk_size = cfg.get("decoder_chunk_size", 8)  # chunk size
            self.decoder_left_chunk_num = cfg.get(
                "decoder_left_chunk_num", 4
            )  # number of encoder frames to look at at current step
            self.decoder_right_peak = cfg.get(
                "decoder_right_peak", 1
            )  # number of future peaks to look at at current step
        self.ctc_max_len = cfg.get("ctc_max_len", False)
        self.no_repeat_ngram_size = cfg.get("no_repeat_ngram_size", 0)

        self.beam_size = inference_cfg.get('las_beam_size', 10)
        self.lm_weight = inference_cfg.get('lm_weight', 1.0)
        self.nbest_out = inference_cfg.get('nbest_out', False)
        self.filter_list = inference_cfg.get('filter_list', [])
        self.temperature = inference_cfg.get('temperature', 1.0)
        self.max_len_a = inference_cfg.get('max_len_a', 1.0)
        self.max_len_b = inference_cfg.get('max_len_b', 10)
        self.min_len = inference_cfg.get('min_len', 1)
        self.normalize_scores = inference_cfg.get('normalize_scores', True)
        self.len_penalty = inference_cfg.get('len_penalty', 1.0)
        self.unk_penalty = inference_cfg.get('unk_penalty', 0.0)
        self.eos_factor = inference_cfg.get('eos_factor', None)
        self.coverage_weight = inference_cfg.get('coverage_weight', 0.0)
        self.lingvo_beamsearch = inference_cfg.get('lingvo_beamsearch', False)
        self.beam_width = inference_cfg.get('beam_width', 3.0)
        # pylint:disable=invalid-name
        self.valid_eos_max_logit_delta = inference_cfg.get('valid_eos_max_logit_delta', 5.0)
        self.max_len = inference_cfg.get('max_len', 200)
        self.buffers = {}
        self.search = BeamSearch(cfg.tgt_dict)

    @staticmethod
    def reorder_encoder_out(encoder_out, new_order):
        '''encoder_out shape (B,T,N) new_order shape (B*beam)'''
        for name, tensor in encoder_out.items():
            encoder_out[name] = tensor.index_select(0, new_order)
        return encoder_out

    def reorder_states(self, encoder_outs, encoder_out_masks, new_order, decoder_module=None):
        '''reorder encoder and decoder states'''
        if decoder_module is not None:
            decoder_module.reorder_incremental_state(new_order)

        encoder_out_dict = {}
        if isinstance(encoder_outs, list):
            for i, encoder_out in enumerate(encoder_outs):
                encoder_out_dict['encoder_out_stream{}'.format(i)] = encoder_out
                encoder_out_dict['encoder_out_mask_stream{}'.format(i)] = encoder_out_masks[i]
        else:
            encoder_out_dict['encoder_out'] = encoder_outs
            encoder_out_dict['encoder_out_mask'] = encoder_out_masks
        encoder_out_dict = self.reorder_encoder_out(encoder_out_dict, new_order)
        if isinstance(encoder_outs, list):
            encoder_outs = [
                encoder_out_dict['encoder_out_stream{}'.format(i)] for i in range(len(encoder_outs))
            ]
            encoder_out_masks = [
                encoder_out_dict['encoder_out_mask_stream{}'.format(i)]
                for i in range(len(encoder_out_masks))
            ]
        else:
            encoder_outs = encoder_out_dict['encoder_out']
            encoder_out_masks = encoder_out_dict['encoder_out_mask']
        return encoder_outs, encoder_out_masks

    @staticmethod
    def calculate_banned_tokens(
        tokens,
        step: int,
        gen_ngrams: List[Dict[str, List[int]]],
        no_repeat_ngram_size: int,
        bbsz_idx: int,
    ):
        '''calculate_banned_tokens'''
        tokens_list: List[int] = tokens[
            bbsz_idx, step + 2 - no_repeat_ngram_size : step + 1
        ].tolist()
        # before decoding the next token, prevent decoding of ngrams that have already appeared
        ngram_index = ",".join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    @staticmethod
    def transpose_list(l: List[List[int]]):
        '''transpose_list'''
        # GeneratorExp aren't supported in TS so ignoring the lint
        min_len = min(len(x) for x in l)  # noqa
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        '''no_repeat_ngram'''
        # for each beam and batch sentence, generate a list of previous ngrams
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {}) for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list(
                [gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
            ):
                key = ",".join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
                    key, torch.jit.annotate(List[int], [])
                ) + [ngram[-1]]

        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(
                    tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
                )
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
            ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][torch.tensor(banned_tokens[bbsz_idx]).long()] = torch.tensor(
                -math.inf
            ).to(lprobs)
        return lprobs

    def step_enc_dec_mask(self, peak_times, step, cur_time, encoder_out_mask, device):
        '''step_enc_dec_mask'''
        chunk_width = self.decoder_chunk_size
        left_chunk_num = self.decoder_left_chunk_num
        if len(peak_times) > 0:
            if step >= len(peak_times):
                peak_idx = -1
            else:
                peak_idx = step
            left = max(
                0,
                (
                    (
                        peak_times[peak_idx] // chunk_width
                        if peak_times[peak_idx] % chunk_width
                        else peak_times[peak_idx] // chunk_width - 1
                    )
                    - left_chunk_num
                )
                * chunk_width,
            )
        else:
            left = 0

        if step + 1 < len(peak_times):
            right = (
                min(
                    cur_time,
                    (peak_times[step + self.decoder_right_peak] // chunk_width + 1) * chunk_width,
                )
                if peak_times[step + self.decoder_right_peak] % chunk_width
                else peak_times[step + self.decoder_right_peak]
            )
        else:
            right = cur_time
        att_boundary = [left, right]
        enc_dec_mask = torch.zeros_like(encoder_out_mask).to(device)
        enc_dec_mask[:, att_boundary[0] : att_boundary[1]] = 1

        return enc_dec_mask

    def buffer(self, name, type_of):  # noqa
        '''buffer'''
        if name not in self.buffers:
            self.buffers[name] = type_of.new()
        return self.buffers[name]

    def is_finished(self, sent, finalized):
        """
        Check whether we've finished generation for a given sentence, by
        comparing the worst score among finalized hypotheses to the best
        possible score among unfinalized hypotheses.
        """
        assert len(finalized[sent]) <= self.beam_size
        if len(finalized[sent]) == self.beam_size:
            return True
        return False

    def finalize_hypos(self, step, bbsz_idx, eos_scores, tokens, attn, scores, finished, finalized):
        """
        Finalize the given hypotheses at this step, while keeping the total
        number of finalized hypotheses per sentence <= beam_size.

        Note: the input must be in the desired finalization order, so that
        hypotheses that appear earlier in the input are preferred to those
        that appear later.

        Args:
            step: current time step
            bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                indicating which hypotheses to finalize
            eos_scores: A vector of the same size as bbsz_idx containing
                scores for each hypothesis
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors
        tokens_clone = tokens.index_select(0, bbsz_idx)
        tokens_clone = tokens_clone[:, 1 : step + 2]  # skip the first index, which is EOS
        assert not tokens_clone.eq(self.eos).any()
        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2] if attn is not None else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            if self.streaming_decoder:
                eos_scores /= (5 + step) ** self.len_penalty / 6**self.len_penalty
            else:
                eos_scores /= (step + 1) ** self.len_penalty

        cum_unfin = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        sents_seen = set()
        for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
            unfin_idx = idx // self.beam_size
            sent = unfin_idx + cum_unfin[unfin_idx]

            sents_seen.add((sent, unfin_idx))

            # pylint: disable=cell-var-from-loop
            def get_hypo():

                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = None

                return {
                    'tokens': tokens_clone[i],
                    'score': score,
                    'attention': hypo_attn,  # src_len x tgt_len
                    'alignment': None,
                    'positional_scores': pos_scores[i],
                }

            if len(finalized[sent]) < self.beam_size:
                finalized[sent].append(get_hypo())

        newly_finished = []
        for sent, unfin_idx in sents_seen:
            # check termination conditions for this sentence
            if not finished[sent] and self.is_finished(sent, finalized):
                finished[sent] = True
                newly_finished.append(unfin_idx)
        return newly_finished

    def __call__(
        self,
        encoder_out,
        encoder_out_mask,
        mtl_logits=None,
    ):
        '''call'''
        # pylint:disable=too-many-branches,too-many-locals,too-many-statements
        assert (
            self.eos_factor is None or self.eos_factor >= 1.0
        ), '--eos-factor must be >= 1.0 if set'

        if isinstance(encoder_out, list):
            bsz = encoder_out[1].shape[0]  # use rnnt nbest src length as length constrain
            src_lengths = encoder_out_mask[1].sum(dim=1)
            src_len = encoder_out[1].shape[1]
            device = encoder_out[1].device
        else:
            bsz = encoder_out.shape[0]  # (B, T, ndim)
            src_lengths = encoder_out_mask.sum(dim=1)  # real input length
            src_len = encoder_out.shape[1]  # padded input length
            device = encoder_out.device

        # max output length
        max_len = min(
            int(self.max_len_a * src_len + self.max_len_b),
            # exclude the EOS marker
            self.max_decoder_positions - 1,
        )

        ############ Acoustic ############
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)  # (B, beam)
        new_order = new_order.to(device).long()
        encoder_out, encoder_out_mask = self.reorder_states(
            encoder_out, encoder_out_mask, new_order
        )  # (B*beam, T, dim)

        # get ctc peak times
        if (self.streaming_decoder and mtl_logits is not None) or self.ctc_max_len:
            ys_hat_ctc = torch.argmax(mtl_logits, dim=2)
            peak_times = []
            cur_time = 0
            for x in groupby(ys_hat_ctc[0]):  # batch size =1 for streaming decoder
                cur_time += len(list(x[1]))
                if x[0] != self.blank:
                    peak_times.append(cur_time)
            if len(peak_times) == 0 and (self.streaming_decoder and mtl_logits is not None):
                return [[]]

            if self.streaming_decoder:
                max_len = len(peak_times) + 2
            elif self.ctc_max_len:
                max_len = int(len(peak_times) * 1.05) + 3

        if self.lingvo_beamsearch:
            max_len = self.max_len
        # initialize buffers
        scores = torch.zeros(bsz * self.beam_size, max_len + 1, device=device).float()
        scores_buf = scores.clone()
        tokens = torch.ones(bsz * self.beam_size, max_len + 2, device=device).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.bos
        attn, attn_buf = None, None
        coverage, coverage_buf = None, None

        if self.lingvo_beamsearch:
            # mark the done hyps when beam search
            # done_hyp [B, hyps_per_beam, max_len+2]
            hyp_tokens = (
                torch.ones(bsz, self.beam_size, max_len + 2, device=device).long().fill_(self.pad)
            )
            hyp_scores = torch.zeros(bsz, self.beam_size, max_len + 2, device=device).fill_(
                -math.inf
            )
            is_done_hyps = torch.zeros(bsz, self.beam_size, max_len + 2, device=device).eq(-1)
            done_scores = torch.zeros(bsz, self.beam_size, max_len + 2, device=device).fill_(
                -math.inf
            )
            hyp_pre = torch.ones(bsz, self.beam_size, max_len + 2, device=device).long().fill_(0)
        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = torch.zeros(bsz, self.beam_size, device=device).eq(-1)
        # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * self.beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * self.beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        self.buffers = {}

        new_order = None
        batch_idxs = None
        if hasattr(self.decoder_module, 'reset_state'):
            self.decoder_module.reset_state()
        if hasattr(self.decoder_module, 'incremental_states'):
            self.decoder_module.incremental_states = None

        # lm fusion part
        # if hasattr(model.nnlm_model, 'incremental_states'):
        #     model.nnlm_model.incremental_states = None

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams,
            if new_order is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    new_order.view(-1, self.beam_size).add_(corr.unsqueeze(-1) * self.beam_size)

                encoder_out, encoder_out_mask = self.reorder_states(
                    encoder_out, encoder_out_mask, new_order, self.decoder_module
                )

            if self.streaming_decoder:
                enc_dec_mask = self.step_enc_dec_mask(
                    peak_times, step, cur_time, encoder_out_mask, device
                )

            lprobs, avg_attn_scores = self.decoder_module.forward_step(
                tokens[:, : step + 1],
                encoder_out,
                encoder_out_mask if not self.streaming_decoder else enc_dec_mask,
                temperature=self.temperature,
                streaming=self.streaming_decoder,
            )

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle min and max length constraints
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf
            elif step < self.min_len or (self.streaming_decoder and step < max_len - 4):
                # minimum length constraint (does not apply if using prefix_tokens)
                if not self.lingvo_beamsearch:
                    lprobs[:, self.eos] = -math.inf
            elif self.eos_factor is not None:
                # only consider EOS if its score is no less than a specified
                # factor of the best candidate score
                disallow_eos_mask = lprobs[:, self.eos] < self.eos_factor * lprobs.max(dim=1)[0]
                lprobs[disallow_eos_mask, self.eos] = -math.inf

            # # handle prefix tokens (possibly with different lengths)
            # prefix_tokens = None
            # TODO (houjunfeng) add prefix token

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * self.beam_size, src_len, max_len + 2)
                    coverage = scores.new_full([bsz * self.beam_size, src_len], 0.0)
                    attn_buf = attn.clone()
                    coverage_buf = coverage.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)
                if self.coverage_weight > 0:
                    coverage.add_(avg_attn_scores)
                    # TODO: hard-code the numbers below for now
                    frames_covered = (coverage > 0.5).float().sum(1, keepdim=True)
                    frames_covered -= (
                        torch.where(
                            coverage - 1.0 > 0.0, coverage - (1.0 - 0.7), coverage.new([0.0])
                        )
                    ).sum(1, keepdim=True)
                    lprobs.add_(self.coverage_weight, frames_covered)

            if self.lingvo_beamsearch:
                lprobs_eos = lprobs.clone()
                lprobs[:, self.eos] -= math.inf
            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = self.buffer('eos_bbsz_idx', type_of=tokens)
            eos_scores = self.buffer('eos_scores', type_of=scores)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, self.beam_size, step)

            self.search.set_src_lengths(src_lengths)

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, lprobs.shape[-1]),
                scores.view(bsz, self.beam_size, -1)[:, :, :step],
            )
            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            if self.lingvo_beamsearch:
                beam_scores = lprobs_eos.view(bsz, -1, lprobs_eos.shape[-1])
                if step != 0:
                    beam_scores += scores.view(bsz, self.beam_size, -1)[:, :, step - 1].unsqueeze(
                        -1
                    )
                topk_per_hyp_scores = self.buffer("topk_per_hyp_scores", type_of=lprobs)
                topk_per_hyp_indices = self.buffer("topk_per_hyp_indices", type_of=tokens)
                torch.topk(
                    beam_scores,
                    k=self.beam_size + 1,
                    dim=2,
                    out=(topk_per_hyp_scores, topk_per_hyp_indices),
                )
                scores_valid = topk_per_hyp_scores.ge(
                    topk_per_hyp_scores.max(-1, keepdims=True)[0] - self.valid_eos_max_logit_delta
                )
                eos_valid = topk_per_hyp_indices.eq(self.eos)
                valid_beam_end_with_eos = scores_valid.logical_and(eos_valid).any(-1)
                is_done_hyps[:, :, step] = valid_beam_end_with_eos
                if step == 0:
                    is_done_hyps[:, 1:, step] = False
                done_scores[:, :, step] = beam_scores[:, :, self.eos]
                done_scores[:, :, step].masked_fill_(~is_done_hyps[:, :, step], -math.inf)
                hyp_scores[:, :, step] = cand_scores[:, : self.beam_size]
                hyp_tokens[:, :, step] = cand_indices[:, : self.beam_size]
                hyp_pre[:, :, step] = cand_beams[:, : self.beam_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            # finalize hypotheses that end in eos (except for blacklisted ones)
            eos_mask = cand_indices.eq(self.eos)
            eos_mask[:, : self.beam_size][blacklist] = 0
            # only consider eos when it's among the top beam_size indices
            torch.masked_select(
                cand_bbsz_idx[:, : self.beam_size],
                mask=eos_mask[:, : self.beam_size],
                out=eos_bbsz_idx,
            )

            finalized_sents = set()
            eos_scores.resize_(0)
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, : self.beam_size],
                    mask=eos_mask[:, : self.beam_size],
                    out=eos_scores,
                )
                finalized_sents = self.finalize_hypos(
                    step, eos_bbsz_idx, eos_scores, tokens, attn, scores, finished, finalized
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if not self.lingvo_beamsearch:
                assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                # if prefix_tokens is not None:
                #     prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * self.beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * self.beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * self.beam_size, attn.size(1), -1
                    )
                    attn_buf.resize_as_(attn)
                if coverage is not None:
                    coverage = coverage.view(bsz, -1)[batch_idxs].view(new_bsz * self.beam_size, -1)
                    coverage_buf.resize_as_(coverage)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            active_mask = self.buffer('active_mask', type_of=tokens)
            active_mask.resize_(0)
            eos_mask[:, : self.beam_size] |= blacklist
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = self.buffer('active_hypos', type_of=tokens), self.buffer(
                'new_blacklist', type_of=tokens
            )
            torch.topk(
                active_mask,
                k=self.beam_size,
                dim=1,
                largest=False,
                out=(new_blacklist, active_hypos),
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, : self.beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = self.buffer('active_bbsz_idx', type_of=tokens)
            torch.gather(
                cand_bbsz_idx,
                dim=1,
                index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores,
                dim=1,
                index=active_hypos,
                out=scores[:, step].view(bsz, self.beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, : step + 1],
                dim=0,
                index=active_bbsz_idx,
                out=tokens_buf[:, : step + 1],
            )
            torch.gather(
                cand_indices,
                dim=1,
                index=active_hypos,
                out=tokens_buf.view(bsz, self.beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step],
                    dim=0,
                    index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores,
                dim=1,
                index=active_hypos,
                out=scores_buf.view(bsz, self.beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, : step + 2],
                    dim=0,
                    index=active_bbsz_idx,
                    out=attn_buf[:, :, : step + 2],
                )
            if coverage is not None:
                torch.index_select(coverage, dim=0, index=active_bbsz_idx, out=coverage_buf)

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn
            if coverage is not None:
                coverage, coverage_buf = coverage_buf, coverage

            # reorder incremental state in decoder
            new_order = active_bbsz_idx
            if self.lingvo_beamsearch:
                done_scores_tmp = done_scores.view(bsz, -1).max(-1)[0] - self.beam_width
                cand_scores_tmp = cand_scores[:, self.beam_size - 1]
                if done_scores_tmp.ge(cand_scores_tmp).all() or step == max_len:
                    break
        # sort by score descending

        for sent, finalized_term in enumerate(finalized):
            finalized[sent] = sorted(finalized_term, key=lambda r: r['score'], reverse=True)
        if self.lingvo_beamsearch:
            finalized = self.get_finalized_topk_hyps(
                bsz, is_done_hyps, done_scores, hyp_scores, hyp_tokens, hyp_pre, max_len
            )
        return finalized

    def get_finalized_topk_hyps(
        self, bsz, is_done_hyps, done_scores, hyp_scores, hyp_tokens, hyp_pre, max_len
    ):
        '''get hyps for lingvo beam search'''
        finalized = [[] for i in range(bsz)]
        for i in range(bsz):
            for j in range(self.beam_size):
                for k in range(max_len + 1):
                    if is_done_hyps[i, j, k]:
                        beam_id = j
                        token_ids = [self.eos]
                        token_scores = [done_scores[i, j, k].tolist()]
                        for t in reversed(range(k)):
                            token_ids.append(hyp_tokens[i, beam_id, t].tolist())
                            token_scores.append(hyp_scores[i, beam_id, t].tolist())
                            beam_id = hyp_pre[i, beam_id, t]
                        token_ids = torch.Tensor(token_ids).long().flip(-1)
                        token_scores = torch.Tensor(token_scores).flip(-1)
                        positional_scores = token_scores.clone()
                        positional_scores[1:] = positional_scores[1:] - positional_scores[:-1]
                        norm_score = (
                            token_scores[-1].tolist()
                            / (6 + k) ** self.len_penalty
                            * 5**self.len_penalty
                        )
                        token_confidence = torch.exp(positional_scores)
                        sentence_confidence = torch.mean(token_confidence[:-1])
                        valid_res = {
                            'tokens': token_ids.tolist(),
                            'score': norm_score,
                            'attention': None,  # src_len x tgt_len
                            'alignment': None,
                            'positional_scores': positional_scores.tolist(),
                            'token_confidence': token_confidence.tolist(),
                            'sentence_confidence': sentence_confidence.tolist(),
                        }
                        finalized[i].append(valid_res)
            finalized[i] = sorted(finalized[i], key=lambda r: r['score'], reverse=True)[
                : self.beam_size
            ]
        return finalized
