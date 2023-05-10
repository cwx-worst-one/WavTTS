''' beam search '''
# pylint: disable=line-too-long, cell-var-from-loop, too-many-statements
# pylint: disable=too-many-branches, too-many-nested-blocks, too-many-locals
import copy
import numpy as np
import torch
from core.utils import FalconDict
from core.models.layers.transformer import CifSelfAttentionModule
from core.models.asr.cif_decoder import attention_bias_lower_triangle
from .hypothesis import HotwordToken

INT_MAX = 2147483647


def log_sum(tensor_a, tensor_b):
    '''log_sum'''
    return torch.logsumexp(torch.tensor([tensor_a, tensor_b]), 0)


def log_prob_from_logits(logits, dim=2):
    '''log_prob_from_logits'''
    return logits - torch.logsumexp(logits, dim=dim, keepdim=True)


def gather_nd(x, indices):
    '''gather_nd'''
    newshape = indices.shape[:-1] + x.shape[indices.shape[-1] :]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    # pylint:disable=unnecessary-dunder-call
    out = torch.cat([x.__getitem__(tuple(i)) for i in indices])
    return out.reshape(newshape)


def reform(value, coordinates, shape):
    '''reform'''
    flatten_value = torch.reshape(value, shape)
    reformed_value = torch.reshape(
        gather_nd(flatten_value, coordinates), (shape[0] * shape[1], shape[2], shape[3])
    )
    return reformed_value


def compute_batch_indices(batch_size, beam_size, device='cuda'):
    """Computes the i'th coodinate that contains the batch index for gathers.

    Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    batch the beam item is in. This will create the i of the i,j coordinate
    needed for the gather.

    Args:
      batch_size: Batch size
      beam_size: Size of the beam.
    Returns:
      batch_pos: [batch_size, beam_size] tensor of ids
    """
    batch_pos = torch.arange(batch_size * beam_size, device=device) // beam_size
    batch_pos = torch.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos


class Hypothesis(FalconDict):
    """Hypothesis definition for CIF beam search"""

    def __init__(self, **kwargs):
        '''init Hypothesis'''
        self.label_seq = [0]  # yseq, predicted tokens
        self.score = 0.0
        self.lm_states = None
        # for fst-fusion
        self.hotword_score = 0.0
        self.coldword_state = 0

        # for multi step search
        self.prev_total_lm_score = 0.0
        self.lm_tokens = []
        self.matched_hotwords = []
        self.lm_token_timestamp = 0
        super().__init__(**kwargs)

    def copy(self):
        '''return a mem-copy obj'''
        hyp = Hypothesis(
            label_seq=self.label_seq[:],
            score=self.score,
            lm_states=self.lm_states,
            hotword_score=self.hotword_score,
            coldword_state=self.coldword_state,
        )
        hyp.prev_total_lm_score = self.prev_total_lm_score
        hyp.lm_tokens = self.lm_tokens[:]
        hyp.matched_hotwords = self.matched_hotwords[:]
        hyp.lm_token_timestamp = self.lm_token_timestamp  # used to sort the lm tokens
        return hyp

    @property
    def yseq(self):
        '''string the tokens list'''
        return ' '.join(map(str, self.label_seq))

    def init_lm_tokens(self, fst_num, fst_states):
        """Init lm tokens"""
        assert fst_num == len(fst_states)
        self.lm_tokens = []
        for i in range(0, fst_num):
            token = HotwordToken()
            token.fst_idx = i
            token.state = fst_states[i]
            self.lm_tokens.append(token)


class BaseBeamSearch:
    '''base class of beam search of cif-based model'''

    def __init__(
        self,
        args,
        cfg,
        decoder_module,
        lm_solution=None,
    ):
        '''cfg: beam search required configuration file'''
        self.args = args
        self.decoder_module = decoder_module
        self.lm_solution = lm_solution

        self.beam_size = cfg.get('beam_size', 0)
        self.nbest = cfg.get('nbest', 1)
        self.cif_temperature = cfg.get('cif_temperature', 1.0)

    def __call__(self, cif_outputs, not_padding_after_cif, *args, **kwargs):
        '''the interface of beam search call'''
        batch_size = cif_outputs.size(0)
        decode_length = cif_outputs.size(1)

        cif_outputs = self.expand_inputs_beamsize(cif_outputs)
        running_hyp = self.get_init_hyp(batch_size, decode_length)

        def beam_search_inner_loop(i, running_hyp):
            alive_seq = running_hyp.alive_ids
            alive_log_probs = running_hyp.alive_log_probs
            cache = running_hyp.running_cache

            cur_ids = torch.reshape(alive_seq[:, :, -1], (batch_size * self.beam_size,))

            cur_logits, _, cache = self.decoder_module(i, cur_ids, cif_outputs, cache)

            cur_logits = torch.reshape(cur_logits, (batch_size, self.beam_size, -1))
            candidate_log_probs = log_prob_from_logits(cur_logits * self.cif_temperature)

            # Multiply the probabilites by the current probabilites of the beam.
            # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
            log_probs = candidate_log_probs + alive_log_probs.unsqueeze(-1)

            # Flatten output (beam_size, vocab_size) probs into a list of possibilities
            flat_cur_scores = torch.reshape(log_probs, [-1, self.beam_size * self.args.vocab_size])
            topk_scores, topk_ids = torch.topk(flat_cur_scores, k=self.beam_size)

            # Work out what beam the top probs are in.
            topk_beam_index = topk_ids // self.args.vocab_size
            topk_ids %= self.args.vocab_size  # Unflatten the ids

            # The next two steps are to create coordinates for gather_nd to pull
            # out the correct seqences from id's that we need to grow.
            batch_pos = compute_batch_indices(batch_size, self.beam_size, 'cuda')

            # top beams will give us the actual coordinates to do the gather.
            # stacking will create a tensor of dimension batch * beam , where the
            # last dimension contains the i,j gathering coordinates.
            topk_coordinates = torch.stack([batch_pos, topk_beam_index], dim=2)

            # Gather up the most probable beams both for the ids
            topk_seq = gather_nd(alive_seq, topk_coordinates)

            # Reform logits and bias logits and k,v in cache and bias cache
            for key in cache:
                if "decoder_layer" in key:
                    cache[key]["k"] = reform(
                        cache[key]["k"],
                        topk_coordinates,
                        (batch_size, self.beam_size, i + 1, self.args.hidden_size),
                    )
                    cache[key]["v"] = reform(
                        cache[key]["v"],
                        topk_coordinates,
                        (batch_size, self.beam_size, i + 1, self.args.hidden_size),
                    )

            # Append the most probable value
            topk_seq = torch.cat((topk_seq, topk_ids.unsqueeze(2).int()), dim=2)

            # Handle the padding case
            not_padding_mask = not_padding_after_cif[:, i].float()
            not_padding_masks = torch.stack(
                [not_padding_mask for _ in range(self.beam_size)], axis=1
            ).int()

            # correct the scores or padding part
            topk_scores = torch.where(not_padding_masks == 1, topk_scores, alive_log_probs)
            topk_seq = torch.where(
                not_padding_masks.unsqueeze(-1) == 1,
                topk_seq,
                torch.cat(
                    (
                        alive_seq,
                        torch.zeros(
                            batch_size, self.beam_size, 1, dtype=torch.int32, device='cuda'
                        ),
                    ),
                    dim=-1,
                ),
            )

            running_hyp['alive_ids'] = topk_seq
            running_hyp['alive_log_probs'] = topk_scores
            running_hyp['running_cache'] = cache

            return running_hyp

        for i in range(decode_length):
            running_hyp = beam_search_inner_loop(i, running_hyp)

        if self.nbest == 1:
            return running_hyp.alive_ids[:, 0, 1:]
        raise ValueError('Not implement')

    def expand_inputs_beamsize(self, cif_outputs):
        '''expand inputs'''
        cif_outputs = cif_outputs.unsqueeze(1).expand(-1, self.beam_size, -1, -1)
        cif_outputs = torch.reshape(cif_outputs, (-1, cif_outputs.shape[-2], cif_outputs.shape[-1]))
        return cif_outputs

    def get_decoder_self_attention_bias(self, decode_length):
        '''generate decoder self-attention bias'''
        decoder_self_attention_bias = attention_bias_lower_triangle(decode_length, device='cuda')
        if self.args.get('limited_decoder_states', 0):
            limited_decoder_states = self.args.limited_decoder_states
            masking_matrix = torch.ones((1, 1, decode_length, decode_length))
            masking_matrix_left_all = 1.0 - torch.triu(masking_matrix, 1)
            masking_matrix_left_limited = 1.0 - torch.triu(masking_matrix, -limited_decoder_states)
            decoder_self_attention_bias = masking_matrix_left_all - masking_matrix_left_limited
            # Note, should convert to 0 and -Inf for softmax
            decoder_self_attention_bias = (1.0 - decoder_self_attention_bias) * -1e9
            decoder_self_attention_bias = decoder_self_attention_bias.cuda()
        if self.args.proximity_bias:
            decoder_self_attention_bias = (
                decoder_self_attention_bias
                + CifSelfAttentionModule.attention_bias_proximal(
                    decode_length, decoder_self_attention_bias.device
                )
            )
        return decoder_self_attention_bias

    def get_init_hyp(self, batch_size, decode_length):
        '''get initial hypothesis'''
        running_cache = {
            f"decoder_layer_{n}": {"k": None, "v": None}
            for n in range(self.args.num_decoder_layers)
        }

        running_cache['decoder_self_attention_bias'] = self.get_decoder_self_attention_bias(
            decode_length
        )

        hyp = FalconDict(
            alive_ids=torch.ones(batch_size, self.beam_size, 1, dtype=torch.int32, device='cuda'),
            alive_log_probs=torch.cat(
                (
                    torch.zeros(batch_size, 1, dtype=torch.float32, device='cuda'),
                    torch.full(
                        (batch_size, self.beam_size - 1),
                        -99999.0,
                        dtype=torch.float32,
                        device='cuda',
                    ),
                ),
                axis=1,
            ),
            running_cache=running_cache,
        )
        return hyp


class NonBatchBeamSearch(BaseBeamSearch):
    '''nonbatch class of beam search of cif-based model'''

    def __init__(
        self,
        args,
        cfg,
        decoder_module,
        lm_solution=None,
    ):
        '''cfg: beam search required configuration file'''
        super().__init__(
            args,
            cfg,
            decoder_module,
            lm_solution=lm_solution,
        )
        # for shallow-fusion and ilme
        self.nnlm_path = cfg.get('nnlm_path', "")
        self.nnlm_weight = cfg.get('nnlm_weight', 0)
        self.use_ilme = cfg.get('use_ilme', False)
        self.internal_lm_weight = cfg.get('internal_lm_weight', 0)
        # for fst-fusion: coldwords and hotwords
        self.hotword_fst_path = cfg.get('hotword_fst_path', "")
        self.hotword_fst_weight = cfg.get('hotword_fst_weight', '')

        self.coldword_fst_path = cfg.get('coldword_fst_path', "")
        self.coldword_fst_weight = cfg.get('coldword_fst_weight', '')
        self.use_lm_beam = cfg.get('use_lm_beam', False)
        if not (self.hotword_fst_path or self.coldword_fst_path):
            self.use_lm_beam = False
        self.lm_beam_size = cfg.get('lm_beam_size', 20)
        # decoder params
        self.beam_size = cfg.get('beam_size', 0)
        self.nbest = cfg.get('nbest', 1)
        self.cif_temperature = cfg.get('cif_temperature', 1.0)

        if lm_solution:
            hotword_fst_length = lm_solution.get_hotword_fst_number()
            hotword_weight_length = len(lm_solution.hotword_weight)
            if self.hotword_fst_path and hotword_fst_length != hotword_weight_length:
                raise ValueError(
                    f"Hotword fst length not match: {hotword_fst_length} vs {hotword_weight_length}"
                )
        elif self.hotword_fst_path or self.coldword_fst_path or self.nnlm_path:
            raise ValueError("lm model can not work.")

    def get_nonbatch_init_hyp(self):
        '''get initial hypothesis'''
        hyps = []
        for n in range(self.beam_size):
            init_hyp = Hypothesis(label_seq=[1])
            if n > 0:
                init_hyp.score = -99999.0
            # initialize fst_state
            if self.lm_solution is not None:
                hotword_fst_num = self.lm_solution.get_hotword_fst_number()
                hotword_start_states = self.lm_solution.hotword_fst_start()
                init_hyp.init_lm_tokens(hotword_fst_num, hotword_start_states)
                init_hyp.coldword_state = self.lm_solution.coldword_fst_start()
            hyps.append(init_hyp)
        return hyps

    def expand_nonbatch_inputs_beamsize(self, cif_outputs):
        '''expand inputs'''
        cif_outputs = cif_outputs.unsqueeze(1).expand(-1, self.beam_size, -1, -1)
        cif_outputs = torch.reshape(cif_outputs, (-1, cif_outputs.shape[-2], cif_outputs.shape[-1]))
        return cif_outputs

    def fusion_word_fst(self, best_token, hyp, is_last):
        """Apply fusion on hyp using a best token."""
        matched = False
        if self.lm_solution.finish_non_greedy_search(is_last, best_token, hyp.lm_tokens):
            # Clear tokens. This means all tokens will start from start state in the next frame.
            hotword_fst_num = self.lm_solution.get_hotword_fst_number()
            hotword_start_states = self.lm_solution.hotword_fst_start()
            hyp.init_lm_tokens(hotword_fst_num, hotword_start_states)
            if best_token.total_lm_score < 0:
                matched = True
        hyp.score = hyp.score + hyp.prev_total_lm_score
        hyp.score = hyp.score - best_token.total_lm_score
        hyp.prev_total_lm_score = best_token.total_lm_score
        if matched:
            hyp.prev_total_lm_score = 0.0
            hyp.hotword_score += best_token.total_lm_score
            for hotword in best_token.matched_hotwords:
                hyp.matched_hotwords.append(hotword)

    def hotwords_score_backoff(self, hyps):
        """Add hotword backoff score.
        This function is worked at the last decode step,
        to fix the case that hotword is not full matched at sentence end.
        Return a dict to store roads' keys and backoff scores"""
        hotword_backoff = {}
        # for fst demo, only one fst is used, fst_index is 0
        if self.lm_solution and self.hotword_fst_path:
            for hyp in hyps:
                backoff_score = self.lm_solution.hotword_fst_backoff(hyp.lm_tokens)
                hotword_backoff[hyp.yseq] = backoff_score
                hyp.score -= backoff_score
        return hotword_backoff

    def __call__(self, cif_outputs, not_padding_after_cif, *args, **kwargs):
        '''the interface of beam search call'''
        batch_size = cif_outputs.size(0)
        decode_length = cif_outputs.size(1)
        # use top_beam
        top_beam_size = self.beam_size
        if self.use_lm_beam:
            top_beam_size = self.lm_beam_size

        # get decoder bias
        hyp_label_seqs = []
        for i in range(batch_size):
            running_hyps = self.get_nonbatch_init_hyp()
            cif_output = cif_outputs[i : i + 1]
            cif_output = self.expand_nonbatch_inputs_beamsize(cif_output)

            cache = dict()
            cache['decoder_self_attention_bias'] = self.get_decoder_self_attention_bias(
                decode_length
            )

            if self.use_ilme:
                cache_ilme = dict()
                cache_ilme['decoder_self_attention_bias'] = self.get_decoder_self_attention_bias(
                    decode_length
                )

            for n in range(self.args.num_decoder_layers):
                cache[f"decoder_layer_{n}"] = {
                    "k": torch.zeros(
                        self.beam_size, 0, self.args.hidden_size, dtype=torch.float32, device='cuda'
                    ),
                    "v": torch.zeros(
                        self.beam_size, 0, self.args.hidden_size, dtype=torch.float32, device='cuda'
                    ),
                }
                if self.use_ilme:
                    cache_ilme[f"decoder_layer_{n}"] = {
                        "k": torch.zeros(
                            self.beam_size,
                            0,
                            self.args.hidden_size,
                            dtype=torch.float32,
                            device='cuda',
                        ),
                        "v": torch.zeros(
                            self.beam_size,
                            0,
                            self.args.hidden_size,
                            dtype=torch.float32,
                            device='cuda',
                        ),
                    }

            for j in range(decode_length):
                if not_padding_after_cif[i][j] == 0:
                    break

                cur_id = []
                for k in range(self.beam_size):
                    cur_id.append(torch.tensor(running_hyps[k].label_seq[-1:]).cuda())
                cur_id = torch.cat(cur_id, dim=0)
                cur_id = torch.reshape(cur_id, [self.beam_size])

                # get logits for every beam
                cur_logit, _, cache = self.decoder_module(j, cur_id, cif_output, cache)

                score = []
                for k in range(self.beam_size):
                    score.append(running_hyps[k].score)
                score = torch.tensor(score, device='cuda')
                score = score.unsqueeze(-1) + log_prob_from_logits(
                    cur_logit * self.cif_temperature, dim=1
                )

                if self.use_ilme:
                    if j == 0:
                        lm_states = None
                    else:
                        lm_states = []
                        for n in range(len(running_hyps[0].lm_states)):
                            h = torch.cat(
                                [
                                    running_hyps[k].lm_states[n][0].unsqueeze(0)
                                    for k in range(self.beam_size)
                                ],
                                dim=0,
                            )
                            c = torch.cat(
                                [
                                    running_hyps[k].lm_states[n][1].unsqueeze(0)
                                    for k in range(self.beam_size)
                                ],
                                dim=0,
                            )
                            lm_states.append((h, c))

                    cur_lm_lprobs, lm_states = self.lm_solution.step_nn(
                        cur_id.view(-1).long(), lm_states
                    )
                    score += self.nnlm_weight * cur_lm_lprobs[:, : score.size(-1)]

                    if self.internal_lm_weight > 0:
                        inter_lm_logits, _, cache_ilme = self.decoder_module(
                            j, cur_id, torch.zeros_like(cif_output), cache_ilme
                        )
                        inter_lm_logits = torch.reshape(inter_lm_logits, (self.beam_size, -1))
                        inter_lm_log_probs = log_prob_from_logits(inter_lm_logits, dim=1)
                        score -= self.internal_lm_weight * inter_lm_log_probs

                score = torch.reshape(score, [-1])
                token_score, token_index = torch.topk(score, k=top_beam_size)

                update_hyp = []
                update_hyp_accum = []
                cur_cache = dict()
                cur_cache['decoder_self_attention_bias'] = self.get_decoder_self_attention_bias(
                    decode_length
                )

                cur_cache_ilme = None
                cur_lm_states = None
                if self.use_ilme:
                    cur_cache_ilme = dict()
                    cur_cache_ilme[
                        'decoder_self_attention_bias'
                    ] = self.get_decoder_self_attention_bias(decode_length)
                    cur_lm_states = [None for _ in range(len(lm_states))]

                for n in range(self.args.num_decoder_layers):
                    cur_cache[f"decoder_layer_{n}"] = {"k": [], "v": []}
                    if self.use_ilme:
                        cur_cache_ilme[f"decoder_layer_{n}"] = {"k": [], "v": []}

                for k in range(top_beam_size):
                    index = token_index[k]
                    hyp_node_idx = int(index / self.args.vocab_size)
                    hyp_node = running_hyps[hyp_node_idx]
                    new_hyp_node = hyp_node.copy()
                    vocab_idx = int(index % self.args.vocab_size)
                    # update score and label_seq for k-th beam
                    new_hyp_node.score = token_score[k]
                    new_hyp_node.label_seq.append(vocab_idx)
                    if self.use_ilme:
                        # pylint: disable=consider-using-enumerate
                        for n in range(len(lm_states)):
                            cur_lm_states[n] = (
                                lm_states[n][0][hyp_node_idx],
                                lm_states[n][1][hyp_node_idx],
                            )
                    new_hyp_node.lm_states = copy.deepcopy(cur_lm_states)

                    # add Fst-fusion for cif
                    if self.lm_solution:
                        if self.hotword_fst_path:
                            is_last = j == decode_length - 1
                            best_token = self.lm_solution.step_hotword(
                                vocab_idx, new_hyp_node, is_last
                            )
                            if best_token is not None:
                                self.fusion_word_fst(best_token, new_hyp_node, is_last)
                        if self.coldword_fst_path:
                            coldword_fst_state = new_hyp_node.coldword_state
                            coldword_fst_state, coldword_fst_score = self.lm_solution.step_coldword(
                                vocab_idx, coldword_fst_state
                            )
                            new_hyp_node.score -= coldword_fst_score * self.coldword_fst_weight
                            new_hyp_node.coldword_state = coldword_fst_state

                    update_hyp.append(new_hyp_node)

                # sort and get top beam hyps
                score_accum = []
                for hyp in update_hyp:
                    score_accum.append(hyp.score)
                _, token_index_accum = torch.topk(torch.tensor(score_accum), k=self.beam_size)
                for idx in token_index_accum.tolist():
                    update_hyp_accum.append(update_hyp[idx])
                    # here cache is equal to the ptr in implementation0 of penguin but use dict as the structure
                    cur_hyp_node_idx = int(token_index[idx] / self.args.vocab_size)
                    for n in range(self.args.num_decoder_layers):
                        cur_cache[f"decoder_layer_{n}"]["k"].append(
                            cache[f"decoder_layer_{n}"]["k"][cur_hyp_node_idx].unsqueeze(0)
                        )
                        cur_cache[f"decoder_layer_{n}"]["v"].append(
                            cache[f"decoder_layer_{n}"]["v"][cur_hyp_node_idx].unsqueeze(0)
                        )
                        if self.use_ilme:
                            cur_cache_ilme[f"decoder_layer_{n}"]["k"].append(
                                cache_ilme[f"decoder_layer_{n}"]["k"][cur_hyp_node_idx].unsqueeze(0)
                            )
                            cur_cache_ilme[f"decoder_layer_{n}"]["v"].append(
                                cache_ilme[f"decoder_layer_{n}"]["v"][cur_hyp_node_idx].unsqueeze(0)
                            )
                for n in range(self.args.num_decoder_layers):
                    cur_cache[f"decoder_layer_{n}"]["k"] = torch.cat(
                        cur_cache[f"decoder_layer_{n}"]["k"], dim=0
                    )
                    cur_cache[f"decoder_layer_{n}"]["v"] = torch.cat(
                        cur_cache[f"decoder_layer_{n}"]["v"], dim=0
                    )
                    if self.use_ilme:
                        cur_cache_ilme[f"decoder_layer_{n}"]["k"] = torch.cat(
                            cur_cache_ilme[f"decoder_layer_{n}"]["k"], dim=0
                        )
                        cur_cache_ilme[f"decoder_layer_{n}"]["v"] = torch.cat(
                            cur_cache_ilme[f"decoder_layer_{n}"]["v"], dim=0
                        )
                cache = cur_cache
                cache_ilme = cur_cache_ilme
                running_hyps = update_hyp_accum

            if j == decode_length - 1:
                self.hotwords_score_backoff(running_hyps)

            if self.nbest == 1:
                score = []
                for k in range(self.beam_size):
                    score.append(running_hyps[k].score)
                score = torch.tensor(score).cpu().numpy()
                idx = np.argmax(score)
                hyp_label_best_seq = running_hyps[idx].label_seq
                best_seq_length = len(hyp_label_best_seq)
                hyp_label_best_seq = hyp_label_best_seq + (decode_length + 1 - best_seq_length) * [
                    0
                ]
                hyp_label_seqs.append(
                    torch.tensor(hyp_label_best_seq, device='cuda:0', dtype=torch.int32).unsqueeze(
                        0
                    )
                )
            else:
                raise ValueError('Not implement')

        return torch.cat(hyp_label_seqs, dim=0)
