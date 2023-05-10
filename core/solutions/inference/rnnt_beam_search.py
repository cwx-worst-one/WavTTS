''' beam search '''
# pylint: disable=line-too-long, cell-var-from-loop
# pylint:disable=too-many-lines
import math
import torch
import numpy as np
from core.extensions import beam_search_unique_roads as recombine_hyps_cpp
from core.utils import FalconDict
from .hypothesis import Hypothesis


def log_sum(tensor_a, tensor_b):
    '''log_sum'''
    return torch.logsumexp(torch.tensor([tensor_a, tensor_b]), 0)


class BaseBeamSearch:
    '''base class of beam search'''

    def __init__(
        self,
        cfg,
        predictor_module,
        jointer_module,
        criterion_module,
        lm_solution=None,
        rw_bias_moudule=None,
    ):
        '''cfg: ineference config'''
        self.predictor_module = predictor_module
        self.jointer_module = jointer_module
        self.criterion_module = criterion_module

        self.beam_size = cfg.get('beam_size', 0)
        self.nbest = cfg.get('nbest', 1)
        attn_rescore_weight = cfg.get('attn_rescore_weight', 0.0)
        if attn_rescore_weight > 0.0:
            self.nbest = self.beam_size
        # lm params.
        self.lm_solution = lm_solution
        self.rw_bias_moudule = rw_bias_moudule
        if rw_bias_moudule is not None:
            self.rw_embed = None
            self.rw_bias_weight = cfg.get('rw_bias_weight', 1.0)
        self.hotword_fst_path = cfg.get('hotword_fst_path', "")
        self.coldword_fst_path = cfg.get('coldword_fst_path', "")
        self.coldword_fst_weight = cfg.get('coldword_fst_weight', '')
        self.hotword_fst_greed_search = cfg.get('hotword_fst_greed_search', False)
        self.ngram_fst_path = cfg.get('ngram_fst_path', "")
        self.ngram_fst_ilm = cfg.get('ngram_fst_minus_internal_lm', True)
        self.ngram_fst_weight = cfg.get('ngram_fst_weight', 0)
        self.class_lm_fst_path = cfg.get('class_lm_fst_path', '')
        self.class_lm_fst_fusion_weight = cfg.get('class_lm_fst_fusion_weight', 0)
        self.class_lm_fst_word_weight = cfg.get('class_lm_fst_word_weight', 0)
        self.nnlm_path = cfg.get('nnlm_path', "")
        self.nnlm_weight = cfg.get('nnlm_weight', 0)
        self.nnlm_ilme = cfg.get('nnlm_ilme', True)
        self.ilm_weight = cfg.get('internal_lm_weight', 0)
        self.use_lm_beam = cfg.get('use_lm_beam', False)
        if not (self.hotword_fst_path or self.ngram_fst_path or self.coldword_fst_path):
            self.use_lm_beam = False
        self.lm_beam_size = cfg.get('lm_beam_size', 20)

        self.blank_scale = cfg.get('blk_scale', 1.0)
        self.len_penalty_scale = cfg.get('len_penalty_scale', 0.01)
        self.rnnt_temperature = cfg.get('rnnt_temperature', 1.0)
        self.recombine_sum = cfg.get('recombine_sum', True)
        self.use_recombine = cfg.get('use_recombine', True)
        self.blank_scale_add = cfg.get('blk_scale_add', True)
        self.endpoints_indexs = cfg.get('endpoints_indexs', None)
        self.endpoints_scales = cfg.get('endpoints_scales', None)
        self.nbest_align_info = cfg.get('nbest_align_info', False)
        self.only_head = cfg.get('only_head', False)
        self.adapt_softmax_thresh = cfg.get('adapt_softmax_thresh', 0.995)
        self.prefetch = cfg.get('enable_prefetch', False)
        self.log_prefetch_thresh = math.log(cfg.get('prefetch_thresh', 1e-4))
        self.unique_road_backend = cfg.get('unique_road_backend', 'panther')
        self.fixed_prefix_beam_search = cfg.get('fixed_prefix_beam_search', False)
        self.fixed_prefix_time_freq = cfg.get('fixed_prefix_time_freq', 10)
        self.fixed_prefix_changeable_token = cfg.get('fixed_prefix_changeable_token', 0)
        self.enable_endpoint = cfg.get('enable_endpoint', False)
        self.ep_alpha = cfg.get('endpoint_alpha', 2.0)
        self.log_ep_beta = math.log(cfg.get('endpoint_beta', 0.65))
        self.limited_context = cfg.get('limited_context', None)

        if self.endpoints_indexs is None:
            self.endpoints_indexs = []
            self.endpoints_scales = []

        eos_scale = cfg.get('eos_scale', None)
        if eos_scale is not None:
            self.endpoints_indexs.append(2)
            self.endpoints_scales.append(eos_scale)
        if lm_solution:
            hotword_fst_num = lm_solution.get_hotword_fst_number()
            hotword_weight_num = len(lm_solution.hotword_weight)
            if hotword_fst_num != hotword_weight_num:
                raise ValueError(
                    f"the num of hotword fsts and weights are not match: {hotword_fst_num} vs {hotword_weight_num}"
                )

    def __call__(self, *args, **kwargs):
        '''the interface of beam search call'''
        raise NotImplementedError()

    def get_init_hyp(self, batch_size):
        '''get initial hypothesis'''
        hyp = FalconDict(
            batch_size=batch_size,
            scores=torch.zeros(batch_size, self.beam_size, dtype=torch.float32, device='cuda'),
            label_seq=torch.zeros(batch_size, self.beam_size, 1, dtype=torch.int64, device='cuda'),
        )
        hyp.pred_feats, hyp.pred_states = self.predictor_forward_step(hyp.label_seq)
        if self.lm_solution is not None and self.nnlm_path:
            hyp.lm_scores = torch.zeros(
                batch_size, self.beam_size, dtype=torch.float32, device='cuda'
            )
            hyp.lm_lprobs, hyp.lm_states = self.lm_solution.step_nn(hyp.label_seq.view(-1))
        return hyp

    def expand_inputs_beamsize(self, acoustic_outs, acoustic_masks=None):
        '''expand inputs'''
        acoustic_outs = (
            acoustic_outs.unsqueeze(1).expand(-1, self.beam_size, -1, -1).permute(2, 0, 1, 3)
        )  # (t, batch, beam, v)
        acoustic_outs = acoustic_outs.contiguous().view(
            acoustic_outs.shape[0], -1, acoustic_outs.shape[-1]
        )
        if acoustic_masks is not None:
            acoustic_masks = (
                acoustic_masks.unsqueeze(1).expand(-1, self.beam_size, -1).permute(2, 0, 1)
            )  # (t, batch, beam)
            return acoustic_outs, acoustic_masks
        return acoustic_outs

    def predictor_forward_step(self, label_seq, states=None, **kwargs):
        '''Predictor forward step'''
        context_embd = kwargs.get('context_embd')
        context_mask = kwargs.get('context_mask')
        context_encoder = kwargs.get('context_encoder')
        bid = kwargs.get('bid', 0)
        beam_num = kwargs.get('beam_num', 1)
        bias = self.rw_bias_moudule is not None and self.rw_embed is not None
        if context_embd is None:
            pred_feats, pred_states = self.predictor_module.forward_step(
                label_seq.view(-1), states, return_embed=bias
            )
        else:
            pred_feats, pred_states = self.predictor_module.forward_step(
                label_seq.view(-1),
                states,
                context_embd=context_embd[bid].repeat(beam_num, 1, 1),
                context_mask=context_mask[bid].repeat(beam_num, 1),
                context_encoder=context_encoder,
            )
        if self.limited_context:
            return pred_feats, pred_states
        if bias:
            prev_embd = pred_feats[0]
            pred_feats = pred_feats[1]
            bias_feats = self.rw_bias_moudule.forward_step(
                pred_feats, self.rw_embed, query2=prev_embd
            )
            pred_feats += self.rw_bias_weight * bias_feats
        return pred_feats, list(pred_states)

    def get_lprobs(self, acoustic_outs, pred_feats):
        '''calculate the probability according to the joiner out'''
        jointer_out = self.jointer_module.forward_step(acoustic_outs, pred_feats)
        lprobs = self.criterion_module.get_log_prob(
            jointer_out,
            rnnt_temperature=self.rnnt_temperature,
            only_head=self.only_head,
            adapt_softmax_thresh=self.adapt_softmax_thresh,
        )
        return lprobs

    def apply_blank_scale(self, lprobs):
        '''apply blank scale'''
        if self.blank_scale_add:
            lprobs[:, 0] += math.log(self.blank_scale)
        else:
            lprobs[:, 0] *= self.blank_scale

    def apply_endpoints_scale(self, lprobs):
        '''apply endpoints scale'''
        for idx, scale in zip(self.endpoints_indexs, self.endpoints_scales):
            lprobs[:, idx] += math.log(scale)

    def check_eos_strong_enough(self, eos_score):
        '''check eos strong enough to be emitted'''
        if eos_score >= self.log_ep_beta:
            return True
        return False

    def t_step(self, lprobs, hyp):
        '''do t step'''
        t_step_hyp = hyp.copy()
        t_step_hyp.scores = hyp.scores + lprobs[:, 0].view(-1, self.beam_size)
        return t_step_hyp

    def u_step(self, lprobs, hyp):
        '''do u step'''
        u_step_hyp = hyp.copy()
        # get the probability of all cases
        scores = lprobs + hyp.scores.view(-1, 1)
        # filter out the first beam cases, which is step U
        u_step_hyp.scores, beam_idx, token_idx = self.top_beam(scores, hyp.batch_size)
        u_step_hyp = self.gather_hyp(beam_idx, u_step_hyp)
        u_step_hyp.label_seq = torch.cat([u_step_hyp.label_seq, token_idx.unsqueeze(2)], dim=2)
        u_step_hyp.pred_feats, u_step_hyp.pred_states = self.predictor_forward_step(
            token_idx, u_step_hyp.pred_states
        )
        if self.lm_solution and self.nnlm_path:
            u_step_hyp.lm_scores += torch.gather(
                u_step_hyp.lm_lprobs, 2, token_idx.unsqueeze(2)
            ).view(-1, self.beam_size)
            u_step_hyp.lm_lprobs, u_step_hyp.lm_states = self.lm_solution.step_nn(
                token_idx.view(-1), u_step_hyp.lm_states
            )
        return u_step_hyp

    def top_beam(self, lprobs, bsz=1, beam_size=0):
        '''find out the probability of large front beam and its location'''
        if beam_size == 0:
            beam_size = self.beam_size
        # set blank lprob to -inf
        lprobs[:, 0] -= 1e8
        voc = lprobs.shape[1]
        topk_lprobs, topk_idx = lprobs.view(bsz, -1).topk(beam_size, 1)
        beam_idx, token_idx = topk_idx // voc, topk_idx % voc
        return topk_lprobs, beam_idx, token_idx

    def concat_hyps(self, *hyps):
        '''concat tuple(pred_feat) | tuple(pred_state) | tuple(road) | tuple(score)'''
        hyp_num = len(hyps)
        for i in range(hyp_num - 1):
            hyps[i].label_seq = torch.cat(
                (
                    hyps[i].label_seq,
                    torch.zeros(
                        hyps[i].batch_size,
                        self.beam_size,
                        hyp_num - i - 1,
                        dtype=torch.int64,
                        device='cuda',
                    ),
                ),
                dim=-1,
            )
        concat_hyp = FalconDict(
            batch_size=hyps[0].batch_size,
            scores=torch.cat([hyp.scores for hyp in hyps], dim=1),
            label_seq=torch.cat([hyp.label_seq for hyp in hyps], dim=1),
            pred_feats=torch.cat(
                [hyp.pred_feats.view(hyp.label_seq.shape[0], self.beam_size, -1) for hyp in hyps],
                dim=1,
            ),
        )
        lstm_concat_fn = (
            self.predictor_module.concate_t_ut_states
            if len(hyps) == 2
            else self.predictor_module.concate_t_ut_uut_states
        )
        concat_hyp.pred_states = lstm_concat_fn(*[hyp.pred_states for hyp in hyps], self.beam_size)
        if self.lm_solution is not None and self.nnlm_path:
            concat_hyp.lm_scores = torch.cat([hyp.lm_scores for hyp in hyps], dim=1)
            concat_hyp.lm_states = lstm_concat_fn(*[hyp.lm_states for hyp in hyps], self.beam_size)
            concat_hyp.lm_lprobs = torch.cat(
                [hyp.lm_lprobs.view(-1, self.beam_size, hyp.lm_lprobs.shape[-1]) for hyp in hyps],
                dim=1,
            )
        return concat_hyp

    def gather_hyp(self, index, hyp, process_pred_feats=False):
        '''according to the index select roads | scores | pred_states | pred_feats'''
        out_hyp = hyp.copy()
        road_lens, feat_dims = hyp.label_seq.shape[-1], hyp.pred_feats.shape[-1]
        out_hyp.label_seq = torch.gather(
            hyp.label_seq, 1, index.unsqueeze(2).repeat(1, 1, road_lens)
        )
        out_hyp.pred_states = self.predictor_module.reorder_beam_states(hyp.pred_states, index)
        if process_pred_feats:
            out_hyp.pred_feats = torch.gather(
                hyp.pred_feats, 1, index.unsqueeze(2).repeat(1, 1, feat_dims)
            ).view(-1, feat_dims)
        if self.lm_solution is not None and self.nnlm_path:
            out_hyp.lm_scores = torch.gather(hyp.lm_scores, 1, index)
            if hyp.lm_lprobs.dim() == 2:
                hyp.lm_lprobs = hyp.lm_lprobs.view(-1, self.beam_size, hyp.lm_lprobs.shape[-1])
            out_hyp.lm_lprobs = torch.gather(
                hyp.lm_lprobs, 1, index.unsqueeze(2).repeat(1, 1, hyp.lm_lprobs.shape[-1])
            )
            out_hyp.lm_states = self.predictor_module.reorder_beam_states(hyp.lm_states, index)
        return out_hyp

    def recombine_hyps(self, hyp):
        '''unique road'''
        unique_hyp = hyp.copy()
        if self.unique_road_backend in ('panther', 'falconpai'):
            unique_hyp.scores, unique_hyp.label_seq, unique_hyp.tokens_num = recombine_hyps_cpp(
                hyp.scores, hyp.label_seq, self.beam_size, self.recombine_sum
            )
        else:
            (
                unique_hyp.scores,
                unique_hyp.label_seq,
                unique_hyp.tokens_num,
            ) = self.recombine_hyps_python(hyp.scores, hyp.label_seq)
        return unique_hyp

    def recombine_hyps_python(self, scores, label_seq):
        '''unique the road (according set the score to -inf)'''
        scores, label_seq = scores.cpu(), label_seq.cpu()
        bsz, expand_beam_size = scores.shape
        tokens_nums = []
        label_seq_list = [
            [tuple(item for item in beam if item) for beam in batch] for batch in label_seq.tolist()
        ]
        for bid in range(bsz):
            label_seq_batch, tokens_num = label_seq_list[bid], []
            compose_dict = dict()
            for beam_idx in range(self.beam_size):
                token = label_seq_batch[beam_idx]
                compose_dict[token] = beam_idx
                tokens_num.append(len(token))
            scores_beams, label_seq_beams = scores[bid], label_seq[bid]
            for beam_idx in range(self.beam_size, expand_beam_size):
                token = label_seq_batch[beam_idx]
                beam_idx_ = compose_dict.get(token)
                if beam_idx_ is not None:
                    scores_beam_idx_value = scores_beams[beam_idx]
                    if self.recombine_sum:
                        scores_beams[beam_idx_] = log_sum(
                            scores_beams[beam_idx_], scores_beam_idx_value
                        )
                    elif scores_beams[beam_idx_] < scores_beam_idx_value:
                        scores_beams[beam_idx_] = scores_beam_idx_value
                        label_seq_beams[beam_idx_] = label_seq_beams[beam_idx]
                    scores_beams[beam_idx] = -1e8
                else:
                    compose_dict[token] = beam_idx
                tokens_num.append(len(token))
            tokens_nums.append(tokens_num)
        scores, label_seq = scores.cuda(), label_seq.cuda()
        tokens_nums = torch.tensor(tokens_nums, dtype=scores.dtype, device='cuda')
        return scores, label_seq, tokens_nums

    def add_length_penalty(self, hyp):
        '''add road length penalty'''
        len_diff = torch.abs(hyp.tokens_num - hyp.tokens_num.mean(dim=1, keepdim=True))
        hyp.scores -= self.len_penalty_scale * len_diff
        del hyp['tokens_num']
        return hyp

    def get_valid_lprobs(self, acoustics, hyp):
        '''do get log probs'''
        lprobs = self.get_lprobs(acoustics.feat, hyp.pred_feats)
        if acoustics.index == 0:
            lprobs.view(hyp.batch_size, -1, lprobs.shape[-1])[:, 1:] -= 1e8
        return lprobs

    def get_top_beam_hyp(self, hyp):
        '''do top beam'''
        top_beam_hyp = hyp.copy()
        if self.lm_solution is not None and self.nnlm_path:
            top_beam_hyp.scores += hyp.lm_scores * self.nnlm_weight
        top_beam_hyp.scores, topk_concat_idx = top_beam_hyp.scores.topk(self.beam_size, 1)
        top_beam_hyp = self.gather_hyp(topk_concat_idx, top_beam_hyp, process_pred_feats=True)
        if self.lm_solution is not None and self.nnlm_path:
            top_beam_hyp.lm_lprobs = top_beam_hyp.lm_lprobs.view(
                -1, top_beam_hyp.lm_lprobs.shape[-1]
            )
            top_beam_hyp.scores -= self.nnlm_weight * top_beam_hyp.lm_scores
        top_beam_hyp.pred_feats = top_beam_hyp.pred_feats.view(
            -1, top_beam_hyp.pred_feats.shape[-1]
        )
        return top_beam_hyp


class BatchBeamSearch(BaseBeamSearch):
    '''batch version beam search'''

    def merge_hyp_with_mask(self, hyp_a, hyp_b, mask):
        '''merge tensor with mask'''
        out_hyp = hyp_a.copy()
        out_hyp.scores = hyp_a.scores * mask.long() + (1 - mask.long()) * hyp_b.scores
        if self.lm_solution and self.nnlm_path:
            out_hyp.lm_scores = hyp_a.lm_scores * mask.long() + (1 - mask.long()) * hyp_b.lm_scores
        mask = mask.unsqueeze(2)
        hyp_b.label_seq = torch.cat(
            [
                hyp_b.label_seq,
                torch.zeros(hyp_b.batch_size, self.beam_size, 1, dtype=torch.int64, device='cuda'),
            ],
            dim=-1,
        )
        out_hyp.label_seq = hyp_a.label_seq * mask.long() + (1 - mask.long()) * hyp_b.label_seq
        return out_hyp

    def get_nbest_list(self, hyp):
        '''get nbest list'''
        num_beam = min(self.beam_size, self.nbest)
        batch_nbest_list = []
        for bid in range(hyp.batch_size):
            roads_bid, scores_bid = hyp.label_seq[bid], hyp.scores[bid]
            nbest_list = [(roads_bid[i].tolist(), scores_bid[i].item()) for i in range(num_beam)]
            batch_nbest_list.append(nbest_list)
        return batch_nbest_list

    def get_nbest_align_info(self, hyp, masks):
        '''get nebst align info'''
        frame_lengths = masks.sum(-1).int()
        nbest_roads_infos = []
        for bid in range(hyp.batch_size):
            nbest_roads_info = {}
            for beam_idx in range(self.beam_size):
                road = hyp.label_seq[bid, beam_idx, 1 : frame_lengths[bid] + 1].tolist()
                u_steps, t_steps, u_road_steps = [0], [], []
                for idx, step in enumerate(road):
                    t_steps.append(idx)
                    u_steps.append(u_steps[-1])
                    if step != 0:
                        u_steps.append(u_steps[-1] + 1)
                        t_steps.append(idx)
                        u_road_steps.append(step)
                    u_road_steps.append(0)
                road = [str(r) for r in road if r]
                key = '0 {}'.format(' '.join(road))
                nbest_roads_info[key] = {
                    'u_t_len': frame_lengths[bid] + len(road),
                    'u_steps': np.array(u_steps[1:]),
                    't_steps': np.array(t_steps),
                    'hyp_steps': np.array(u_road_steps),
                }
            nbest_roads_infos.append(nbest_roads_info)
        return nbest_roads_infos

    def format_output(self, hyp, acoustic_masks):
        '''do return'''
        if not self.nbest_align_info:
            batch_best_lists = self.get_nbest_list(hyp)
            return hyp.label_seq[:, 0, :], batch_best_lists
        nbest_roads_infos = self.get_nbest_align_info(hyp, acoustic_masks)
        return hyp.label_seq[:, 0, :], nbest_roads_infos

    def __call__(self, acoustic_outs, acoustic_masks, **kwargs):
        '''call'''
        batch_size = acoustic_outs.size(0)
        self.rw_embed = kwargs.get("rw_embed", None)
        expand_acoustic_outs, expand_acoustic_masks = self.expand_inputs_beamsize(
            acoustic_outs, acoustic_masks
        )
        running_hyp = self.get_init_hyp(batch_size)
        frame_lengths = expand_acoustic_masks.sum(0).int()
        min_frame_length = frame_lengths.min()
        for frame_idx, (acoustic_out, acoustic_mask) in enumerate(
            zip(expand_acoustic_outs, expand_acoustic_masks)
        ):
            acoustics = FalconDict(index=frame_idx, feat=acoustic_out, mask=acoustic_mask)
            # get log_prob
            first_lprobs = self.get_valid_lprobs(acoustics, running_hyp)
            self.apply_blank_scale(first_lprobs)
            # t step
            t_step_hyp = self.t_step(first_lprobs, running_hyp)
            # u step
            u_step_hyp = self.u_step(first_lprobs, running_hyp)
            # concat t_step and u_step
            concat_hyp = self.concat_hyps(t_step_hyp, u_step_hyp)
            concat_hyp = self.recombine_hyps(concat_hyp)
            concat_hyp = self.add_length_penalty(concat_hyp)
            top_beam_hyp = self.get_top_beam_hyp(concat_hyp)
            if frame_idx >= min_frame_length:
                running_hyp = self.merge_hyp_with_mask(top_beam_hyp, running_hyp, acoustic_mask)
            else:
                running_hyp = top_beam_hyp
        best_hyp, nbest_hyps_infos = self.format_output(running_hyp, acoustic_masks)
        return best_hyp, {'nbest': nbest_hyps_infos}


class PenguinBeamSearch(BaseBeamSearch):
    '''methods of beam searching for Penguin'''

    def __init__(
        self,
        config,
        predictor_module=None,
        jointer_module=None,
        criterion_module=None,
        lm_solution=None,
        rw_bias_moudule=None,
    ):
        '''beam search init'''
        super().__init__(
            cfg=config,
            predictor_module=predictor_module,
            jointer_module=jointer_module,
            criterion_module=criterion_module,
            lm_solution=lm_solution,
            rw_bias_moudule=rw_bias_moudule,
        )

    def __call__(self, *args, **kwargs):
        '''the interface of beam search call'''
        raise NotImplementedError()

    def recombine_hyps(self, t_hyps, u_hyps):
        '''recombine hypotheses'''
        label_str2idx = dict()
        u_hyps, running_hyps = [], u_hyps
        for i, hyp in enumerate(running_hyps):
            label_str2idx[hyp.label_str] = i
        for hyp in t_hyps:
            if hyp.label_str not in label_str2idx:
                running_hyps.append(hyp)
            else:
                renew_hyp = running_hyps[label_str2idx[hyp.label_str]]
                if renew_hyp.score < hyp.score:
                    renew_hyp.timestamp = hyp.timestamp
                    renew_hyp.lm_tokens = hyp.lm_tokens[:]
                    renew_hyp.prev_total_lm_score = hyp.prev_total_lm_score
                    renew_hyp.lm_token_timestamp = hyp.lm_token_timestamp
                    renew_hyp.matched_hotwords = hyp.matched_hotwords
                renew_hyp.score = np.logaddexp(renew_hyp.score, hyp.score)
                renew_hyp.nnlm_score = np.logaddexp(renew_hyp.nnlm_score, hyp.nnlm_score)
                # the 'hyp.eos_score' comes from t_step hyps, it's newer
                renew_hyp.eos_score = hyp.eos_score
        return running_hyps

    def add_length_penalty(self, hyps):
        '''add length penalty'''
        hyps_len_mean = np.mean([len(hyp.label_seq) for hyp in hyps])
        for hyp in hyps:
            hyp.score -= self.len_penalty_scale * abs(len(hyp.label_seq) - hyps_len_mean)
        return hyps

    def fusion_hotword_fst(self, best_token, hyp, is_last):
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
        if self.lm_solution and self.hotword_fst_path:
            for hyp in hyps:
                backoff_score = self.lm_solution.hotword_fst_backoff(hyp.lm_tokens)
                hotword_backoff[hyp.label_str] = backoff_score
                hyp.score -= backoff_score
        return hotword_backoff

    def get_fixed_prefix(self, hyps, keep_beam_size=False):
        '''maintain (top one length - changeable_token) length prefix, drop other hyps.
        make sure that hyps are sorted by hyp.score,
        keep_beam_size: if True, append to hyps list with dummy hyp
        '''
        changeable_token = self.fixed_prefix_changeable_token
        fixed_prefix = []
        fixed_prefix_hyps = []
        best_label_seq = hyps[0].label_seq
        best_label_len = len(best_label_seq)
        if best_label_len <= changeable_token:
            for hyp in hyps:
                hyp.fixed_prefix = []
                hyp.changeable_token = -1
            return hyps, fixed_prefix
        fixed_prefix = best_label_seq[: best_label_len - changeable_token]
        for hyp in hyps:
            if (
                best_label_len <= len(hyp.label_seq)
                and hyp.label_seq[: len(fixed_prefix)] == fixed_prefix
            ):
                hyp.fixed_prefix = fixed_prefix
                hyp.changeable_token = changeable_token
                fixed_prefix_hyps.append(hyp)
        while keep_beam_size and len(fixed_prefix_hyps) < self.beam_size:
            fixed_prefix_hyps.append(fixed_prefix_hyps[-1].copy())
            fixed_prefix_hyps[-1].score = -100000.0
            fixed_prefix_hyps[-1].label_seq = [1]  # <pad>, dummy token
        return fixed_prefix_hyps, fixed_prefix

    def fusion_class_lm_fst(self, hyps):
        """fusion class fst, add class fst score"""
        for hyp in hyps:
            last_lm_label = hyp.label_seq[-1] + 1
            class_lm_fst_hyps = hyp.class_lm_fst_hyps
            # Get raw rnnt score.
            hyp.score -= class_lm_fst_hyps[hyp.class_lm_fst_best_hyp_index].class_lm_fst_score
            new_class_lm_fst_hyps = []
            best_score = -math.inf
            for class_lm_fst_hyp in class_lm_fst_hyps:
                class_lm_fst_results = self.lm_solution.step_class_lm_fst(
                    class_lm_fst_hyp.class_lm_fst_idx,
                    last_lm_label,
                    class_lm_fst_hyp.class_lm_fst_states,
                )
                for current_result in class_lm_fst_results:
                    new_class_lm_fst_hyp = class_lm_fst_hyp.copy()
                    fst_idx, states, scores, labels = current_result
                    labels = [x - 1 for x in labels]
                    new_class_lm_fst_hyp.tokens += labels
                    new_class_lm_fst_hyp.class_lm_fst_idx = fst_idx
                    new_class_lm_fst_hyp.class_lm_fst_states = states
                    new_class_lm_fst_hyp.class_lm_fst_score += (
                        +self.class_lm_fst_fusion_weight * scores[0]
                        - self.class_lm_fst_word_weight * scores[1]
                    )
                    new_class_lm_fst_hyps.append(new_class_lm_fst_hyp)
                    if new_class_lm_fst_hyp.class_lm_fst_score > best_score:
                        best_score = new_class_lm_fst_hyp.class_lm_fst_score
                        hyp.class_lm_fst_best_hyp_index = len(new_class_lm_fst_hyps) - 1

            hyp.class_lm_fst_hyps = new_class_lm_fst_hyps
            hyp.score += new_class_lm_fst_hyps[hyp.class_lm_fst_best_hyp_index].class_lm_fst_score

    def class_lm_check_last_frame(self, hyps):
        """Check if class lm fst is full match at last frame"""
        for hyp in hyps:
            class_lm_fst_hyps = hyp.class_lm_fst_hyps
            hyp.score -= class_lm_fst_hyps[hyp.class_lm_fst_best_hyp_index].class_lm_fst_score
            valid = False
            for class_lm_fst_hyp in sorted(
                class_lm_fst_hyps, key=lambda x: x.class_lm_fst_score, reverse=True
            ):
                lm_labels = [x + 1 for x in class_lm_fst_hyp.tokens]
                valid = self.lm_solution.class_lm_fst_check_labels(lm_labels)
                if valid:
                    hyp.score += class_lm_fst_hyp.class_lm_fst_score
                    break
            assert valid is True


class NonBatchBeamSearch(PenguinBeamSearch):
    '''non batch version beam search'''

    def __init__(
        self,
        cfg,
        predictor_module,
        jointer_module,
        criterion_module,
        lm_solution=None,
        rw_bias_moudule=None,
    ):

        '''non batch version beam search init'''
        super().__init__(
            cfg,
            predictor_module,
            jointer_module,
            criterion_module,
            lm_solution=lm_solution,
            rw_bias_moudule=rw_bias_moudule,
        )
        self.blank_thresh = cfg.get('blank_thresh', 1.0)
        self.log_blank_thresh = math.log(self.blank_scale * self.blank_thresh)

    def initialize_hyp(self, device, **kwargs):
        '''initialize the running_hyps'''
        init_hyp = Hypothesis(label_seq=[0])
        prev_token = torch.tensor([0], dtype=torch.long, device=device)
        pred_feats, pred_states = self.predictor_forward_step(prev_token, None, **kwargs)
        init_hyp.pred_feat = pred_feats
        init_hyp.pred_state = pred_states
        if self.lm_solution is not None:
            hotword_fst_num = self.lm_solution.get_hotword_fst_number()
            hotword_start_states = self.lm_solution.hotword_fst_start()
            init_hyp.init_lm_tokens(hotword_fst_num, hotword_start_states)

            init_hyp.coldword_state = self.lm_solution.coldword_fst_start()

            init_hyp.ngram_state = self.lm_solution.ngram_fst_start()

            class_lm_fst_state = self.lm_solution.class_lm_fst_start()
            init_hyp.init_class_lm_fst_hyps(class_lm_fst_state)

            if self.nnlm_path:
                init_hyp.nnlm_lprobs, init_hyp.nnlm_state = self.lm_solution.step_nn(prev_token)
        return [init_hyp]

    def skip_u_step(self, rnnt_lprobs, running_hyps, t_blk_lprobs):
        '''if t_prob > threshold, no more go to u-step'''
        remain_hyps, remain_idx = [], []
        for beam_idx, hyp in enumerate(running_hyps):
            if t_blk_lprobs[beam_idx] < self.log_blank_thresh:
                remain_hyps.append(hyp)
                remain_idx.append(beam_idx)
        return rnnt_lprobs[remain_idx], remain_hyps

    @staticmethod
    def gather_hyps_attributes(hyps, key, dtype=torch.float32, device='cuda'):
        '''gather hypotheses attributes to torch.tensor'''
        items = [item[key] for item in hyps]
        if isinstance(items[0], torch.Tensor):
            return torch.cat(items, dim=0)
        if isinstance(items[0], list):
            # the item is of lstm-state in this case
            lstm_states = []
            for layer_idx in range(len(items[0])):
                h = torch.cat([item[layer_idx][0] for item in items], dim=0)
                c = torch.cat([item[layer_idx][1] for item in items], dim=0)
                lstm_states.append([h, c])
            return lstm_states
        return torch.tensor(items, dtype=dtype, device=device)

    def split_lstm_states(self, states, idx):
        '''get pred_states[idx]'''
        if self.limited_context:
            return states[idx].unsqueeze(dim=0)
        return [[item[idx : idx + 1] for item in state] for state in states]

    @staticmethod
    def tensor2list(*args):
        '''one dim tensor to list'''
        if len(args) == 1:
            return args[0].view(-1).tolist()
        return [item.view(-1).tolist() for item in args]

    def recombine_hyps(self, t_hyps, u_hyps):
        # recombine hypotheses
        label_str2idx = dict()
        u_hyps, running_hyps = [], u_hyps
        for i, hyp in enumerate(running_hyps):
            label_str2idx[hyp.label_str] = i
        for hyp in t_hyps:
            if hyp.label_str not in label_str2idx:
                running_hyps.append(hyp)
            else:
                renew_hyp = running_hyps[label_str2idx[hyp.label_str]]
                if self.recombine_sum:
                    if renew_hyp.score < hyp.score:
                        renew_hyp.timestamp = hyp.timestamp
                        renew_hyp.lm_tokens = hyp.lm_tokens[:]
                        renew_hyp.prev_total_lm_score = hyp.prev_total_lm_score
                        renew_hyp.lm_token_timestamp = hyp.lm_token_timestamp
                        renew_hyp.matched_hotwords = hyp.matched_hotwords
                    renew_hyp.score = np.logaddexp(renew_hyp.score, hyp.score)
                    # the 'hyp.eos_score' comes from t_step hyps, it's newer
                    renew_hyp.eos_score = hyp.eos_score
                    if self.lm_solution and self.nnlm_path:
                        renew_hyp.nnlm_score = np.logaddexp(renew_hyp.nnlm_score, hyp.nnlm_score)
                elif renew_hyp.score < hyp.score:
                    running_hyps[label_str2idx[hyp.label_str]] = hyp
        return running_hyps

    def topk_hyps(self, hyps, beam_size):
        '''get top-k hyps'''
        hyps.sort(
            reverse=True,
            key=self.get_hyp_total_score,
        )
        return hyps[:beam_size]

    def get_prefetch_info(
        self, best_label_seq, eos_score, pred_feat, acous_feat, prev_best_label_seq
    ):
        '''get prefetch info'''
        enable_prefetch = True
        # skip prefetch if there is no change of best hyp
        if prev_best_label_seq[1:] == best_label_seq[1:]:  # label_seq[0] is 0 for all hyps
            enable_prefetch = False
        if enable_prefetch and eos_score is None:
            prefetch_lprobs = self.get_lprobs(acous_feat[0:1, :], pred_feat)
            eos_score = float(prefetch_lprobs[0, 2].item())
        if enable_prefetch and eos_score <= self.log_prefetch_thresh:
            enable_prefetch = False
        return enable_prefetch

    def weighted_ngram_score(self, ngram_score, ilm_score):
        """Transform ngram socre to decoder bias."""
        bias = 0
        if self.ngram_fst_ilm:
            bias = -(ilm_score - max(ngram_score, ilm_score)) * self.ngram_fst_weight
        else:
            bias = ngram_score * self.ngram_fst_weight
        return bias

    @staticmethod
    def merge_score(score1, score2):
        '''merge score using logsumexp
        score1 and score2's shape: [B, 1]
        '''
        return torch.logsumexp(torch.cat((score1, score2), dim=1), dim=1)

    def get_nbest_hyps_info(self, nbest_hyps):
        '''get info of nbest hyps'''
        nbest_info = [
            (
                hyp.label_seq,
                self.get_hyp_total_score(hyp),
                hyp.hotword_score,
                hyp.timestamp,
                hyp.confidence,
            )
            for hyp in nbest_hyps
        ]
        return nbest_info

    def get_hyp_total_score(self, hyp):
        '''add ilme score to get total score.'''
        total_score = (
            hyp.score + hyp.nnlm_score * self.nnlm_weight - hyp.ilm_score * self.ilm_weight
        )
        return total_score

    @torch.no_grad()
    def __call__(self, acoustic_outs, acoustic_masks, **kwargs):
        '''non batch version beam search'''
        # pylint:disable=too-many-branches,too-many-statements,too-many-nested-blocks,too-many-locals
        bsz, _, _ = acoustic_outs.size()
        self.rw_embed = kwargs.get("rw_embed", None)
        frames = acoustic_masks.sum(dim=1).int().tolist()
        batch_nbest, batch_best_list = {}, []
        for res_key in ("nbest", "prefetch", "fixed_prefix", "endpoint"):
            batch_nbest[res_key] = []
        stable_metric_list = kwargs.get('stable_metric_list')
        for bid in range(bsz):
            curr_acoustic_outs = acoustic_outs[bid, : frames[bid], :]
            kwargs['bid'] = bid
            kwargs['beam_num'] = 1
            running_hyps = self.initialize_hyp(acoustic_outs.device, **kwargs)
            prev_best_label_seq = []
            prefetch_list = []
            endpoint_list = []
            fixed_prefix_list = []
            for frame_idx in range(frames[bid]):
                acous_feats = curr_acoustic_outs[frame_idx : frame_idx + 1, :]
                acous_feats = acous_feats.repeat(len(running_hyps), 1)
                t_hyps, u_hyps = [], []
                # collect pred_feat from dict
                pred_feats = self.gather_hyps_attributes(running_hyps, 'pred_feat')
                # get log probs of rnnt
                rnnt_lprobs = self.get_lprobs(acous_feats, pred_feats)
                # merge the score of </s> to blk when enabling prefetch
                if self.prefetch and not self.enable_endpoint:
                    rnnt_lprobs[:, 0] = self.merge_score(rnnt_lprobs[:, 0:1], rnnt_lprobs[:, 2:3])
                elif self.enable_endpoint:
                    eos_score = [lp * self.ep_alpha for lp in rnnt_lprobs[:, 2].cpu().tolist()]
                    for bi in range(rnnt_lprobs.shape[0]):
                        if self.prefetch and not self.check_eos_strong_enough(eos_score[bi]):
                            rnnt_lprobs[bi : bi + 1, 0] = self.merge_score(
                                rnnt_lprobs[bi : bi + 1, 0:1], rnnt_lprobs[bi : bi + 1, 2:3]
                            )

                self.apply_blank_scale(rnnt_lprobs)

                # get internal lm log-probs
                ilm_lprobs = None
                if self.nnlm_ilme or self.ngram_fst_ilm:
                    ilm_lprobs = self.get_lprobs(torch.zeros_like(acous_feats), pred_feats)
                    ilm_lprobs = self.criterion_module.get_log_prob_noblk(ilm_lprobs)
                    ilm_lprobs[:, 0] = 0.0
                    if self.nnlm_ilme:
                        for idx, hyp in enumerate(running_hyps):
                            hyp.ilm_lprobs = ilm_lprobs[idx : idx + 1]

                # t-step, running_hyps -> t_hyps, emit blank
                t_blk_lprobs = self.tensor2list(rnnt_lprobs[:, 0])
                for beam_idx, hyp in enumerate(running_hyps):
                    new_hyp = hyp.copy()
                    new_hyp.score += t_blk_lprobs[beam_idx]
                    new_hyp.eos_score = float(rnnt_lprobs[beam_idx, 2].item())
                    t_hyps.append(new_hyp)

                # if t_step's prob > blk_thresh, no more u-step
                if self.blank_thresh < 1.0:
                    rnnt_lprobs, running_hyps = self.skip_u_step(
                        rnnt_lprobs, running_hyps, t_blk_lprobs
                    )
                    if len(running_hyps) == 0:
                        continue

                # u-step, running_hyps -> u_hyps, emit non-blank tokens
                scores = self.gather_hyps_attributes(
                    running_hyps, 'score', pred_feats.dtype, pred_feats.device
                )
                # disable to emit eos in top beams when enabling prefetch
                if self.prefetch and not self.enable_endpoint:
                    rnnt_lprobs[:, 2] = -math.inf
                elif self.enable_endpoint:
                    rnnt_lprobs[:, 2] = self.ep_alpha * rnnt_lprobs[:, 2]
                    for beam_idx in range(rnnt_lprobs.shape[0]):
                        if not self.check_eos_strong_enough(eos_score[beam_idx]):
                            rnnt_lprobs[beam_idx, 2] = -math.inf

                u_scores = scores.unsqueeze(1) + rnnt_lprobs  # (beam, vocab_size)
                if self.lm_solution and self.nnlm_path:
                    nnlm_scores = self.gather_hyps_attributes(running_hyps, 'nnlm_score')
                    nnlm_lprobs = self.gather_hyps_attributes(running_hyps, 'nnlm_lprobs')
                    nnlm_scores = nnlm_scores.unsqueeze(1) + nnlm_lprobs
                    # shallow fusion
                    u_scores = u_scores + self.nnlm_weight * nnlm_scores
                    if self.nnlm_ilme:
                        ilm_scores = self.gather_hyps_attributes(running_hyps, 'ilm_score')
                        ilm_lprobs = self.gather_hyps_attributes(running_hyps, 'ilm_lprobs')
                        ilm_scores = ilm_scores.unsqueeze(1) + ilm_lprobs
                        u_scores = u_scores - self.ilm_weight * ilm_scores

                # u step top_beam
                u_beam_size = self.beam_size
                if self.use_lm_beam:
                    u_beam_size = self.lm_beam_size
                u_scores, u_beam_idx, u_token_idx = self.top_beam(u_scores, 1, u_beam_size)
                u_scores, u_beam_idx, u_token_idx = self.tensor2list(
                    u_scores, u_beam_idx, u_token_idx
                )
                new_token_list = []
                for i in range(u_beam_size):
                    beam_idx = u_beam_idx[i]
                    new_token = u_token_idx[i]
                    # create a new hyp at u_step
                    hyp = running_hyps[beam_idx]
                    new_hyp = hyp.copy()
                    new_hyp.label_seq.append(new_token)
                    new_hyp.score = u_scores[i]
                    new_hyp.timestamp.append(frame_idx)
                    token_lprob = rnnt_lprobs[beam_idx, new_token].item()
                    new_hyp.confidence.append(np.exp(token_lprob))

                    if self.lm_solution and self.nnlm_path:
                        new_hyp.nnlm_score += float(hyp.nnlm_lprobs[0, new_token])
                        if self.nnlm_ilme:
                            new_hyp.ilm_score += float(hyp.ilm_lprobs[0, new_token])
                        new_hyp.score -= (
                            new_hyp.nnlm_score * self.nnlm_weight
                            - new_hyp.ilm_score * self.ilm_weight
                        )
                    if self.lm_solution:
                        # NOTE: compare to dict reorder_tgt_dict, reorder_idx.dict
                        # have extra <eps> symbol, so the lm_token = new_token + 1
                        lm_token = new_token + 1
                        if self.ngram_fst_path:
                            (
                                new_hyp.ngram_state,
                                ngram_score,
                            ) = self.lm_solution.step_ngram(lm_token, new_hyp.ngram_state)
                            ilm_scores = None
                            if self.ngram_fst_ilm and ilm_lprobs is not None:
                                ilm_scores = float(ilm_lprobs[beam_idx, new_token].cpu())
                            ngram_bias = self.weighted_ngram_score(ngram_score, ilm_scores)
                            new_hyp.score += ngram_bias
                        if self.hotword_fst_path:
                            is_last = frame_idx == frames[bid] - 1
                            best_token = self.lm_solution.step_hotword(lm_token, new_hyp, is_last)
                            if best_token is not None:
                                self.fusion_hotword_fst(best_token, new_hyp, is_last)
                        if self.coldword_fst_path:
                            # cold-FST: greedy search for beam
                            coldword_fst_state = new_hyp.coldword_state
                            coldword_fst_state, coldword_fst_score = self.lm_solution.step_coldword(
                                lm_token, coldword_fst_state
                            )
                            new_hyp.score -= coldword_fst_score * self.coldword_fst_weight
                            new_hyp.coldword_state = coldword_fst_state

                    u_hyps.append(new_hyp)
                    new_token_list.append(new_token)

                new_token_list = torch.tensor(
                    new_token_list, device=acoustic_outs.device, dtype=torch.long
                )
                # predictor / nnlm forwards one step after u-step
                u_pred_states = self.gather_hyps_attributes(u_hyps, 'pred_state')
                kwargs['beam_num'] = len(u_hyps)
                u_pred_feats, u_pred_states = self.predictor_forward_step(
                    new_token_list, u_pred_states, **kwargs
                )
                for idx, hyp in enumerate(u_hyps):
                    hyp.pred_state = self.split_lstm_states(u_pred_states, idx)
                    hyp.pred_feat = u_pred_feats[idx : idx + 1]
                if self.lm_solution and self.nnlm_path:
                    u_lm_states = self.gather_hyps_attributes(u_hyps, 'nnlm_state')
                    u_lm_lprobs, u_lm_states = self.lm_solution.step_nn(
                        new_token_list.view(-1), u_lm_states
                    )
                    for idx, hyp in enumerate(u_hyps):
                        hyp.nnlm_state = self.split_lstm_states(u_lm_states, idx)
                        hyp.nnlm_lprobs = u_lm_lprobs[idx : idx + 1]
                # class fst
                if self.lm_solution and self.class_lm_fst_path:
                    self.fusion_class_lm_fst(u_hyps)
                # do recombine
                running_hyps = self.recombine_hyps(t_hyps, u_hyps)

                # add len penalty
                if self.len_penalty_scale > 0.0:
                    running_hyps = self.add_length_penalty(running_hyps)

                # endpoint should before prefetch
                # if endpoint is triggered, we don't need prefetch
                if self.enable_endpoint and frame_idx < frames[bid] - 1:
                    hotword_backoff = self.hotwords_score_backoff(running_hyps)
                    running_hyps = self.topk_hyps(running_hyps, self.beam_size)
                    best_hyp_label_seq = running_hyps[0].label_seq
                    if best_hyp_label_seq[-1] == 2:
                        # stop decoding once </s> emitted in 1-best
                        endpoint_list.append((frame_idx, best_hyp_label_seq))
                        break
                    # restore partial fst score
                    if hotword_backoff:
                        for hyp in running_hyps:
                            hyp.score += hotword_backoff.get(hyp.label_str, 0)

                if self.prefetch and frame_idx < frames[bid] - 1:
                    hotword_backoff = self.hotwords_score_backoff(running_hyps)
                    running_hyps = self.topk_hyps(running_hyps, self.beam_size)
                    best_hyp = running_hyps[0]
                    best_hyp_label_seq = best_hyp.label_seq
                    enable_prefetch = self.get_prefetch_info(
                        best_hyp_label_seq,
                        best_hyp.eos_score,
                        best_hyp.pred_feat[-1],
                        acous_feats,
                        prev_best_label_seq,
                    )
                    if enable_prefetch:
                        nbest_info = self.get_nbest_hyps_info(running_hyps[: self.nbest])
                        prefetch_list.append((frame_idx, best_hyp_label_seq, nbest_info))
                        prev_best_label_seq = best_hyp_label_seq
                    if hotword_backoff:
                        for hyp in running_hyps:
                            hyp.score += hotword_backoff.get(hyp.label_str, 0)

                if frame_idx == frames[bid] - 1:
                    self.hotwords_score_backoff(running_hyps)
                    if self.lm_solution and self.class_lm_fst_path:
                        self.class_lm_check_last_frame(running_hyps)

                # sort and get top beam hyps
                running_hyps = self.topk_hyps(running_hyps, self.beam_size)

                if self.fixed_prefix_beam_search and frame_idx % self.fixed_prefix_time_freq == 0:
                    running_hyps, fixed_prefix = self.get_fixed_prefix(running_hyps)
                    fixed_prefix_list.append((frame_idx, fixed_prefix))

                # try stable metric update if neccessary
                if stable_metric_list is not None:
                    stable_metric_list[bid].update(running_hyps[0].label_seq, frame_idx)

            # get best branch
            nbest_hyps = self.topk_hyps(running_hyps, self.nbest)
            nbest_info = self.get_nbest_hyps_info(nbest_hyps)
            best_label_seq = nbest_hyps[0].label_seq
            if self.prefetch:
                prefetch_list.append((-1, best_label_seq, nbest_info))
                batch_nbest["prefetch"].append(prefetch_list)
            if self.fixed_prefix_beam_search:
                fixed_prefix_list.append((frame_idx, best_label_seq))
                batch_nbest["fixed_prefix"].append(fixed_prefix_list)
            if self.enable_endpoint:
                endpoint_list.append((-1, best_label_seq))
                endpoint_list.append(nbest_info)
                batch_nbest["endpoint"].append(endpoint_list)
            batch_nbest["nbest"].append(nbest_info)
            batch_best_list.append(best_label_seq)
        return batch_best_list, batch_nbest
