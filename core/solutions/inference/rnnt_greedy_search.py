"""rnnt_greedy_search"""
import math
import torch
from core.extensions import edit_distance
from core.solutions.inference.utils import rnnt_rlt_neaten


class GreedySearch:
    """Greedy Search"""

    def __init__(self, args, criterion_module, predictor_module, jointer_module):
        """__init__"""
        self.args = args
        self.rnntce = criterion_module
        self.predictor_module = predictor_module
        self.jointer_module = jointer_module
        self.blank_scale = args.get('cer_blank_scale', 1.0)

    def __call__(self, encoder_out, target, input_dict, **kwargs):
        """__call__"""
        error_dist = 0
        total_dist = 0

        greedy_infer_rlt = self.greedy_infer(
            encoder_out, self.predictor_module, self.jointer_module, **kwargs
        )
        error_dist, total_dist = self.greedy_infer_post_process(
            greedy_infer_rlt, target, input_dict
        )
        return error_dist, total_dist

    def greedy_infer_post_process(self, greedy_infer_rlt, target, input_dict):
        '''post process of greedy search'''
        total_dist = 0
        bsz = greedy_infer_rlt.size(0)
        target_lens = input_dict['target_lengths']
        refs = []
        hyps = []
        for bid in range(bsz):
            hyp_token_list = greedy_infer_rlt[bid]
            hyp_token_list = rnnt_rlt_neaten(hyp_token_list, True)
            tgt_token_list = target[bid][: target_lens[bid]]
            # delete the eos in target and infer result is use_eos
            if self.args.use_eos:
                if len(hyp_token_list) >= 1 and hyp_token_list[-1] == 2:
                    hyp_token_list = hyp_token_list[:-1]
                tgt_token_list = tgt_token_list[:-1]
            # TODO(huanglu) use the edit_distance in inference
            refs.append(tgt_token_list.contiguous())
            hyps.append(hyp_token_list.contiguous())
            total_dist += target_lens[bid]
        result = edit_distance(refs, hyps)
        return result[:, :1].sum().item(), total_dist

    def greedy_infer(self, acoustic_out, predictor_module, jointer_module, **kwargs):
        """greedy inference

        Args:

            - acoustic_out: [B, T, N], the output of encoder
            - predictor_module: the module of predictor
            - jointer_module: the module of jointer

        Return:

            - output: [B, T], greedy search results
        """
        (bsz, frame, _) = acoustic_out.size()
        output = acoustic_out.data.new(bsz, frame).zero_().int()
        prev_token = (
            acoustic_out.data.new(
                bsz,
            )
            .zero_()
            .long()
        )
        predicter_feat, predicter_states = predictor_module.forward_step(prev_token, None, **kwargs)
        log_blank_scale = math.log(self.blank_scale)
        for frame_idx in range(frame):
            jointer_out = jointer_module.forward_step(acoustic_out[:, frame_idx, :], predicter_feat)
            # do combine_weight in log_softmax_fc.get_log_prob
            lprobs = self.rnntce.get_log_prob(jointer_out)
            lprobs[:, 0] += log_blank_scale
            hyp = torch.argmax(lprobs, dim=-1)
            output[:, frame_idx] = hyp
            hyp = hyp.view(
                bsz,
            )
            mask = hyp == self.args.tgt_dict.bos()
            mask = mask.unsqueeze(1)
            predicter_feat_next, predicter_states_next = predictor_module.forward_step(
                hyp, predicter_states, **kwargs
            )
            predicter_feat = torch.where(mask, predicter_feat, predicter_feat_next)
            predicter_states = predictor_module.select_non_blank_states(
                predicter_states, predicter_states_next, mask
            )
        self.rnntce.clear_combined_weight()
        return output
