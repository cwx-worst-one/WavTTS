"""rnnt decoder processor"""
# pylint: disable=no-member,too-many-branches,too-many-locals,too-many-statements,abstract-method
import math
import copy
import numpy as np
from core.solutions.inference.penguin.core.processor.top_k import TopK
from core.solutions.inference.penguin.core.processor.processor import Processor
from core.solutions.inference.penguin.core.processor.message import Message
from core.solutions.inference.penguin.core.register import Registers
from core.solutions.inference.penguin.core.factory import Factory

from core.solutions.inference.penguin.core.processor.lm.search import get_log_prob_noblk
from core.solutions.inference.penguin.tasks.rnnt.tools.serializer import serializer
from core.solutions.lm.lm_solution import LmSolution
from core.solutions.inference.hypothesis import Hypothesis
from core.solutions.inference.rnnt_beam_search import PenguinBeamSearch
from core.solutions.asr.utils import force_align

INT_MAX = 2147483647
LAS_RESCORE_MAX_FRAMES = 750


def log_add_exp(value_a, value_b):
    '''equivalent to np.logaddexp'''
    return max(value_a, value_b) + math.log1p(math.exp(-math.fabs(value_a - value_b)))


@Registers.processor.register('decoder')
class Decoder(Processor, PenguinBeamSearch):
    '''rnn-t beam searcher'''

    def __init__(self, config):
        '''init'''
        Processor.__init__(self)
        PenguinBeamSearch.__init__(self, config=config)
        self.config = config
        self.backoff_eps = INT_MAX
        self.encoder_type = config.encoder_type
        self.beam_size = int(config.beam_size)
        self.u_step_expand_times = int(config.u_steps)
        self.enable_prefetch = False  # Flag for prefetch
        self.len_penalty_scale = float(config.len_penalty_scale)
        self.log_blank_scale = math.log(config.blank_scale)
        self.prefetch = bool(config.prefetch)
        self.prefetch_thresh = math.log(config.prefetch_thresh)
        self.fixed_prefix = bool(config.fixed_prefix)
        self.fixed_prefix_time_freq = int(config.fixed_prefix_time_freq)
        self.fixed_prefix_changeable_token = int(config.fixed_prefix_changeable_token)
        self.fixed_prefix_lists = []
        self.output_confidence = bool(config.output_confidence)
        # add ilme
        self.ilme = bool(config.ilme)
        self.domain_nnlm_scale = config.domain_nnlm_scale
        self.internal_nnlm_scale = config.internal_nnlm_scale
        self.use_domain_lm_nnlm = bool(config.use_domain_lm_nnlm)
        # multi fst
        self.use_hotword_fst = bool(config.use_hotword_fst)
        self.use_fst_skip = bool(config.use_fst_skip)
        self.fst_skip_length = int(config.fst_skip_length)
        self.domain_lm_fst_path = config.domain_lm_fst_path
        self.use_domain_lm_fst = bool(config.use_domain_lm_fst)
        self.ngram_fst_minus_internal_lm = bool(config.ngram_fst_minus_internal_lm)
        # class_lm
        self.use_class_lm_fst = config.use_class_lm_fst
        self.class_lm_fst_fusion_weight = config.class_lm_fst_fusion_weight
        self.class_lm_fst_word_weight = config.class_lm_fst_word_weight
        inference_cfg = config.inference_cfg
        self.lm_solution = None
        if self.use_hotword_fst or self.use_domain_lm_fst or self.use_class_lm_fst:
            self.lm_solution = LmSolution(config.lm_cfg)
            self.lm_solution.load_from_inference_cfg(inference_cfg)
            self.domain_lm_scale = float(config.domain_lm_scale)

        self.output_speed = bool(config.output_speed)
        self.use_ce_timestamp_conf = bool(config.use_ce_timestamp_conf)
        self.use_las_rescore = bool(config.use_las_rescore)
        if self.use_las_rescore and self.prefetch:
            self.las_rescore_prefetch_cache = []
            self.prefetch_lists_twopass = []
            # the index map between 1st and second pass prefetch
            # -1 for unvalid prefetch in one segment, -2 for the prefetch
            # longer than LAS_RESCORE_MAX_FRAMES, others for valid prefetch
            self.prefetch_first2second_idx = {}
        self.use_las_fst = bool(config.use_las_fst)
        self.use_las_g2p = bool(config.use_las_g2p)
        self.las_fst_scale = float(config.las_fst_scale)
        self.fw_decoder_scale = float(config.fw_decoder_scale)
        self.bw_decoder_scale = float(config.bw_decoder_scale)
        self.las_bos_idx = int(config.las_bos_idx)
        self.las_eos_idx = int(config.las_eos_idx)
        self.las_encoder_in = []

        predictor = Factory.get_processor('predictor', config)
        predictor_input = predictor.get_input_all()
        predictor_state = np.zeros((1, predictor_input[1].shape[-1])).astype(np.float32)

        # reduced_embed_predictor
        if bool(config.reduced_embed_predictor):
            embed = np.loadtxt('{}'.format(config.embed_txt_path), delimiter='\n').astype(
                np.float32
            )
            predictor_state = embed.reshape((1, predictor_input[1].shape[-1]))

        predictor_out = predictor.process(
            Message(
                {
                    predictor_input[0].name: np.zeros((1)).astype(np.int64),
                    predictor_input[1].name: predictor_state,
                }
            )
        )
        pred_out_names = predictor.get_output()
        self.init_pred_feat = predictor_out[pred_out_names[0]]
        self.init_pred_state = predictor_out[pred_out_names[1]]

        self.init_nnlm_probs = []
        self.init_nnlm_state = []
        if self.use_domain_lm_nnlm:
            nnlm_inference = Factory.get_processor('nnlm', config)
            nnlm_input = nnlm_inference.get_input_all()
            nnlm_out = nnlm_inference.process(
                Message(
                    {
                        nnlm_input[0].name: np.zeros((1)).astype(np.int64),
                        nnlm_input[1]
                        .name: np.zeros((1, nnlm_input[1].shape[-1]))
                        .astype(np.float32),
                    }
                )
            )
            nnlm_out_names = nnlm_inference.get_output()
            self.init_nnlm_probs = nnlm_out[nnlm_out_names[0]]
            self.init_nnlm_state = nnlm_out[nnlm_out_names[1]]
        self.pred_label = [0] * self.beam_size
        self.running_hyps = []
        self.prefetch_lists = []
        self.time = 0
        self.total_time = 0
        self.output_streaming_stable_metric = config.output_streaming_stable_metric

    def __call__(self, input_data):
        '''ASR decoding, decodes one wave'''
        self.wav_name = input_data[0]
        decoder_in = input_data[1]
        if self.use_las_rescore or self.use_las_g2p:
            self.las_encoder_in = input_data[2]
        self.reset()
        self.get_total_time(decoder_in)
        if self.output_streaming_stable_metric:
            label_seq_by_segs = []
        for seg_data in decoder_in:
            # decode each segment for an utterance
            self.process(seg_data)
            if self.output_streaming_stable_metric:
                label_seq_by_segs.append(self.running_hyps[0].label_seq)
        if self.prefetch:  # skip prefetch at the last segment
            self.enable_prefetch = False
        if self.use_las_rescore:
            # if prefetch successed, skip las rescore
            prefetch_seqs = [p[1] for p in self.prefetch_lists]
            top1_seq = self.running_hyps[0]["label_seq"]
            need_las_rescore = True
            if self.prefetch and len(prefetch_seqs) > 0 and top1_seq in prefetch_seqs:
                # from newest to oldest
                for first_idx, prefetch_seq in enumerate(prefetch_seqs):
                    if prefetch_seq == top1_seq:
                        second_idx = self.prefetch_first2second_idx[first_idx]
                        if second_idx >= 0:
                            # skip las_rescore for valid prefetch with
                            # 2pass results
                            prefetch_cache = self.las_rescore_prefetch_cache[second_idx]
                            self.running_hyps = copy.deepcopy(prefetch_cache)
                            need_las_rescore = False
                        # second_idx = -2 will skip las_rescore due to
                        # LAS_RESCORE_MAX_FRAMES, and second_idx = -1 is the
                        # invalid prefetch in one segment and need rescore
                        break
            if need_las_rescore:
                self.las_rescore()
        label_seq, extra = self.get_best_label_seq()
        if self.output_streaming_stable_metric:
            if self.use_las_rescore:
                label_seq_by_segs.append(label_seq)
            extra['label_seq_by_segs'] = label_seq_by_segs
        if self.use_las_g2p:
            pronounce_seq = self.las_g2p(label_seq)
            extra['pronounce_seq'] = pronounce_seq
        if self.use_ce_timestamp_conf:
            ce_encoder_in = input_data[2]
            ce_logits = self.ce_encoder(ce_encoder_in)
            out_list = self.apply_ce_timestamp(ce_logits.ce_output, label_seq)
            extra['ce_out_list'] = out_list
        return self.wav_name, label_seq, extra

    def process(self, seg_data):
        '''decode one segment of an utterance
        seg_data: [1, 5, D] in streaming ASR
                  [1, T, D] in non-streaming ASR, T is the length of the whole utterance
        '''
        for _, segment in enumerate(seg_data):
            for i in range(len(segment)):
                one_frame = segment[i : i + 1]
                self.beam_search_impl(one_frame, self.time)
                self.time += 1
            # set the flag of enable_prefetch at the end of one segment
            self.prefetch_after_one_segment()
            self.prev_segment_time = self.time

    def get_total_time(self, decoder_in):
        '''get num of frames of an utterance'''
        self.total_time = 0
        for data in decoder_in:
            self.total_time += len(data[0])

    def ce_encoder(self, ce_encoder_in, encoder_name='ce_encoder'):
        '''ce encoder forward'''
        ce_encoder_in = np.concatenate(ce_encoder_in, axis=1)
        ce_encoder_inference = Factory.get_inference(encoder_name, self.config)
        ce_encoder_input_name = ce_encoder_inference.get_input()
        if len(ce_encoder_input_name) > 1:
            global_state_in = np.zeros((1, ce_encoder_in.shape[-1])).astype(np.float32)
            ce_encoder_data = Message(
                {
                    ce_encoder_input_name[0]: ce_encoder_in,
                    ce_encoder_input_name[1]: global_state_in,
                }
            )
        else:
            ce_encoder_data = Message(
                {
                    ce_encoder_input_name[0]: ce_encoder_in,
                }
            )
        input_data = [ce_encoder_data[name] for name in ce_encoder_input_name]
        ce_logits_out = ce_encoder_inference.run(input_data)
        return ce_logits_out

    def las_encoder(self, encoder_all, encoder_name='las_encoder'):
        '''las encoder forward'''
        las_encoder_inference = Factory.get_inference(encoder_name, self.config)
        las_encoder_input_name = las_encoder_inference.get_input()
        if len(las_encoder_input_name) > 1:
            global_state_in = np.zeros((1, encoder_all.shape[-1])).astype(np.float32)
            las_encoder_data = Message(
                {
                    las_encoder_input_name[0]: encoder_all,
                    las_encoder_input_name[1]: global_state_in,
                }
            )
        else:
            las_encoder_data = Message(
                {
                    las_encoder_input_name[0]: encoder_all,
                }
            )
        input_data = [las_encoder_data[name] for name in las_encoder_input_name]
        las_encoder_out = las_encoder_inference.run(input_data)
        return las_encoder_out

    # pylint: disable=no-self-use
    def las_decoder(
        self,
        las_encoder_out,
        decoder_input,
        decoder_input_mask,
        decoder_input_bw,
        las_decoder_inference,
    ):
        '''lasr decoder forward'''
        las_encoder_output_name = las_encoder_out.key_list()
        las_decoder_input_name = las_decoder_inference.get_input()
        las_decoder_data = Message(
            {
                las_decoder_input_name[0]: las_encoder_out[las_encoder_output_name[0]],
                las_decoder_input_name[1]: decoder_input,
                las_decoder_input_name[2]: decoder_input_bw,
            }
        )
        input_data = [las_decoder_data[name] for name in las_decoder_input_name]
        las_decoder_out = las_decoder_inference.run(input_data)
        las_score = (las_decoder_out.output * decoder_input_mask).sum(1) / (
            decoder_input_mask.sum(1)
        )
        las_score_bw = (las_decoder_out.output_bw * decoder_input_mask).sum(1) / (
            decoder_input_mask.sum(1)
        )
        return las_score, las_score_bw

    def las_rescore(self, prefetch=False):
        '''las rescore'''
        # skip las_rescore for long audio
        if self.time > LAS_RESCORE_MAX_FRAMES:
            if prefetch:
                # 1st prefetch don't have 2nd prefetch for long audio
                self.prefetch_first2second_idx[len(self.prefetch_lists) - 1] = -2
            return
        # las encoder
        if prefetch:
            encoder_all = np.concatenate(self.las_encoder_in, axis=1)[:, : self.time, :]
            # cannot change the score of hyps during decoding when prefetching
            running_hyps = copy.deepcopy(self.running_hyps)
        else:
            encoder_all = np.concatenate(self.las_encoder_in, axis=1)
            running_hyps = self.running_hyps
        las_encoder_out = self.las_encoder(encoder_all)
        # las decoder
        max_seq_len = 0
        rnnt_hyps = []
        rnnt_scores = []
        las_fst_score = []
        for hyp in running_hyps:
            max_seq_len = max(max_seq_len, len(hyp.label_seq))
            rnnt_hyps.append(hyp.label_seq)
            # rnnt_score 归一化
            rnnt_scores.append(hyp.score / self.time)
            las_fst_score.append(hyp.hotword_score)
            hyp.nnlm_score /= self.time
            hyp.ilm_score /= self.time
        decoder_input = (
            np.ones((len(rnnt_hyps), max_seq_len + 2)).astype(np.int64) * self.las_eos_idx
        )
        decoder_input_bw = (
            np.ones((len(rnnt_hyps), max_seq_len + 2)).astype(np.int64) * self.las_bos_idx
        )
        decoder_input_mask = np.zeros((len(rnnt_hyps), max_seq_len + 1)).astype(np.float32)
        for idx, seq in enumerate(rnnt_hyps):
            decoder_input[idx, 0] = self.las_bos_idx
            decoder_input[idx, 1 : len(seq) + 1] = seq
            decoder_input_bw[idx, 0] = self.las_eos_idx
            decoder_input_bw[idx, 1 : len(seq) + 1] = seq[::-1]
            decoder_input_mask[idx, : len(seq) + 1] = 1.0
        # get decoder scores
        las_decoder_inference = Factory.get_inference('las_decoder', self.config)
        las_score_fw, las_score_bw = self.las_decoder(
            las_encoder_out,
            decoder_input,
            decoder_input_mask,
            decoder_input_bw,
            las_decoder_inference,
        )
        total_score = (
            np.array(rnnt_scores)
            - self.las_fst_scale * np.array(las_fst_score)
            + self.fw_decoder_scale * las_score_fw
            + self.bw_decoder_scale * las_score_bw
        )
        beam_idx = 0
        for hyp in running_hyps:
            hyp.score = total_score[beam_idx]
            beam_idx += 1
        if prefetch:
            # reordering to get two-pass top-1 hyp for sending to downstream
            running_hyps.sort(
                key=lambda Take: Take.score
                + self.domain_nnlm_scale * Take.nnlm_score
                - self.internal_nnlm_scale * Take.ilm_score,
                reverse=True,
            )
            self.las_rescore_prefetch_cache.append(running_hyps)
            # when prefetching with las rescore, two-pass top-1
            # running_hyps[0].label_seq is send to downstream in place of
            # first-pass top-1 self.running_hyps[0].label_seq
            self.prefetch_lists_twopass.append((self.time - 1, running_hyps[0].label_seq))
            self.prefetch_first2second_idx[len(self.prefetch_lists) - 1] = (
                len(self.prefetch_lists_twopass) - 1
            )

    # pylint: disable=no-self-use
    def las_g2p_decoder(self, las_encoder_out, decoder_input, las_decoder_inference):
        '''las g2p decoder'''
        las_decoder_input_name = las_decoder_inference.get_input()
        las_encoder_output_name = las_encoder_out.key_list()
        las_decoder_data = Message(
            {
                las_decoder_input_name[0]: las_encoder_out[las_encoder_output_name[0]],
                las_decoder_input_name[1]: decoder_input,
            }
        )
        input_data = [las_decoder_data[name] for name in las_decoder_input_name]
        las_decoder_out = las_decoder_inference.run(input_data)
        return las_decoder_out.output

    def las_g2p(self, label_seq):
        '''las g2p'''
        # las encoder
        encoder_all = np.concatenate(self.las_encoder_in, axis=1)
        las_encoder_out = self.las_encoder(encoder_all, 'las_g2p_encoder')
        # las decoder
        seq_len = len(label_seq)
        decoder_input = np.zeros((1, seq_len)).astype(np.int64)
        decoder_input[0, :] = label_seq
        # get decoder results
        las_decoder_inference = Factory.get_inference('las_g2p_decoder', self.config)
        las_score = self.las_g2p_decoder(las_encoder_out, decoder_input, las_decoder_inference)
        pronounce_seq = np.argmax(las_score, axis=2)[0, :]
        return pronounce_seq

    def get_best_label_seq(self):
        '''get best labels of hyps'''
        self.running_hyps.sort(
            key=lambda Take: Take.score
            + self.domain_nnlm_scale * Take.nnlm_score
            - self.internal_nnlm_scale * Take.ilm_score,
            reverse=True,
        )
        extra = {}
        extra['score'] = (
            self.running_hyps[0].score
            + self.domain_nnlm_scale * self.running_hyps[0].nnlm_score
            - self.internal_nnlm_scale * self.running_hyps[0].ilm_score
        )
        if self.prefetch:
            extra["prefetch"] = self.prefetch_lists
            if self.use_las_rescore:
                extra["prefetch_twopass"] = self.prefetch_lists_twopass
        if self.output_confidence:
            extra["confidence"] = self.running_hyps[0].confidence
        if self.output_speed:
            extra["total_time"] = self.total_time
        if self.fixed_prefix:
            extra["fixed_prefix_info"] = self.fixed_prefix_lists
        return self.running_hyps[0].label_seq, extra

    def top_k_or_pad_k(self):
        '''get top_k hyps, or pad to k hyps'''
        while len(self.running_hyps) < self.beam_size:
            self.running_hyps.append(self.running_hyps[-1].copy())
            self.running_hyps[-1].score = -100000.0
        self.running_hyps.sort(
            key=lambda Take: Take.score
            + self.domain_nnlm_scale * Take.nnlm_score
            - self.internal_nnlm_scale * Take.ilm_score,
            reverse=True,
        )

        self.running_hyps = self.running_hyps[0 : self.beam_size]

    def prefetch_after_one_segment(self):
        '''prefetch after one segment done'''
        self.enable_prefetch = False
        if not self.prefetch or len(self.prefetch_lists) == 0:
            return
        # last_prefetch_seq is the latest prefetch result
        prefetch_frame, last_prefetch_seq = self.prefetch_lists[-1]
        # top1_seq is the 1st pass top-1
        top1_seq = self.running_hyps[0]["label_seq"]
        # enable_prefetch when top-1 at the end of segment is the same as
        # last_prefetch_seq and last_prefetch_seq is in current segment
        # i.e top1_seq == last_prefetch_seq and
        # prefetch_frame >= self.prev_segment_time
        if top1_seq != last_prefetch_seq or prefetch_frame < self.prev_segment_time:
            return
        self.enable_prefetch = True
        if self.use_las_rescore:
            self.las_rescore(prefetch=True)

    def reset(self):
        '''reset for each utterance'''
        self.time = 0
        self.total_time = 0
        self.prev_segment_time = 0
        self.running_hyps = []
        self.prefetch_lists = []
        self.fixed_prefix_hyp = []
        self.fixed_prefix_lists = []
        self.enable_prefetch = False
        if self.use_las_rescore and self.prefetch:
            self.las_rescore_prefetch_cache = []
            self.prefetch_lists_twopass = []
            self.prefetch_first2second_idx = {}

        init_hyp = Hypothesis(
            label_seq=[],
            timestamp=[],
            pred_feat=self.init_pred_feat[0][0],
            pred_state=self.init_pred_state[0],
            nnlm_lprobs=self.init_nnlm_probs,
            nnlm_state=self.init_nnlm_state,
        )
        if self.lm_solution is not None:
            hotword_fst_num = self.lm_solution.get_hotword_fst_number()
            hotword_start_states = self.lm_solution.hotword_fst_start()
            init_hyp.init_lm_tokens(hotword_fst_num, hotword_start_states)
            init_hyp.coldword_state = self.lm_solution.coldword_fst_start()
            init_hyp.ngram_state = self.lm_solution.ngram_fst_start()
            class_lm_fst_state = self.lm_solution.class_lm_fst_start()
            init_hyp.init_class_lm_fst_hyps(class_lm_fst_state)

        self.running_hyps.append(init_hyp)
        for _ in range(self.beam_size - 1):
            hyp = self.running_hyps[0].copy()
            hyp.score = -100000
            hyp.label_seq = [1]  # <pad>, dummy token
            self.running_hyps.append(hyp)

    def get_eos_score(self, jointer_output):
        '''get eos score'''
        num_hyps = jointer_output.shape[0]
        eos_score = []
        for i in range(num_hyps):
            eos_score.append(jointer_output[i][2])
            if self.prefetch:
                jointer_output[i][0] = np.logaddexp(jointer_output[i][0], jointer_output[i][2])
                jointer_output[i][2] = -math.inf
        return eos_score

    @staticmethod
    def gather_hyps_attributes(hyps, key, hyp_orders=None, dtype=np.float32):
        '''gather hypotheses attributes to np.array'''
        if hyp_orders is None:
            hyp_orders = range(len(hyps))
        assert len(hyp_orders) == len(hyps)
        vals = []
        for i in hyp_orders:
            vals.append(hyps[i][key])
        return np.array(vals, dtype=dtype)

    def beam_search_impl(self, encoder_out, time_step):
        '''
        encoder_out: one frame of audio encoder output
        time_step: is the frame_idx of acoustic backbone_out
        '''
        for _ in range(self.u_step_expand_times):
            t_step_hyps, u_step_hyps = [], []
            pred_feat = self.gather_hyps_attributes(self.running_hyps, 'pred_feat')

            # get jointer out
            jointer_inference = Factory.get_processor('jointer', self.config)
            jointer_input_name = jointer_inference.get_input()
            jointer_output_name = jointer_inference.get_output()
            # jointer_input_name: encoder_out, predictor_out
            # jointer_output_name: output, new_score
            jointer_input = Message(
                {
                    jointer_input_name[0]: np.tile(encoder_out, (self.beam_size, 1)),
                    jointer_input_name[1]: pred_feat,
                }
            )
            serializer.add_data(
                self.wav_name, encoder_out, time_step=time_step, name=jointer_input_name[0]
            )
            serializer.add_data(
                self.wav_name, pred_feat, time_step=time_step, name=jointer_input_name[1]
            )
            jointer_out = jointer_inference.process(jointer_input)
            eos_score = self.get_eos_score(jointer_out['output'])

            # get internal lm #
            # TODO: if self.ilme:  # no matter external lm is nnlm or ngram
            if self.use_domain_lm_fst or self.ilme:
                # need panther version >= 1.4.2 && new model
                if len(jointer_output_name) == 2:
                    jointer_out_ilm_lprobs = jointer_out[
                        jointer_output_name[1]
                    ]  # beam x vocab_size
                else:
                    encoder_shape = encoder_out.shape
                    zero_encoder_out = np.zeros(encoder_shape).astype(encoder_out.dtype)
                    jointer_inference_ilm = Factory.get_processor('jointer', self.config)
                    jointer_input_name_ilm = jointer_inference.get_input()

                    jointer_input_ilm = Message(
                        {
                            jointer_input_name_ilm[0]: zero_encoder_out,
                            jointer_input_name_ilm[1]: pred_feat,
                        }
                    )
                    jointer_out_ilm = jointer_inference_ilm.process(jointer_input_ilm)
                    jointer_out_ilm_lprobs = get_log_prob_noblk(jointer_out_ilm.output)
                    jointer_out_ilm.output = jointer_out_ilm_lprobs

            # Domain lm
            if self.use_domain_lm_nnlm:
                nnlm_scores = self.gather_hyps_attributes(self.running_hyps, 'nnlm_score')
                nnlm_lprobs = self.gather_hyps_attributes(self.running_hyps, 'nnlm_lprobs')
                nnlm_scores = nnlm_lprobs.squeeze() + np.expand_dims(nnlm_scores, axis=1)
                nnlm_scores[:, 0] = 0
                u_scores = jointer_out['output'] + self.domain_nnlm_scale * nnlm_scores
                if self.ilme:
                    ilm_lprobs = jointer_out_ilm_lprobs
                    ilm_scores = self.gather_hyps_attributes(self.running_hyps, 'ilm_score')
                    ilm_scores = np.expand_dims(ilm_scores, axis=1) + ilm_lprobs
                    ilm_scores[:, 0] = 0
                    u_scores = u_scores - self.internal_nnlm_scale * ilm_scores
            else:
                u_scores = jointer_out['output']

            # get TopK
            vocab_size = jointer_out.output.shape[1]
            hyp_scores = np.array([[hyp.score for hyp in self.running_hyps]])
            topk_input = Message({"jointer_output": np.array([u_scores]), "hyp_scores": hyp_scores})
            serializer.add_data(
                self.wav_name, jointer_out['output'], time_step=time_step, name="topk_input_0"
            )
            serializer.add_data(self.wav_name, hyp_scores, time_step=time_step, name="topk_input_1")
            topk_inference = TopK()
            topk_inference.process(topk_input, self.beam_size)
            t_hyps_scores = topk_inference.t_hyps_scores
            u_hyps_scores = topk_inference.u_hyps_scores
            u_hyps_topk_idx = topk_inference.u_hyps_topk_idx

            # t_step_expanding, running_hyps -> t_step_hyps
            for i in range(self.beam_size):
                hyp = self.running_hyps[i].copy()
                # add blank scale
                hyp.score = t_hyps_scores[0][i] + self.log_blank_scale
                hyp.eos_score = eos_score[i]
                t_step_hyps.append(hyp)

            # u_step_expanding, running_hyps -> u_step_expanding
            # new emited tokens
            self.pred_label = [idx % vocab_size for idx in u_hyps_topk_idx[0]]
            # reordered hyps indices list
            hyp_orders = [int(idx / vocab_size) for idx in u_hyps_topk_idx[0]]
            pred_states = self.gather_hyps_attributes(self.running_hyps, 'pred_state', hyp_orders)

            # predictor forward one step after u_step_expanding
            predictor_inference = Factory.get_processor('predictor', self.config)
            predictor_input_name = predictor_inference.get_input()
            predictor_output_name = predictor_inference.get_output()
            predictor_input = Message(
                {
                    predictor_input_name[0]: np.array(self.pred_label).astype(np.int64),
                    predictor_input_name[1]: pred_states,
                }
            )
            serializer.add_data(
                self.wav_name,
                self.pred_label,
                time_step=time_step,
                name=predictor_input_name[0],
            )
            serializer.add_data(
                self.wav_name, pred_states, time_step=time_step, name=predictor_input_name[1]
            )
            predictor_out = predictor_inference.process(predictor_input)

            # update u_step_hyps
            for i in range(self.beam_size):
                hyp_idx = hyp_orders[i]
                hyp = self.running_hyps[hyp_idx]
                new_hyp = hyp.copy()
                new_token = self.pred_label[i]
                new_score = u_hyps_scores[0][i]
                new_lm_score = hyp.nnlm_score
                new_ilm_score = hyp.ilm_score
                if self.use_domain_lm_nnlm:
                    nnlm_states = self.gather_hyps_attributes(
                        self.running_hyps, 'nnlm_state', hyp_orders
                    )
                    nnlm_states = nnlm_states.squeeze(axis=1)
                    new_lm_score += float(hyp['nnlm_lprobs'][0, 0, new_token])
                    new_score -= self.domain_nnlm_scale * new_lm_score
                    if self.ilme:
                        new_ilm_score += ilm_lprobs[hyp_idx][new_token]
                        new_score += self.internal_nnlm_scale * new_ilm_score

                # Milme lm
                domain_lm_score = 0.0
                ngram_state = 0
                if self.use_domain_lm_fst:
                    cur_internal_score = jointer_out_ilm_lprobs[hyp_idx, new_token]
                    ngram_state, domain_lm_score = self.lm_solution.step_ngram(
                        new_token + 1, hyp.ngram_state
                    )
                    if self.ngram_fst_minus_internal_lm:
                        domain_lm_score = -cur_internal_score + max(
                            domain_lm_score, cur_internal_score
                        )
                    new_score += domain_lm_score * self.domain_lm_scale

                # hotwords
                new_hyp.score = new_score
                new_hyp.label_seq.append(new_token)
                new_hyp.nnlm_score = new_lm_score
                new_hyp.ilm_score = new_ilm_score
                new_hyp.ngram_state = ngram_state
                new_hyp.pred_feat = predictor_out[predictor_output_name[0]][i][0]
                new_hyp.pred_state = predictor_out[predictor_output_name[1]][i]
                new_hyp.timestamp.append(time_step)
                new_hyp.confidence.append(np.exp(jointer_out['output'][hyp_idx, new_token]))

                is_last = time_step == self.total_time - 1
                if self.use_hotword_fst:
                    best_token = self.lm_solution.step_hotword(new_token + 1, new_hyp, is_last)
                    if not self.use_fst_skip:
                        if best_token is not None:
                            self.fusion_hotword_fst(best_token, new_hyp, is_last)
                    else:  # (TODO) refact fst_skip
                        pass
                u_step_hyps.append(new_hyp)

            # nnlm forward one step after u_step_expanding
            if self.use_domain_lm_nnlm:
                nnlm_inference = Factory.get_processor('nnlm', self.config)
                nnlm_input = nnlm_inference.get_input_all()
                nnlm_out = nnlm_inference.process(
                    Message(
                        {
                            nnlm_input[0].name: np.array(self.pred_label).astype(np.int64),
                            nnlm_input[1].name: nnlm_states,
                        }
                    )
                )
                nnlm_out_names = nnlm_inference.get_output()
                for i in range(self.beam_size):
                    u_step_hyps[i].nnlm_lprobs = nnlm_out[nnlm_out_names[0]][i].reshape(
                        1, 1, nnlm_out[nnlm_out_names[0]].shape[-1]
                    )
                    u_step_hyps[i].nnlm_state = nnlm_out[nnlm_out_names[1]][i].reshape(
                        1, nnlm_input[1].shape[-1]
                    )

            serializer.add_data(
                self.wav_name, predictor_out.output, time_step=time_step, name="pred_feat"
            )
            serializer.add_data(
                self.wav_name,
                predictor_out[predictor_output_name[-1]],
                time_step=time_step,
                name="pred_hidden",
            )

        if self.use_class_lm_fst:
            self.fusion_class_lm_fst(u_step_hyps)

        self.running_hyps = self.recombine_hyps(t_step_hyps, u_step_hyps)
        if self.len_penalty_scale > 0.0:
            self.running_hyps = self.add_length_penalty(self.running_hyps)
        if self.prefetch and time_step < self.total_time - 1:
            hotword_backoff = self.hotwords_score_backoff(self.running_hyps)
            self.apply_prefetch(encoder_out, time_step)
            if hotword_backoff:
                for hyp in self.running_hyps:
                    hyp.score += hotword_backoff.get(hyp.label_str, 0)
        if time_step == self.total_time - 1:
            self.hotwords_score_backoff(self.running_hyps)
            if self.use_class_lm_fst:
                self.class_lm_check_last_frame(self.running_hyps)

        self.top_k_or_pad_k()

        if self.fixed_prefix and time_step % self.fixed_prefix_time_freq == 0:
            self.running_hyps, fixed_prefix_label_seq = self.get_fixed_prefix(
                self.running_hyps, keep_beam_size=True
            )
            self.fixed_prefix_lists.append((time_step, fixed_prefix_label_seq))

    def apply_prefetch(self, encoder_out, time_step):
        '''apply prefetch'''
        prev_seq = []
        if len(self.prefetch_lists) > 0:
            prev_seq = self.prefetch_lists[-1][1]
        # alreay has prefetched
        if self.running_hyps[0]["label_seq"] == prev_seq:
            return
        if self.running_hyps[0]["eos_score"] is None:
            try:
                jointer_small_inference = Factory.get_processor('jointer_small', self.config)
            except Exception:
                jointer_small_inference = Factory.get_processor('jointer', self.config)
            jointer_small_input_name = jointer_small_inference.get_input()
            pred_feat = [self.running_hyps[0]["pred_feat"]]
            jointer_input = Message(
                {
                    jointer_small_input_name[0]: encoder_out,
                    jointer_small_input_name[1]: np.array(pred_feat),
                }
            )
            jointer_output = jointer_small_inference.process(jointer_input)
            self.running_hyps[0].eos_score = jointer_output['output'][0][2]
        if self.running_hyps[0].eos_score > self.prefetch_thresh:
            if self.use_las_rescore:
                self.prefetch_first2second_idx[len(self.prefetch_lists)] = -1
            self.prefetch_lists.append((time_step, self.running_hyps[0]["label_seq"]))

    def apply_ce_timestamp(self, ce_logits, label_seq):
        '''apply ce timestamp'''
        aligned_list, align_score = force_align(ce_logits[0], label_seq)
        out_list = [
            [tmp[0], tmp[1], tmp[2], np.sum(np.exp(align_score[tmp[1] : tmp[2]]))]
            for tmp in aligned_list
        ]
        return out_list
