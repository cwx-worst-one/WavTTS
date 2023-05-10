''' RNNTTwoPassRunner '''

import itertools
import torch
from core.dataset import ValidHDFSDataset
from core.utils import dist_barrier, logging
from core.utils.cer.cer_metric import (
    EditDistanceCalculator,
    TextFormator,
    output_result,
    output_nbest_result,
    merge_result,
    merge_nbest_result,
)
from core.utils.metric_output import (
    StreamingStableMetricOneSample,
    ASRModelRichOutWriter,
    merge_stable_metric,
    merge_asr_rich_info,
)
from core.utils.misc import infer_text_format
from ..base_runner import RUNNERS
from .rnnt_runner import RNNTRunner
from ..utils import (
    get_oracle_ed_info,
    get_time,
)


@RUNNERS.register_module()
class RNNTLASRescoreRunner(RNNTRunner):
    '''RNNT LAS Rescore for ASR'''

    # pylint: disable=line-too-long

    @get_time("time")
    def train_iteration(self):
        '''
        train iteration
        1. use prepare_rnnt_encoder_out before train
        2. use las_forward in place of default forward
        '''
        i = 0
        while i < self.grad_accum_step:
            batch_data = self.next_train_batch()
            try:
                # precompute encoder_out
                with torch.no_grad():
                    self.solution.eval()
                    self.solution.prepare_rnnt_encoder_out(batch_data)

                # las forward
                self.solution.train()
                self.solution_out = self.solution.las_forward(batch_data)
                self.loss = self.solution_out['backward_loss'] / self.grad_accum_step
                self.dist_handler.backward(self.loss, unscale=(i + 1 == self.grad_accum_step))
            except RuntimeError as e:
                # destroy auto grad graph through delete loss,
                # so the gpu memory could be free.
                self.handle_error(e, batch_data)
                continue
            self.train_log_buffer.update(self.solution_out)
            i = i + 1
        # clip the grad
        if self.opt_util_cfg.grad_clip:
            gnorm = self.clip_grads()
            self.train_log_buffer.update({'gnorm': gnorm})
        # optimizer step
        self.dist_handler.step(iters=self.iter)

    def train(self):
        '''train loop'''
        # pylint: disable=not-callable
        self.build_beam_search('solution')
        self.before_train()
        self.mode = 'train'
        self.call_hook('before_epoch')
        while self.iter < self.args.train.max_iters:
            # train step begin
            self.call_hook('before_train_iter')
            self.train_iteration()
            self.call_hook('after_train_iter')
            self.log_metric(self.train_log_buffer)
            # validation
            if (self.iter + 1) % self.args.valid.interval == 0:
                self.validation()
            self._iter += 1
            self._inner_iter += 1
        self.after_train()

    @torch.no_grad()
    def validation(self):
        # pylint: disable=not-callable
        # Switch to eval model

        self.mode = 'val'
        self.solution.eval()
        self.solution.criterion_module.combine_weight()
        self.call_hook('before_val_epoch')
        self.valid_log_buffer.reset()
        self.valid_data_loader.reset()
        batch_data = self.valid_data_loader.next()
        self._val_iter = 0

        while batch_data is not None:
            self.call_hook('before_val_iter')
            try:
                self.solution.prepare_rnnt_encoder_out(batch_data)
                validation_out = self.solution.las_forward(batch_data)
                self.valid_log_buffer.update(validation_out)
                self._val_iter += 1
                batch_data = self.valid_data_loader.next()
                self.call_hook('after_val_iter')

            except RuntimeError as e:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                continue

        self.call_hook('after_val_epoch')
        self.log_metric(self.valid_log_buffer)
        self.solution.criterion_module.clear_combined_weight()
        # switch back to train mode
        self.mode = 'train'
        self.solution.train()

    def init_test_data_loader(self, test_file, inference_cfg):
        '''create test set data loader'''
        # pylint: disable=not-callable
        if hasattr(inference_cfg, 'bucket_schedule'):
            inf_bucket_schedule = inference_cfg.bucket_schedule
        else:
            inf_bucket_schedule = self.dataset_cfg.get('bucket_schedule', '') + ',100000'
        test_data_loader = ValidHDFSDataset(
            [test_file],
            inf_bucket_schedule,
            self.dataset_cfg,
            self.parse_fn_eval,
            self.draw_batch_fn_inference,
            split_path_list_by_rank=False,
        )
        test_data_loader.reset()
        return test_data_loader

    @torch.no_grad()
    def inference_once_rescore(self, test_name, language, test_data_loader, inference_cfg):
        '''rescore mode inference for twopass model'''
        # pylint:disable=too-many-branches,too-many-locals,too-many-statements
        batch_data = test_data_loader.next()
        # inference config
        nbest = inference_cfg.get('nbest', 10)
        filter_list = inference_cfg.get('filter_list', [])
        keep_non_proun_tokens = inference_cfg.get('keep_non_proun_tokens', None)
        twopass_infer_cfg = inference_cfg.get('twopass_infer_cfg', None)
        rnnt_score_scale = inference_cfg.get('rnnt_score_scale', 1.0)
        prefetch = inference_cfg.get('enable_prefetch', False)
        las_forward_score_scale, las_backward_score_scale, las_fst_score_scale = None, None, 0
        if twopass_infer_cfg is not None:
            las_fst_score_scale = twopass_infer_cfg.get('las_fst_score_scale', 0.0)
            las_forward_score_scale = twopass_infer_cfg.get('las_forward_score_scale', None)
            las_backward_score_scale = twopass_infer_cfg.get('las_backward_score_scale', None)
        output_rnnt_confidence = inference_cfg.get('output_rnnt_confidence', False)
        output_streaming_stable_metric = inference_cfg.get('output_streaming_stable_metric', False)
        rich_info_writer = ASRModelRichOutWriter(self.args)
        if output_streaming_stable_metric:
            ms_per_packet = inference_cfg.get('ms_per_packet', 200)
            ms_per_frame = self.solution_cfg.downsampling_size * 10
            stable_metric_testset_list = []
        ed_calculator = EditDistanceCalculator()
        formator = TextFormator(language, keep_non_proun_tokens)
        # '@@ ' for a continuous word
        # '@@' for last word
        default_filter_list = ['^', '@@ ', '<s>', '</s>', '<pad>', '<unk>', '@@']
        filter_list = default_filter_list + filter_list

        if self.reorder_tgt_dict:
            tgt_dict = self.reorder_tgt_dict
        else:
            tgt_dict = self.tgt_dict

        # process every batch
        decode_count = 0
        ed_info_list = []
        align_info_list = []
        score_info_list = []
        inf_nbest_list = []
        rnnt_score_list = []
        fst_score_list = []
        las_fw_score_list = []
        las_bw_score_list = []
        uttid_list = []
        ref_list = []
        while batch_data is not None:
            stable_metric_batch_list = None
            if output_streaming_stable_metric:
                bsz = batch_data['src'].shape[0]
                stable_metric_batch_list = [
                    StreamingStableMetricOneSample(
                        ms_per_packet, ms_per_frame, tgt_dict, filter_list, formator
                    )
                    for _ in range(bsz)
                ]
            (
                inf_nbest,
                rnnt_score,
                fst_score,
                las_fw_score,
                las_bw_score,
            ) = self.solution.beam_inference(
                batch_data,
                prefetch=prefetch,
                nbest=nbest,
                twopass_infer_cfg=twopass_infer_cfg,
                stable_metric_list=stable_metric_batch_list,
            )
            inf_nbest_list.extend(inf_nbest)
            rnnt_score_list.extend(rnnt_score)
            fst_score_list.extend(fst_score)
            las_fw_score_list.extend(las_fw_score)
            las_bw_score_list.extend(las_bw_score)
            uttid_list.extend(batch_data['uttid'])
            ref_list.extend(batch_data['ref'])
            # compute cer for each hyp
            for bid, hyps in enumerate(inf_nbest):
                # every sentence
                decode_count += 1
                if decode_count % 50 == 0:
                    logging.info('rank %d decode %d sentence', self.rank, decode_count)
                ref_format = infer_text_format(batch_data['ref'][bid], filter_list, formator)
                for nid, hyp in enumerate(hyps):
                    hyp_str = tgt_dict.string(hyp)
                    res_format = infer_text_format(hyp_str, filter_list, formator)
                    ed_info, align_info = ed_calculator.show_alignment(
                        batch_data['uttid'][bid], ref_format, res_format
                    )
                    ed_info_list.append(ed_info)
                    align_info_list.append(align_info)
                    score_info_list.append(
                        (
                            rnnt_score[bid][nid],
                            fst_score[bid][nid],
                            las_fw_score[bid][nid],
                            las_bw_score[bid][nid],
                        )
                    )
                    logging.info(
                        'rank %d, uttid %s, %s',
                        self.rank,
                        batch_data['uttid'][bid],
                        ' '.join(res_format),
                    )
            if output_streaming_stable_metric:
                stable_metric_testset_list += stable_metric_batch_list
            batch_data = test_data_loader.next()
        test_data_loader.terminate()
        # output nbest
        out_stat_file = 'rank{}_nbest_result_{}.txt'.format(self.rank, test_name)
        output_nbest_result(align_info_list, ed_info_list, score_info_list, out_stat_file)

        # wait all rank finish
        dist_barrier()
        # merge nbest
        (
            ed_info_all,
            rnnt_score_all,
            las_fst_score_all,
            las_fw_score_all,
            las_bw_score_all,
        ) = merge_nbest_result(self.world_size, test_name)
        dist_barrier()

        # grid search best CER
        w_min, w_max, stride = eval(twopass_infer_cfg.get('grid_search_params', '[0.0, 2.0, 0.1]'))
        w_range = [w_min + i * stride for i in range(int((w_max - w_min) / stride))]
        fw_range = [0]
        bw_range = [0]
        if self.args.solution.twopass_decoder_args.get('las_forward_decoder', True):
            fw_range = w_range
        else:
            las_forward_score_scale = 0.0
        if self.args.solution.twopass_decoder_args.get('las_backward_decoder', False):
            bw_range = w_range
        else:
            las_backward_score_scale = 0.0
        if las_forward_score_scale is None or las_backward_score_scale is None:
            best_cer = 100.0
            for fw_scale, bw_scale in itertools.product(fw_range, bw_range):
                total_word = 0.0
                total_error = 0.0
                for ed_info_nbest, r_scores, las_fst_scores, f_scores, b_scores in zip(
                    ed_info_all,
                    rnnt_score_all,
                    las_fst_score_all,
                    las_fw_score_all,
                    las_bw_score_all,
                ):
                    best_score = float('-inf')
                    best_nid = 0
                    for nid, (r_score, las_fst_score, f_score, b_score) in enumerate(
                        zip(r_scores, las_fst_scores, f_scores, b_scores)
                    ):
                        r_score = float(r_score.split('|', 1)[0])
                        cur_score = (
                            rnnt_score_scale * r_score
                            - las_fst_score_scale * float(las_fst_score)
                            + fw_scale * f_score
                            + bw_scale * b_score
                        )
                        if cur_score > best_score:
                            best_score = cur_score
                            best_nid = nid
                    total_word += ed_info_nbest[best_nid][0]
                    total_error += ed_info_nbest[best_nid][1]
                cur_cer = total_error / total_word
                logging.info(
                    'rnnt_scale %f, las_fst_score_scale %f,las_fw_scale %f, las_bw_scale %f, CER %.2f%%',
                    rnnt_score_scale,
                    las_fst_score_scale,
                    fw_scale,
                    bw_scale,
                    cur_cer * 100,
                )
                if cur_cer < best_cer:
                    best_cer = cur_cer
                    las_forward_score_scale = fw_scale
                    las_backward_score_scale = bw_scale

        # output best CER
        ed_info_list = []
        align_info_list = []
        for bid, (hyps, r_scores, las_fst_scores, f_scores, b_scores) in enumerate(
            zip(
                inf_nbest_list,
                rnnt_score_list,
                fst_score_list,
                las_fw_score_list,
                las_bw_score_list,
            )
        ):
            ref_format = infer_text_format(ref_list[bid], filter_list, formator)
            rich_info_writer.update_one_sample_info(uttid_list[bid], 'ref_format', ref_format)
            best_score = float('-inf')
            best_nid = 0
            best_rnnt_confidence = None
            for nid, (r_score, las_fst_score, f_score, b_score) in enumerate(
                zip(r_scores, las_fst_scores, f_scores, b_scores)
            ):
                if len(hyps[nid]) == 0:
                    rnnt_confidence = []
                else:
                    rnnt_confidence = [float(p) for p in r_score.split('|')[1:]]
                r_score = float(r_score.split('|')[0])
                cur_score = (
                    rnnt_score_scale * r_score
                    - las_fst_score_scale * float(las_fst_score)
                    + las_forward_score_scale * f_score
                    + las_backward_score_scale * b_score
                )
                if cur_score > best_score:
                    best_score = cur_score
                    best_nid = nid
                    best_rnnt_confidence = rnnt_confidence
            hyp_str = tgt_dict.string(hyps[best_nid])
            rich_info_writer.update_one_sample_info(uttid_list[bid], 'infer_label', hyps[best_nid])
            if output_streaming_stable_metric:
                stable_metric_testset_list[bid].update(hyps[best_nid], las_rescore=True)
                logging.info(
                    "uttid %s upwr %.3f upsr %.3f"
                    % (
                        uttid_list[bid],
                        stable_metric_testset_list[bid].upwr,
                        stable_metric_testset_list[bid].upsr,
                    )
                )
                rich_info_writer.update_one_sample_info(
                    uttid_list[bid], 'upwr', stable_metric_testset_list[bid].upwr
                )
                rich_info_writer.update_one_sample_info(
                    uttid_list[bid], 'upsr', stable_metric_testset_list[bid].upsr
                )
            res_format = infer_text_format(hyp_str, filter_list, formator)
            rich_info_writer.update_one_sample_info(uttid_list[bid], 'infer_format', res_format)
            ed_info, align_info = ed_calculator.show_alignment(
                uttid_list[bid], ref_format, res_format
            )
            ed_info_list.append(ed_info)
            align_info_list.append(align_info)
            if output_rnnt_confidence:
                _, rnnt_prob = self.rnnt_confidence_post_process(
                    tgt_dict, hyps[best_nid], best_rnnt_confidence, filter_list
                )
                rich_info_writer.update_one_sample_info(uttid_list[bid], 'confidence', rnnt_prob)
        if output_streaming_stable_metric:
            upwr, upsr = merge_stable_metric(stable_metric_testset_list)
            logging.info(
                "rank%d testset %s upwr %.3f upsr %.3f" % (self.rank, test_name, upwr, upsr)
            )
            rich_info_writer.update_general_info('upwr', upwr)
            rich_info_writer.update_general_info('upsr', upsr)
        # output to file
        out_stat_file = 'rank{}_cer_result_{}.txt'.format(self.rank, test_name)
        output_result(align_info_list, ed_info_list, out_stat_file, lang=language)
        if rich_info_writer.if_necessary_to_write():
            rich_info_writer.write_to("rank{}_rich_info_{}.json".format(self.rank, test_name))
        # wait all rank finish
        dist_barrier()
        # merge the CER
        if self.rank == 0:
            merge_result(self.world_size, test_name)
            if rich_info_writer.if_necessary_to_write():
                merge_asr_rich_info(
                    self.world_size,
                    test_name,
                    input_file_tmp="rank{}_rich_info_{}.json",
                    output_file="rich_info_{}.json".format(test_name),
                )
        dist_barrier()

        logging.info(
            'rank %d, testset %s, rnnt_scale %f, las_fst_scale %f, las_fw_scale %f, las_bw_scale %f',
            self.rank,
            test_name,
            rnnt_score_scale,
            las_fst_score_scale,
            las_forward_score_scale,
            las_backward_score_scale,
        )

    @torch.no_grad()
    def inference_once_beam_search(self, test_name, language, test_data_loader, inference_cfg):
        '''search mode for twopass model'''
        # pylint: disable=not-callable,too-many-locals,too-many-branches,too-many-statements
        batch_data = test_data_loader.next()
        # inference config
        twopass_infer_cfg = inference_cfg.get('twopass_infer_cfg', dict())
        rnnt_nbest = twopass_infer_cfg.get('rnnt_beam_size', 10)
        beam_size = twopass_infer_cfg.get('las_beam_size', 10)
        nbest_out = twopass_infer_cfg.get('nbest_out', False)
        keep_non_proun_tokens = inference_cfg.get('keep_non_proun_tokens', None)
        filter_list = inference_cfg.get('filter_list', [])
        ed_calculator = EditDistanceCalculator()
        formator = TextFormator(language, keep_non_proun_tokens)
        # '@@ ' for a continuous word
        # '@@' for last word
        default_filter_list = ['^', '@@ ', '<s>', '</s>', '<pad>', '<unk>', '@@']
        filter_list = default_filter_list + filter_list

        if self.reorder_tgt_dict:
            tgt_dict = self.reorder_tgt_dict
        else:
            tgt_dict = self.tgt_dict

        # process every batch
        decode_count = 0
        ed_info_list = []
        align_info_list = []
        ed_info_list_oracle = []
        align_info_list_oracle = []
        while batch_data is not None:
            if beam_size > 0:
                inf_res = self.solution.beam_inference(
                    batch_data, nbest=rnnt_nbest, twopass_infer_cfg=twopass_infer_cfg
                )
            else:
                ## TODO (houjunfeng) add greedy_inference
                inf_res = self.solution.greedy_inference(batch_data)
            # post process
            if nbest_out:
                inf_res_top1, inf_res_nbest = inf_res
                inf_res = inf_res_top1
                for bid, hyp_nbest in enumerate(inf_res_nbest):
                    ref_format = infer_text_format(batch_data['ref'][bid], filter_list, formator)
                    ed_info_oracle, align_info_oracle = get_oracle_ed_info(
                        hyp_nbest,
                        batch_data['uttid'][bid],
                        ref_format,
                        tgt_dict,
                        filter_list,
                        formator,
                        ed_calculator,
                        self.rank,
                    )
                    ed_info_list_oracle.append(ed_info_oracle)
                    align_info_list_oracle.append(align_info_oracle)

            for bid, hyp in enumerate(inf_res):
                # every sentence
                decode_count += 1
                if decode_count % 50 == 0:
                    logging.info('rank %d decode %d sentence', self.rank, decode_count)
                hyp_str = tgt_dict.string(hyp)
                res_format = infer_text_format(hyp_str, filter_list, formator)
                ref_format = infer_text_format(batch_data['ref'][bid], filter_list, formator)

                logging.info(
                    'rank %d, uttid %s, RES: %s',
                    self.rank,
                    batch_data['uttid'][bid],
                    ' '.join(res_format),
                )

                ed_info, align_info = ed_calculator.show_alignment(
                    batch_data['uttid'][bid], ref_format, res_format
                )
                ed_info_list.append(ed_info)
                align_info_list.append(align_info)

            batch_data = test_data_loader.next()
        test_data_loader.terminate()
        # output to file
        out_stat_file = 'rank{}_cer_result_{}.txt'.format(self.rank, test_name)
        output_result(align_info_list, ed_info_list, out_stat_file, lang=language)

        if nbest_out:
            oracle_test_name = '{}_oracle'.format(test_name)
            out_stat_file = 'rank{}_cer_result_{}.txt'.format(self.rank, oracle_test_name)
            output_result(align_info_list_oracle, ed_info_list_oracle, out_stat_file, lang=language)

        # wait all rank finish
        dist_barrier()
        # merge the CER
        if self.rank == 0:
            merge_result(self.world_size, test_name, falcon_report=self.falcon_report)

            if nbest_out:
                merge_result(self.world_size, oracle_test_name, falcon_report=self.falcon_report)

        dist_barrier()

    @torch.no_grad()
    def inference_once(self, test_file, test_name, language, inference_cfg):
        '''inference a test set'''
        test_data_loader = self.init_test_data_loader(test_file, inference_cfg)
        if inference_cfg.twopass_infer_cfg.get('enable_twopass_rescore', True):
            self.inference_once_rescore(test_name, language, test_data_loader, inference_cfg)
        else:
            self.inference_once_beam_search(test_name, language, test_data_loader, inference_cfg)

    def build_beam_search(self, key='inference'):
        '''build beam search'''
        cfg = self.args.get(key, None)
        if key == 'inference':
            cfg.setdefault('beam_size', 10)
            cfg.nbest = cfg.twopass_infer_cfg.get('nbest', 10)
        elif key == 'solution':
            cfg.beam_size = cfg.get('mbr_beam_size', 10)
            cfg.nbest = cfg.beam_size
            cfg.setdefault('lm_weight', 0.1)
            cfg.setdefault('use_batch_beam', True)
            cfg.nbest_align_info = False
        super().build_beam_search(key=key)


@RUNNERS.register_module()
class RNNTDeliberationRunner(RNNTLASRescoreRunner):
    '''RNNT LAS Deliberation for ASR'''

    def pre_build_dataset(self, dataset_cfg):
        super().pre_build_dataset(dataset_cfg)
        self.grad_accum_step = self.train_cfg.get('grad_accum_step', 8)
        orig_max_batch_size = dataset_cfg.get('max_batch_size', 1)
        dataset_cfg['max_batch_size'] = self.grad_accum_step * orig_max_batch_size

    def slice_data_bulk(self, batch_data):
        '''slice batch_data into list of batch_data'''
        bsz = min(len(v) for k, v in batch_data.items() if v is not None)
        split_num = min(bsz, self.grad_accum_step)
        if split_num != self.grad_accum_step:
            batch_info = ' '.join(
                [
                    '{}:{}'.format(k, v.shape)
                    for k, v in batch_data.items()
                    if isinstance(v, torch.Tensor)
                ]
            )
            logging.all_rank_info(
                "rank %d iter %d split %d acc %d batch_info %r",
                self.rank,
                self._iter,
                split_num,
                self.grad_accum_step,
                batch_info,
            )
        batch_data_ls = []
        for las_encoder_idx in range(split_num):
            this_batch_data = {}
            for key, item in batch_data.items():
                if item is None:
                    this_batch_data[key] = None
                else:
                    begin_idx = las_encoder_idx * len(item) // split_num
                    end_idx = (las_encoder_idx + 1) * len(item) // split_num
                    this_batch_data[key] = item[begin_idx:end_idx]
            batch_data_ls.append(this_batch_data)
        return batch_data_ls, split_num

    @get_time("time")
    def train_iteration(self):
        '''
        train iteration
        1. use prepare_rnnt_encoder_out before train
        2. use las_forward in place of default forward
        '''
        i = 0
        batch_data = self.next_train_batch()
        with torch.no_grad():
            self.solution.eval()
            self.solution.prepare_rnnt_encoder_out(batch_data)
        self.solution.prepare_rnnt_nbest_data(batch_data)
        batch_data, split_num = self.slice_data_bulk(batch_data)
        while i < split_num:
            try:
                # las forward
                self.solution.train()
                self.solution_out = self.solution.las_forward(batch_data[i])
                self.loss = self.solution_out['backward_loss'] / split_num
                self.dist_handler.backward(self.loss, unscale=(i + 1 == split_num))
            except RuntimeError as e:
                # destroy auto grad graph through delete loss,
                # so the gpu memory could be freed.
                self.handle_error(e, batch_data[i])
                continue
            self.train_log_buffer.update(self.solution_out)
            i = i + 1
        # clip the grad
        if self.opt_util_cfg.grad_clip:
            gnorm = self.clip_grads()
            self.train_log_buffer.update({'gnorm': gnorm})
        # optimizer step
        self.dist_handler.step(iters=self.iter)

    def build_beam_search(self, key='inference'):
        '''build beam search'''
        cfg = self.args.get(key, None)
        if key == 'inference':
            cfg.beam_size = cfg.twopass_infer_cfg.get('rnnt_beam_size', 10)
            cfg.nbest = cfg.beam_size
        elif key == 'solution':
            beam_schedule = cfg.get('beam_schedule', dict())
            cfg.beam_size = beam_schedule.get('rnnt_beam_size', 10)
            cfg.nbest = cfg.beam_size
            cfg.nbest_input_size = beam_schedule.get('nbest_input_size', 4)
            cfg.setdefault('lm_weight', 0.1)
            cfg.setdefault('use_batch_beam', True)
            cfg.nbest_align_info = False
        super().build_beam_search(key=key)
