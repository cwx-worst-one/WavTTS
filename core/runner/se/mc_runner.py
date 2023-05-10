''' multichannel RNNTRunner '''
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, unused-import

import os
import time
import glob
import warnings
import torch
import numpy as np
from core.dataset import ValidHDFSDataset, build_draw_batch_fn
from core.utils import (
    dist_barrier,
    hdfs_put,
    hdfs_mkdir,
    hdfs_test,
    compute_confidence,
    logging,
    get_local_rank,
)
from core.utils.cer.cer_metric import (
    EditDistanceCalculator,
    TextFormator,
    output_result,
    merge_result,
)
from core.extensions import ext_ctc_force_alignment
from core.runner.metric.asr_metric import UniversalAsrMetric
from core.utils.metric_output import (
    StreamingStableMetricOneSample,
    ASRModelRichOutWriter,
    merge_stable_metric,
    merge_asr_rich_info,
)
from core.solutions.asr.utils import (
    force_align,
    aligned_id_to_char,
)
from core.utils.misc import infer_text_format
from ..base_runner import RUNNERS
from ..asr.rnnt_runner import RNNTRunner
from ..utils import (
    get_word_boundary,
    get_oracle_ed_info,
)


@RUNNERS.register_module()
class MCRunner(RNNTRunner):
    '''Multi-channel for RNNTRunner'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        '''train func.'''
        self.before_train()
        while self.iter < self.args.train.max_iters:
            # train step begin
            self.call_hook('before_train_iter')
            self.train_iteration()
            self.call_hook('after_train_iter')
            self.log_metric(self.train_log_buffer)
            if hasattr(self.args.valid, 'interval'):
                if (self.iter + 1) % self.args.valid.interval == 0:
                    self.validation()
            elif hasattr(self.args.train, 'iters_per_epoch'):
                if (self.iter + 1) % self.args.train.iters_per_epoch == 0:
                    self.validation()
            if (
                hasattr(self.args.train, 'iters_per_epoch')
                and self._inner_iter + 1 == self.args.train.iters_per_epoch
            ):
                self._epoch += 1
                self.call_hook('after_train_epoch')
                # epoch ending or data iter error
                self.train_data_loader.reset()
                if self._epoch >= self.args.train.max_epochs:
                    break
                self.call_hook('before_train_epoch')
                self.train_log_buffer.reset()
                # Check current learning rate
                max_lr = max(param_group['lr'] for param_group in self.optimizer.param_groups)
                if max_lr < self.args.train.get('final_lr', 1e-7):
                    break
                self._inner_iter = 0
            else:
                self._inner_iter += 1
            self._iter += 1
        self.after_train()

    @torch.no_grad()
    def inference_once(self, test_file, inference_cfg):
        '''inference a test set'''
        # pylint:disable=too-many-branches,too-many-locals,too-many-statements
        # data set
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
        batch_data = test_data_loader.next()
        # inference config
        beam_size = inference_cfg.get('beam_size', 0)
        nbest = inference_cfg.get('nbest', 1)
        filter_list = inference_cfg.get('filter_list', [])
        language = inference_cfg.get('language', 'zh')
        confidence_scale = inference_cfg.get('confidence_scale', 0.5)
        rnnt_score_scale = inference_cfg.get('rnnt_score_scale', 1.0)
        ctc_score_scale = inference_cfg.get('ctc_score_scale', 0.5)
        keep_non_proun_tokens = inference_cfg.get('keep_non_proun_tokens', None)
        output_timestamp = inference_cfg.get('output_timestamp', False)
        output_rnnt_confidence = inference_cfg.get('output_rnnt_confidence', False)
        output_speed = inference_cfg.get('output_speed', False)
        output_wordboundary_type = inference_cfg.get('output_wordboundary_type', None)
        output_streaming_stable_metric = inference_cfg.get('output_streaming_stable_metric', False)
        prefetch = inference_cfg.get('enable_prefetch', False)
        fixed_prefix = inference_cfg.get('fixed_prefix_beam_search', False)
        output_latency_metric = inference_cfg.get('output_latency_metric', False)
        rich_info_writer = ASRModelRichOutWriter(self.args)
        if output_latency_metric:
            output_timestamp = True
        if output_streaming_stable_metric:
            ms_per_packet = inference_cfg.get('ms_per_packet', 200)
            ms_per_frame = self.solution_cfg.downsampling_size * 10
            stable_metric_testset_list = []
        if output_timestamp or output_rnnt_confidence:
            if nbest > 1:
                logging.warning("nbest not support > 1 for output timestamp or confidence")
                nbest = 1
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
        time_elapsed = 0
        utt_count = 0
        prefetch_info = None
        fixed_prefix_info = None
        res_timestamps = []
        ref_timestamps = []
        while batch_data is not None:
            st = time.time()
            if beam_size > 0:
                stable_metric_batch_list = None
                if output_streaming_stable_metric:
                    bsz = batch_data['src'].shape[0]
                    stable_metric_batch_list = [
                        StreamingStableMetricOneSample(
                            ms_per_packet, ms_per_frame, tgt_dict, filter_list, formator
                        )
                        for _ in range(bsz)
                    ]
                inf_res, logits, frames = self.solution.beam_inference(
                    batch_data,
                    prefetch=prefetch,
                    fixed_prefix=fixed_prefix,
                    nbest=nbest,
                    output_timestamp=output_timestamp,
                    output_rnnt_confidence=output_rnnt_confidence,
                    stable_metric_list=stable_metric_batch_list,
                )
            else:
                inf_res = self.solution.greedy_inference(batch_data)
            time_elapsed += time.time() - st
            utt_count += len(batch_data['uttid'])
            nbest_out = False
            if nbest > 1 and isinstance(inf_res[0][0], tuple):
                nbest_out = True
            if prefetch:
                prefetch_info = inf_res[0]
                inf_res = inf_res[1]
            if fixed_prefix:
                fixed_prefix_info = inf_res[0]
                inf_res = inf_res[1]
            if output_rnnt_confidence:
                rnnt_confidence_info = inf_res[0]
                inf_res = inf_res[1]
            # post process
            frame_lens = batch_data["src_mask"].sum(dim=1).int().tolist()
            for bid, hyp in enumerate(inf_res):
                # every sentence
                decode_count += 1
                if decode_count % 50 == 0:
                    logging.info('rank %d decode %d sentence', self.rank, decode_count)
                ref_format = infer_text_format(batch_data['ref'][bid], filter_list, formator)
                uttid = batch_data['uttid'][bid]
                rich_info_writer.update_one_sample_info(uttid, 'ref_format', ref_format)
                if output_timestamp:
                    hyp_timestamp = hyp[1]
                    hyp = hyp[0]  # for below process
                    res_str, time_list = self.timestamp_post_process(
                        tgt_dict, hyp, hyp_timestamp, filter_list
                    )
                    rich_info_writer.update_one_sample_info(uttid, 'timestamp', time_list)
                    if output_wordboundary_type == 'base':
                        word_boundary = get_word_boundary(
                            res_str, time_list, frame_lens[bid], self.solution_cfg.downsampling_size
                        )
                        rich_info_writer.update_one_sample_info(
                            uttid, 'wordboundary', word_boundary
                        )
                    if output_latency_metric:
                        ref_timestamp = []
                        if 'timestamp' in batch_data:
                            ref_timestamp = batch_data['timestamp'][bid]
                        res_timestamp = [
                            (int(item) + 1) * self.solution_cfg.downsampling_size * 0.01
                            for item in time_list
                        ]
                        ref_timestamps.append(ref_timestamp)
                        res_timestamps.append(res_timestamp)
                if output_rnnt_confidence:
                    hyp_confidence = rnnt_confidence_info[bid]
                    _, rnnt_prob = self.rnnt_confidence_post_process(
                        tgt_dict, hyp, hyp_confidence, filter_list
                    )
                    rich_info_writer.update_one_sample_info(uttid, 'confidence', rnnt_prob)
                if not nbest_out:
                    hyp_str = tgt_dict.string(hyp)
                    rich_info_writer.update_one_sample_info(uttid, 'infer_label', hyp)
                    res_format = infer_text_format(hyp_str, filter_list, formator)
                    rich_info_writer.update_one_sample_info(uttid, 'infer_format', res_format)
                    logging.info(
                        'rank %d, uttid %s, %s',
                        self.rank,
                        batch_data['uttid'][bid],
                        ' '.join(res_format),
                    )
                    ed_info, align_info = ed_calculator.show_alignment(
                        batch_data['uttid'][bid], ref_format, res_format
                    )
                    if output_speed:
                        minutes = float(frames[bid]) * self.solution_cfg.downsampling_size / 6000.0
                        speed = len(res_format) / minutes
                        rich_info_writer.update_one_sample_info(uttid, 'speed', speed)
                    if output_wordboundary_type == 'ce':
                        hyp_new = hyp
                        align_out, align_score = force_align(
                            logits[bid].cpu()[0 : frames[bid]], hyp_new
                        )
                        final_align = aligned_id_to_char(
                            tgt_dict, align_out, align_score, filter_list
                        )
                        rich_info_writer.update_one_sample_info(uttid, 'wordboundary', final_align)
                elif inference_cfg.get('use_confidence', False):
                    device = logits.device
                    bsz = len(hyp)
                    hyp_ids = [x[0] if len(x[0]) != 0 else [0] for x in hyp]
                    target_lens = [len(x) for x in hyp_ids]
                    max_char_length = max(target_lens)
                    target = torch.zeros(bsz, max_char_length, dtype=torch.int32, device=device)
                    for b in range(bsz):
                        target[b, : target_lens[b]] = torch.tensor(hyp_ids[b], dtype=torch.int32)
                    ctc_logprobs = logits[bid : bid + 1, : frames[bid]]
                    ctc_logprobs = torch.log_softmax(ctc_logprobs, dim=2).repeat(bsz, 1, 1)
                    input_lengths = torch.full(
                        (bsz,), frames[bid], dtype=torch.int32, device=device
                    )
                    ctc_alignments = ext_ctc_force_alignment(
                        ctc_logprobs.transpose(1, 2),
                        target,
                        input_lengths,
                        target_lens,
                        soft=False,
                        left=0,
                        right=0,
                    )
                    best_hyp, _ = compute_confidence(
                        hyp,
                        ctc_logprobs[0],
                        ctc_alignments.tolist(),
                        alpha=confidence_scale,
                        beta=rnnt_score_scale,
                        gamma=ctc_score_scale,
                    )
                    hyp_str = tgt_dict.string(best_hyp)
                    rich_info_writer.update_one_sample_info(uttid, 'infer_label', best_hyp)
                    res_format = infer_text_format(hyp_str, filter_list, formator)
                    rich_info_writer.update_one_sample_info(uttid, 'infer_format', res_format)
                    logging.info(
                        'rank %d, uttid %s, %s',
                        self.rank,
                        batch_data['uttid'][bid],
                        ' '.join(res_format),
                    )
                    ed_info, align_info = ed_calculator.show_alignment(
                        batch_data['uttid'][bid], ref_format, res_format
                    )
                else:
                    ed_info, align_info = get_oracle_ed_info(
                        hyp,
                        batch_data['uttid'][bid],
                        ref_format,
                        tgt_dict,
                        filter_list,
                        formator,
                        ed_calculator,
                        self.rank,
                    )

                ed_info_list.append(ed_info)
                align_info_list.append(align_info)
                uttid = batch_data['uttid'][bid]
                if prefetch and prefetch_info is not None:
                    prefetch_infolist = []
                    for (frame, prefetch_hyp, _) in prefetch_info[bid]:
                        prefetch_hyp_str = tgt_dict.string(prefetch_hyp)
                        prefetch_infolist.append([frame, prefetch_hyp_str])
                        logging.info("prefetch: %s %d %s" % (uttid, frame, prefetch_hyp_str))
                    rich_info_writer.update_one_sample_info(uttid, 'prefetch', prefetch_infolist)
                if fixed_prefix and fixed_prefix_info is not None:
                    fixed_prefix_infolist = []
                    for (frame, fixed_prefix_hyp) in fixed_prefix_info[bid]:
                        fixed_prefix_hyp_str = tgt_dict.string(fixed_prefix_hyp)
                        fixed_prefix_infolist.append([frame, fixed_prefix_hyp_str])
                        logging.info(
                            "fixed prefix: %s %d %s" % (uttid, frame, fixed_prefix_hyp_str)
                        )
                    rich_info_writer.update_one_sample_info(
                        uttid, 'fixed_prefix', fixed_prefix_infolist
                    )
                if output_streaming_stable_metric:
                    logging.info(
                        "uttid %s upwr %.3f upsr %.3f"
                        % (
                            uttid,
                            stable_metric_batch_list[bid].upwr,
                            stable_metric_batch_list[bid].upsr,
                        )
                    )
                    rich_info_writer.update_one_sample_info(
                        uttid, 'upwr', stable_metric_batch_list[bid].upwr
                    )
                    rich_info_writer.update_one_sample_info(
                        uttid, 'upsr', stable_metric_batch_list[bid].upsr
                    )

            if output_streaming_stable_metric:
                stable_metric_testset_list += stable_metric_batch_list
            # next batch
            batch_data = test_data_loader.next()
        test_data_loader.terminate()
        if output_streaming_stable_metric:
            upwr, upsr = merge_stable_metric(stable_metric_testset_list)
            logging.info(
                "rank%d testset %s upwr %.3f upsr %.3f"
                % (self.rank, test_file.split('/')[-1], upwr, upsr)
            )
            rich_info_writer.update_general_info('upwr', round(upwr, 3))
            rich_info_writer.update_general_info('upsr', round(upsr, 3))
        # output to file
        test_name = test_file.split('/')[-1]
        out_stat_file = 'rank{}_cer_result_{}.txt'.format(self.rank, test_name)
        output_result(align_info_list, ed_info_list, out_stat_file, lang='zh')
        if output_latency_metric:
            latency_info = self.latency_post_process(
                res_timestamps, ref_timestamps, align_info_list
            )
            rich_info_writer.update_general_info('first_token_latency', round(latency_info[0], 2))
            rich_info_writer.update_general_info('last_token_latency', round(latency_info[1], 2))
            rich_info_writer.update_general_info('avg_token_latency', round(latency_info[2], 2))
            rich_info_writer.update_general_info('p50_token_latency', round(latency_info[3], 2))
            rich_info_writer.update_general_info('p90_token_latency', round(latency_info[4], 2))
            logging.info("rank %d first token latency: %.2f", self.rank, latency_info[0])
            logging.info("rank %d last token latency: %.2f", self.rank, latency_info[1])
            logging.info("rank %d avg token latency: %.2f", self.rank, latency_info[2])
            logging.info("rank %d p50 token latency: %.2f", self.rank, latency_info[3])
            logging.info("rank %d p90 token latency: %.2f", self.rank, latency_info[4])
        if rich_info_writer.if_necessary_to_write():
            rich_info_writer.write_to("rank{}_rich_info_{}.json".format(self.rank, test_name))
        # wait all rank finish
        dist_barrier()
        # merge the CER (and rich info json)
        if self.rank == 0:
            merge_result(self.world_size, test_name, falcon_report=self.falcon_report)
            if rich_info_writer.if_necessary_to_write():
                merge_asr_rich_info(
                    self.world_size,
                    test_name,
                    input_file_tmp="rank{}_rich_info_{}.json",
                    output_file="rich_info_{}.json".format(test_name),
                )
        dist_barrier()
        logging.info(
            'rank %d, testset %s, total num of utt: %d, total infer forward time: %.3f s',
            self.rank,
            test_file,
            utt_count,
            time_elapsed,
        )

    @torch.no_grad()
    def inference(self):
        '''inference'''
        self.build_beam_search()
        inference_cfg = self.args.inference
        # resume RNN-T solution
        self.solution.eval()
        self.solution.criterion_module.clear_combined_weight()
        # test data sets
        test_sets = inference_cfg.test_sets.strip().split('|')
        test_file_path = self.get_test_file_list(self.dataset_cfg, test_sets)

        # inference every test set
        inf_draw_batch_cfg = self.dataset_cfg.inference_batch_transform
        self.draw_batch_fn_inference = build_draw_batch_fn(inf_draw_batch_cfg)
        for test_file in test_file_path:
            self.inference_once(test_file, inference_cfg)
        remote_stat_dir = inference_cfg.get('remote_stat_dir', '')
        if remote_stat_dir and get_local_rank() == 0:
            if hdfs_test(remote_stat_dir, '-f') == 0:
                logging.warning(
                    'remote_stat_dir='
                    + remote_stat_dir
                    + ' is a file, changed to '
                    + remote_stat_dir
                    + '_dir'
                )
                remote_stat_dir = remote_stat_dir + '_dir'
            if hdfs_test(remote_stat_dir, '-e') != 0:
                hdfs_mkdir(remote_stat_dir)
            hdfs_put('cer_*.txt', remote_stat_dir, sync=True)
            if inference_cfg.get('enable_prefetch', False):
                hdfs_put('rank*_prefetch_result_*.txt', remote_stat_dir, sync=True)
                hdfs_put('*boundary.*', remote_stat_dir, sync=True)
            if inference_cfg.get('fixed_prefix_beam_search', False):
                hdfs_put('rank*_fixed_prefix_result_*.txt', remote_stat_dir, sync=True)
            if glob.glob('rich_info_*.json'):
                hdfs_put('rich_info_*.json', remote_stat_dir, sync=True)
