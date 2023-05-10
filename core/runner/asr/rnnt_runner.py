''' RNNTRunner '''
import time
import glob
import warnings
import torch
from core.dataset import (
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
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
    CalculatorCombiner,
    output_result,
    merge_result,
    output_json_result,
    merge_json_result,
)
from core.extensions import ext_ctc_force_alignment
from core.runner.metric.asr_metric import UniversalAsrMetric, AsrCaseMetric
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
from .base_rnnt_runner import BaseRNNTRunner
from ..utils import (
    get_word_boundary,
    get_oracle_ed_info,
)


@RUNNERS.register_module()
class RNNTRunner(BaseRNNTRunner):
    '''RNNT for ASR'''

    @torch.no_grad()
    def inference_once(self, test_file, test_name, language, inference_cfg):
        '''inference a test set'''
        # pylint:disable=too-many-branches,too-many-locals,too-many-statements
        # data set
        if hasattr(inference_cfg, 'bucket_schedule'):
            inf_bucket_schedule = inference_cfg.bucket_schedule
        else:
            inf_bucket_schedule = self.dataset_cfg.get('bucket_schedule', '') + ',100000'

        # item transform
        parse_eval_cfg = self.dataset_cfg.get('valid_item_transform', [])
        parse_eval_cfg = self.dataset_cfg.get('inference_item_transform', parse_eval_cfg)
        for item_trans in parse_eval_cfg:
            if item_trans['type'] == 'DomainAdd':
                item_trans['domain_list'] = [inference_cfg.get('domain', 0)]
        # batch transform
        draw_batch_cfg = self.dataset_cfg.get("batch_transform", [])
        draw_valid_batch_cfg = self.dataset_cfg.get("valid_batch_transform", draw_batch_cfg)
        draw_infer_batch_cfg = self.dataset_cfg.get(
            "inference_batch_transform", draw_valid_batch_cfg
        )
        # device transform
        device_trans_cfg = self.dataset_cfg.get('device_transform', [])
        valid_device_trans_cfg = self.dataset_cfg.get('valid_device_transform', device_trans_cfg)
        infer_device_trans_cfg = self.dataset_cfg.get(
            'inference_device_transform', valid_device_trans_cfg
        )

        parse_fn_infer = build_item_augmentation(parse_eval_cfg, self.meta_data)
        draw_batch_fn_inference = build_draw_batch_fn(draw_infer_batch_cfg, self.meta_data)
        infer_device_trans = build_device_augmentation(infer_device_trans_cfg, self.meta_data)

        test_data_loader = ValidHDFSDataset(
            [test_file],
            inf_bucket_schedule,
            self.dataset_cfg,
            parse_fn_infer,
            draw_batch_fn_inference,
            device_transforms=infer_device_trans,
            split_path_list_by_rank=False,
        )
        test_data_loader.reset()
        batch_data = test_data_loader.next()
        # inference config
        beam_size = inference_cfg.get('beam_size', 0)
        nbest = inference_cfg.get('nbest', 1)
        filter_list = inference_cfg.get('filter_list', [])
        decode_char = inference_cfg.get('decode_char', False)
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
        enable_endpoint = inference_cfg.get('enable_endpoint', False)
        ep_frames_appended = inference_cfg.get('ep_frames_appended', 0)
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
        # intial text process and cer calculator (asr_eval_tool)
        self.asr_eval_tool_config = inference_cfg.get('asr_eval_tool_cfg', None)
        if self.asr_eval_tool_config:
            cal_combiner = CalculatorCombiner(
                self.asr_eval_tool_config.lang, self.asr_eval_tool_config
            )
        ed_calculator = EditDistanceCalculator()
        if inference_cfg.get('format', True):
            formator = TextFormator(language, keep_non_proun_tokens)
        else:
            formator = None
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
        endpoint_info_list = []
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
                output, logits, frames = self.solution.beam_inference(
                    batch_data,
                    prefetch=prefetch,
                    fixed_prefix=fixed_prefix,
                    nbest=nbest,
                    output_timestamp=output_timestamp,
                    output_rnnt_confidence=output_rnnt_confidence,
                    endpoint=enable_endpoint,
                    stable_metric_list=stable_metric_batch_list,
                )
                inf_res = output["inf_res"]
            else:
                inf_res = self.solution.greedy_inference(batch_data)
                output = {}
            time_elapsed += time.time() - st
            utt_count += len(batch_data['uttid'])
            nbest_out = False
            nbest_res = []
            if nbest > 1 and "nbest" in output:
                nbest_out = True
                nbest_res = output["nbest"]
            if prefetch:
                assert 'prefetch' in output
                prefetch_info = output["prefetch"]
            if fixed_prefix:
                assert 'fixed_prefix' in output
                fixed_prefix_info = output["fixed_prefix"]
            if output_rnnt_confidence:
                assert 'confidence' in output
                rnnt_confidence_info = output["confidence"]
            if enable_endpoint:
                assert 'endpoint' in output
                endpoint_info = output["endpoint"]
            if output_timestamp:
                assert 'timestamp' in output
                timestamp_info = output["timestamp"]
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
                    hyp_timestamp = timestamp_info[bid]
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
                    if decode_char:
                        hyp_str = hyp_str.replace(' ', '').replace('<space>', ' ')
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
                    if self.asr_eval_tool_config:
                        # TODO(shenchen.0622): adaptor other metrics by calculator combiner
                        raw_ref_str = ' '.join(
                            infer_text_format(batch_data['ref'][bid], filter_list, None)
                        )
                        raw_hyp_str = ' '.join(infer_text_format(hyp_str, filter_list, None))
                        cal_combiner.add_calculator(
                            batch_data['uttid'][bid], raw_ref_str, raw_hyp_str
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
                    if decode_char:
                        hyp_str = hyp_str.replace(' ', '').replace('<space>', ' ')
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
                else:  # nbest_out == True
                    ed_info, align_info = get_oracle_ed_info(
                        nbest_res[bid],
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
                if enable_endpoint and endpoint_info is not None:
                    uttid = batch_data['uttid'][bid]
                    gt_eos = batch_data['eos'][bid].item()
                    ep_frame, _ = endpoint_info[bid][0]
                    ori_frame_len = int(batch_data['src_mask'][bid].sum().item())
                    endpoint_info_list.append((uttid, gt_eos, ep_frame, ori_frame_len))
                if output_streaming_stable_metric:
                    logging.info(
                        "uttid %s upwr %.3f upsr %.3f",
                        uttid,
                        stable_metric_batch_list[bid].upwr,
                        stable_metric_batch_list[bid].upsr,
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
        if enable_endpoint:
            self.endpoint_post_process(endpoint_info_list, ep_frames_appended)
        if output_streaming_stable_metric:
            upwr, upsr = merge_stable_metric(stable_metric_testset_list)
            logging.info("rank%d testset %s upwr %.3f upsr %.3f", self.rank, test_name, upwr, upsr)
            rich_info_writer.update_general_info('upwr', round(upwr, 3))
            rich_info_writer.update_general_info('upsr', round(upsr, 3))
        # output to file
        out_stat_file = 'rank{}_cer_result_{}.txt'.format(self.rank, test_name)
        output_result(align_info_list, ed_info_list, out_stat_file, lang='zh')

        if self.asr_eval_tool_config:
            cal_combiner.calculate()
            json_content = cal_combiner.combiner.to_dict()
            out_stat_file = 'rank{}_json_result_{}.json'.format(self.rank, test_name)
            output_json_result(json_content, out_stat_file)

        if output_latency_metric:
            latency_info = self.latency_post_process(
                res_timestamps, ref_timestamps, align_info_list
            )
            rich_info_writer.update_general_info(
                'p50_first_token_latency_start', round(latency_info[0], 2)
            )
            rich_info_writer.update_general_info(
                'p90_first_token_latency_start', round(latency_info[1], 2)
            )
            rich_info_writer.update_general_info(
                'p50_first_token_latency', round(latency_info[2], 2)
            )
            rich_info_writer.update_general_info(
                'p90_first_token_latency', round(latency_info[3], 2)
            )
            rich_info_writer.update_general_info(
                'p50_last_token_latency', round(latency_info[4], 2)
            )
            rich_info_writer.update_general_info(
                'p90_last_token_latency', round(latency_info[5], 2)
            )
            rich_info_writer.update_general_info('p50_token_latency', round(latency_info[6], 2))
            rich_info_writer.update_general_info('p90_token_latency', round(latency_info[7], 2))
            logging.info("rank %d p50 first token latency start: %.2f", self.rank, latency_info[0])
            logging.info("rank %d p90 first token latency start: %.2f", self.rank, latency_info[1])
            logging.info("rank %d p50 first token latency: %.2f", self.rank, latency_info[2])
            logging.info("rank %d p90 first token latency: %.2f", self.rank, latency_info[3])
            logging.info("rank %d p50 last token latency: %.2f", self.rank, latency_info[4])
            logging.info("rank %d p90 last token latency: %.2f", self.rank, latency_info[5])
            logging.info("rank %d p50 token latency: %.2f", self.rank, latency_info[6])
            logging.info("rank %d p90 token latency: %.2f", self.rank, latency_info[7])
        if rich_info_writer.if_necessary_to_write():
            rich_info_writer.write_to("rank{}_rich_info_{}.json".format(self.rank, test_name))
        # wait all rank finish
        dist_barrier()
        self.rich_info_writed[test_name] = rich_info_writer.if_necessary_to_write()
        logging.info(
            'rank %d, testset %s, total num of utt: %d, total infer forward time: %.3f s',
            self.rank,
            test_file,
            utt_count,
            time_elapsed,
        )

    def merge_inference_results(self, test_names):
        '''merge inference results'''
        if self.rank == 0:
            # merge the CER (and rich info json)
            for test_name in test_names:
                merge_result(self.world_size, test_name, falcon_report=self.falcon_report)
                if self.rich_info_writed.get(test_name, False):
                    merge_asr_rich_info(
                        self.world_size,
                        test_name,
                        input_file_tmp="rank{}_rich_info_{}.json",
                        output_file="rich_info_{}.json".format(test_name),
                    )
                if self.asr_eval_tool_config:
                    merge_json_result(
                        self.world_size,
                        test_name,
                        in_file_name="rank{}_json_result_{}.json",
                        out_file_name="json_result_{}.json".format(test_name),
                    )
        dist_barrier()

    def save_inference_stat(self, inference_cfg):  # pylint: disable=no-self-use
        '''save inference stat'''
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

    @torch.no_grad()
    def inference(self):
        '''inference'''
        self.build_beam_search()
        self.rich_info_writed = dict()
        self.solution.criterion_module.clear_combined_weight()
        super().inference()

    def endpoint_post_process(self, endpoint_info_list, ep_frames_appended):
        '''post process for endpointer'''
        ep_latency_lst = []
        ep_too_early_cnt = 0
        ep_converage_cnt = 0
        ep_too_early_debug = []
        for info in endpoint_info_list:
            # uttid,
            # gt_eos, model_eos(both after downsampling 40ms),
            # ori_frame_len(10ms),
            uttid = info[0]
            gt_eos = info[1]
            model_eos = info[2]
            ori_frame_len = info[3] // 4
            if gt_eos == 0:
                gt_eos = ori_frame_len - ep_frames_appended // 4
                logging.info(
                    "EPDETAIL %s endpoint fake gt_eos, ori_frame_len %d gt_eos %d"
                    % (uttid, ori_frame_len, gt_eos)
                )
            if model_eos != -1:
                ep_latency_lst.append(model_eos - gt_eos)
                logging.info(
                    "EPDETAIL %s endpoint latency detail, model_eos %d gt_eos %d latency %d"
                    % (uttid, model_eos, gt_eos, model_eos - gt_eos)
                )
                ep_converage_cnt += 1
                if model_eos < gt_eos:
                    ep_too_early_cnt += 1
                    logging.info(
                        "EPDETAIL %s endpoint too early, model_eos %d gt_eos %d"
                        % (uttid, model_eos, gt_eos)
                    )
                    ep_too_early_debug.append(model_eos - gt_eos)

        if len(ep_latency_lst) > 0:
            ep_latency_lst.sort()
            avg_ep_latency = sum(ep_latency_lst) / len(ep_latency_lst) * 40
            p50_ep_latency = ep_latency_lst[len(ep_latency_lst) // 2] * 40
            p90_ep_latency = ep_latency_lst[int(len(ep_latency_lst) * 0.9)] * 40
            ep_too_early_ratio = 100.0 * ep_too_early_cnt / len(ep_latency_lst)
            ep_converage_ratio = 100.0 * ep_converage_cnt / len(endpoint_info_list)
            logging.info(
                "rank %d endpoint statis: avg latency %.1f ms, "
                "p50 latency %.1f ms, p90 latency %.1f ms, "
                "too early ratio %.2f%%, EOU %.2f%%.",
                self.rank,
                avg_ep_latency,
                p50_ep_latency,
                p90_ep_latency,
                ep_too_early_ratio,
                ep_converage_ratio,
            )
            if len(ep_too_early_debug) > 0:
                logging.info(
                    "EPDETAIL endpoint too early debug, cnt %d avg %.2f ms"
                    % (ep_too_early_cnt, 40 * sum(ep_too_early_debug) / ep_too_early_cnt)
                )

    @staticmethod
    def timestamp_post_process(tgt_dict, hyp, hyp_time, filter_list):
        '''post process for timestamp'''
        hyp_token = []
        hyp_time_filter = []
        word = ''
        for idx, (token_id, t) in enumerate(zip(hyp, hyp_time)):
            token = tgt_dict[token_id]
            if token not in filter_list:
                word += token
                if token[-1] != '@' or idx == len(hyp) - 1:
                    hyp_token.append(word)
                    hyp_time_filter.append(t)
                    word = ''
        res_str = ' '.join(hyp_token).replace('@@', '')
        return res_str, hyp_time_filter

    @staticmethod
    def rnnt_confidence_post_process(tgt_dict, hyp, hyp_confidence, filter_list):
        '''post process for rnnt confidence'''
        hyp_token = []
        hyp_prob_filter = []
        word = ''
        word_prob = 0.0
        word_bpe_count = 0

        for idx, (token_id, prob) in enumerate(zip(hyp, hyp_confidence)):
            token = tgt_dict[token_id]
            if token not in filter_list:
                word += token
                word_prob += prob
                word_bpe_count += 1
                if token[-1] != '@' or idx == len(hyp) - 1:
                    hyp_token.append(word)
                    hyp_prob_filter.append(word_prob / word_bpe_count)
                    word = ''
                    word_prob = 0.0
                    word_bpe_count = 0
        res_str = ' '.join(hyp_token).replace('@@', '')
        return res_str, hyp_prob_filter

    @staticmethod
    def latency_post_process(res_timestamps, ref_timestamps, align_info_list):
        '''post process for latency'''
        first_token_latency_start_list = []
        first_token_latency_list = []
        last_token_latency_list = []
        avg_token_latency_list = []
        for i, res_timestamp in enumerate(res_timestamps):
            try:
                ref_start_timestamp = [w[0] for w in ref_timestamps[i]]
                ref_end_timestamp = [w[1] for w in ref_timestamps[i]]
                if len(res_timestamp) == 0 or len(ref_end_timestamp) == 0:
                    continue
                cor_index = align_info_list[i]['cor_index']
                for ref_index, res_index, _ in cor_index:
                    cur_latency = res_timestamp[res_index] - ref_end_timestamp[ref_index]
                    avg_token_latency_list.append(cur_latency)
                    if ref_index == 0 and res_index == 0:
                        first_token_latency_start_list.append(
                            res_timestamp[res_index] - ref_start_timestamp[ref_index]
                        )
                        first_token_latency_list.append(cur_latency)
                    if (
                        ref_index == len(ref_end_timestamp) - 1
                        and res_index == len(res_timestamp) - 1
                    ):
                        last_token_latency_list.append(cur_latency)
            except Exception:
                logging.warning("ref_index out of range")
        p50_first_token_latency_start = 0
        p90_first_token_latency_start = 0
        if len(first_token_latency_start_list) > 0:
            first_token_latency_start_list.sort()
            p50_first_token_latency_start = first_token_latency_start_list[
                len(first_token_latency_start_list) // 2
            ]
            p90_first_token_latency_start = first_token_latency_start_list[
                int(len(first_token_latency_start_list) * 0.9)
            ]
        p50_first_token_latency = 0
        p90_first_token_latency = 0
        if len(first_token_latency_list) > 0:
            first_token_latency_list.sort()
            p50_first_token_latency = first_token_latency_list[len(first_token_latency_list) // 2]
            p90_first_token_latency = first_token_latency_list[
                int(len(first_token_latency_list) * 0.9)
            ]
        p50_last_token_latency = 0
        p90_last_token_latency = 0
        if len(last_token_latency_list) > 0:
            last_token_latency_list.sort()
            p50_last_token_latency = last_token_latency_list[len(last_token_latency_list) // 2]
            p90_last_token_latency = last_token_latency_list[
                int(len(last_token_latency_list) * 0.9)
            ]
        p50_token_latency = 0
        p90_token_latency = 0
        if len(avg_token_latency_list) > 0:
            avg_token_latency_list.sort()
            p50_token_latency = avg_token_latency_list[len(avg_token_latency_list) // 2]
            p90_token_latency = avg_token_latency_list[int(len(avg_token_latency_list) * 0.9)]
        return (
            p50_first_token_latency_start,
            p90_first_token_latency_start,
            p50_first_token_latency,
            p90_first_token_latency,
            p50_last_token_latency,
            p90_last_token_latency,
            p50_token_latency,
            p90_token_latency,
        )


@RUNNERS.register_module()
class TelRunner(RNNTRunner):
    '''alias for RNNTRunner'''

    def __init__(self, *args, **kwargs):
        warnings.warn("TelRunner will be deprecated, please use RNNTRunner")
        super().__init__(*args, **kwargs)


@RUNNERS.register_module()
class UniversalRNNTRunner(RNNTRunner):
    '''alias for RNNTRunner'''

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = UniversalAsrMetric()
        self.valid_log_buffer = UniversalAsrMetric()


@RUNNERS.register_module()
class RNNTCASERunner(RNNTRunner):
    '''alias for RNNTRunner'''

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = AsrCaseMetric()
        self.valid_log_buffer = AsrCaseMetric()
