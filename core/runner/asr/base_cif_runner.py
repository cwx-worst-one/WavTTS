''' base asr runner. '''
import copy
import io
import os.path as osp
import time
import glob
import torch

from subword_nmt.apply_bpe import BPE

from core.extensions import GradientsAccumulator  # pylint: disable=import-error

from core.dataset import (
    HDFSDataset,
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
from core.runner.metric.asr_metric import (
    AsrCifTrainMetric,
    AsrCifValMetric,
    UniversalAsrCifTrainMetric,
    UniversalAsrCifValMetric,
)
from core.dataset import get_meta
from core.utils import (
    dist_barrier,
    hdfs_put,
    hdfs_mkdir,
    hdfs_test,
    logging,
    get_local_rank,
)
from core.utils.cer.cer_metric import output_result, output_wordboundary
from core.solutions.asr.utils import force_align, aligned_id_to_char
from .base_asr_runner import BaseAsrRunner
from ..utils import get_time
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class BaseCifRunner(BaseAsrRunner):
    '''Base CIF Runner.'''

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        meta_data_root = dataset_cfg.get("meta_data_root", dataset_cfg.get("data_root", None))
        if isinstance(meta_data_root, (list, tuple)):
            meta_data_root = meta_data_root[0]
        meta_file = osp.join(meta_data_root, dataset_cfg.meta_file)
        self.meta_data = get_meta(meta_file)
        self.tgt_dict = self.meta_data['tgt_dict']
        self.id_map = self.meta_data.get('id_map', None)
        is_reorder_dict = self.solution_cfg.get('reorder_dict_by_freq', 1)
        if is_reorder_dict:
            self.reorder_tgt_dict = self.meta_data['reorder_tgt_dict']
            self.reorder_dict_map = self.meta_data['reorder_dict_map']
        else:
            self.reorder_tgt_dict = None
            self.reorder_dict_map = None
        total_code = self.meta_data['total.code']
        self.bpe_fn = BPE(io.StringIO(total_code))
        self.cmvn_mean = self.meta_data['cmvn_mean']
        self.cmvn_var = self.meta_data['cmvn_var']

        self.solution_cfg.setdefault('fbank_dim', dataset_cfg.fbank_dim)
        self.solution_cfg.setdefault('fbank_channel', dataset_cfg.fbank_channel)
        self.solution_cfg.setdefault('use_eos', dataset_cfg.use_eos)
        self.solution_cfg.setdefault('tgt_dict', self.tgt_dict)
        self.solution_cfg.setdefault('id_map', self.id_map)
        # align to 8
        self.tgt_vocab_size = len(self.tgt_dict)
        self.tgt_vocab_size = ((self.tgt_vocab_size + 8 - 1) // 8) * 8
        logging.info('Dictionary total vocab size %d', self.tgt_vocab_size)
        self.solution_cfg.setdefault('tgt_vocab_size', self.tgt_vocab_size)

        self.solution_cfg.setdefault('reorder_tgt_dict', self.reorder_tgt_dict)
        self.in_out_ratio = dataset_cfg.get('in_out_ratio', 8)
        if self.in_out_ratio < self.solution_cfg.get('downsampling_size', 8):
            self.in_out_ratio = self.solution_cfg.get('downsampling_size', 8)
        # set cif_temperature to solution_cfg
        inference_cfg = self.args.get('inference', None)
        if inference_cfg is not None:
            self.solution_cfg.setdefault(
                'cif_temperature', inference_cfg.get('cif_temperature', 1.0)
            )

        self.setup_transform_cfg(dataset_cfg)

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        self.pre_build_dataset(dataset_cfg)

        parse_eval_cfg = dataset_cfg.valid_item_transform
        self.parse_fn_eval = build_item_augmentation(parse_eval_cfg, self.meta_data)

        if not self.need_build_data_loader:
            self.train_data_loader = None
            self.valid_data_loader = None
            return

        parse_cfg = dataset_cfg.train_item_transform
        self.parse_fn = build_item_augmentation(parse_cfg, self.meta_data)
        draw_batch_cfg = dataset_cfg.batch_transform
        self.draw_batch_fn = build_draw_batch_fn(draw_batch_cfg, self.meta_data)
        device_trans_cfg = dataset_cfg.get('train_device_transform', None)
        self.device_trans = build_device_augmentation(device_trans_cfg, self.meta_data)

        self.get_data_list(dataset_cfg)
        if hasattr(dataset_cfg, 'bucket_schedule_val'):
            val_bucket_schedule = dataset_cfg.bucket_schedule_val
        else:
            val_bucket_schedule = dataset_cfg.bucket_schedule

        split_path_list_by_rank = dataset_cfg.get('split_path_list_by_rank', 1)
        self.train_data_loader = HDFSDataset(
            self.train_file_list,
            dataset_cfg.bucket_schedule,
            dataset_cfg,
            self.parse_fn,
            self.draw_batch_fn,
            device_transforms=self.device_trans,
            split_path_list_by_rank=split_path_list_by_rank,
            shuffle=True,
        )

        valid_split_each_dataset = self.solution_cfg.get('valid_multi_cer', False)
        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            val_bucket_schedule,
            dataset_cfg,
            self.parse_fn_eval,
            self.draw_batch_fn,
            split_path_list_by_rank=False,
            split_each_dataset=valid_split_each_dataset,
        )

    def before_train(self):
        '''override before_train'''
        super().before_train()
        self.grads_accumulator = GradientsAccumulator(
            grad_clip_mode=self.opt_util_cfg.get('grad_clip_mode', 'accum_after'),
            clip_grads_fn=self.clip_grads,
        )

    @get_time('time')
    def train_iteration(self):
        '''train iteration.'''
        i = 0
        loss_scale = self.dist_handler.get_scale(scaler_idx=0)
        enable_grad_clip = self.opt_util_cfg.grad_clip and self.iter > self.lr_cfg.get(
            'warmup_steps', 0
        )
        self.grads_accumulator.reset(enable_grad_clip=enable_grad_clip, loss_scale=loss_scale)
        if hasattr(self.solution, 'set_num_updates'):
            self.solution.set_num_updates(self.iter)
        while i < self.grad_accum_step:
            batch_data = self.next_train_batch()
            try:
                self.solution.train()
                batch_data['loss_scale'] = loss_scale
                self.solution_out = self.solution(batch_data)

                # in cif, the accumulated gradients are not averaged by accumulation steps,
                # and the gradients are cliped before accumulation, these decease the final
                # cer by about 1%
                if self.grad_accum_mode == 'AVG':
                    self.loss = self.solution_out['backward_loss'] / self.grad_accum_step
                else:
                    self.loss = self.solution_out['backward_loss']
                self.dist_handler.backward(self.loss, unscale=False)
                self.grads_accumulator.clip_and_accumulate(
                    self.dist_handler.parameters,
                    optimizer=self.optimizer,
                    last_step=(i + 1 == self.grad_accum_step),
                )
            except RuntimeError as e:
                self.handle_error(e, batch_data)
                continue
            with torch.no_grad():
                self.train_log_buffer.update(self.solution_out)
            i = i + 1
        # get grads_norm and clip grads if needed
        gnorm = self.grads_accumulator.after_accumulation(self.dist_handler.parameters, dict())
        self.solution_out.update(gnorm)
        with torch.no_grad():
            self.train_log_buffer.update(gnorm)
        # optimizer step
        self.dist_handler.step(iters=self.iter)

    @torch.no_grad()
    def validation(self):
        '''valid'''
        # Switch to eval model
        self.mode = 'val'
        self.solution.eval()
        super().validation()
        # switch back to train mode
        self.mode = 'train'
        self.solution.train()

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = AsrCifTrainMetric()
        self.valid_log_buffer = AsrCifValMetric()

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError('lr is not applicable because optimizer does not exist.')

        # show layer_wise lr
        ret_lr_list = []
        if 'params' in self.optimizer_cfg:
            for item in self.optimizer_cfg['params']:
                lr = item['lr']
                if lr not in ret_lr_list:
                    ret_lr_list.append(lr)
            return [ret_lr_list]

        return [group['lr'] for group in self.optimizer.param_groups]

    def update_best_metric(self):
        '''Update best metric to save checkpoint'''
        cur_loss = self.valid_log_buffer.get_value('loss')
        if cur_loss == 0.0:
            logging.error("rank %d catch cur loss is 0.0, not update best metric", self.rank)
            return False
        if self._best_metric is None or cur_loss <= self._best_metric:
            self._best_metric = cur_loss
            return True
        return False

    def run(self):
        '''Entrypoint of runner'''
        self.call_hook('before_run')
        if self.is_export_onnx:
            self.export_onnx()
        elif self.is_inference:
            self.inference()
        else:
            self.train()
            self.train_data_loader.terminate()
            self.valid_data_loader.terminate()
        self.call_hook('after_run')
        self.report_metric()

    @torch.no_grad()
    def inference_once(self, test_file, test_name, language, inference_cfg):
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
        ed_info_list = []
        align_info_list = []
        utt_count = 0
        # inference config
        beam_size = inference_cfg.get('beam_size', 10)
        nbest = inference_cfg.get('nbest', 1)
        filter_list = inference_cfg.get('filter_list', [])
        output_ce_wordboundary = inference_cfg.get('output_ce_wordboundary', False)
        default_filter_list = ['^', '@@ ', '<s>', '</s>', '<pad>', '<unk>', '@@']
        concate_en_letters = inference_cfg.get('concate_en_letters', None)
        filter_list = default_filter_list + filter_list
        stream_mode = inference_cfg.get('stream_mode', False)  # for dual mode infer
        if stream_mode:
            self.solution.encoder_backbone.set_stream_mode(True)

        # inference timestamp config
        output_latency_metric = inference_cfg.get('output_latency_metric', False)
        output_timestamp = output_latency_metric

        if output_timestamp or output_ce_wordboundary:
            assert nbest == 1 or beam_size <= 1
        ref_timestamps = []
        res_timestamps = []
        word_boundary_list = []
        # inference preappear config
        output_preappear_metric = inference_cfg.get('output_preappear_metric', None)
        output_preappear_result = inference_cfg.get('output_preappear_result', None)
        frames_in_each_chunk = inference_cfg.get('frames_in_each_chunk', None)
        given_wait_void_chunk = inference_cfg.get('given_wait_void_chunk', 1)
        token_count = 0
        tail_preappear_count, tail_preappear_correct = 0, 0
        not_tail_preappear_count, not_tail_preappear_correct = 0, 0

        use_llm_inference = inference_cfg.get("use_llm_inference", False)
        if not use_llm_inference:
            if beam_size > 1:
                self.build_beam_search()

        st = time.time()
        while batch_data is not None:
            (
                hyp_strs,
                tgt_strs,
                curr_ed_info_list,
                curr_align_info_list,
                out_rlt_dict,
            ) = self.solution.inference(
                batch_data,
                mode='test',
                beam_size=beam_size,
                nbest=nbest,
                language=language,
                filter_list=filter_list,
                concate_en_letters=concate_en_letters,
                output_timestamp=output_timestamp,
                output_preappear_metric=output_preappear_metric,
            )
            uttid = batch_data['uttid']
            bsz = len(uttid)
            for bid in range(bsz):
                logging.info('%s hyp - %s' % (uttid[bid], hyp_strs[bid]))
                logging.info('%s tgt - %s' % (uttid[bid], tgt_strs[bid]))
                if output_timestamp:
                    timestamp = out_rlt_dict['timestamp']
                    hyp_idx = out_rlt_dict['tokens']
                    hyp_timestamp = timestamp[bid]
                    hyp = hyp_idx[bid].cpu().tolist()[: len(hyp_timestamp)]
                    _, time_list = self.timestamp_post_process(
                        self.tgt_dict, hyp, hyp_timestamp, filter_list
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
                if output_ce_wordboundary:
                    logits = out_rlt_dict["logits"]
                    frames = out_rlt_dict["frames"]
                    hyp_idx = out_rlt_dict['tokens']
                    res_lengths = out_rlt_dict["res_lengths"]
                    align_out, align_score = force_align(
                        logits[bid].cpu()[0 : frames[bid]],
                        hyp_idx[bid][0 : res_lengths[bid]],
                    )
                    final_align = aligned_id_to_char(
                        self.tgt_dict,
                        align_out,
                        align_score,
                        filter_list,
                        frame_shift=20,
                        frame_length=self.solution_cfg.get('downsampling_size', 4) * 10,
                    )
                    logging.info('%s word boundary: %s' % (uttid[bid], final_align))
                    word_boundary_list.append(uttid[bid] + " " + final_align + "\n")
                if output_preappear_metric:
                    assert (
                        ('cif_boundary_marks' in out_rlt_dict)
                        and ('ctc_rlt' in out_rlt_dict)
                        and ('tokens' in out_rlt_dict)
                    )
                    assert frames_in_each_chunk
                    assert given_wait_void_chunk
                    boundary_marks = out_rlt_dict['cif_boundary_marks'][bid].tolist()
                    ctc_rlt = out_rlt_dict['ctc_rlt'][bid].tolist()
                    frame_count = len(boundary_marks)
                    tokens = out_rlt_dict['tokens'][bid].tolist()
                    chunk_size = frames_in_each_chunk
                    eos_id = self.solution_cfg.get('eos_id', 2)
                    blk_id = self.solution_cfg.get('blk_id', 0)

                    chunked_boundary_marks = [
                        boundary_marks[i : i + chunk_size]
                        for i in range(0, frame_count, chunk_size)
                    ]
                    chunked_ctc_rlt = [
                        ctc_rlt[i : i + chunk_size] for i in range(0, frame_count, chunk_size)
                    ]
                    no_eos_tokens = (
                        tokens[0 : list(tokens).index(eos_id)]
                        if (eos_id in list(tokens))
                        else tokens
                    )

                    (
                        tail_preappear_count,
                        tail_preappear_correct,
                        not_tail_preappear_count,
                        not_tail_preappear_correct,
                        token_count,
                    ) = self.preappear_process(
                        chunked_ctc_rlt,
                        chunked_boundary_marks,
                        no_eos_tokens,
                        blk_id,
                        given_wait_void_chunk,
                        not_tail_preappear_count,
                        not_tail_preappear_correct,
                        tail_preappear_count,
                        tail_preappear_correct,
                        output_preappear_result,
                        token_count,
                    )
            ed_info_list += curr_ed_info_list
            align_info_list += curr_align_info_list
            utt_count += len(batch_data['uttid'])
            batch_data = test_data_loader.next()
        time_elapsed = time.time() - st
        test_data_loader.terminate()
        # output to file
        out_stat_file = 'rank{}_cer_result_{}.txt'.format(self.rank, test_name)
        output_result(align_info_list, ed_info_list, out_stat_file, lang='zh')

        if output_latency_metric:
            latency_info = self.latency_post_process(
                res_timestamps, ref_timestamps, align_info_list
            )
            logging.info("rank %d first token latency: %.2f", self.rank, latency_info[0])
            logging.info("rank %d last token latency: %.2f", self.rank, latency_info[1])
            logging.info("rank %d avg token latency: %.2f", self.rank, latency_info[2])
            logging.info("rank %d p50 token latency: %.2f", self.rank, latency_info[3])
            logging.info("rank %d p90 token latency: %.2f", self.rank, latency_info[4])
        if output_ce_wordboundary:
            out_word_boundary_file = 'rank{}_word_boundary_{}.txt'.format(self.rank, test_name)
            output_wordboundary(word_boundary_list, out_word_boundary_file)

        if output_preappear_metric:
            logging.info(
                "tail preappear ratio (utt-level): %.4f" % (float(tail_preappear_count) / utt_count)
            )
            if tail_preappear_count > 0:
                logging.info(
                    "tail preappear accuracy: %.4f"
                    % (float(tail_preappear_correct) / tail_preappear_count)
                )
            logging.info(
                "false preappear ratio (token-level): %.4f"
                % (float(not_tail_preappear_count) / token_count)
            )
            if not_tail_preappear_count > 0:
                logging.info(
                    "not_tail preappear accuracy: %.4f"
                    % (float(not_tail_preappear_correct) / not_tail_preappear_count)
                )
        # wait all rank finish
        dist_barrier()
        logging.info(
            'rank %d, testset %s, total num of utt: %d, total infer forward time: %.3f s',
            self.rank,
            test_file,
            utt_count,
            time_elapsed,
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
    def latency_post_process(res_timestamps, ref_timestamps, align_info_list):
        '''post process for latency'''
        first_token_latency_list = []
        last_token_latency_list = []
        avg_token_latency_list = []
        for i, res_timestamp in enumerate(res_timestamps):
            try:
                ref_timestamp = ref_timestamps[i]
                if len(res_timestamp) == 0 or len(ref_timestamp) == 0:
                    continue
                cor_index = align_info_list[i]['cor_index']
                for ref_index, res_index, _ in cor_index:
                    cur_latency = res_timestamp[res_index] - ref_timestamp[ref_index]
                    avg_token_latency_list.append(cur_latency)
                    if ref_index == 0 and res_index == 0:
                        first_token_latency_list.append(cur_latency)
                    if ref_index == len(ref_timestamp) - 1 and res_index == len(res_timestamp) - 1:
                        last_token_latency_list.append(cur_latency)
            except Exception as e:
                logging.error('latency_post_process, error %s', str(e))
                continue
        first_token_latency = 0
        if len(first_token_latency_list) > 0:
            first_token_latency = sum(first_token_latency_list) / len(first_token_latency_list)
        last_token_latency = 0
        if len(last_token_latency_list) > 0:
            last_token_latency = sum(last_token_latency_list) / len(last_token_latency_list)
        avg_token_latency = 0
        p50_token_latency = 0
        p90_token_latency = 0
        if len(avg_token_latency_list) > 0:
            avg_token_latency = sum(avg_token_latency_list) / len(avg_token_latency_list)
            avg_token_latency_list.sort()
            p50_token_latency = avg_token_latency_list[len(avg_token_latency_list) // 2]
            p90_token_latency = avg_token_latency_list[int(len(avg_token_latency_list) * 0.9)]
        return (
            first_token_latency,
            last_token_latency,
            avg_token_latency,
            p50_token_latency,
            p90_token_latency,
        )

    @staticmethod
    def preappear_process(
        chunked_ctc_rlt,
        chunked_boundary_marks,
        no_eos_tokens,
        blk_id,
        given_wait_void_chunk,
        not_tail_preappear_count,
        not_tail_preappear_correct,
        tail_preappear_count,
        tail_preappear_correct,
        output_preappear_result,
        token_count,
    ):
        '''show and stat the preappear of each utterance'''
        # pylint:disable=too-many-branches,too-many-nested-blocks
        is_start = True
        void_chunk_count = 0
        last_token_index = -1
        tail_token_index = len(no_eos_tokens) - 1
        last_valid_ctc_token = blk_id

        is_on_preappear = False
        org_chunked_output = []
        preappear_chunked_output = []
        for cid, _ in enumerate(chunked_ctc_rlt):
            if sum(chunked_boundary_marks[cid]) == 0:
                if is_start:
                    if last_valid_ctc_token != blk_id:
                        is_start = False
                else:
                    void_chunk_count += 1
                    if void_chunk_count >= given_wait_void_chunk and last_valid_ctc_token != blk_id:
                        if (
                            last_token_index == -1
                            or no_eos_tokens[last_token_index] != last_valid_ctc_token
                        ):
                            if not is_on_preappear:
                                if last_token_index < tail_token_index - 1:
                                    not_tail_preappear_count += 1
                                    if last_valid_ctc_token == no_eos_tokens[last_token_index + 1]:
                                        not_tail_preappear_correct += 1
                                else:
                                    tail_preappear_count += 1
                                    if last_valid_ctc_token == no_eos_tokens[last_token_index + 1]:
                                        tail_preappear_correct += 1
                            if output_preappear_result:
                                preappear_chunked_output = copy.deepcopy(org_chunked_output)
                                preappear_chunked_output.append(last_valid_ctc_token)
                                is_on_preappear = True
            else:
                void_chunk_count = 0
                is_start = False

                if output_preappear_result:
                    if is_on_preappear:
                        preappear_chunked_output.pop()
                        is_on_preappear = False
                    for ii in range(sum(chunked_boundary_marks[cid])):
                        tmp_index = last_token_index + 1 + ii
                        if tmp_index < len(no_eos_tokens):
                            org_chunked_output.append(no_eos_tokens[tmp_index])
                            preappear_chunked_output.append(no_eos_tokens[tmp_index])

            # pylint:disable=unnecessary-list-index-lookup
            for k in range(len(chunked_ctc_rlt[cid]) - 1, -1, -1):
                if chunked_ctc_rlt[cid][k] != blk_id and chunked_ctc_rlt[cid][k] != 0:
                    last_valid_ctc_token = chunked_ctc_rlt[cid][k]
                    break

            if output_preappear_result:
                logging.info("org result till to chunk %d: %s" % (cid, org_chunked_output))
                logging.info("pre result till to chunk %d: %s" % (cid, preappear_chunked_output))

            last_token_index += sum(chunked_boundary_marks[cid])
            if last_token_index >= tail_token_index:
                break

        token_count += len(no_eos_tokens)

        return (
            tail_preappear_count,
            tail_preappear_correct,
            not_tail_preappear_count,
            not_tail_preappear_correct,
            token_count,
        )

    def build_beam_search(self, key='inference'):
        '''build beam search'''
        cfg = self.args.get(key, None)
        if cfg is not None:
            if hasattr(self, 'reorder_dict_map'):
                cfg.reorder_dict_map = getattr(self, 'reorder_dict_map')
            self.solution.init_beam_search(cfg, self.lm_solution)

    def save_inference_stat(self, inference_cfg):
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
            if glob.glob('*word_boundary_*.txt'):
                hdfs_put('*word_boundary_*.txt', remote_stat_dir, sync=True)


@RUNNERS.register_module()
class UniversalCifRunner(BaseCifRunner):
    '''alias for RNNTRunner'''

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = UniversalAsrCifTrainMetric()
        self.valid_log_buffer = UniversalAsrCifValMetric()
