"""BestrqPretrainRunner"""

import os.path as osp
import time
import torch
from core.extensions import clear_cuda_error
from core.runner.pretrain.wav2vec_pretrain_runner import Wav2vecPretrainRunner
from core.runner.metric.asr_metric import SpokenLmAffixMetric, SpokenlmMetric
from core.dataset.dictionary import Dictionary, ScpDictionary
from ..base_runner import RUNNERS
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
from core.utils.metric_output import (
    StreamingStableMetricOneSample,
    ASRModelRichOutWriter,
    merge_stable_metric,
    merge_asr_rich_info,
)
from core.utils.cer.cer_metric import (
    EditDistanceCalculator,
    TextFormator,
    output_result,
    merge_result,
)
from core.solutions.asr.utils import (
    force_align,
    aligned_id_to_char,
)
from core.utils.misc import infer_text_format
from ..utils import (
    get_word_boundary,
    get_oracle_ed_info,
)


@RUNNERS.register_module()
class SpokenLmPretrainRunner(Wav2vecPretrainRunner):
    """SpokenLmPretrainRunner"""

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = SpokenLmAsrMetric()
        self.valid_log_buffer = SpokenLmAsrMetric()

    def pre_build_dataset(self, dataset_cfg):
        '''prepare for build dataset.'''
        super().pre_build_dataset(dataset_cfg)
        if self.meta_data:
            new_dict = Dictionary(extra_special_symbols=dataset_cfg.extra_special_symbols)
            new_dict.update(self.meta_data.get('tgt_dict'))
            self.meta_data["tgt_dict"] = new_dict
            self.solution_cfg.tgt_dict = self.meta_data["tgt_dict"]
            self.tgt_dict = self.meta_data.get('tgt_dict')
            if "reorder_tgt_dict" in self.meta_data:
                new_dict = ScpDictionary(extra_special_symbols=dataset_cfg.extra_special_symbols)
                new_dict.update(self.meta_data.get('reorder_tgt_dict'))
                self.meta_data["reorder_tgt_dict"] = new_dict
                self.solution_cfg.reorder_tgt_dict = self.meta_data["reorder_tgt_dict"]
                self.reorder_tgt_dict = self.meta_data.get('reorder_tgt_dict')
                self.reorder_dict_map = self.meta_data.get('reorder_dict_map')
            if self.tgt_dict:
                self.tgt_vocab_size = len(self.tgt_dict)
                # align to 8
                self.tgt_vocab_size = ((self.tgt_vocab_size + 8 - 1) // 8) * 8
                logging.info('Dictionary total vocab size %d', self.tgt_vocab_size)
                self.solution_cfg.setdefault('tgt_vocab_size', self.tgt_vocab_size)
                self.solution_cfg.setdefault('reorder_tgt_dict', self.reorder_tgt_dict)
                self.in_out_ratio = dataset_cfg.get('in_out_ratio', 4)
                if self.in_out_ratio < self.solution_cfg.get('downsampling_size', 4):
                    self.in_out_ratio = self.solution_cfg.get('downsampling_size', 4)
            self.setup_transform_cfg(dataset_cfg)

    @torch.no_grad()
    def validation(self):
        '''validation func.'''
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        self.valid_log_buffer.reset()
        batch_data = self.valid_data_loader.next()
        self._val_iter = 0
        dataset_names = [osp.basename(p) for p in self.valid_data_loader.origin_path_list]
        for dataset_name in dataset_names:
            while batch_data is not None:
                self.call_hook('before_val_iter')
                try:
                    self.solution.eval()
                    validation_out = self.solution(batch_data)
                    self._val_iter += 1
                    batch_data = self.valid_data_loader.next()
                    self.call_hook('after_val_iter')
                    self.valid_log_buffer.update(validation_out)
                except RuntimeError as e:
                    clear_cuda_error()
                    torch.cuda.empty_cache()
                    logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                    continue
            self.log_metric(self.valid_log_buffer, name=dataset_name)
        self.after_validation()

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
        keep_non_proun_tokens = inference_cfg.get('keep_non_proun_tokens', None)
        rich_info_writer = ASRModelRichOutWriter(self.args)
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
        while batch_data is not None:
            st = time.time()
            if beam_size > 0:
                inf_res = self.solution.beam_inference(
                    batch_data,
                    nbest=nbest,
                )
                # inf_res = output["inf_res"]
                output = {}

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
            # next batch
            batch_data = test_data_loader.next()
        test_data_loader.terminate()
        # output to file
        out_stat_file = 'rank{}_cer_result_{}.txt'.format(self.rank, test_name)
        output_result(align_info_list, ed_info_list, out_stat_file, lang='zh')
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
        dist_barrier()

    @torch.no_grad()
    def inference(self):
        '''inference'''
        self.build_beam_search()
        self.rich_info_writed = dict()
        super().inference()


@RUNNERS.register_module()
class SpokenLmTranslationRunner(Wav2vecPretrainRunner):
    """SpokenLmTranslationRunner"""

    def build_metrics(self):
        '''build metric'''
        if (
            getattr(self.solution_cfg, 'loss_type', 'SpokenLlmAffixXentropy')
            == "SpokenLlmAffixXentropy"
        ):
            self.train_log_buffer = SpokenLmAffixMetric()
            self.valid_log_buffer = SpokenLmAffixMetric()
        else:
            self.train_log_buffer = SpokenlmMetric()
            self.valid_log_buffer = SpokenlmMetric()

    def pre_build_dataset(self, dataset_cfg):
        '''prepare for build dataset.'''
        super().pre_build_dataset(dataset_cfg)

        if self.meta_data:  # only special token and speech tokens for s2st
            logging.info(
                "Adding %d acoustic tokens into Dictionary." % dataset_cfg.acoustic_vocab_size
            )
            acoustic_symbols = ["<{}>".format(i) for i in range(dataset_cfg.acoustic_vocab_size)]
            new_dict = Dictionary(
                extra_special_symbols=dataset_cfg.extra_special_symbols + acoustic_symbols
            )
            for trans in dataset_cfg.valid_item_transform:
                if trans['type'] == "BPE":
                    trans["skip_list"] = dataset_cfg.extra_special_symbols + acoustic_symbols
                    break
            for trans in dataset_cfg.train_item_transform:
                if trans['type'] == "BPE":
                    trans["skip_list"] = dataset_cfg.extra_special_symbols + acoustic_symbols
                    break
            dataset_cfg.only_acoustic_token = getattr(dataset_cfg, 'only_acoustic_token', True)
            if not dataset_cfg.only_acoustic_token:
                logging.info("Loading tgt_dict.")
                new_dict.update(self.meta_data.get('tgt_dict'))
            self.meta_data["tgt_dict"] = new_dict
            self.solution_cfg.tgt_dict = self.meta_data["tgt_dict"]
            self.tgt_dict = self.meta_data.get('tgt_dict')
            if "reorder_tgt_dict" in self.meta_data:
                new_dict = ScpDictionary(
                    extra_special_symbols=dataset_cfg.extra_special_symbols + acoustic_symbols
                )
                new_dict.update(self.meta_data.get('reorder_tgt_dict'))
                self.meta_data["reorder_tgt_dict"] = new_dict
                self.solution_cfg.reorder_tgt_dict = self.meta_data["reorder_tgt_dict"]
                self.reorder_tgt_dict = self.meta_data.get('reorder_tgt_dict')
                self.reorder_dict_map = self.meta_data.get('reorder_dict_map')
            if self.tgt_dict:
                self.tgt_vocab_size = len(self.tgt_dict)
                # align to 8
                self.tgt_vocab_size = ((self.tgt_vocab_size + 8 - 1) // 8) * 8
                logging.info('Dictionary total vocab size %d', self.tgt_vocab_size)
                self.solution_cfg.setdefault('tgt_vocab_size', self.tgt_vocab_size)
                self.solution_cfg.setdefault('reorder_tgt_dict', self.reorder_tgt_dict)
                self.in_out_ratio = dataset_cfg.get('in_out_ratio', 4)
                if self.in_out_ratio < self.solution_cfg.get('downsampling_size', 4):
                    self.in_out_ratio = self.solution_cfg.get('downsampling_size', 4)
            self.setup_transform_cfg(dataset_cfg)

    @torch.no_grad()
    def validation(self):
        '''validation func.'''
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        self.valid_log_buffer.reset()
        batch_data = self.valid_data_loader.next()
        self._val_iter = 0
        dataset_names = [osp.basename(p) for p in self.valid_data_loader.origin_path_list]
        for dataset_name in dataset_names:
            while batch_data is not None:
                self.call_hook('before_val_iter')
                try:
                    self.solution.eval()
                    validation_out = self.solution(batch_data)
                    self._val_iter += 1
                    batch_data = self.valid_data_loader.next()
                    self.call_hook('after_val_iter')
                    self.valid_log_buffer.update(validation_out)
                except RuntimeError as e:
                    clear_cuda_error()
                    torch.cuda.empty_cache()
                    logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                    continue
            self.log_metric(self.valid_log_buffer, name=dataset_name)
        self.after_validation()
