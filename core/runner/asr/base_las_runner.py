''' BaseLASRunner '''

import torch

from core.dataset import (
    HDFSDataset,
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)

from core.utils import dist_barrier, hdfs_put, hdfs_mkdir, hdfs_test, logging, get_local_rank
from core.runner.metric.asr_metric import LasMetric
from core.utils.cer.cer_metric import (
    EditDistanceCalculator,
    TextFormator,
    output_result,
    merge_result,
)
from core.utils.misc import (
    get_file_key,
    infer_text_format,
)
from .base_asr_runner import BaseAsrRunner
from ..base_runner import RUNNERS
from ..utils import get_oracle_ed_info, file_pattern


@RUNNERS.register_module()
class BaseLASRunner(BaseAsrRunner):
    '''LAS Runner'''

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

    # TODO(zhengyijie): refactor the file organization
    @torch.no_grad()
    def inference_once(self, test_file, test_name, language, inference_cfg, lm_solution=None):
        '''inference a test set'''
        # pylint:disable=too-many-branches,too-many-locals,too-many-statements,too-many-nested-blocks
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
        beam_size = inference_cfg.get('las_beam_size', 0)
        lm_weight = inference_cfg.get('lm_weight', 1.0)
        nbest_out = inference_cfg.get('nbest_out', False)
        filter_list = inference_cfg.get('filter_list', [])
        merge_asr_results = inference_cfg.get('merge_asr_results', False)
        ed_calculator = EditDistanceCalculator()
        formator = TextFormator(language)

        # pylint:disable=consider-using-with
        rec_file = open(
            'rank{}_rec_result_{}.txt'.format(self.rank, test_name), 'w', encoding='utf-8'
        )
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
        ref_cache = {}
        while batch_data is not None:
            if beam_size > 0:
                inf_res = self.solution.beam_inference(
                    batch_data,
                    beam_size,
                    lm_solution=lm_solution,
                    lm_weight=lm_weight,
                    reorder_dict_map=self.reorder_dict_map,
                    nbest_out=nbest_out,
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
                rec_file.write('{} {}\n'.format(batch_data['uttid'][bid], '-'.join(res_format)))
                ref_format = infer_text_format(batch_data['ref'][bid], filter_list, formator)
                if merge_asr_results:
                    uttrk = batch_data['uttid'][bid].strip().split('_')[-1]
                    uttid = batch_data['uttid'][bid][: -len(uttrk) - 1]
                    if uttrk == '0':
                        ref_cache[uttid] = batch_data['ref'][bid]

                logging.info(
                    'rank %d, uttid %s, %s',
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
        rec_file.close()
        dist_barrier()

        # merge split result
        if merge_asr_results:
            # 1. merge rec results in rank0
            if self.rank == 0:
                temp = {}
                for rk in range(self.world_size):
                    with open(
                        'rank{}_rec_result_{}.txt'.format(rk, test_name), 'r', encoding='utf-8'
                    ) as f:
                        lines = f.readlines()
                    for line in lines:
                        line = line.split(' ')
                        if len(line) == 2:
                            uttrk = line[0].strip().split('_')[-1]
                            uttid = line[0][: -len(uttrk) - 1]
                            if uttid not in temp:
                                temp[uttid] = {}
                            temp[uttid][uttrk] = line[1].strip()
                with open('las_inference_res_{}.txt'.format(test_name), 'w', encoding='utf-8') as f:
                    for utt, v in temp.items():
                        res_format = '-'.join([s[1] for s in sorted(v.items())])
                        f.write('{} {}\n'.format(utt, res_format.strip()))
            dist_barrier()
            # 2. postprocess infernce results to get res_format
            res = {}
            with open('las_inference_res_{}.txt'.format(test_name), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(' ')
                    if len(line) == 2:
                        res[line[0]] = line[1].strip().split('-')
            # 3. calculate ed_info for current rank
            ed_info_list = []
            align_info_list = []
            for uttid, ref in ref_cache.items():
                ref_format = infer_text_format(ref, filter_list, formator)
                res_format = res.get(uttid, [])
                logging.info(
                    'rank %d, uttid %s, %s',
                    self.rank,
                    uttid,
                    ' '.join(res_format),
                )
                ed_info, align_info = ed_calculator.show_alignment(uttid, ref_format, res_format)
                ed_info_list.append(ed_info)
                align_info_list.append(align_info)
            ref_cache = None
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

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = LasMetric()
        self.valid_log_buffer = LasMetric()

    @torch.no_grad()
    def inference(self):
        '''inference'''
        self.build_beam_search()
        inference_cfg = self.args.inference
        # resume RNN-T solution
        self.call_hook('before_run')
        self.solution.eval()
        # self.solution.criterion_module.log_softmax_fc.combine_weight()
        # test data sets
        test_sets = inference_cfg.test_sets.strip().split('|')
        test_sets_langs = inference_cfg.get('language', 'zh-CN').split('|')
        assert len(test_sets_langs) == 1 or len(test_sets) == len(test_sets_langs)
        if len(test_sets_langs) == 1:
            test_sets_langs = test_sets_langs * len(test_sets)
        test_file_path = self.get_test_file_list(self.dataset_cfg, test_sets)
        test_names = [get_file_key(test_file) for test_file in test_sets]

        # inference every test set
        inf_draw_batch_cfg = self.dataset_cfg.inference_batch_transform
        self.draw_batch_fn_inference = build_draw_batch_fn(inf_draw_batch_cfg)
        for test_file, test_name, lang in zip(test_file_path, test_names, test_sets_langs):
            self.inference_once(test_file, test_name, lang, inference_cfg, self.lm_solution)
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
        self.call_hook('after_run')
        self.report_metric()

    def build_beam_search(self, key='inference'):
        '''build beam search'''
        cfg = self.args.get(key, None)
        if cfg is not None:
            self.solution.init_beam_search(cfg)

    def get_data_list(self, dataset_cfg):
        '''
        get valid_file_list,train_file_list
        '''
        data_root = dataset_cfg.get('data_root', '')
        train_data_root = dataset_cfg.get('train_data_root', data_root)
        train_file_list = dataset_cfg.get('train_file_list', None)
        train_file_list = file_pattern(train_data_root, train_file_list)
        dataset_cfg.train_file_list = train_file_list
        super().get_data_list(dataset_cfg)
