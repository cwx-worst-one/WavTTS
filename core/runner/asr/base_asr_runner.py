''' base asr runner. '''
import os
import os.path as osp
import shutil
from abc import ABCMeta

import torch
from core.dataset import build_draw_batch_fn
from core.runner.metric.asr_metric import AsrMetric
from core.solutions.lm.lm_solution import LmSolution
from core.utils import (
    dist_barrier,
    get_local_rank,
    dist_hdfs_get,
    hdfs_ls,
    hdfs_mkdir,
    hdfs_put,
    hdfs_test,
    logging,
    Config,
)
from core.utils.cer.cer_metric import merge_result, sort_asr_metrics
from core.utils.misc import get_file_key

from ..base_runner import BaseRunner

# abstract class, no need to register
class BaseAsrRunner(BaseRunner, metaclass=ABCMeta):
    '''Base ASR Runner.'''

    def __init__(self, cfg, inference=False, export_onnx=False):
        """Init."""
        super().__init__(cfg, inference=inference, export_onnx=export_onnx)
        lm_cfg = cfg.get('lm', Config())
        inference_cfg = cfg.get('inference', {})
        self.lm_solution = None
        if isinstance(lm_cfg, dict) and "nnlm" in lm_cfg.keys():
            lm_cfg["nnlm"]["tgt_vocab_size"] = self.tgt_vocab_size
        onnx_dir = osp.join(
            self.train_cfg.save_root, self.train_cfg.save_dir, self.train_cfg.save_name, 'onnx'
        )

        lm_cfg.setdefault('hotword_fst', Config())
        lm_cfg.setdefault('coldword_fst', Config())
        lm_cfg.hotword_fst.setdefault('lm_token_beam_size', 10)
        lm_cfg.hotword_fst.setdefault('max_active_lm_token_num', 5000)
        lm_cfg.hotword_fst.setdefault('enable_multi_step_search', True)
        lm_cfg.hotword_fst.hotword_weight = inference_cfg.get('hotword_fst_weight', '')

        if lm_cfg:
            lm_cfg.setdefault('backend', self.solution_cfg.get('backend', 'torch'))
            lm_cfg.setdefault('onnx_dir', self.solution_cfg.get('onnx_dir', onnx_dir))
        self.lm_solution = LmSolution(lm_cfg)
        self.lm_solution.load_from_inference_cfg(inference_cfg)

    def extra_parse_config(self):
        '''parse config'''
        front_end_type = self.solution_cfg.get('front_end_type')
        self.solution_cfg.setdefault('input_concat_size', 1)
        if front_end_type is not None and front_end_type != "TimeReduceLSTMP":
            self.solution_cfg['input_concat_size'] = 1
        fbank_dim = self.dataset_cfg.get('fbank_dim', 80)
        self.solution_cfg['onnx_stack_frame'] = self.solution_cfg.input_concat_size * fbank_dim
        self.solution_cfg.setdefault('onnx_dir', 'onnx')

    def build_solution(self):
        '''build_solution'''
        self.extra_parse_config()
        super().build_solution()

    def export_onnx(self):
        '''export onnx'''
        out_dir = self.solution_cfg.get('onnx_dir')
        self.solution.register_infers()
        self.solution.export(data_loader=self.train_data_loader)
        if self.lm_solution is not None:
            self.lm_solution.register_infers()
            self.lm_solution.export(data_loader=self.train_data_loader)
        self.save_onnx(out_dir)

    def setup_transform_cfg(self, dataset_cfg):
        '''setup transforms for dataset preprocessing'''
        for cfg in dataset_cfg.batch_transform:
            if cfg.type == 'PreCharCollate':
                cfg['args'] = self.solution_cfg

    @staticmethod
    def get_test_file_list(dataset_cfg, test_sets):
        '''get test file list.'''
        data_root = dataset_cfg.get('data_root', None)
        if isinstance(data_root, list):
            data_root = data_root[0]
        test_data_root = dataset_cfg.get('test_data_root', data_root)
        test_files = []
        for test_file in test_sets:
            test_files += [osp.join(test_data_root, test_file)]
        return test_files

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        super().pre_build_dataset(dataset_cfg)

        train_domain_list = dataset_cfg.get("train_domain_list", [])
        valid_domain_list = dataset_cfg.get("valid_domain_list", [])
        self.train_domain_list = []
        for train_domain in train_domain_list:
            self.train_domain_list += eval(train_domain)
        self.valid_domain_list = []
        for valid_domain in valid_domain_list:
            self.valid_domain_list += eval(valid_domain)

        for item_trans in dataset_cfg.get('train_item_transform', []):
            if item_trans['type'] == 'DomainAdd':
                item_trans['domain_list'] = self.train_domain_list
        for item_trans in dataset_cfg.get('valid_item_transform', []):
            if item_trans['type'] == 'DomainAdd':
                item_trans['domain_list'] = self.valid_domain_list

        meta_data = self.meta_data
        self.tgt_dict = meta_data.get('tgt_dict')
        is_reorder_dict = self.solution_cfg.get('reorder_dict_by_freq', 1)
        if is_reorder_dict:
            self.reorder_tgt_dict = meta_data.get('reorder_tgt_dict')
            self.reorder_dict_map = meta_data.get('reorder_dict_map')
        else:
            self.reorder_tgt_dict = None
            self.reorder_dict_map = None

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

        chunk_size = dataset_cfg.get('chunk_size', 40)
        if chunk_size > 100:
            logging.warning(
                "HDFSDataset get chunk_size (%d) too large, "
                "May cause memory crash and IO failure results, "
                "If you want to set to something else, "
                "modify the chunk_size in the configuration file, "
                "or configure --data.chunk_size at the beginning of trail. ",
                chunk_size,
            )
            dataset_cfg.chunk_size = 40

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = AsrMetric()
        self.valid_log_buffer = AsrMetric()

    # TODO(mst) inference_once should be abstractmethod
    @torch.no_grad()
    def inference_once(self, test_file, test_name, language, inference_cfg):
        '''inference once'''
        # pylint: disable=unused-argument, no-self-use
        logging.error('inference_once not implement')

    def merge_inference_results(self, test_names):
        '''merge inference results'''
        if self.rank == 0:
            for test_name in test_names:
                merge_result(self.world_size, test_name, falcon_report=self.falcon_report)
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

    @torch.no_grad()
    def inference_one_chkpt(self):
        '''inference func.'''
        inference_cfg = self.args.inference
        self.solution.eval()
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
        self.draw_batch_fn_inference = build_draw_batch_fn(inf_draw_batch_cfg, self.meta_data)
        for test_file, test_name, lang in zip(test_file_path, test_names, test_sets_langs):
            self.inference_once(test_file, test_name, lang, inference_cfg)
        self.merge_inference_results(test_names)
        self.save_inference_stat(inference_cfg)

    @torch.no_grad()
    def inference(self):
        '''inference multiple checkpoints.'''
        # pylint: disable=too-many-branches
        infer_cfg = self.args.inference
        chkpts_path = infer_cfg.get('chkpts_path', None)
        override = infer_cfg.get('override', False)
        remote_stat_dir = infer_cfg.get('remote_stat_dir', None)
        if not chkpts_path:
            self.inference_one_chkpt()
            return
        local_asr_results_dir = './asr_results'
        if get_local_rank() == 0:
            os.system('rm -rf ' + local_asr_results_dir)
            os.makedirs(local_asr_results_dir, exist_ok=False)
        chkpts_components = chkpts_path.split('|')
        if len(chkpts_components) > 1:
            chkpts_root = chkpts_components[0]
            chkpts = []
            for ckpt in chkpts_components[1].split(','):
                path = osp.join(chkpts_root, ckpt)
                chkpts.append(path)
        else:
            raise KeyError(f'Wrong config chkpts_path: {chkpts_path}')
        chkpts = [p for p in chkpts if p.endswith('.pth')]
        test_sets = [get_file_key(t) for t in infer_cfg.get('test_sets', '').split('|')]
        testset_stats = {'cer_result_{}.txt'.format(t) for t in test_sets}
        checkpoint_dir = osp.join(
            self.train_cfg.save_root,
            self.train_cfg.save_dir,
            self.train_cfg.save_name,
            'checkpoints',
        )
        for chkpt in chkpts:
            step = osp.basename(chkpt).replace('.pth', '').replace('step_', '')
            step_stat_dir = '%s/%s_asr_results' % (remote_stat_dir, step)
            cer_stats = {osp.basename(t) for t in hdfs_ls(step_stat_dir, ptype='dir')}
            if testset_stats.issubset(cer_stats):
                if not (override or step == 'best'):
                    logging.info('not override: %s', step_stat_dir)
                    dist_hdfs_get(step_stat_dir, local_asr_results_dir)
                    continue
            if self._resume_from_pretrain(chkpt, checkpoint_dir):
                logging.info('resume chkpt: %s', chkpt)
                logging.info('save cer_stats to %s', step_stat_dir)
                self.args.inference['remote_stat_dir'] = step_stat_dir
                self.inference_one_chkpt()
                if hasattr(self.solution.criterion_module, "clear_combined_weight"):
                    self.solution.criterion_module.clear_combined_weight()
                dist_hdfs_get(step_stat_dir, local_asr_results_dir)
                if get_local_rank() == 0:
                    shutil.rmtree(checkpoint_dir)
                    os.makedirs(checkpoint_dir, exist_ok=False)
            else:
                logging.info('resume failed: %s', chkpt)
        if get_local_rank() == 0:
            sort_asr_metrics(local_asr_results_dir, testset_stats)
