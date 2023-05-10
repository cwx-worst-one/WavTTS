"""BestrqPretrainRunner"""

import os.path as osp
import io
from subword_nmt.apply_bpe import BPE
from core.utils import logging
from core.runner.asr.base_cif_runner import BaseCifRunner
from core.runner.metric.asr_metric import AuLlmValMetric
from core.dataset import get_meta
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class AuLlmPretrainRunner(BaseCifRunner):
    """SpokenLmPretrainRunner"""

    def __init__(self, cfg, inference=False, export_onnx=False):
        if cfg.data.get('add_sc', False):
            cfg.solution.vocab_size += 1
        super().__init__(cfg, inference=inference, export_onnx=export_onnx)

    def pre_build_dataset(self, dataset_cfg):
        '''The only difference from BaseCifRunner is that
        we add $ token into meta_data['tgt_dict'].
        '''
        meta_data_root = dataset_cfg.get("meta_data_root", dataset_cfg.get("data_root", None))
        if isinstance(meta_data_root, (list, tuple)):
            meta_data_root = meta_data_root[0]
        meta_file = osp.join(meta_data_root, dataset_cfg.meta_file)
        self.meta_data = get_meta(meta_file)
        if self.dataset_cfg.get('add_sc', False):
            # Add sc token into tgt_dict and reorder_tgt_dict
            sc_token = self.dataset_cfg.get('sc_token', '$')
            self.meta_data['tgt_dict'].add_symbol(sc_token)
            self.meta_data['reorder_tgt_dict'].add_symbol(sc_token)

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

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = AuLlmValMetric()
        self.valid_log_buffer = AuLlmValMetric()

    def train_iteration(self):
        self.solution.set_num_updates(self.iter)
        super().train_iteration()
