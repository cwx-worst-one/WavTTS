''' CIFPretrainRunner '''

from ..base_runner import RUNNERS
from .base_cif_runner import BaseCifRunner


@RUNNERS.register_module()
class CIFCePretrainRunner(BaseCifRunner):
    '''pretrain acoustic encoder for CIF'''

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        self.solution_cfg.setdefault('fbank_dim', dataset_cfg.fbank_dim)
        self.solution_cfg.setdefault('fbank_channel', dataset_cfg.fbank_channel)
        self.solution_cfg.setdefault('use_eos', dataset_cfg.use_eos)
        self.vocab_size = dataset_cfg.vocab_size
        # align to 8
        self.solution_cfg.setdefault('vocab_size', self.vocab_size)

        # self.solution_cfg.setdefault('reorder_tgt_dict', self.reorder_tgt_dict)
        self.in_out_ratio = dataset_cfg.get('in_out_ratio', 8)
        if self.in_out_ratio < self.solution_cfg.get('downsampling_size', 8):
            self.in_out_ratio = self.solution_cfg.get('downsampling_size', 8)

        self.setup_transform_cfg(dataset_cfg)
