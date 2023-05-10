''' RNNTContextAwareRunner '''

from transformers import BertTokenizer as BertTokenizer_huggingface
from core.utils.dist_hdfs import dist_hdfs_get
from ..base_runner import RUNNERS
from .rnnt_runner import RNNTRunner


@RUNNERS.register_module()
class RNNTContextAwareRunner(RNNTRunner):
    '''Runner of context-aware RNN-T'''

    def setup_transform_cfg(self, dataset_cfg):
        '''setup transforms for dataset preprocessing'''
        bert_vocab_dict = self.solution_cfg.get('bert_vocab_dict', '')
        local_vocab_file = None
        if bert_vocab_dict:
            local_vocab_file = dist_hdfs_get(bert_vocab_dict, './tmp/', 'vocab_dict.txt')

        transform_cfgs = (
            dataset_cfg.train_item_transform,
            dataset_cfg.valid_item_transform,
        )
        for transform_cfg in transform_cfgs:
            for cfg in transform_cfg:
                if cfg.type == 'DialogHistToContext':
                    cfg['vocab'] = self.reorder_tgt_dict.symbols
                elif cfg.type in ('BertTokenizer', 'DocBertTokenizer'):
                    assert local_vocab_file is not None
                    bert_tokenizer = BertTokenizer_huggingface(local_vocab_file)
                    cfg['bert_tokenizer'] = bert_tokenizer

        for cfg in dataset_cfg.batch_transform:
            if cfg.type == 'PreCharCollate':
                cfg['args'] = self.solution_cfg
            elif cfg.type == 'ContextMakePairsCollate':
                context_loss_scale = self.solution_cfg.get('context_loss_scale', 0)
                if context_loss_scale > 0:
                    cfg['context_loss_scale'] = context_loss_scale
