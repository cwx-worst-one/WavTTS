''' NNLMRunner '''

import os
import torch
import numpy as np
from core.dataset import (
    HDFSDataset,
    ValidHDFSDataset,
    build_draw_batch_fn,
)
from core.extensions import clear_cuda_error
from core.runner.metric.asr_metric import AsrMetric
from core.utils import (
    dist_barrier,
    hdfs_put,
    hdfs_test,
    logging,
    hdfs_mkdir,
    get_local_rank,
)
from ..base_runner import BaseRunner, RUNNERS
from ..utils import file_pattern


@RUNNERS.register_module()
class BaseLMRunner(BaseRunner):
    '''base lm runner'''

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        super().pre_build_dataset(dataset_cfg)
        meta_data = self.meta_data
        self.tgt_dict = meta_data['tgt_dict']
        is_reorder_dict = self.solution_cfg.get('reorder_dict_by_freq', 0)
        if is_reorder_dict:
            self.reorder_tgt_dict = meta_data['reorder_tgt_dict']
            self.reorder_dict_map = meta_data['reorder_dict_map']
        else:
            self.reorder_tgt_dict = None
            self.reorder_dict_map = None

        self.tgt_vocab_size = len(self.tgt_dict)
        # align to 8
        self.tgt_vocab_size = ((self.tgt_vocab_size + 8 - 1) // 8) * 8
        logging.info('Dictionary total vocab size %d', self.tgt_vocab_size)
        self.solution_cfg.setdefault('tgt_vocab_size', self.tgt_vocab_size)
        self.solution_cfg.setdefault('reorder_tgt_dict', self.reorder_tgt_dict)

        self.setup_transform_cfg(dataset_cfg)

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = AsrMetric()
        self.valid_log_buffer = AsrMetric()

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

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        super().build_dataset(dataset_cfg)

        if not self.need_build_data_loader:
            self.train_data_loader = None
            self.valid_data_loader = None
            return

        # LM text data is small, so can set
        # chunk_size to large
        if dataset_cfg.get('chunk_size', 20) < 1000:
            dataset_cfg.chunk_size = 2000

        self.train_data_loader = HDFSDataset(
            self.train_file_list,
            dataset_cfg.bucket_schedule,
            dataset_cfg,
            self.train_item_trans,
            self.draw_batch_fn,
            shuffle=True,
        )

        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            self.val_bucket_schedule,
            dataset_cfg,
            self.valid_item_trans,
            self.draw_batch_fn,
            split_path_list_by_rank=False,
        )

    @torch.no_grad()
    def validation(self):
        '''valid'''
        # Switch to eval model
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        self.valid_log_buffer.reset()
        batch_data = self.valid_data_loader.next()
        self._val_iter = 0
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
        self.call_hook('after_val_epoch')
        self.log_metric(self.valid_log_buffer)
        # switch back to train mode
        self.mode = 'train'
        self.solution.train()

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

    @staticmethod
    def get_test_file_list(dataset_cfg, test_sets):
        '''get test file list.'''
        data_root = dataset_cfg.get('data_root', None)
        test_data_root = dataset_cfg.get('test_data_root', data_root)
        if isinstance(test_data_root, str):
            test_data_root = [test_data_root]
            test_sets = [test_sets]

        test_files = []
        for test_root, test_file in zip(test_data_root, test_sets):
            test_files += [os.path.join(test_root, p) for p in test_file]
        return test_files

    @torch.no_grad()
    def get_ppl(self, test_file, inference_cfg):
        '''ppl calculation of a test set'''
        # data set
        # pylint: disable=not-callable,too-many-locals,too-many-branches
        if hasattr(inference_cfg, 'bucket_schedule'):
            inf_bucket_schedule = inference_cfg.bucket_schedule
        else:
            inf_bucket_schedule = self.dataset_cfg.get('bucket_schedule', '') + ',100000'
        test_data_loader = ValidHDFSDataset(
            [test_file],
            inf_bucket_schedule,
            self.dataset_cfg,
            self.valid_item_trans,
            self.draw_batch_fn_inference,
            split_path_list_by_rank=False,
        )
        test_data_loader.reset()
        batch_data = test_data_loader.next()

        # process every batch
        logp_sum = 0.0
        word_num_sum = 0.0
        utt_num_sum = 0.0
        while batch_data is not None:
            batch_logp = self.solution.get_log_prob(batch_data).sum().item()
            word_num = batch_data['char_mask'].sum().item()
            logp_sum += batch_logp
            word_num_sum += word_num
            utt_num_sum += batch_data['char_mask'].shape[0]

            # next batch
            batch_data = test_data_loader.next()
        test_data_loader.terminate()
        # output to file
        test_name = test_file.split('/')[-1]
        out_stat_file = 'rank{}_logp_result_{}.txt'.format(self.rank, test_name)
        with open(out_stat_file, 'w', encoding='utf-8') as fout:
            fout.write('word num {} logp {} utt num {}'.format(word_num_sum, logp_sum, utt_num_sum))
        # wait all rank finish
        dist_barrier()
        # merge the CER
        # pylint:disable=consider-using-with
        if self.rank == 0:
            in_f_list = [
                open('rank{}_logp_result_{}.txt'.format(rank, test_name), encoding='utf-8')
                for rank in range(self.world_size)
            ]
            total_logp = 0.0
            total_word_num = 0.0
            total_utt_num = 0.0
            for f in in_f_list:
                for line in f:
                    total_word_num += float(line.split()[2])
                    total_logp += float(line.split()[4])
                    total_utt_num += float(line.split()[7])
            ppl = np.exp(-total_logp / total_word_num)
            with open('ppl_result_{}.txt'.format(test_name), 'w', encoding='utf-8') as out_f:
                out_f.write(
                    'ppl: {}\nutt num: {}\nword num: {}'.format(ppl, total_utt_num, total_word_num)
                )
            logging.info('dataset {} ppl: {}'.format(test_name, ppl))

        dist_barrier()

    @torch.no_grad()
    def inference(self):
        '''
        calculate ppl
        '''
        inference_cfg = self.args.inference
        # resume nnlm solution
        self.solution.eval()
        # test data sets
        test_sets = inference_cfg.test_sets.strip().split('|')
        test_file_path = self.get_test_file_list(self.dataset_cfg, test_sets)

        # inference every test set
        inf_draw_batch_cfg = self.dataset_cfg.inference_batch_transform
        for cfg in inf_draw_batch_cfg:
            if cfg.type == 'LMPreCharCollate':
                cfg['tgt_dict'] = self.tgt_dict
            elif cfg.type == 'LMCharCollate':
                cfg['tgt_dict'] = self.tgt_dict
            elif cfg.type == 'PreCharCollate':
                cfg['args'] = self.solution_cfg
                cfg['tgt_dict'] = self.tgt_dict

        # pylint: disable=attribute-defined-outside-init
        self.draw_batch_fn_inference = build_draw_batch_fn(inf_draw_batch_cfg)
        for test_file in test_file_path:
            self.get_ppl(test_file, inference_cfg)
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
            hdfs_put('ppl_*.txt', remote_stat_dir, sync=True)
