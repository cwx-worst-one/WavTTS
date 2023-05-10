''' KwsRnntRunner '''
import io
import os.path as osp
import torch
import kaldi_io
from subword_nmt.apply_bpe import BPE
from core.dataset import ValidHDFSDataset, get_meta, build_draw_batch_fn
from core.runner.metric.kws_metric import KwsRnntMetric
from core.utils import dist_barrier, hdfs_put, hdfs_mkdir, hdfs_test, logging
from .base_kws_runner import BaseKwsRunner
from ..base_runner import RUNNERS
from ...utils import logging


@RUNNERS.register_module()
class KwsRnntRunner(BaseKwsRunner):
    '''RNNT for KWS'''

    def setup_transform_cfg(self, dataset_cfg):
        '''setup transform config.'''
        for cfg in dataset_cfg.batch_transform:
            if cfg.type == 'PreCharCollate':
                cfg['args'] = self.solution_cfg

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        super().pre_build_dataset(dataset_cfg)
        meta_data_root = dataset_cfg.get("meta_data_root", dataset_cfg.get("data_root", None))
        if isinstance(meta_data_root, str):
            meta_file = osp.join(meta_data_root, dataset_cfg.meta_file)
        else:
            meta_file = osp.join(meta_data_root[0], dataset_cfg.meta_file)
        meta_data = get_meta(meta_file)
        self.tgt_dict = meta_data['tgt_dict']
        self.cmvn_mean = meta_data['cmvn_mean']
        self.cmvn_var = meta_data['cmvn_var']
        if dataset_cfg.get('use_bpe', False):
            total_code = meta_data['total.code']
            self.bpe_fn = BPE(io.StringIO(total_code))
            self.cmvn_mean = meta_data['cmvn_mean']
            self.cmvn_var = meta_data['cmvn_var']
        is_reorder_dict = self.solution_cfg.get('reorder_dict_by_freq', 0)
        if is_reorder_dict:
            self.reorder_tgt_dict = meta_data['reorder_tgt_dict']
            self.reorder_dict_map = meta_data['reorder_dict_map']
        else:
            self.reorder_tgt_dict = None
            self.reorder_dict_map = None

        self.solution_cfg.setdefault('fbank_dim', dataset_cfg.fbank_dim)
        self.solution_cfg.setdefault('tgt_dict', self.tgt_dict)
        self.tgt_vocab_size = dataset_cfg.tgt_vocab_size
        # self.tgt_vocab_size = len(self.tgt_dict)
        # # align to 8
        # self.tgt_vocab_size = ((self.tgt_vocab_size + 8 - 1) // 8) * 8
        logging.info('Dictionary total vocab size %d', self.tgt_vocab_size)
        self.solution_cfg.setdefault('tgt_vocab_size', self.tgt_vocab_size)
        self.in_out_ratio = dataset_cfg.get('in_out_ratio', 3)
        self.solution_cfg.setdefault('reorder_tgt_dict', self.reorder_tgt_dict)

        self.setup_transform_cfg(dataset_cfg)

    def build_metrics(self):
        '''build metric'''
        self.train_log_buffer = KwsRnntMetric()
        self.valid_log_buffer = KwsRnntMetric()

    def get_keyword_ids(self, keyword_phones):
        '''get keyword ids'''
        phones = keyword_phones.split(',')
        ids = [0]
        for phone in phones:
            ids.append(self.tgt_dict[phone])
        return torch.tensor(ids).long()

    @staticmethod
    def merge_prob(world_size, test_name):
        '''merge prob'''
        # pylint:disable=consider-using-with
        prob_handler = open('prob_{}.ark'.format(test_name), 'wb')
        for i in range(world_size):
            curr_prob_file = 'rank{}_prob_{}.ark'.format(i, test_name)
            for utt_id, probs in kaldi_io.read_mat_ark(curr_prob_file):
                kaldi_io.write_mat(prob_handler, probs, key=utt_id)
        prob_handler.close()

    @torch.no_grad()
    def prob_inference(self, test_file, inference_cfg):
        '''inference a test set'''
        # data set
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
        test_name = test_file.split('/')[-1]
        # pylint:disable=consider-using-with
        prob_handler = open('rank{}_prob_{}.ark'.format(self.rank, test_name), 'wb')
        # inference config
        keyword_phones = inference_cfg.get('keyword_phones', 'da,li,da,li')
        keyword_ids = self.get_keyword_ids(keyword_phones)
        keep_valid_probs = inference_cfg.get('keep_valid_probs', True)
        utt_cnt = 0
        while batch_data is not None:
            probs, mask = self.solution.prob_beam_forward(batch_data, keyword_ids, keep_valid_probs)
            bsz = probs.size(0)
            utt_cnt += bsz
            for bid in range(bsz):
                curr_mask = mask[bid]
                valid_length = int(curr_mask.sum().item())
                curr_prob = probs[bid, :, 0:valid_length, :]
                curr_prob = torch.flatten(curr_prob, 0, 1)
                curr_length = curr_prob.size(0)
                curr_utt_id = batch_data['uttid'][bid]
                kaldi_io.write_mat(
                    prob_handler, curr_prob[0:curr_length, :].cpu().numpy(), key=curr_utt_id
                )
            logging.info('rank {}, processed {} utts'.format(self.rank, utt_cnt))
            # next batch
            batch_data = test_data_loader.next()
        prob_handler.close()
        test_data_loader.terminate()
        dist_barrier()
        # merge the CER
        if self.rank == 0:
            self.merge_prob(self.world_size, test_name)
        dist_barrier()
        logging.info(
            'rank %d, testset %s, utt num %d, inference done', self.rank, test_file, utt_cnt
        )

    @torch.no_grad()
    def inference(self):
        '''inference'''
        inference_cfg = self.args.inference
        # resume RNN-T solution
        self.solution.skeleton_model.eval()
        # test data sets
        test_sets = inference_cfg.test_sets.strip().split('|')
        test_file_path = self.get_test_file_list(self.dataset_cfg, test_sets)

        # inference every test set
        inf_draw_batch_cfg = self.dataset_cfg.inference_batch_transform
        self.draw_batch_fn_inference = build_draw_batch_fn(inf_draw_batch_cfg)
        inference_type = inference_cfg.get('inference_type', 'prob')
        for test_file in test_file_path:
            if inference_type == 'prob':
                self.prob_inference(test_file, inference_cfg)
            else:
                logging.info('{} inference not supported yet'.format(inference_type))
        remote_stat_dir = inference_cfg.get('remote_stat_dir', '')
        if remote_stat_dir:
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
            hdfs_put('prob_*', remote_stat_dir, sync=True)
