"""Wav2vecCtcRunner"""

import os
import os.path as osp
import time
import math
import torch
import numpy as np
from kaldiio import WriteHelper
from core.utils import logging, get_rank, get_world_size, dist_barrier
from core.dataset import (
    HDFSDataset,
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
from core.utils.dist_hdfs import dist_hdfs_get
from core.dataset.dictionary import ScpDictionary
from core.utils.cer.cer_metric import EditDistanceCalculator, TextFormator, output_result
from core.runner.asr.base_rnnt_runner import BaseRNNTRunner
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class Wav2vecCtcRunner(BaseRNNTRunner):
    """Wav2vecCtcRunner"""

    def setup_transform_cfg(self, dataset_cfg):
        '''setup transform config.'''
        transform_cfgs = (
            dataset_cfg.train_item_transform,
            dataset_cfg.valid_item_transform,
            dataset_cfg.get('train_device_transform', []),
        )
        for transform_cfg in transform_cfgs:
            for cfg in transform_cfg:
                if cfg.type == 'W2vPhone2charLabel':
                    cfg['tgt_dict'] = self.tgt_dict
                    cfg['lexicon'] = self.lexicon

        for cfg in dataset_cfg.batch_transform:
            if cfg.type == 'CharCollate':
                cfg['tgt_dict'] = self.tgt_dict

    def pre_build_dataset(self, dataset_cfg):
        """pre_build_dataset"""
        tgt_dict_file = os.path.join(
            dataset_cfg.data_root[0].replace('hdfs_data', ''), dataset_cfg.tgt_dict_dir
        )
        dist_hdfs_get(tgt_dict_file)
        self.tgt_dict = ScpDictionary.load(dataset_cfg.tgt_dict_dir)
        self.solution_cfg.setdefault('tgt_dict', self.tgt_dict)
        self.tgt_vocab_size = len(self.tgt_dict)

        # align to 8
        self.tgt_vocab_size = ((self.tgt_vocab_size + 8 - 1) // 8) * 8
        logging.info('Dictionary total vocab size %d', self.tgt_vocab_size)
        self.solution_cfg.setdefault('tgt_vocab_size', self.tgt_vocab_size)

        # setup lexicon
        lexicon_file = os.path.join(
            dataset_cfg.data_root[0].replace('hdfs_data', ''), dataset_cfg.lexicon
        )
        dist_hdfs_get(lexicon_file)
        lexicon = dict()
        with open(dataset_cfg.lexicon, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split()
                lexicon[line[0]] = line[1:]  # word to phones
        self.lexicon = lexicon

        # build wfst graph
        dist_hdfs_get(self.solution_cfg.wfst_decoder, local_file='./latgen.ken')
        dist_hdfs_get(self.solution_cfg.wfst_graph, local_dir='wfst_graph/')
        os.system('chmod u+x ./latgen.ken')
        self.solution_cfg.wfst_graph = (
            f'wfst_graph/{os.path.basename(self.solution_cfg.wfst_graph)}'
        )

        self.setup_transform_cfg(dataset_cfg)

    def build_dataset(self, dataset_cfg):
        '''build data set'''
        self.pre_build_dataset(dataset_cfg)

        parse_eval_cfg = dataset_cfg.valid_item_transform
        self.parse_fn_eval = build_item_augmentation(parse_eval_cfg, self.meta_data)

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

    def build_test_dataset(self, test_file):
        '''build inference dataset'''
        if hasattr(self.dataset_cfg, 'bucket_schedule_val'):
            val_bucket_schedule = self.dataset_cfg.bucket_schedule_val
        else:
            val_bucket_schedule = self.dataset_cfg.bucket_schedule

        valid_split_each_dataset = self.solution_cfg.get('valid_multi_cer', False)
        self.test_data_loader = ValidHDFSDataset(
            [test_file],
            val_bucket_schedule,
            self.dataset_cfg,
            self.parse_fn_eval,
            self.draw_batch_fn,
            split_path_list_by_rank=False,
            split_each_dataset=valid_split_each_dataset,
        )
        self.test_data_loader.reset()

    @torch.no_grad()
    def validation(self):
        '''valid'''
        self.mode = 'val'
        self.solution.eval()
        self.call_hook('before_val_epoch')
        self.valid_data_loader.reset()
        self._val_iter = 0
        if self.solution_cfg.get('valid_multi_cer', False):
            dataset_names = [osp.basename(p) for p in self.valid_data_loader.origin_path_list]
        else:
            dataset_names = [None]
        for dataset_name in dataset_names:
            self.valid_log_buffer.reset()
            batch_data = self.valid_data_loader.next()
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
                    torch.cuda.empty_cache()
                    logging.error("rank %d catch %s", self.rank, str(e), exc_info=True)
                    continue
            self.log_metric(self.valid_log_buffer, name=dataset_name)
        self.mode = 'train'
        self.solution.train()
        self.call_hook('after_val_epoch')

    def inference(self):
        """inference"""
        # search params to get best am_scale and blk_scale
        if self.args.inference.get('search_params', True):
            self.get_data_list(self.dataset_cfg)
            self.test_data_name = os.path.basename(self.valid_file_list[0])
            self.test_data_loader = self.valid_data_loader
            self.search_params()

        test_sets = self.args.inference.test_sets.strip().split('###')
        test_file_path = self.get_test_file_list(self.dataset_cfg, test_sets)
        for name, path in zip(test_sets, test_file_path):
            self.test_data_name = name
            self.build_test_dataset(path)
            self.inference_once()

        if get_rank() == 0:
            self.calculate_inference_res(test_sets)

    def search_params(self):
        """search_params"""
        if self.solution_cfg.w2l_decoder == 'wfst':
            self.search_params_wfst()
            return

        lmweight_range = np.arange(0, 8.1, 1)
        wordscore_range = np.arange(-12, 3.1, 2)

        params = self._search_params(lmweight_range, wordscore_range)
        best_param = sorted(params, key=lambda a: a[2])[0]
        logging.info('First iter best param: %r', best_param)

        self.set_lm_params(best_param)

    def _inference_param(self, lmweight, wordscore, name):
        """_inference_param"""
        self.solution_cfg.lm_weight = lmweight
        self.solution_cfg.word_score = wordscore
        self.solution.criterion_module.build_decoder(self.solution_cfg)

        self.inference_once()
        with os.popen(f'tail -n1 cer_rlt_top1_{name}.txt') as fd:
            wer = float(fd.read().split()[1].split('%')[0])
        return wer

    def _search_params(self, lmweight_range, wordscore_range):
        '''searching the decoding params on dev set'''
        name = self.test_data_name
        param_list = []
        for lmweight in lmweight_range:
            for wordscore in wordscore_range:
                if self.solution_cfg.w2l_decoder == 'wfst':
                    wer = self._inference_param_wfst(lmweight, wordscore, name)
                else:
                    wer = self._inference_param(lmweight, wordscore, name)

                param = (lmweight, wordscore, wer)
                logging.info('search param: %r', param)
                param_list += [param]

        return param_list

    def search_params_wfst(self):
        """search_params_wfst"""
        am_scale_range = np.around(np.arange(0.2, 1.01, 0.2), decimals=2)
        blk_scale_range = np.around(np.arange(0.2, 1.01, 0.2), decimals=2)

        params = self._search_params(am_scale_range, blk_scale_range)
        best_param = sorted(params, key=lambda a: a[2])[0]

        # refine params
        amscale, blkscale, _wer = best_param
        am_scale_range = np.around(np.arange(amscale - 0.1, amscale + 0.11, 0.05), decimals=2)
        blk_scale_range = np.around(np.arange(blkscale - 0.1, blkscale + 0.11, 0.05), decimals=2)
        params1 = self._search_params(am_scale_range, blk_scale_range)
        best_param = sorted(params + params1, key=lambda a: a[2])[0]

        logging.info('Param search iter 1: %r', '\n'.join(str(param) for param in params))
        logging.info('Param search iter 2: %r', '\n'.join(str(param) for param in params1))
        logging.info('Best param: %r', best_param)
        self.set_lm_params(best_param)

    def _inference_param_wfst(self, am_scale, blk_scale, name):
        """_inference_param_wfst"""
        self.args.inference.am_scale = am_scale
        self.args.inference.blk_scale = blk_scale

        self.inference_once()
        with os.popen(f'tail -n1 cer_rlt_top1_{name}.txt') as fd:
            wer = float(fd.read().split()[1].split('%')[0])
        return wer

    def inference_once(self):
        """inference_once"""
        if self.solution_cfg.w2l_decoder == 'wfst':
            self.inference_once_wfst()
            return

        sot = time.time()
        self.test_data_loader.reset()
        batch_data = self.test_data_loader.next()

        ed_calculator = EditDistanceCalculator()

        lang = self.solution_cfg.lang_formator
        formator = TextFormator(lang)
        top1_align_info_list = []
        top1_ed_info_list = []

        decode_count = 0
        batch_count = 0
        self.solution.eval()
        while batch_data is not None:
            batch_count += 1
            with torch.no_grad():
                _prefix_tokens = None
                hyps = self.solution.inference(batch_data)
                for bid, hyp in enumerate(hyps):
                    decode_count += 1
                    if decode_count % 50 == 0:
                        logging.info("rank:{} decode {} sentence".format(get_rank(), decode_count))

                    curr_utt_id = batch_data['utt_id'][bid]
                    ref = batch_data['text'][bid]
                    ref_format = formator(ref.lower().strip())

                    res_format = formator(hyp.lower().strip())
                    ed_info, align_info = ed_calculator.show_alignment(
                        curr_utt_id, ref_format, res_format
                    )

                    curr_ins_err = int(ed_info['ins_err'])
                    curr_del_err = int(ed_info['del_err'])
                    curr_sub_err = int(ed_info['sub_err'])
                    _curr_err_count = curr_ins_err + curr_del_err + curr_sub_err

                    top1_ed_info_list.append(ed_info)
                    top1_align_info_list.append(align_info)

            batch_data = self.test_data_loader.next()

        top1_out_stat_file = 'rank{}_cer_rlt_top1_{}.txt'.format(get_rank(), self.test_data_name)

        output_result(top1_align_info_list, top1_ed_info_list, top1_out_stat_file, lang=lang)
        dist_barrier()
        if get_rank() == 0:
            self.merge_rnnt_rlt(key='top1')

        eot = time.time()
        logging.info(
            "RNN-T inference testset {} cost time {} seconds".format(self.test_data_name, eot - sot)
        )

    def set_lm_params(self, best_param):
        '''set lm parameters'''
        am_scale, blk_scale, _wer = best_param
        self.args.inference.am_scale = am_scale
        self.args.inference.blk_scale = blk_scale

    @staticmethod
    def filter_batch(batch_data, prefix):
        """filter_batch"""
        if prefix is None or len(prefix) == 0:
            return batch_data
        mask = [utt.startswith(prefix) for utt in batch_data['utt_id']]
        th_mask = torch.tensor(mask, dtype=torch.bool)
        new_batch_data = {}
        for key, val in batch_data.items():
            if isinstance(val, torch.Tensor):
                th_mask = th_mask.to(val.device)
                new_batch_data[key] = val[th_mask]
            else:
                assert isinstance(val, list)
                new_batch_data[key] = [v for v, m in zip(val, mask) if m]
        if len(new_batch_data['utt_id']) == 0:
            return None
        return new_batch_data

    def inference_once_wfst(self, filter_prefix=None):
        """inference_once_wfst"""
        # pylint: disable=too-many-locals
        sot = time.time()
        self.test_data_loader.reset()
        batch_data = self.test_data_loader.next()

        ed_calculator = EditDistanceCalculator()

        lang = self.solution_cfg.lang_formator
        formator = TextFormator(lang)
        top1_align_info_list = []
        top1_ed_info_list = []

        decode_count = 0
        batch_count = 0
        self.solution.eval()

        rank = get_rank()
        graph = self.solution_cfg.wfst_graph
        blk_scale = math.log(self.args.inference.blk_scale)
        writer = WriteHelper(
            f"ark:|"
            f"./latgen.ken --acoustic-scale {self.args.inference.am_scale} "
            "--lattice-beam 5 --beam 50 "
            f"--rescore '{graph}/G.ken.small|{graph}/G.ken' "
            "--max-active 1500 --min-active 1 --num-threads 10 "
            f"{graph}/tokens.txt {graph}/words.txt {graph}/lexicon.txt {graph}/TLG.fst "
            f"- units{rank}.out words{rank}.out lattice{rank}.out"
        )

        # words
        word_dict = {}
        with open(f'{graph}/words.txt', 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split()
                word_dict[int(words[1])] = words[0]

        ref_dict = {}
        while batch_data is not None:
            batch_data = self.filter_batch(batch_data, filter_prefix)
            while batch_data is None:
                batch_data = self.test_data_loader.next()
                if batch_data is None:
                    break
                batch_data = self.filter_batch(batch_data, filter_prefix)
            if batch_data is None:
                break

            batch_count += 1
            with torch.no_grad():
                logits, lengths = self.solution.inference(batch_data)
                logits[:, :, 0] += blk_scale

            for i in range(lengths.size(0)):
                decode_count += 1
                if decode_count % 50 == 0:
                    logging.info("rank:{} decode {} sentence".format(get_rank(), decode_count))
                utt = batch_data['utt_id'][i]
                ref_dict[utt] = batch_data['text'][i]
                if filter_prefix and not utt.startswith(filter_prefix):
                    continue
                writer(utt, logits[i][: lengths[i]].numpy())

            batch_data = self.test_data_loader.next()
        writer.close()

        # now we get all decoded hyps

        # read decoded outputs
        with open(f'words{rank}.out', 'r', encoding='utf-8') as f:
            for line in f:
                rec = line.split()
                curr_utt_id = rec[0]
                words = [word_dict[int(i)] for i in rec[1:]]
                hyp = ' '.join(words)
                ref = ref_dict[curr_utt_id]

                ref_format = formator(ref.lower().strip())
                res_format = formator(hyp.lower().strip())
                ed_info, align_info = ed_calculator.show_alignment(
                    curr_utt_id, ref_format, res_format
                )

                curr_ins_err = int(ed_info['ins_err'])
                curr_del_err = int(ed_info['del_err'])
                curr_sub_err = int(ed_info['sub_err'])
                _curr_err_count = curr_ins_err + curr_del_err + curr_sub_err

                top1_ed_info_list.append(ed_info)
                top1_align_info_list.append(align_info)

        top1_out_stat_file = 'rank{}_cer_rlt_top1_{}.txt'.format(get_rank(), self.test_data_name)

        output_result(top1_align_info_list, top1_ed_info_list, top1_out_stat_file, lang=lang)
        dist_barrier()
        if get_rank() == 0:
            self.merge_rnnt_rlt(key='top1')

        eot = time.time()
        logging.info(
            "RNN-T inference testset {} cost time {} seconds".format(self.test_data_name, eot - sot)
        )

    def merge_rnnt_rlt(self, key='top1', path='.'):
        """merge_rnnt_rlt"""
        # pylint:disable=consider-using-with
        in_f_list = [
            open(
                '{}/rank{}_cer_rlt_{}_{}.txt'.format(path, rank, key, self.test_data_name),
                encoding='utf-8',
            )
            for rank in range(get_world_size())
        ]
        out_f = open(
            '{}/cer_rlt_{}_{}.txt'.format(path, key, self.test_data_name), 'w', encoding='utf-8'
        )
        total_words = 0
        total_correct = 0
        total_err = 0
        total_ins = 0
        total_del = 0
        total_sub = 0
        for f in in_f_list:
            for line in f:
                if 'TOTAL Words:' in line:
                    items = line.split(':')
                    total_words += int(items[1].split()[0])
                    total_correct += int(items[2].split()[0])
                    total_err += int(items[3].split()[0])
                elif 'TOTAL Insertions:' in line:
                    items = line.split(':')
                    total_ins += int(items[1].split()[0])
                    total_del += int(items[2].split()[0])
                    total_sub += int(items[3].split()[0])
                elif 'CER:' in line:
                    pass
                elif 'SER:' in line:
                    pass
                else:
                    out_f.write(line)
        #
        cer = total_err / float(total_words)
        out_f.write(
            "TOTAL Words: {} Correct: {} Errors: {}\n".format(total_words, total_correct, total_err)
        )
        out_f.write(
            "TOTAL Insertions: {} Deletions: {} Substitutions: {}\n".format(
                total_ins, total_del, total_sub
            )
        )
        out_f.write("CER: %.2f%%, CCR: %.2f%%\n" % (cer * 100, (1 - cer) * 100))
        out_f.close()

    @staticmethod
    def calculate_inference_res(test_sets):
        '''calculate inference result'''
        cer_map = {}
        total_cer = 0

        logging.info('Top1 Insertions/Deletions/Substitutions')
        for name in test_sets:
            # pylint:disable=consider-using-with
            f = open('cer_rlt_top1_{}.txt'.format(name), encoding='utf-8')
            for line in f:
                if 'CER:' in line:
                    cur_cer = float(line.split(':')[1].split('%')[0].strip())
                    cer_map[name] = cur_cer
                    total_cer += cur_cer
                elif 'TOTAL Insertions:' in line:
                    items = line.split(':')
                    total_ins = int(items[1].split()[0])
                    total_del = int(items[2].split()[0])
                    total_sub = int(items[3].split()[0])
                    logging.info('{}: {}/{}/{}'.format(name, total_ins, total_del, total_sub))
            f.close()

        logging.info('\nTop1 wer:')
        for name, cer in cer_map.items():
            logging.info('{}: {}'.format(name, cer))
        logging.info("Average: %r", total_cer / len(test_sets))
