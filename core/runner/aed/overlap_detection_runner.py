''' OverlapDetectionRunner '''

import os.path as osp
import numpy as np
import torch
from core.dataset import get_meta
from core.dataset import ValidHDFSDataset, build_draw_batch_fn
from core.runner.asr.base_rnnt_runner import BaseRNNTRunner
from core.utils.misc import get_file_key
from core.utils import logging, compute_recall_precision
from ..base_runner import RUNNERS


@RUNNERS.register_module()
class OverlapDetectionRunner(BaseRNNTRunner):
    '''OverlapDetectionRunner'''

    def pre_build_dataset(self, dataset_cfg):
        '''prepare something before load dataset'''
        super().pre_build_dataset(dataset_cfg)
        meta_data_root = dataset_cfg.get("meta_data_root", dataset_cfg.get("data_root", None))
        if isinstance(meta_data_root, (list, tuple)):
            meta_data_root = meta_data_root[0]
        meta_file = osp.join(meta_data_root, dataset_cfg.meta_file)
        meta_data = get_meta(meta_file)
        self.cmvn_mean = meta_data['cmvn_mean']
        self.cmvn_var = meta_data['cmvn_var']
        self.solution_cfg.setdefault('fbank_dim', dataset_cfg.fbank_dim)
        self.tgt_size = self.solution_cfg.tgt_size
        self.setup_transform_cfg(dataset_cfg)

    def build_test_dataset(self, test_file, inference_cfg):
        '''build test dataset'''
        if hasattr(inference_cfg, 'bucket_schedule'):
            inf_bucket_schedule = inference_cfg.bucket_schedule
        else:
            inf_bucket_schedule = self.dataset_cfg.get('bucket_schedule', '')
        self.test_data_loader = ValidHDFSDataset(
            [test_file],
            inf_bucket_schedule,
            self.dataset_cfg,
            self.parse_fn_eval,
            self.draw_batch_fn_inference,
            split_path_list_by_rank=False,
        )
        self.test_data_loader.reset()

    @torch.no_grad()
    def get_inference_results(self, test_file, inference_cfg):
        '''get original inference results'''
        self.build_test_dataset(test_file, inference_cfg)
        batch_data = self.test_data_loader.next()
        labels = []
        probs = []
        frame_probs = []
        uttids = []
        while batch_data is not None:
            inf_res, frame_res = self.solution.inference(batch_data)
            probs.extend(inf_res)
            labels.extend(batch_data['overlap'])
            uttids.extend(batch_data['uttid'])
            for i, prob in enumerate(inf_res):
                logging.info(
                    "uttid: %s, label %d, predict score: %.3f"
                    % (batch_data['uttid'][i], batch_data['overlap'][i], prob)
                )
                if frame_res is not None:
                    frame_probs.append(np.array(frame_res[i, :].tolist()))
            # next batch
            batch_data = self.test_data_loader.next()
        self.test_data_loader.terminate()
        return uttids, labels, probs, frame_probs

    def inference_once(self, test_file, test_name, inference_cfg):
        '''inference for one test set'''
        uttids, labels, probs, frame_probs = self.get_inference_results(test_file, inference_cfg)
        # inference config
        prob_thres = inference_cfg.get("prob_thres", 0.5)
        min_frame = inference_cfg.get("min_frame", 1)
        # search thres
        if isinstance(prob_thres, str):
            st_et = eval(prob_thres)
            all_prob_thres = np.arange(st_et[0], st_et[1] + st_et[2], st_et[2])
        else:
            all_prob_thres = [prob_thres]
        if isinstance(min_frame, str):
            min_frame = eval(min_frame)
        else:
            min_frame = [min_frame]
        best_f1_score, best_predict, best_dur = self.search_prob_frame_thres(
            min_frame, all_prob_thres, labels, frame_probs, probs
        )
        final_str = 'BEST config: frame %d, thres %.2f, recall %.2f%%, ' % (
            best_f1_score[0],
            best_f1_score[1],
            best_f1_score[2],
        )
        final_str += 'precision %.2f%%, f1-score: %.2f%%' % (best_f1_score[3], best_f1_score[4])
        logging.info(final_str)
        out_file = 'overlap_detection_results_%s.txt' % test_name
        downsampling_size = self.solution_cfg.get("downsampling_size", 4)
        with open(out_file, 'w', encoding='utf-8') as fout:
            for i, uttid in enumerate(uttids):
                fout.write(
                    "%s %d %.3f\n"
                    % (uttid, best_predict[i], best_dur[i] * downsampling_size / 100.0)
                )
            fout.write(final_str)

    @staticmethod
    def frame_predict_to_label(frame_predict, min_frame):
        '''frame-level prediction to sentence-level prediction with event duration'''
        predict_dur = [0]
        cur_dur = 0
        for p in frame_predict:
            if p == 1:
                cur_dur += 1
            else:
                if cur_dur > 0:
                    predict_dur.append(cur_dur)
                cur_dur = 0
        return max(predict_dur) >= min_frame, sum(predict_dur)

    def search_prob_frame_thres(self, min_frame, all_prob_thres, labels, frame_probs, probs):
        '''search all_prob_thres and min_frame for best F1'''
        best_f1_score = [None, None, 0, 0, 0]
        best_predict = None
        best_dur = None
        labels = np.array(labels)
        sen_infer = self.args.inference.get("sen_infer", False)
        if sen_infer:
            probs = np.array(probs)
        for frame in min_frame:
            for prob_thres in all_prob_thres:
                if sen_infer:
                    sen_predict = probs > prob_thres
                predict_label = []
                predict_dur = []
                for i, frame_prob in enumerate(frame_probs):
                    frame_predict = (frame_prob > prob_thres).astype(np.int64)
                    res = self.frame_predict_to_label(frame_predict, frame)
                    if sen_infer:
                        predict_label.append(int(res[0] or sen_predict[i]))
                    else:
                        predict_label.append(int(res[0]))
                    predict_dur.append(max(res[1], 1) if predict_label[-1] else 0)
                predict = np.array(predict_label)
                logging.info('Frame %d, Thres %.2f' % (frame, prob_thres))
                f1_score, recall_rate, precision_rate = compute_recall_precision(
                    labels, predict, show_log=True
                )
                logging.info('-' * 20)
                if f1_score > best_f1_score[4] or best_f1_score[1] is None:
                    best_f1_score = [frame, prob_thres, recall_rate, precision_rate, f1_score]
                    best_predict = predict
                    best_dur = predict_dur
        return best_f1_score, best_predict, best_dur

    @torch.no_grad()
    def inference(self):
        '''inference main function'''
        inference_cfg = self.args.inference
        test_sets = inference_cfg.test_sets.strip().split('|')
        test_file_path = self.get_test_file_list(self.dataset_cfg, test_sets)
        test_names = [get_file_key(test_file) for test_file in test_sets]

        # inference every test set
        inf_draw_batch_cfg = self.dataset_cfg.inference_batch_transform
        self.draw_batch_fn_inference = build_draw_batch_fn(inf_draw_batch_cfg)
        for test_file, test_name in zip(test_file_path, test_names):
            self.inference_once(test_file, test_name, inference_cfg)
