''' used to eer test '''

import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from .dcf import get_dcf


class EER:
    '''EER class'''

    # read trial-data from trial path
    @staticmethod
    def read_trials(trial_path):
        '''read trials'''
        trial_data = []
        utts = []
        with open(trial_path, mode='r', encoding='utf-8') as f:
            pairs = f.readlines()
            for pair in pairs:
                pair_list = pair.strip().split(' ')
                label = 0
                # check trial type
                if pair_list[2] == 'nontarget' or pair_list[2] == 'target':
                    if pair_list[2] == 'target':
                        label = 1
                else:
                    label = int(pair_list[2])

                trial_data.append([pair_list[0], pair_list[1], label])
                utts.append(pair_list[0])
                utts.append(pair_list[1])
        utts = list(set(utts))
        return trial_data, len(utts)

    def get_scores(self, trial_data, embedding: dict):
        '''get scores'''
        # make scores from embedding following trial-datas
        scores = []
        for pair in trial_data:
            scores.append(self.cosine_similarity(embedding[pair[0]], embedding[pair[1]]))
        return scores

    def calculate_evaluation(self, labels, scores):
        '''

        Args:
            labels: The labels of each pairs from sid trials
            scores: The scores of each pairs from sid trials

        Returns:


        note that: labels is trial_data[:,2]
        used labels = np.array([pairs[2] for pairs in trial_data])
        '''
        # calculate evaluation matrix for SID
        # calculate EER
        eer, thresh = self.cal_eer(labels, scores)
        # calculate SRE10
        dcf_001 = get_dcf(labels, scores, c_miss=10.0, c_fa=1.0, p_target=0.01)
        # calculate SRE16
        dcf_0001 = get_dcf(labels, scores, c_miss=1.0, c_fa=1.0, p_target=0.001)
        return eer, thresh, dcf_001, dcf_0001

    @staticmethod
    def cal_eer(y_true, y_pred):
        '''cal eer'''
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        thresh = interp1d(fpr, thresholds)(eer)
        return eer, thresh

    @staticmethod
    def cosine_similarity(a, b):
        '''CosineSimilarity'''
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return sim
