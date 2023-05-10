''' recall, precision and f1 score '''

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from core.utils import logging


def compute_recall_precision(labels, predict, show_log=False):
    '''compute recall, precision and F1-score
    labels and predict are all np.array, where 1 means True
    '''
    total_label_num = np.sum(labels).item()
    total_predict_num = np.sum(predict).item()
    recall_num = np.sum((predict + labels) == 2).item()
    if recall_num == 0 or total_label_num == 0 or total_predict_num == 0:
        return 0.0, 0.0, 0.0
    recall_rate = 100 * recall_num / total_label_num
    precision_rate = 100 * recall_num / total_predict_num
    f1_score = 2 * recall_rate * precision_rate / (recall_rate + precision_rate)
    if show_log:
        recall_str = 'Recall: %d / %d = %.2f%%' % (recall_num, total_label_num, recall_rate)
        logging.info(recall_str)
        precision_str = 'Precision: %d / %d = %.2f%%' % (
            recall_num,
            total_predict_num,
            precision_rate,
        )
        logging.info(precision_str)
        logging.info('F1-Score: %.2f%%' % (f1_score))
    return f1_score, recall_rate, precision_rate


def compute_emotion_metrics(
    predict=None, label=None, confusion_matrix=None, show_log=False, mode=0
):
    '''compute weighted and unweighted recall, precision and F1-score
    labels and predict are all np.array,
    mode 0: input predict, label, output emotion metrics
    mode 1: input predict, label, output confusion_matrix
    mode 2: input confusion_matrix, output emotion metrics
    '''
    if mode < 2:
        if label is None or predict is None:
            raise ValueError("predict or label can not be None")
        emotion_num = label.shape[1]
        data_num = label.shape[0]
        confusion_matrix = np.zeros((emotion_num, emotion_num))
        for j, _ in enumerate(predict):
            confusion_matrix[np.argmax(label[j])][np.argmax(predict[j])] += 1
        if mode == 1:
            return confusion_matrix
    else:
        if confusion_matrix is None:
            raise ValueError("confusion_matrix can not be None")
        emotion_num = confusion_matrix.shape[1]
        data_num = confusion_matrix.sum()

    f1, all_recall, all_accuracy, all_precision = 0, 0, 0, 0
    right_predict = np.zeros(emotion_num, dtype=float)
    true_label = np.zeros(emotion_num, dtype=float)
    all_predict = np.zeros(emotion_num, dtype=float)
    for j, _ in enumerate(confusion_matrix):
        right_predict[j] = confusion_matrix[j][j]
        true_label[j] = confusion_matrix[j].sum()
        all_predict[j] = confusion_matrix[:, j].sum()
    confusion_matrix = confusion_matrix.astype(int)
    if show_log:
        for i in range(emotion_num):
            logging.info(str(list(confusion_matrix[i])))

    w_precision = 0
    w_recall = 0
    w_f1 = 0
    log_pre_emotion = []
    for i in range(emotion_num):
        precision = 0 if all_predict[i] == 0 else right_predict[i] / all_predict[i]
        recall = 0 if true_label[i] == 0 else right_predict[i] / true_label[i]
        f1_score = 0 if recall + precision == 0 else 2 * (precision * recall) / (precision + recall)

        f1 += f1_score
        all_recall += recall
        all_accuracy += right_predict[i]
        all_precision += precision
        w_precision += precision * true_label[i]
        w_recall += recall * true_label[i]
        w_f1 += f1_score * true_label[i]

        if show_log:
            metrics_str = (
                '%d :\t precision: %.4f, \t recall: %.4f\
                \t f1_score: %.4f \t right_predict:  %d \t true_label:  %d\
                \t all_predict: %d'
                % (i, precision, recall, f1_score, right_predict[i], true_label[i], all_predict[i])
            )
            logging.info(metrics_str)
        log_pre_emotion.append(
            [i, precision, recall, f1_score, right_predict[i], true_label[i], all_predict[i]]
        )

    w_precision = w_precision / data_num
    w_recall = w_recall / data_num
    w_f1 = w_f1 / data_num
    if show_log:
        metrics_str = (
            'WA: %.4f \t unweighted_precision(UA): %.4f\
            \t unweighted_recall: %.4f \t  unweighted_f1: %.4f'
            % (
                all_accuracy / data_num,
                all_precision / emotion_num,
                all_recall / emotion_num,
                f1 / emotion_num,
            )
        )
        logging.info(metrics_str)

        weighted_metrics_str = 'weighted_P: %.4f \t weighted_R: %.4f \t weighted_F1: %.4f' % (
            w_precision,
            w_recall,
            w_f1,
        )
        logging.info(weighted_metrics_str)

    return {
        'UA': all_precision / emotion_num,
        'UR': all_recall / emotion_num,
        'UF': f1 / emotion_num,
        'WP': w_precision,
        'WA': w_recall,
        'WF': w_f1,
    }  # UP(UA), UR, UF, WP, WR(WA), WF,


def compute_dimemotion_metrics(x, y, show_log=False):
    '''Computes the metrics CCC, PCC, and RMSE between the sequences x and y
     CCC:  Concordance correlation coeffient
     PCC:  Pearson's correlation coeffient
     RMSE: Root mean squared error
    Input:  x,y: numpy arrays (one-dimensional)
    Output: CCC,PCC,RMSE
    '''
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean((x - x_mean) * (y - y_mean))

    x_var = (
        1.0 / (len(x) - 1) * np.nansum((x - x_mean) ** 2)
    )  # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    y_var = 1.0 / (len(y) - 1) * np.nansum((y - y_mean) ** 2)

    ccc = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    x_std = np.sqrt(x_var)
    y_std = np.sqrt(y_var)

    pcc = covariance / (x_std * y_std)

    rmse = np.sqrt(np.nanmean((x - y) ** 2))

    if show_log:
        metrics_str = (
            'CCC: %.4f \t PCC: %.4f\
            \t RMSE: %.4f'
            % (
                ccc,
                pcc,
                rmse,
            )
        )
        logging.info(metrics_str)

    return {'CCC': ccc, 'RMSE': rmse}


def compute_average_precision(predictions, labels, num_classes):
    '''mAP'''
    if predictions.size == 0:
        return -1, []
    aps = []
    for k in range(num_classes):
        if np.sum(labels[:, k]) > 0:
            avg_precision = average_precision_score(labels[:, k], predictions[:, k], average=None)
            aps.append(avg_precision)
        else:
            aps.append(-1)
    aps_valid = [ap for ap in aps if ap != -1]
    return round(float(np.mean(aps_valid)), 6), aps


def compute_area_under_curve(predictions, labels, num_classes):
    '''AUC'''
    if predictions.size == 0:
        return -1, []
    aucs = []
    for k in range(num_classes):
        if np.sum(labels[:, k]) > 0:
            auc = roc_auc_score(labels[:, k], predictions[:, k], average=None)
            aucs.append(auc)
        else:
            aucs.append(-1)
    aucs_valid = [auc for auc in aucs if auc != -1]
    return round(float(np.mean(aucs_valid)), 6), aucs


def compute_equal_error_rate(predictions, labels, num_classes):
    '''equal error rate.
    ref: https://www.pythonf.cn/read/165447'''
    if predictions.size == 0:
        return -1, []
    eers = []
    for k in range(num_classes):
        if np.sum(labels[:, k]) > 0:
            # pylint: disable=cell-var-from-loop
            fpr, tpr, _ = roc_curve(labels[:, k], predictions[:, k], pos_label=1)
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            eers.append(eer)
        else:
            eers.append(-1)
    eers_valid = [eer for eer in eers if eer != -1]
    return round(float(np.mean(eers_valid)), 6), eers


def compute_d_prime(auc):
    '''d-prime'''
    if auc < 0:
        return -1
    standard_normal = stats.norm()
    d_prime_val = standard_normal.ppf(auc) * np.sqrt(2.0)
    return round(float(d_prime_val), 6)
