import pickle
import mir_eval
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from recipes.structure.preprocess.utils import get_boundary_labels

EVAL_METRICS = [
    "summary",
    "valid_loss",
    "HR5F",
    "HR30F",
    "mHR5F",
    "mHR30F",
    "ACC0",
    "ACC",
    "Intro",
    "Inst",
    "Verse",
    "ChorusF1",
    "Chorus",
    "ChoBoF1",
]

SUMMARY_WEIGHTS = {"HR5F": 2, "ACC": 2, "ChorusF1": 1, "mHR5F": 1}


def merge_temporal_prob(prob, window_stride=5):
    T = prob.shape[0]
    L = prob.shape[1]
    total_len = int((T - 1) * window_stride + L)
    list_time_dict_prob = []
    for t in range(T):
        time_dict_prob = {}
        start = int(t * window_stride)
        for i in range(L):
            time_dict_prob[start + i] = prob[t, i]
        list_time_dict_prob.append(time_dict_prob)

    final_prob = []
    for gt in range(total_len):
        gt_probs = []
        for t in range(T):
            if gt in list_time_dict_prob[t]:
                gt_probs.append(list_time_dict_prob[t][gt])
        if gt_probs:
            final_prob.append(np.mean(gt_probs))
        else:
            final_prob.append(0.0)

    return np.array(final_prob)


def merge_multiple_temporal_probs(probs, window_stride=5):
    multi_probs = merge_temporal_prob(probs[:, :, 0], window_stride)
    for i in range(1, probs.shape[2]):
        prob = merge_temporal_prob(probs[:, :, i], window_stride)
        multi_probs = np.vstack((multi_probs, prob))
    return multi_probs.T


def select_peaks(prob, top=10, hop_sec=0.2, window_sec=12, step_sec=5, duration=-1):
    prob = prob.astype(np.float64)
    if duration > 0:
        prob = prob[: int(round(duration / hop_sec))]

    top = int(round(len(prob) * hop_sec / 180 * top))  # 180 secones
    window_len = int(round(window_sec / hop_sec))
    cands = {}

    start_idx = 0
    for i in range(start_idx, len(prob), int(step_sec / hop_sec)):
        idx = int(np.argmax(prob[i : i + window_len])) + i  # i = 0 is at 1 * hop_sec
        cands[idx] = prob[idx]
    # cands[idx+1] because time starts at 1 but array index starts at 0

    if len(cands) == 0:
        return np.array([])

    if window_sec > 10:  # for regular mode
        for i in cands.keys():
            avg = np.mean(
                prob[max(i - window_len, 0) : min(i + window_len // 2, len(prob))]
            )
            cands[i] -= avg

    sorted_cands = sorted(cands.items(), key=lambda x: x[0])  # sorted by time
    rm_idx = []
    for i in range(len(sorted_cands) - 1):
        if sorted_cands[i + 1][0] - sorted_cands[i][0] < 3 / hop_sec:
            if sorted_cands[i + 1][1] > sorted_cands[i][1]:
                rm_idx.append(i)
            else:
                rm_idx.append(i + 1)
    for i in sorted(rm_idx, reverse=True):
        del sorted_cands[i]

    # sorted_idx = np.argsort(np.array(sorted_cands))
    if sorted_cands[0][0] > 2 / hop_sec:  # 2 secones
        sorted_cands.append((0, 1.0))  # add the start boundary at 0
    else:
        sorted_cands[0] = (0, 1.0)  # move the first boundary to 0 index

    sorted_cands = sorted(sorted_cands, key=lambda x: -x[1])  # sorted by probabilities
    top_cands = sorted_cands[:top]
    top_index = sorted(top_cands, key=lambda x: x[0])  # sorted by times
    cands_index = sorted(sorted_cands, key=lambda x: x[0])  # sorted by times

    return top_index, cands_index  # list of (index, boundary_prob)


def predict_segment(
    index_prob,
    funct_prob,
    hop_sec=0.192,
    forChorus=False,
    segment_classes=["silence", "chorus", "verse", "bridge", "inst", "outro", "intro"],
):
    """
    index_prob: list of (index, boundary_prob) for boundary
    funct_prob: T-by-9 probability matrix of segment function
    beat_times: list of beat timestamps in second (for adjustmemt)
    """

    if type(index_prob) is not list:
        index_prob = index_prob.astype(np.float64)
    if type(funct_prob) is not list:
        funct_prob = funct_prob.astype(np.float64)

    raws = []
    count = 0
    for i in range(len(index_prob) - 1):
        if index_prob[i + 1][0] > index_prob[i][0]:
            p = np.mean(funct_prob[index_prob[i][0] : index_prob[i + 1][0], :], axis=0)
            funct = int(np.argmax(p))
            if forChorus and funct != 1:
                funct = 0
                funct_name = "other"
            else:
                funct_name = segment_classes[funct]

            raws.append(
                {
                    "interval": [index_prob[i][0], index_prob[i + 1][0]],
                    "duration": (index_prob[i + 1][0] - index_prob[i][0]) * hop_sec,
                    "start_prob": round(index_prob[i][1], 4),
                    "end_prob": round(index_prob[i + 1][1], 4),
                    "funct_prob": round(p.tolist()[funct], 4),
                    "funct_label": funct,
                    "funct_name": funct_name,
                    "res_duration": (len(funct_prob) - 1 - index_prob[i][0]) * hop_sec,
                }
            )
            count += 1

    pred_labels = np.zeros(
        funct_prob.shape[0], dtype=int
    )  # initialized with all 'silence' index = 0
    for seg in raws:
        pred_labels[seg["interval"][0] : seg["interval"][1]] = seg["funct_label"]

    # merging consecutive raw segments with the same labels
    segments = []
    ss = 0
    for i in range(len(raws)):
        if i == len(raws) - 1 or raws[i]["funct_label"] != raws[i + 1]["funct_label"]:
            ee = i
            funct = raws[i]["funct_label"]
            start = raws[ss]["interval"][0]
            end = raws[ee]["interval"][1]
            p = np.mean(funct_prob[start:end, funct])
            segment = {
                "interval": [round(start * hop_sec, 3), round(end * hop_sec, 3)],
                "duration": round((end - start) * hop_sec, 3),
                "funct_label": funct,
                "funct_prob": round(p, 5),
                "funct_name": segment_classes[funct],
                "start_prob": raws[ss]["start_prob"],
                "end_prob": raws[ee]["end_prob"],
                "res_duration": (len(funct_prob) - 1 - start) * hop_sec,
            }
            segments.append(segment)
            ss = i + 1

    for r in raws:
        r["interval"][0] = round(r["interval"][0] * hop_sec, 3)
        r["interval"][1] = round(r["interval"][1] * hop_sec, 3)

    return segments, raws, pred_labels


def post_process(
    probs_bound,
    probs_funct,
    n_top_bound,
    label_hop,
    downbeat_align=False,
    beat_pred=None,
):

    top_index, _ = select_peaks(probs_bound[:, 0], top=n_top_bound, hop_sec=label_hop)
    segments, raw_segments, pred_labels = predict_segment(
        top_index, probs_funct, hop_sec=label_hop
    )

    choruses = [seg for seg in raw_segments if seg["funct_name"] == "chorus"]

    pred_choruses = []
    for pre_label in pred_labels:
        if pre_label == 1:
            pred_choruses.append(1)
        else:
            pred_choruses.append(0)

    return pred_labels, pred_choruses, segments, choruses, raw_segments


def eval_boundary_f1(ann_inter, est_inter, key, ChorusOnly=False):
    ann_inter = np.array(ann_inter)
    est_inter = np.array(est_inter)
    if min(len(ann_inter), len(est_inter)) == 0:
        return None, None
    if not ChorusOnly and not np.allclose(est_inter.max(), ann_inter.max()):
        if ann_inter[-1][-1] > est_inter[-1][-1]:
            est_inter = np.concatenate(
                (est_inter, np.array([[est_inter[-1][-1], ann_inter[-1][-1]]])), axis=0
            )
        elif ann_inter[-1][-1] == est_inter[-1][-1]:
            pass
        else:
            idx = len(est_inter)
            while ann_inter[-1][-1] <= est_inter[idx - 1][0]:
                idx -= 1
            est_inter = est_inter[:idx]
            est_inter[-1][-1] = ann_inter[-1][-1]
    _, _, f1_5 = mir_eval.segment.detection(
        ann_inter, est_inter, window=0.5, trim=False
    )
    _, _, f1_30 = mir_eval.segment.detection(ann_inter, est_inter, window=3, trim=False)
    return f1_5, f1_30


def function_accuracy(prob_true, prob_pred):
    """
    prob_true: T-by-7 probability matrix
    """
    y0 = np.sum(prob_true, 1)
    y_valid = y0 > 0
    if len(prob_pred.shape) > 1:
        acc = accuracy_score(
            np.argmax(prob_true[y_valid, :], 1), np.argmax(prob_pred[y_valid, :], 1)
        )
    else:
        acc = accuracy_score(np.argmax(prob_true[y_valid, :], 1), prob_pred[y_valid])
    return acc


def framewise_f1(pred_binary, truth_label):
    truth_binary = np.zeros((truth_label.shape), dtype=np.int)
    truth_binary[truth_label >= 0.999] = 1
    return f1_score(
        truth_binary, pred_binary, labels=np.unique(pred_binary), average="micro"
    )


def chorus_bound_framewise_f1(pred_binary, intervals, label_hop, coverage=5):
    if not intervals:  # no chorus timestamps
        return None
    cover_len = int(np.round(coverage / label_hop))
    preds = np.empty(0)
    trues = np.empty(0)
    for inter in intervals:
        idx = int(np.round(inter[0] / label_hop))  # front
        start = max(idx - cover_len, 0)
        end = min(idx + cover_len, len(pred_binary) - 1)
        trues = np.append(trues, np.zeros(max(idx - start, 0)))
        trues = np.append(trues, np.ones(max(end - idx + 1, 0)))
        preds = np.append(preds, pred_binary[start : end + 1])
        idx = int(np.round(inter[1] / label_hop))  # rare
        start = max(idx - cover_len, 0)
        end = min(idx + cover_len, len(pred_binary) - 1)
        trues = np.append(trues, np.ones(max(idx - start + 1, 0)))
        trues = np.append(trues, np.zeros(max(end - idx, 0)))
        preds = np.append(preds, pred_binary[start : end + 1])
    if len(preds) > len(trues):
        preds = preds[: len(trues)]
    elif len(trues) > len(preds):
        trues = trues[: len(preds)]
    return framewise_f1(preds, trues)


def binarize_prob(prob, threshold=0.9999):
    binary = np.zeros((prob.shape), dtype=np.int32)
    binary[prob >= threshold] = 1
    return binary


def aroc_ap(truth_labels, pred_probs):
    truth_binary = binarize_prob(truth_labels)
    if min(len(truth_binary.shape), len(pred_probs.shape)) < 2:
        if np.sum(truth_binary) != 0:
            auc = np.round(roc_auc_score(truth_binary, pred_probs), 4)
            aprec = np.round(average_precision_score(truth_binary, pred_probs), 4)
            return auc, aprec
        else:
            return None, None

    n_classes = pred_probs.shape[1]
    aucs, aprecs = [], []
    for i in range(n_classes):
        if np.sum(truth_binary[:, i]) != 0:
            aucs.append(roc_auc_score(truth_binary[:, i], pred_probs[:, i]))
            aprecs.append(average_precision_score(truth_binary[:, i], pred_probs[:, i]))
    if aucs:
        auc = round(np.mean(aucs), 4)
        aprec = round(np.mean(aprecs), 4)
        return auc, aprec
    else:
        return None, None


def get_summary(scores):
    sum_score, sum_weight = 0, 0
    for m in SUMMARY_WEIGHTS:
        if scores[m] is not None:
            sum_score += SUMMARY_WEIGHTS[m] * scores[m]
            sum_weight += SUMMARY_WEIGHTS[m]

    if sum_weight == 0:
        scores['summary'] = 0
    else:
        if scores['summary'] is None:
            scores['summary'] = sum_score / sum_weight
        else:
            scores['summary'] += sum_score / sum_weight
    return scores


def eval_a_song(
    probs_bound,
    probs_funct,
    tar_bound,
    tar_funct,
    tar_chorus_bound,
    segment_type,
    n_top_bound,
    label_hop,
    key,
):
    scores = {}
    for m in EVAL_METRICS:
        scores[m] = None

    # if 'boundary_labels' in pred['truth']:
    pre_funct, pre_choruses, segments, choruses, raw_segments = post_process(
        probs_bound, probs_funct, n_top_bound, label_hop
    )

    chorus_only, short_clip = False, False
    if segment_type == "c":
        chorus_only = True
    if segment_type == "s":
        short_clip = True

    if not short_clip:
        if not chorus_only:
            est_inter = [seg["interval"] for seg in raw_segments]
            scores["HR5F"], scores["HR30F"] = eval_boundary_f1(
                tar_bound, est_inter, key, ChorusOnly=False
            )

            scores["ACC0"] = function_accuracy(tar_funct, probs_funct)
            scores["ACC"] = function_accuracy(tar_funct, pre_funct)

        # chorus boundary evaluation
        # est_inter = [time_prob[0] for time_prob in out['chorus_time_prob']]
        est_inter = [seg["interval"] for seg in choruses]
        scores["mHR5F"], scores["mHR30F"] = eval_boundary_f1(
            tar_chorus_bound, est_inter, key, ChorusOnly=True
        )
        scores["ChorusF1"] = framewise_f1(
            (pre_funct == 1).astype(np.int), tar_funct[:, 1]
        )
        scores["ChoBoF1"] = chorus_bound_framewise_f1(
            (pre_funct == 1).astype(np.int), tar_chorus_bound, label_hop
        )
        scores["Chorus"], _ = aroc_ap(tar_funct[:, 1], probs_funct[:, 1])
        scores["Intro"], _ = aroc_ap(tar_funct[:, 6], probs_funct[:, 6])
        scores["Inst"], _ = aroc_ap(tar_funct[:, 4], probs_funct[:, 4])
        scores["Verse"], _ = aroc_ap(tar_funct[:, 2], probs_funct[:, 2])

    scores = get_summary(scores)

    return scores


def evaluate_scores():
    HR5F_scores, mHR5F_scores = [], []
    for file in os.listdir('../Ripple/data/structure/pop909_labels/'):
        with open('../Ripple/data/structure/pop909_labels/'+file, "rb") as input_file:
            data = pickle.load(input_file)
            intervals_tar = data['intervals']
            chorus_tar = [i for l, i in zip(data['labels'], data['intervals']) if l == 'chorus']

        with open('../Ripple/results/structure/pop909/'+file, "rb") as input_file:
            data = pickle.load(input_file)
            intervals_pre = data['intervals']
            chorus_pre = [i for l, i in zip(data['labels'], data['intervals']) if l == 'chorus']

        key = file.split('.')[0]
        
        intervals_tar = [i for i in intervals_tar if (i[1] - i[0]) > 0]
        #intervals_pre = [i for i in intervals_pre if (i[1] - i[0]) > 0]
        HR5F = eval_boundary_f1(intervals_tar, intervals_pre, key, ChorusOnly=False)[0]
        mHR5F = eval_boundary_f1(chorus_tar, chorus_pre, key, ChorusOnly=True)[0]

        HR5F_scores.append(HR5F)
        mHR5F_scores.append(mHR5F)

    print('HR5F:', sum(HR5F_scores) / len(HR5F_scores))
    print('mHR5F', sum(mHR5F_scores) / len(mHR5F_scores))