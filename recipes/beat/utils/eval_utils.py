import torch
import os
import numpy as np
import recipes.beat.eval.mireval_beat as mireval_beat


def merge_multiple_temporal_probs(probs, chord, sample_len, sample_hop):
    new_probs = torch.zeros(chord.shape).to(chord.device)
    count = torch.zeros(chord.shape).to(chord.device)
    for i in range(len(probs)):
        new_probs[i * sample_hop : i * sample_hop + sample_len] += probs[i][
            : len(new_probs[i * sample_hop : i * sample_hop + sample_len])
        ]
        count[i * sample_hop : i * sample_hop + sample_len] += 1
    new_probs = new_probs / count
    return new_probs


def merge_multiple_temporal_probs_inference(prediction, n_labels, prediction_legnth, sample_len, sample_hop):
    new_probs = torch.zeros((prediction_legnth, n_labels)).to(prediction.device)
    count = torch.zeros((prediction_legnth, n_labels)).to(prediction.device)
    for i in range(len(prediction)):
        new_probs[i * sample_hop : i * sample_hop + sample_len] += prediction[i][
            : len(new_probs[i * sample_hop : i * sample_hop + sample_len])
        ]
        count[i * sample_hop : i * sample_hop + sample_len] += 1
    new_probs = new_probs / count
    return new_probs


def evaluate_scores(dataset):
    Beat_f1_scores, Downbeat_f1_scores = [], []
    for file in os.listdir(f'../Ripple/data/beat/{dataset}_labels/'):
        if '.DS_Store' == file:
            continue
        beat_tar, downbeat_tar, beat_pre, downbeat_pre = [], [], [], []
        for line in open(f'../Ripple/data/beat/{dataset}_labels/'+file, 'r').readlines():
            time, b = line.strip().split('\t')
            beat_tar.append(float(time))
            if b == '1':
                downbeat_tar.append(float(time))

        for line in open(f'../Ripple/results/beat/{dataset}/'+file, 'r').readlines():
            time, b = line.strip().split('\t')
            beat_pre.append(float(time))
            if float(b) == 1:
                downbeat_pre.append(float(time))

        Beat_f1 = mireval_beat.f_measure(
            np.array(beat_tar),
            np.array(beat_pre),
            f_measure_threshold=0.2,
        )

        Downbeat_f1 = mireval_beat.f_measure(
            np.array(downbeat_tar),
            np.array(downbeat_pre),
            f_measure_threshold=0.2,
        )
        Beat_f1_scores.append(Beat_f1)
        Downbeat_f1_scores.append(Downbeat_f1)
    
    print('Beat F1:', sum(Beat_f1_scores)/len(Beat_f1_scores))
    print('Downbeat F1:', sum(Downbeat_f1_scores)/len(Downbeat_f1_scores))
