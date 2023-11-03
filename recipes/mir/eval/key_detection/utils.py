import numpy as np
from mir_tools.key_detection.constants import MODE_MAJMIN


def intervals_to_activation(
    intervals, song_duration, hop_in_sec=0.2, smear_duration=2.0, short=False
):
    """
    intervals: list of interval
    """
    hop_per_sec = 1 / hop_in_sec
    ramp = np.round(np.hanning(np.round(smear_duration * hop_per_sec)), 5)
    # in sample
    smear_half = len(ramp) // 2

    lls = np.round(np.array(intervals) * hop_per_sec).astype(int)
    # sec to sample
    end_idx = np.round(song_duration * hop_per_sec).astype(int)
    timeweights = {}
    for start, end in lls:
        for i in range(smear_half):
            offset = i - smear_half
            idx = int(start + offset)
            if idx >= 0 and idx <= end_idx:
                try:
                    timeweights[idx] = max(timeweights[idx], ramp[i])
                    # if overlapped idx, take the larger one
                except Exception:
                    timeweights[idx] = ramp[i]
        for i in range(smear_half, len(ramp)):
            offset = i - smear_half
            idx = int(end + offset)
            if idx >= 0 and idx <= end_idx:
                try:
                    timeweights[idx] = max(timeweights[idx], ramp[i])
                except Exception:
                    timeweights[idx] = ramp[i]
        for i in range(start, end):
            if i >= 0 and i <= end_idx:
                timeweights[i] = 1.0

    seq_labels = []
    for i in range(end_idx + 1):
        if i in timeweights:
            seq_labels.append(float(timeweights[i]))
        elif short:
            seq_labels.append(-1)
        else:
            seq_labels.append(0.0)

    return seq_labels  # t-by-1 list


def get_keymode_labels(key_labels, intervals, song_duration, hop_in_sec=0.2):
    keymode_labels = []
    # none + 12 keys for major mode
    for key_idx in range(13):
        idx = [
            i
            for i, x in enumerate(key_labels)
            if x[0] == key_idx and (MODE_MAJMIN[x[1]] == 0 or MODE_MAJMIN[x[1]] == 1)
        ]
        inters = [intervals[i] for i in idx]
        keymode_labels.append(
            intervals_to_activation(inters, song_duration, hop_in_sec, smear_duration=0)
        )

    # 12 keys for minor mode
    for key_idx in range(1, 13):
        idx = [
            i
            for i, x in enumerate(key_labels)
            if x[0] == key_idx and MODE_MAJMIN[x[1]] == 2
        ]
        inters = [intervals[i] for i in idx]
        keymode_labels.append(
            intervals_to_activation(inters, song_duration, hop_in_sec, smear_duration=0)
        )

    return np.array(keymode_labels).T


def get_key_training_label(
    key_labels,
    audio_duration_in_s,
    sample_start_time_in_s,
    label_hop,
    sample_len_in_s,
    is_train,
):
    if is_train:
        label_len = int(sample_len_in_s / label_hop)
        sample_idx = int(round(sample_start_time_in_s * (1 / label_hop)))
        key_label = key_labels[sample_idx : sample_idx + label_len, :]
        pad_len = label_len - key_label.shape[0]
    else:
        key_label = key_labels[: int(audio_duration_in_s * (1 / label_hop))]
        pad_len = sample_len_in_s * int(1 / label_hop) - key_label.shape[0]

    if pad_len > 0:
        key_label = np.pad(key_label, [(0, pad_len), (0, 0)], "constant")

    return key_label
