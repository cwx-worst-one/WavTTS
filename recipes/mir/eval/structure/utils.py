import numpy as np


def get_function_weight(train_transformed):
    return None


def convert_structure_label(label):
    seg_map = {
        "silence": "silence",
        "end": "end",
        "build": "verse",
        "fadein": "intro",
        "opening": "intro",
        "stutter": "chorus",
        "slow": "verse",
        "drumroll": "inst",
        "synth": "inst",
        "closing": "outro",
        "interlude": "inst",
        "mantra": "verse",
        "fade-out": "outro",
        "out": "outro",
        "guitar": "inst",
        "head": "inst",
        "loop": "inst",
    }

    substr_map = {
        "other": "other",
        "pre-chorus-and-chorus": "chorus",
        "verse-and-chorus": "chorus",
        "intro": "intro",
        "verse": "verse",
        "prechorus": "verse",
        "refrain": "chorus",
        "pre-chorus": "verse",
        "chorus": "chorus",
        "bridge": "bridge",
        "outro": "outro",
        "fadeout": "outro",
        "ending": "outro",
        "fadein": "intro",
        "inst": "inst",
        "solo": "inst",
        "break": "inst",
        "trans": "bridge",
        "gtr": "inst",
        "section": "verse",
        "riff": "inst",
        "rap": "verse",
        "coda": "outro",
        "interlude": "inst",
        "lead-in": "inst",
        "theme": "chorus",
        "development": "verse",
        "variation": "bridge",
        "impro": "inst",
        "guitar": "inst",
        "spoken": "inst",
        "trumpet": "inst",
        "applause": "inst",
        "voice": "inst",
        "stage": "inst",
        "banjo": "inst",
        "crowd": "inst",
        "pause": "inst",
        "tag": "inst",
    }

    label = label.lower()
    for s in substr_map:
        if s in label:
            return substr_map[s]
    if label in seg_map:
        return seg_map[label]
    else:
        return None


def intervals_to_activation(
    intervals, song_duration, hop_in_sec=0.2, smear_duration=2.0
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
    for label in lls:
        start, end = label
        for i in range(smear_half):
            offset = i - smear_half
            idx = int(start + offset)
            if idx >= 0 and idx <= end_idx:
                if idx in timeweights:
                    timeweights[idx] = max(timeweights[idx], ramp[i])
                else:
                    timeweights[idx] = ramp[i]
        for i in range(smear_half, len(ramp)):
            offset = i - smear_half
            idx = int(end + offset)
            if idx >= 0 and idx <= end_idx:
                if idx in timeweights:
                    timeweights[idx] = max(timeweights[idx], ramp[i])
                else:
                    timeweights[idx] = ramp[i]
        for i in range(start, end):
            if i >= 0 and i <= end_idx:
                timeweights[i] = 1.0

    seq_labels = []
    for i in range(end_idx + 1):
        if i in timeweights:
            seq_labels.append(float(timeweights[i]))
        else:
            seq_labels.append(0.0)

    return seq_labels  # t-by-1 list


def boundary_activation(
    times, song_duration, hop_in_sec=0.2, smear_duration=2.0, sustain_duration=0.6
):
    """
    times: list of float timestamps
    all the remaining arguments are in second
    """

    hop_per_sec = 1 / hop_in_sec
    ramp = np.round(np.hanning(np.round(smear_duration * hop_per_sec)), 5)
    # in sample
    sustain = int(round(sustain_duration * hop_per_sec))  # in sample
    weights = np.concatenate(
        (ramp[: len(ramp) // 2], np.ones(sustain), ramp[len(ramp) // 2 :])
    )
    smear_half = len(weights) // 2

    lls = np.round(np.array(times) * hop_per_sec).astype(int)
    # sec to sample
    end_idx = np.round(song_duration * hop_per_sec).astype(int)
    timeweights = {}
    for label in lls:
        for i in range(len(weights)):
            offset = i - smear_half
            idx = int(label + offset)
            if idx >= 0 and idx <= end_idx:
                if idx in timeweights:
                    timeweights[idx] = max(timeweights[idx], weights[i])
                else:
                    timeweights[idx] = weights[i]

    seq_labels = []
    for i in range(end_idx + 1):
        if i in timeweights.keys():
            seq_labels.append(float(timeweights[i]))
        else:
            seq_labels.append(0.0)

    return seq_labels


def get_boundary_labels(intervals, labels, song_duration, hop_in_sec=0.2):
    all_times = [s[0] for s in intervals] + [intervals[-1][-1]]
    seq_labels = [boundary_activation(all_times, song_duration, hop_in_sec)]
    labels = [convert_structure_label(label) for label in labels]
    chorus_idx = [i for i, x in enumerate(labels) if x == "chorus"]
    chorus_times = []
    chorus_intervals = []
    for i in chorus_idx:
        chorus_intervals.append(intervals[i])
        chorus_times.append(intervals[i][0])
        chorus_times.append(intervals[i][1])
    chorus_times = sorted(list(set(chorus_times)))  # de-dup and sorted by time
    seq_labels.append(boundary_activation(chorus_times, song_duration, hop_in_sec))

    return (
        seq_labels,
        chorus_intervals,
    )  # t-by-2 list (all boundaries and chorus boundaries)


def get_function_labels(
    intervals,
    labels,
    song_duration,
    hop_in_sec=0.2,
    short=False,
    segment_classes=["silence", "chorus", "verse", "bridge", "inst", "outro", "intro"],
):
    labels = [convert_structure_label(label) for label in labels]
    seq_labels = []
    for segment_class in segment_classes:
        idx = [i for i, x in enumerate(labels) if x == segment_class]
        inters = [intervals[i] for i in idx]
        seq_labels.append(intervals_to_activation(inters, song_duration, hop_in_sec))

    if (
        short
    ):  # for short==True, there are unknown sections, where the activations will be -1
        seq_labels = np.array(seq_labels)
        time_activate = np.sum(seq_labels, 0)
        seq_labels[:, time_activate == 0] = -1  # inactive time steps are given -1

    return seq_labels, labels  # t-by-7 list, list of converted labels


def get_structure_training_labels(x, audio_duration_in_s, _label_hop, _enable_short):
    intervals, labels = x["intervals.pickle"], x["labels.pickle"]
    short_clip, chorus_only = False, False
    if x["segment_type.txt"] == "s":
        short_clip = True
    if x["segment_type.txt"] == "c":
        chorus_only = True

    isShort = _enable_short and short_clip

    if not short_clip and intervals[0][0] > 0:
        intervals = [[0.0, intervals[0][0]]] + intervals
        labels = ["silence"] + labels

    bl, chorus_interval = get_boundary_labels(
        intervals, labels, audio_duration_in_s, hop_in_sec=_label_hop
    )
    boundary_label = np.array(bl).T  # original 7-by-T -> latter T-by-7
    fl, converted_labels = get_function_labels(
        intervals, labels, audio_duration_in_s, hop_in_sec=_label_hop, short=isShort
    )
    function_label = np.array(fl).T  # original 7-by-T -> latter T-by-7

    return boundary_label, function_label, chorus_only, chorus_interval, intervals
