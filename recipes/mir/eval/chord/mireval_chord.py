r"""
Chord estimation algorithms produce a list of intervals and labels which denote
the chord being played over each timespan.  They are evaluated by comparing the
estimated chord labels to some reference, usually using a mapping to a chord
subalphabet (e.g. minor and major chords only, all triads, etc.).  There is no
single 'right' way to compare two sequences of chord labels.  Embracing this
reality, every conventional comparison rule is provided.  Comparisons are made
over the different components of each chord (e.g. G:maj(6)/5): the root (G),
the root-invariant active semitones as determined by the quality
shorthand (maj) and scale degrees (6), and the bass interval (5).
This submodule provides functions both for comparing a sequences of chord
labels according to some chord subalphabet mapping and for using these
comparisons to score a sequence of estimated chords against a reference.
Conventions
-----------
A sequence of chord labels is represented as a list of strings, where each
label is the chord name based on the syntax of [#harte2010towards]_.  Reference
and estimated chord label sequences should be of the same length for comparison
functions.  When converting the chord string into its constituent parts,
* Pitch class counting starts at C, e.g. C:0, D:2, E:4, F:5, etc.
* Scale degree is represented as a string of the diatonic interval, relative to
  the root note, e.g. 'b6', '#5', or '7'
* Bass intervals are represented as strings
* Chord bitmaps are positional binary vectors indicating active pitch classes
  and may be absolute or relative depending on context in the code.
If no chord is present at a given point in time, it should have the label 'N',
which is defined in the variable ``mir_eval.chord.NO_CHORD``.
Metrics
-------
* :func:`mir_eval.chord.root`: Only compares the root of the chords.
* :func:`mir_eval.chord.majmin`: Only compares major, minor, and "no chord"
  labels.
* :func:`mir_eval.chord.majmin_inv`: Compares major/minor chords, with
  inversions.  The bass note must exist in the triad.
* :func:`mir_eval.chord.mirex`: A estimated chord is considered correct if it
  shares *at least* three pitch classes in common.
* :func:`mir_eval.chord.thirds`: Chords are compared at the level of major or
  minor thirds (root and third), For example, both ('A:7', 'A:maj') and
  ('A:min', 'A:dim') are equivalent, as the third is major and minor in
  quality, respectively.
* :func:`mir_eval.chord.thirds_inv`: Same as above, with inversions (bass
  relationships).
* :func:`mir_eval.chord.triads`: Chords are considered at the level of triads
  (major, minor, augmented, diminished, suspended), meaning that, in addition
  to the root, the quality is only considered through #5th scale degree (for
  augmented chords). For example, ('A:7', 'A:maj') are equivalent, while
  ('A:min', 'A:dim') and ('A:aug', 'A:maj') are not.
* :func:`mir_eval.chord.triads_inv`: Same as above, with inversions (bass
  relationships).
* :func:`mir_eval.chord.tetrads`: Chords are considered at the level of the
  entire quality in closed voicing, i.e. spanning only a single octave;
  extended chords (9's, 11's and 13's) are rolled into a single octave with any
  upper voices included as extensions. For example, ('A:7', 'A:9') are
  equivlent but ('A:7', 'A:maj7') are not.
* :func:`mir_eval.chord.tetrads_inv`: Same as above, with inversions (bass
  relationships).
* :func:`mir_eval.chord.sevenths`: Compares according to MIREX "sevenths"
  rules; that is, only major, major seventh, seventh, minor, minor seventh and
  no chord labels are compared.
* :func:`mir_eval.chord.sevenths_inv`: Same as above, with inversions (bass
  relationships).
* :func:`mir_eval.chord.overseg`: Computes the level of over-segmentation
  between estimated and reference intervals.
* :func:`mir_eval.chord.underseg`: Computes the level of under-segmentation
  between estimated and reference intervals.
* :func:`mir_eval.chord.seg`: Computes the minimum of over- and
  under-segmentation between estimated and reference intervals.
References
----------
    .. [#harte2010towards] C. Harte. Towards Automatic Extraction of Harmony
        Information from Music Signals. PhD thesis, Queen Mary University of
        London, August 2010.
"""

import functools
import os
import pickle

import mir_eval
import numpy as np
import torch
from mir_tools.chord.utils import get_chord_labels, get_chord_training_label
from mir_eval.chord import encode_many, validate

QUALITIES = {
    #           1     2     3     4  5     6     7
    "maj": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "min": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "aug": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "dim": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "sus4": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    "sus2": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min7": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "minmaj7": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    "maj6": [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    "min6": [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    "dim7": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    "hdim7": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "maj9": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min9": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "9": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "b9": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "#9": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "min11": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "11": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "#11": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "maj13": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min13": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "13": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "b13": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "1": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "5": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

idx2chord = ["N", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

idx2triad = ["N", "maj", "min", "sus4", "sus2", "dim", "aug"]


def customized_score(reference_labels, estimated_labels, qualities, valid_note=8):
    validate(reference_labels, estimated_labels)
    valid_semitones = np.array([QUALITIES[name][:valid_note] for name in qualities])

    ref_roots, ref_semitones = encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = encode_many(estimated_labels, False)[:2]
    ref_semitones = ref_semitones[:, :valid_note]
    est_semitones = est_semitones[:, :valid_note]

    eq_root = ref_roots == est_roots
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    comparison_scores = (eq_root * eq_semitones).astype(np.float)

    # Test for reference chord inclusion
    is_valid = np.array(
        [
            np.all(np.equal(ref_semitones, semitones), axis=1)
            for semitones in valid_semitones
        ]
    )
    # Drop if NOR
    comparison_scores[np.sum(is_valid, axis=0) == 0] = -1
    return comparison_scores


def mireval_score(gt_data, est_data):
    ref_intervals, ref_labels = gt_data
    est_intervals, est_labels = est_data

    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals,
        est_labels,
        ref_intervals.min(),
        ref_intervals.max(),
        mir_eval.chord.NO_CHORD,
        mir_eval.chord.NO_CHORD,
    )

    (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels
    )
    durations = mir_eval.util.intervals_to_durations(intervals)

    major_minor = mir_eval.chord.majmin(ref_labels, est_labels)
    major_minor = mir_eval.chord.weighted_accuracy(major_minor, durations)

    root = mir_eval.chord.root(ref_labels, est_labels)
    root = mir_eval.chord.weighted_accuracy(root, durations)

    major = None
    major = None

    minor = None
    minor = None

    sus4 = None  # customized_score(ref_labels, est_labels, ['sus4'])
    sus4 = None  # mir_eval.chord.weighted_accuracy(sus4, durations)

    sus2 = None  # customized_score(ref_labels, est_labels, ['sus2'])
    sus2 = None  # mir_eval.chord.weighted_accuracy(sus2, durations)

    dim = None  # customized_score(ref_labels, est_labels, ['dim', 'dim7', 'hdim7'])
    dim = None  # mir_eval.chord.weighted_accuracy(dim, durations)

    aug = None  # customized_score(ref_labels, est_labels, ['aug'])
    aug = None  # mir_eval.chord.weighted_accuracy(aug, durations)

    seventh = None  # customized_score(ref_labels, est_labels, ['7'], 12)
    seventh = None  # mir_eval.chord.weighted_accuracy(seventh, durations)

    maj_seventh = None  # customized_score(ref_labels, est_labels, ['maj7'], 12)
    maj_seventh = None  # mir_eval.chord.weighted_accuracy(maj_seventh, durations)

    min_seventh = None  # customized_score(ref_labels, est_labels, ['min7'], 12)
    min_seventh = None  # mir_eval.chord.weighted_accuracy(min_seventh, durations)

    overseg = None  # mir_eval.chord.overseg(tar_seg, pre_seg)
    underseg = None  # mir_eval.chord.underseg(tar_seg, pre_seg)

    return [
        root,
        major_minor,
        major,
        minor,
        sus4,
        sus2,
        dim,
        aug,
        seventh,
        maj_seventh,
        min_seventh,
        overseg,
        underseg,
    ]


def logsumexp(*args):
    M = functools.reduce(torch.max, args)
    mask = M != -np.inf
    M[mask] += torch.log(sum(torch.exp(x[mask] - M[mask]) for x in args))
    return M


def calculate_seventh(triad, seventh):
    if triad == 1:  # maj
        if seventh == 1:  # maj7
            triads = idx2triad[triad] + "7"
        elif seventh == 2:  # 7
            triads = "7"
        else:
            triads = idx2triad[triad]
    elif triad == 2:  # min
        if seventh == 2:  # min7
            triads = idx2triad[triad] + "7"
        else:
            triads = idx2triad[triad]
    else:
        triads = idx2triad[triad]
    return triads


def get_chord_score(prediction, target, label_hop):
    res = {}
    est_labels, interval, tar_labels = [], [], []

    pre_root = np.argmax(prediction["chord_root"], -1)
    pre_triad = np.argmax(prediction["chord_triad"], -1)
    tar_root = np.argmax(target["chord_root"], -1)
    tar_triad = np.argmax(target["chord_triad"], -1)

    for i in range(len(pre_root)):
        if idx2chord[pre_root[i]] == "N" or idx2triad[pre_triad[i]] == "N":
            label = "N"
        else:
            triads = calculate_seventh(pre_triad[i], 0)  # pre_seventh[i])
            label = f"{idx2chord[pre_root[i]]}:{triads}"
        est_labels.append(label)

        if "chord_ignore" in target and target["chord_ignore"][i] == 1:
            tar_labels.append("X")
        elif idx2chord[tar_root[i]] == "N" or idx2triad[tar_triad[i]] == "N":
            tar_labels.append("N")
        else:
            triads = calculate_seventh(tar_triad[i], 0)  # tar_seventh[i])
            tar_labels.append(f"{idx2chord[tar_root[i]]}:{triads}")
        interval.append([i * label_hop, (i + 1) * label_hop])

    (
        root,
        major_minor,
        major,
        minor,
        sus4,
        sus2,
        dim,
        aug,
        seventh,
        maj_seventh,
        min_seventh,
        overseg,
        underseg,
    ) = mireval_score(
        (np.array(interval), tar_labels), (np.array(interval), est_labels)
    )

    res["root"] = root
    res["major_minor"] = major_minor
    res["major"] = major
    res["minor"] = minor
    res["sus4"] = sus4
    res["sus2"] = sus2
    res["dim"] = dim
    res["aug"] = aug
    res["seventh"] = seventh
    res["maj_seventh"] = maj_seventh
    res["min_seventh"] = min_seventh
    res["overseg"] = overseg
    res["underseg"] = underseg

    return res


def evaluate_scores():
    major_minor_scores, root_scores = [], []
    for file in os.listdir("../Ripple/data/chord/leadsheet_labels"):
        with open("../Ripple/data/chord/leadsheet_labels/" + file, "rb") as input_file:
            data = pickle.load(input_file)
            tar_intervals = data["intervals"]
            tar_labels = [l.strip() for l in data["labels"]]

        with open("../Ripple/results/chord/leadsheet/" + file, "rb") as input_file:
            data = pickle.load(input_file)
            pre_intervals = data["intervals"]
            pre_labels = data["labels"]

        (
            root,
            major_minor,
            major,
            minor,
            sus4,
            sus2,
            dim,
            aug,
            seventh,
            maj_seventh,
            min_seventh,
            overseg,
            underseg,
        ) = mireval_score(
            (np.array(tar_intervals), tar_labels), (np.array(pre_intervals), pre_labels)
        )

        root_scores.append(root)
        major_minor_scores.append(major_minor)

    print("Root:", sum(root_scores) / len(root_scores))
    print("Major minor:", sum(major_minor_scores) / len(major_minor_scores))
