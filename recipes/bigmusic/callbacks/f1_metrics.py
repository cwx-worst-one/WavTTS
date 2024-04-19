import json
import numpy as np
import pytorch_lightning as pl
import torch

from collections import defaultdict
from pathlib import Path
from recipes.bigmusic.lightning.embedding_modules import get_mulan_embeds
from recipes.bigmusic.utils.format_utils import concat_metadata_list, update_json
from recipes.bigmusic.datasets.transforms.lyrics_segment import crop_pad_to_seq_length, random_crop_pad_to_seq_length
from recipes.musiclm.inference.utils import load_wav
from typing import Any


# The maximum allowable beat time
MAX_TIME = 30000.0


class F1MetricsCallback(pl.Callback):
    """F1 metric to compare the downbeats from generated audio and target audio."""
    def __init__(self):
        super().__init__()
        self.batch_f1 = []
        self.phase_shift = []
        self.target_downbeats = []
        self.predicted_downbeats = []

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Compare the downbeats using target audio and generate audio
        device = pl_module.device
        generated_audio = outputs['generated_audio_tensor']
        if generated_audio.ndim == 3:
            generated_audio = generated_audio.mean(dim=1)
            generated_audio = generated_audio.squeeze(dim=1)
        generated_audio = generated_audio.cuda()
        batch_size = generated_audio.shape[0]

        target_audio = None
        if "beat_audio" in batch:
            target_audio = batch["beat_audio"]
        elif "target_audio" in batch:
            target_audio = batch["target_audio"]
        elif "style_audio" in batch:
            target_audio = batch["style_audio"]
        if target_audio is None:
            return

        if target_audio.ndim == 3:
            target_audio = target_audio.mean(dim=1)
            target_audio = target_audio.squeeze(dim=1)
        requires = pl_module.semantic_module.requires

        # Gets the beat data in the format of a sequence of (timestamp, beat_id).
        target_beats = requires["beat"].predict_step({"target_audio": target_audio}, 0)
        pred_beats = requires["beat"].predict_step({"target_audio": generated_audio}, 0)

        f1_scores = []
        for tb, pb in zip(target_beats, pred_beats):
            # Extracts the downbeat timestamp that has the beat id as 1.
            target_db_ts = [x[0] for x in tb if x[1] == 1]
            pred_db_ts = [x[0] for x in pb if x[1] == 1]
            self.target_downbeats.append(target_db_ts)
            self.predicted_downbeats.append(pred_db_ts)
            f1_score, phase_shift = f_measure(np.array(target_db_ts), np.array(pred_db_ts))
            self.batch_f1.append(f1_score)
            self.phase_shift.append(phase_shift)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        f1_avg = np.mean(self.batch_f1)
        output_dir = pl_module.extra_params.output_dir
        metrics_fp = Path(output_dir)/'metrics.json'
        # TODO: add the corresponding target audio and generated audio file path.
        update_json(
            metrics_fp,
            {
                "Beat F1": f1_avg,
                "Phase shift": np.mean(self.phase_shift),
                "Target downbeats: ": self.target_downbeats,
                "Predicted downbeats: ": self.predicted_downbeats
                })


# The following code is adopted from sami_ai_models.recipes.beat.eval.mir_eval.beat.f_measure
def f_measure(reference_beats, estimated_beats, f_measure_threshold=0.1):
    """Compute the F-measure of correct vs incorrectly predicted beats.
    "Correctness" is determined over a small window.

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> reference_beats = mir_eval.beat.trim_beats(reference_beats)
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    >>> f_measure = mir_eval.beat.f_measure(reference_beats,
                                            estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        estimated beat times, in seconds
    f_measure_threshold : float
        Window size, in seconds
        (Default value = 0.07)

    Returns
    -------
    f_score : float
        The computed F-measure score

    """
    validate(reference_beats, estimated_beats)
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0 or reference_beats.size == 0:
        return 0.0

    # Compute the phase shift between reference and estimated beats and update
    # the downbeats based on the phase shift.
    reference_beats, estimated_beats, phase_shift = compute_phase_shift(
        reference_beats, estimated_beats)

    # Better align the reference and estimated beat sequence
    estimated_beats = estimated_beats - phase_shift
    # Compute the best-case matching between reference and estimated locations
    matching = match_events(reference_beats, estimated_beats, f_measure_threshold)

    precision = float(len(matching)) / len(estimated_beats)
    recall = float(len(matching)) / len(reference_beats)
    return _f_measure(precision, recall), phase_shift


def compute_phase_shift(ref, est):
    """Compute the phase shift between two beat arrays."""
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    last_ts = min(ref[-1], est[-1])
    est = est[est <= last_ts]
    ref = ref[ref <= last_ts]
    dist = [0] * len(est)
    min_dist = [0] * len(est)
    # The sign to track whether the predicted downbeats are moved to the left or right.
    sign = 0.0
    for i, e in enumerate(est):
        dist[i] = e - ref
        # Filter out the distance that has an opposite direction with
        # the first beat shift
        dist_filter = dist[i] * sign >= 0
        if dist_filter.any():
            dist[i] = dist[i][dist_filter]
        # The adjacent reference beat with the smallest distance.
        adj = np.argmin(np.abs(dist[i]))
        min_dist[i] = dist[i][adj]
        if sign == 0.0:
            sign = min_dist[i]
    min_dist = np.array(min_dist)
    # Averaging the downbeat shifts by removing the smallest and
    # largest distance which might be outliers.
    phase_shift = np.mean(min_dist[1: -1])
    if phase_shift < 0:
        ref = ref[1:]
    # Better align the reference and estimated beat sequence
    est = est - phase_shift
    return ref, est, phase_shift


def _f_measure(precision, recall, beta=1.0):
    """Compute the f-measure from precision and recall scores.

    Parameters
    ----------
    precision : float in (0, 1]
        Precision
    recall : float in (0, 1]
        Recall
    beta : float > 0
        Weighting factor for f-measure
        (Default value = 1.0)

    Returns
    -------
    f_measure : float
        The weighted f-measure

    """

    if precision == 0 and recall == 0:
        return 0.0

    return (1 + beta**2) * precision * recall / ((beta**2) * precision + recall)


def match_events(ref, est, window, distance=None):
    """Compute a maximum matching between reference and estimated event times,
    subject to a window constraint.

    Given two lists of event times ``ref`` and ``est``, we seek the largest set
    of correspondences ``(ref[i], est[j])`` such that
    ``distance(ref[i], est[j]) <= window``, and each
    ``ref[i]`` and ``est[j]`` is matched at most once.

    This is useful for computing precision/recall metrics in beat tracking,
    onset detection, and segmentation.

    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Array of reference values
    est : np.ndarray, shape=(m,)
        Array of estimated values
    window : float > 0
        Size of the window.
    distance : function
        function that computes the outer distance of ref and est.
        By default uses ``|ref[i] - est[j]|``

    Returns
    -------
    matching : list of tuples
        A list of matched reference and event numbers.
        ``matching[i] == (i, j)`` where ``ref[i]`` matches ``est[j]``.

    """
    if distance is not None:
        # Compute the indices of feasible pairings
        hits = np.where(distance(ref, est) <= window)
    else:
        hits = _fast_hit_windows(ref, est, window)

    # Construct the graph input
    G = {}
    for ref_i, est_i in zip(*hits):
        if est_i not in G:
            G[est_i] = []
        G[est_i].append(ref_i)

    # Compute the maximum matching
    matching = sorted(_bipartite_match(G).items())

    return matching


def validate(reference_beats, estimated_beats):
    """Checks that the input annotations to a metric look like valid beat time
    arrays, and throws helpful errors if not.

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        estimated beat times, in seconds
    """

    for beats in [reference_beats, estimated_beats]:
        validate_events(beats, MAX_TIME)


def _fast_hit_windows(ref, est, window):
    """Fast calculation of windowed hits for time events.

    Given two lists of event times ``ref`` and ``est``, and a
    tolerance window, computes a list of pairings
    ``(i, j)`` where ``|ref[i] - est[j]| <= window``.

    This is equivalent to, but more efficient than the following:

    >>> hit_ref, hit_est = np.where(np.abs(np.subtract.outer(ref, est))
    ...                             <= window)

    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Array of reference values
    est : np.ndarray, shape=(m,)
        Array of estimated values
    window : float >= 0
        Size of the tolerance window

    Returns
    -------
    hit_ref : np.ndarray
    hit_est : np.ndarray
        indices such that ``|hit_ref[i] - hit_est[i]| <= window``
    """

    ref = np.asarray(ref)
    est = np.asarray(est)
    ref_idx = np.argsort(ref)
    ref_sorted = ref[ref_idx]

    left_idx = np.searchsorted(ref_sorted, est - window, side="left")
    right_idx = np.searchsorted(ref_sorted, est + window, side="right")

    hit_ref, hit_est = [], []

    for j, (start, end) in enumerate(zip(left_idx, right_idx)):
        hit_ref.extend(ref_idx[start:end])
        hit_est.extend([j] * (end - start))

    return hit_ref, hit_est


def validate_events(events, max_time=30000.0):
    """Checks that a 1-d event location ndarray is well-formed, and raises
    errors if not.

    Parameters
    ----------
    events : np.ndarray, shape=(n,)
        Array of event times
    max_time : float
        If an event is found above this time, a ValueError will be raised.
        (Default value = 30000.)

    """
    # Make sure no event times are huge
    if (events > max_time).any():
        raise ValueError(
            "An event at time {} was found which is greater than "
            "the maximum allowable time of max_time = {} (did you"
            " supply event times in "
            "seconds?)".format(events.max(), max_time)
        )
    # Make sure event locations are 1-d np ndarrays
    if events.ndim != 1:
        raise ValueError(
            "Event times should be 1-d numpy ndarray, "
            "but shape={}".format(events.shape)
        )
    # Make sure event times are increasing
    if (np.diff(events) < 0).any():
        raise ValueError("Events should be in increasing order.")


def _bipartite_match(graph):
    """Find maximum cardinality matching of a bipartite graph (U,V,E).
    The input format is a dictionary mapping members of U to a list
    of their neighbors in V.

    The output is a dict M mapping members of V to their matches in U.

    Parameters
    ----------
    graph : dictionary : left-vertex -> list of right vertices
        The input bipartite graph.  Each edge need only be specified once.

    Returns
    -------
    matching : dictionary : right-vertex -> left vertex
        A maximal bipartite matching.

    """
    # Adapted from:
    #
    # Hopcroft-Karp bipartite max-cardinality matching and max independent set
    # David Eppstein, UC Irvine, 27 Apr 2002

    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break

    while True:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first
        # layer
        preds = {}
        unmatched = []
        pred = dict([(u, unmatched) for u in graph])
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            new_layer = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        new_layer.setdefault(v, []).append(u)
            layer = []
            for v in new_layer:
                preds[v] = new_layer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None
            return matching

        def recurse(v):
            """Recursively search backward through layers to find alternating
            paths.  recursion returns true if found path, false otherwise
            """
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v)
