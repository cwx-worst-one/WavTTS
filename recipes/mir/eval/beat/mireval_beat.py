r"""
The aim of a beat detection algorithm is to report the times at which a typical
human listener might tap their foot to a piece of music. As a result, most
metrics for evaluating the performance of beat tracking systems involve
computing the error between the estimated beat times and some reference list of
beat locations. Many metrics additionally compare the beat sequences at
different metric levels in order to deal with the ambiguity of tempo.

Based on the methods described in:
    Matthew E. P. Davies,  Norberto Degara, and Mark D. Plumbley.
    "Evaluation Methods for Musical Audio Beat Tracking Algorithms",
    Queen Mary University of London Technical Report C4DM-TR-09-06
    London, United Kingdom, 8 October 2009.

See also the Beat Evaluation Toolbox:
    https://code.soundsoftware.ac.uk/projects/beat-evaluation/

Conventions
-----------

Beat times should be provided in the form of a 1-dimensional array of beat
times in seconds in increasing order.  Typically, any beats which occur before
5s are ignored; this can be accomplished using
:func:`mir_eval.beat.trim_beats()`.

Metrics
-------

* :func:`mir_eval.beat.f_measure`: The F-measure of the beat sequence, where an
  estimated beat is considered correct if it is sufficiently close to a
  reference beat
* :func:`mir_eval.beat.cemgil`: Cemgil's score, which computes the sum of
  Gaussian errors for each beat
* :func:`mir_eval.beat.goto`: Goto's score, a binary score which is 1 when at
  least 25% of the estimated beat sequence closely matches the reference beat
  sequence
* :func:`mir_eval.beat.p_score`: McKinney's P-score, which computes the
  cross-correlation of the estimated and reference beat sequences represented
  as impulse trains
* :func:`mir_eval.beat.continuity`: Continuity-based scores which compute the
  proportion of the beat sequence which is continuously correct
* :func:`mir_eval.beat.information_gain`: The Information Gain of a normalized
  beat error histogram over a uniform distribution

"""

import collections
import inspect

import numpy as np
import six

# The maximum allowable beat time
MAX_TIME = 30000.0


def trim_beats(beats, min_beat_time=5.0):
    """Removes beats before min_beat_time.  A common preprocessing step.

    Parameters
    ----------
    beats : np.ndarray
        Array of beat times in seconds.
    min_beat_time : float
        Minimum beat time to allow
        (Default value = 5.)

    Returns
    -------
    beats_trimmed : np.ndarray
        Trimmed beat array.
    """
    # Remove beats before min_beat_time
    return beats[beats >= min_beat_time]


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


def _get_reference_beat_variations(reference_beats):
    """Return metric variations of the reference beats

    Parameters
    ----------
    reference_beats : np.ndarray
        beat locations in seconds

    Returns
    -------
    reference_beats : np.ndarray
        Original beat locations
    off_beat : np.ndarray
        180 degrees out of phase from the original beat locations
    double : np.ndarray
        Beats at 2x the original tempo
    half_odd : np.ndarray
        Half tempo, odd beats
    half_even : np.ndarray
        Half tempo, even beats

    """

    # Create annotations at twice the metric level
    interpolated_indices = np.arange(0, reference_beats.shape[0] - 0.5, 0.5)
    original_indices = np.arange(0, reference_beats.shape[0])
    double_reference_beats = np.interp(
        interpolated_indices, original_indices, reference_beats
    )
    # Return metric variations:
    # True, off-beat, double tempo, half tempo odd, and half tempo even
    return (
        reference_beats,
        double_reference_beats[1::2],
        double_reference_beats,
        reference_beats[::2],
        reference_beats[1::2],
    )


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


def f_measure(reference_beats, estimated_beats, f_measure_threshold=0.07):
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
    # Compute the best-case matching between reference and estimated locations
    matching = match_events(reference_beats, estimated_beats, f_measure_threshold)

    precision = float(len(matching)) / len(estimated_beats)
    recall = float(len(matching)) / len(reference_beats)
    return _f_measure(precision, recall)


def cemgil(reference_beats, estimated_beats, cemgil_sigma=0.04):
    """Cemgil's score, computes a gaussian error of each estimated beat.
    Compares against the original beat times and all metrical variations.

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> reference_beats = mir_eval.beat.trim_beats(reference_beats)
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    >>> cemgil_score, cemgil_max = mir_eval.beat.cemgil(reference_beats,
                                                        estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        query beat times, in seconds
    cemgil_sigma : float
        Sigma parameter of gaussian error windows
        (Default value = 0.04)

    Returns
    -------
    cemgil_score : float
        Cemgil's score for the original reference beats
    cemgil_max : float
        The best Cemgil score for all metrical variations
    """
    validate(reference_beats, estimated_beats)
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0 or reference_beats.size == 0:
        return 0.0, 0.0
    # We'll compute Cemgil's accuracy for each variation
    accuracies = []
    for reference_beats in _get_reference_beat_variations(reference_beats):
        accuracy = 0
        # Cycle through beats
        for beat in reference_beats:
            # Find the error for the closest beat to the reference beat
            beat_diff = np.min(np.abs(beat - estimated_beats))
            # Add gaussian error into the accuracy
            accuracy += np.exp(-(beat_diff**2) / (2.0 * cemgil_sigma**2))
        # Normalize the accuracy
        accuracy /= 0.5 * (estimated_beats.shape[0] + reference_beats.shape[0])
        # Add it to our list of accuracy scores
        accuracies.append(accuracy)
    # Return raw accuracy with non-varied annotations
    # and maximal accuracy across all variations
    return accuracies[0], np.max(accuracies)


def goto(
    reference_beats, estimated_beats, goto_threshold=0.35, goto_mu=0.2, goto_sigma=0.2
):
    """Calculate Goto's score, a binary 1 or 0 depending on some specific
    heuristic criteria

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> reference_beats = mir_eval.beat.trim_beats(reference_beats)
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    >>> goto_score = mir_eval.beat.goto(reference_beats, estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        query beat times, in seconds
    goto_threshold : float
        Threshold of beat error for a beat to be "correct"
        (Default value = 0.35)
    goto_mu : float
        The mean of the beat errors in the continuously correct
        track must be less than this
        (Default value = 0.2)
    goto_sigma : float
        The std of the beat errors in the continuously correct track must
        be less than this
        (Default value = 0.2)

    Returns
    -------
    goto_score : float
        Either 1.0 or 0.0 if some specific criteria are met
    """
    validate(reference_beats, estimated_beats)
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0 or reference_beats.size == 0:
        return 0.0
    # Error for each beat
    beat_error = np.ones(reference_beats.shape[0])
    # Flag for whether the reference and estimated beats are paired
    paired = np.zeros(reference_beats.shape[0])
    # Keep track of Goto's three criteria
    goto_criteria = 0
    for n in range(1, reference_beats.shape[0] - 1):
        # Get previous inner-reference-beat-interval
        previous_interval = 0.5 * (reference_beats[n] - reference_beats[n - 1])
        # Window start - in the middle of the current beat and the previous
        window_min = reference_beats[n] - previous_interval
        # Next inter-reference-beat-interval
        next_interval = 0.5 * (reference_beats[n + 1] - reference_beats[n])
        # Window end - in the middle of the current beat and the next
        window_max = reference_beats[n] + next_interval
        # Get estimated beats in the window
        beats_in_window = np.logical_and(
            (estimated_beats >= window_min), (estimated_beats < window_max)
        )
        # False negative/positive
        if beats_in_window.sum() == 0 or beats_in_window.sum() > 1:
            paired[n] = 0
            beat_error[n] = 1
        else:
            # Single beat is paired!
            paired[n] = 1
            # Get offset of the estimated beat and the reference beat
            offset = estimated_beats[beats_in_window] - reference_beats[n]
            # Scale by previous or next interval
            if offset < 0:
                beat_error[n] = offset / previous_interval
            else:
                beat_error[n] = offset / next_interval
    # Get indices of incorrect beats
    incorrect_beats = np.flatnonzero(np.abs(beat_error) > goto_threshold)
    # All beats are correct (first and last will be 0 so always correct)
    if incorrect_beats.shape[0] < 3:
        # Get the track of correct beats
        track = beat_error[incorrect_beats[0] + 1 : incorrect_beats[-1] - 1]
        goto_criteria = 1
    else:
        # Get the track of maximal length
        track_len = np.max(np.diff(incorrect_beats))
        track_start = np.flatnonzero(np.diff(incorrect_beats) == track_len)[0]
        # Is the track length at least 25% of the song?
        if track_len - 1 > 0.25 * (reference_beats.shape[0] - 2):
            goto_criteria = 1
            start_beat = incorrect_beats[track_start]
            end_beat = incorrect_beats[track_start + 1]
            track = beat_error[start_beat : end_beat + 1]
    # If we have a track
    if goto_criteria:
        # Are mean and std of the track less than the required thresholds?
        if np.mean(np.abs(track)) < goto_mu and np.std(track, ddof=1) < goto_sigma:
            goto_criteria = 3
    # If all criteria are met, score is 100%!
    return 1.0 * (goto_criteria == 3)


def p_score(reference_beats, estimated_beats, p_score_threshold=0.2):
    """Get McKinney's P-score.
    Based on the autocorrelation of the reference and estimated beats

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> reference_beats = mir_eval.beat.trim_beats(reference_beats)
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    >>> p_score = mir_eval.beat.p_score(reference_beats, estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        query beat times, in seconds
    p_score_threshold : float
        Window size will be
        ``p_score_threshold*np.median(inter_annotation_intervals)``,
        (Default value = 0.2)

    Returns
    -------
    correlation : float
        McKinney's P-score

    """
    validate(reference_beats, estimated_beats)

    # When estimated or reference beats have <= 1 beats, can't compute the
    # metric, so return 0
    if estimated_beats.size <= 1 or reference_beats.size <= 1:
        return 0.0
    # Quantize beats to 10ms
    sampling_rate = int(1.0 / 0.010)
    # Shift beats so that the minimum in either sequence is zero
    offset = min(estimated_beats.min(), reference_beats.min())
    estimated_beats = np.array(estimated_beats - offset)
    reference_beats = np.array(reference_beats - offset)
    # Get the largest time index
    end_point = np.int(
        np.ceil(np.max([np.max(estimated_beats), np.max(reference_beats)]))
    )
    # Make impulse trains with impulses at beat locations
    reference_train = np.zeros(end_point * sampling_rate + 1)
    beat_indices = np.ceil(reference_beats * sampling_rate).astype(np.int)
    reference_train[beat_indices] = 1.0
    estimated_train = np.zeros(end_point * sampling_rate + 1)
    beat_indices = np.ceil(estimated_beats * sampling_rate).astype(np.int)
    estimated_train[beat_indices] = 1.0
    # Window size to take the correlation over
    # defined as .2*median(inter-annotation-intervals)
    annotation_intervals = np.diff(np.flatnonzero(reference_train))
    win_size = int(np.round(p_score_threshold * np.median(annotation_intervals)))
    # Get full correlation
    train_correlation = np.correlate(reference_train, estimated_train, "full")
    # Get the middle element - note we are rounding down on purpose here
    middle_lag = train_correlation.shape[0] // 2
    # Truncate to only valid lags (those corresponding to the window)
    start = middle_lag - win_size
    end = middle_lag + win_size + 1
    train_correlation = train_correlation[start:end]
    # Compute and return the P-score
    n_beats = np.max([estimated_beats.shape[0], reference_beats.shape[0]])
    return np.sum(train_correlation) / n_beats


def continuity(
    reference_beats,
    estimated_beats,
    continuity_phase_threshold=0.175,
    continuity_period_threshold=0.175,
):
    """Get metrics based on how much of the estimated beat sequence is
    continually correct.

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> reference_beats = mir_eval.beat.trim_beats(reference_beats)
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    >>> CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(reference_beats,
                                                          estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        query beat times, in seconds
    continuity_phase_threshold : float
        Allowable ratio of how far is the estimated beat
        can be from the reference beat
        (Default value = 0.175)
    continuity_period_threshold : float
        Allowable distance between the inter-beat-interval
        and the inter-annotation-interval
        (Default value = 0.175)

    Returns
    -------
    CMLc : float
        Correct metric level, continuous accuracy
    CMLt : float
        Correct metric level, total accuracy (continuity not required)
    AMLc : float
        Any metric level, continuous accuracy
    AMLt : float
        Any metric level, total accuracy (continuity not required)
    """
    validate(reference_beats, estimated_beats)

    # When estimated or reference beats have <= 1 beats, can't compute the
    # metric, so return 0
    if estimated_beats.size <= 1 or reference_beats.size <= 1:
        return 0.0, 0.0, 0.0, 0.0
    # Accuracies for each variation
    continuous_accuracies = []
    total_accuracies = []
    # Get accuracy for each variation
    for reference_beats in _get_reference_beat_variations(reference_beats):
        # Annotations that have been used
        n_annotations = np.max([reference_beats.shape[0], estimated_beats.shape[0]])
        used_annotations = np.zeros(n_annotations)
        # Whether or not we are continuous at any given point
        beat_successes = np.zeros(n_annotations)
        for m in range(estimated_beats.shape[0]):
            # Is this beat correct?
            beat_success = 0
            # Get differences for this beat
            beat_differences = np.abs(estimated_beats[m] - reference_beats)
            # Get nearest annotation index
            nearest = np.argmin(beat_differences)
            min_difference = beat_differences[nearest]
            # Have we already used this annotation?
            if used_annotations[nearest] == 0:
                # Is this the first beat or first annotation?
                # If so, look forward.
                if m == 0 or nearest == 0:
                    # How far is the estimated beat from the reference beat,
                    # relative to the inter-annotation-interval?
                    if nearest + 1 < reference_beats.shape[0]:
                        reference_interval = (
                            reference_beats[nearest + 1] - reference_beats[nearest]
                        )
                    else:
                        # Special case when nearest + 1 is too large - use the
                        # previous interval instead
                        reference_interval = (
                            reference_beats[nearest] - reference_beats[nearest - 1]
                        )
                    # Handle this special case when beats are not unique
                    if reference_interval == 0:
                        if min_difference == 0:
                            phase = 1
                        else:
                            phase = np.inf
                    else:
                        phase = np.abs(min_difference / reference_interval)
                    # How close is the inter-beat-interval
                    # to the inter-annotation-interval?
                    if m + 1 < estimated_beats.shape[0]:
                        estimated_interval = estimated_beats[m + 1] - estimated_beats[m]
                    else:
                        # Special case when m + 1 is too large - use the
                        # previous interval
                        estimated_interval = estimated_beats[m] - estimated_beats[m - 1]
                    # Handle this special case when beats are not unique
                    if reference_interval == 0:
                        if estimated_interval == 0:
                            period = 0
                        else:
                            period = np.inf
                    else:
                        period = np.abs(1 - estimated_interval / reference_interval)
                    if (
                        phase < continuity_phase_threshold
                        and period < continuity_period_threshold
                    ):
                        # Set this annotation as used
                        used_annotations[nearest] = 1
                        # This beat is matched
                        beat_success = 1
                # This beat/annotation is not the first
                else:
                    # How far is the estimated beat from the reference beat,
                    # relative to the inter-annotation-interval?
                    reference_interval = (
                        reference_beats[nearest] - reference_beats[nearest - 1]
                    )
                    phase = np.abs(min_difference / reference_interval)
                    # How close is the inter-beat-interval
                    # to the inter-annotation-interval?
                    estimated_interval = estimated_beats[m] - estimated_beats[m - 1]
                    reference_interval = (
                        reference_beats[nearest] - reference_beats[nearest - 1]
                    )
                    period = np.abs(1 - estimated_interval / reference_interval)
                    if (
                        phase < continuity_phase_threshold
                        and period < continuity_period_threshold
                    ):
                        # Set this annotation as used
                        used_annotations[nearest] = 1
                        # This beat is matched
                        beat_success = 1
            # Set whether this beat is matched or not
            beat_successes[m] = beat_success
        # Add 0s at the begnning and end
        # so that we at least find the beginning/end of the estimated beats
        beat_successes = np.append(np.append(0, beat_successes), 0)
        # Where is the beat not a match?
        beat_failures = np.nonzero(beat_successes == 0)[0]
        # Take out those zeros we added
        beat_successes = beat_successes[1:-1]
        # Get the continuous accuracy as the longest track of successful beats
        longest_track = np.max(np.diff(beat_failures)) - 1
        continuous_accuracy = longest_track / (1.0 * beat_successes.shape[0])
        continuous_accuracies.append(continuous_accuracy)
        # Get the total accuracy - all sequences
        total_accuracy = np.sum(beat_successes) / (1.0 * beat_successes.shape[0])
        total_accuracies.append(total_accuracy)
    # Grab accuracy scores
    return (
        continuous_accuracies[0],
        total_accuracies[0],
        np.max(continuous_accuracies),
        np.max(total_accuracies),
    )


def information_gain(reference_beats, estimated_beats, bins=41):
    """Get the information gain - K-L divergence of the beat error histogram
    to a uniform histogram

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> reference_beats = mir_eval.beat.trim_beats(reference_beats)
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    >>> information_gain = mir_eval.beat.information_gain(reference_beats,
                                                          estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        query beat times, in seconds
    bins : int
        Number of bins in the beat error histogram
        (Default value = 41)

    Returns
    -------
    information_gain_score : float
        Entropy of beat error histogram
    """
    validate(reference_beats, estimated_beats)

    # When estimated or reference beats have <= 1 beats, can't compute the
    # metric, so return 0
    if estimated_beats.size <= 1 or reference_beats.size <= 1:
        return 0.0
    # Get entropy for reference beats->estimated beats
    # and estimated beats->reference beats
    forward_entropy = _get_entropy(reference_beats, estimated_beats, bins)
    backward_entropy = _get_entropy(estimated_beats, reference_beats, bins)
    # Pick the larger of the entropies
    norm = np.log2(bins)
    if forward_entropy > backward_entropy:
        # Note that the beat evaluation toolbox does not normalize
        information_gain_score = (norm - forward_entropy) / norm
    else:
        information_gain_score = (norm - backward_entropy) / norm
    return information_gain_score


def _get_entropy(reference_beats, estimated_beats, bins):
    """Helper function for information gain
    (needs to be run twice - once backwards, once forwards)

    Parameters
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        query beat times, in seconds
    bins : int
        Number of bins in the beat error histogram

    Returns
    -------
    entropy : float
        Entropy of beat error histogram

    """
    beat_error = np.zeros(estimated_beats.shape[0])
    for n in range(estimated_beats.shape[0]):
        # Get index of closest annotation to this beat
        beat_distances = estimated_beats[n] - reference_beats
        closest_beat = np.argmin(np.abs(beat_distances))
        absolute_error = beat_distances[closest_beat]
        # If the first annotation is closest...
        if closest_beat == 0:
            # Inter-annotation interval - space between first two beats
            interval = 0.5 * (reference_beats[1] - reference_beats[0])
        # If last annotation is closest...
        if closest_beat == (reference_beats.shape[0] - 1):
            interval = 0.5 * (reference_beats[-1] - reference_beats[-2])
        else:
            if absolute_error < 0:
                # Closest annotation is the one before the current beat
                # so look at previous inner-annotation-interval
                start = reference_beats[closest_beat]
                end = reference_beats[closest_beat - 1]
                interval = 0.5 * (start - end)
            else:
                # Closest annotation is the one after the current beat
                # so look at next inner-annotation-interval
                start = reference_beats[closest_beat + 1]
                end = reference_beats[closest_beat]
                interval = 0.5 * (start - end)
        # The actual error of this beat
        beat_error[n] = 0.5 * absolute_error / interval
    # Put beat errors in range (-.5, .5)
    beat_error = np.mod(beat_error + 0.5, -1) + 0.5
    # Note these are slightly different the beat evaluation toolbox
    # (they are uniform)
    histogram_bin_edges = np.linspace(-0.5, 0.5, bins + 1)
    # Get the histogram
    raw_bin_values = np.histogram(beat_error, histogram_bin_edges)[0]
    # Turn into a proper probability distribution
    raw_bin_values = raw_bin_values / (1.0 * np.sum(raw_bin_values))
    # Set zero-valued bins to 1 to make the entropy calculation well-behaved
    raw_bin_values[raw_bin_values == 0] = 1
    # Calculate entropy
    return -np.sum(raw_bin_values * np.log2(raw_bin_values))


def has_kwargs(function):
    r"""Determine whether a function has \*\*kwargs.

    Parameters
    ----------
    function : callable
        The function to test

    Returns
    -------
    True if function accepts arbitrary keyword arguments.
    False otherwise.
    """

    if six.PY2:
        return inspect.getargspec(function).keywords is not None
    else:
        sig = inspect.signature(function)

        for param in sig.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                return True

        return False


def filter_kwargs(_function, *args, **kwargs):
    r"""Given a function and args and keyword args to pass to it, call the function
    but using only the keyword arguments which it accepts.  This is equivalent
    to redefining the function with an additional **kwargs to accept slop
    keyword args.

    If the target function already accepts **kwargs parameters, no filtering
    is performed.

    Parameters
    ----------
    _function : callable
        Function to call.  Can take in any number of args or kwargs

    """

    if has_kwargs(_function):
        return _function(*args, **kwargs)

    # Get the list of function arguments
    func_code = six.get_function_code(_function)
    function_args = func_code.co_varnames[: func_code.co_argcount]
    # Construct a dict of those kwargs which appear in the function
    filtered_kwargs = {}
    for kwarg, value in list(kwargs.items()):
        if kwarg in function_args:
            filtered_kwargs[kwarg] = value
    # Call the function with the supplied args and the filtered kwarg dict
    return _function(*args, **filtered_kwargs)


def evaluate(reference_beats, estimated_beats, **kwargs):
    """Compute all metrics for the given reference and estimated annotations.

    Examples
    --------
    >>> reference_beats = mir_eval.io.load_events('reference.txt')
    >>> estimated_beats = mir_eval.io.load_events('estimated.txt')
    >>> scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)

    Parameters
    ----------
    reference_beats : np.ndarray
        Reference beat times, in seconds
    estimated_beats : np.ndarray
        Query beat times, in seconds
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.

    """

    # Trim beat times at the beginning of the annotations
    reference_beats = filter_kwargs(trim_beats, reference_beats, **kwargs)
    estimated_beats = filter_kwargs(trim_beats, estimated_beats, **kwargs)

    # Now compute all the metrics

    scores = collections.OrderedDict()

    # F-Measure
    scores["F-measure"] = filter_kwargs(
        f_measure, reference_beats, estimated_beats, **kwargs
    )

    # Cemgil
    scores["Cemgil"], scores["Cemgil Best Metric Level"] = filter_kwargs(
        cemgil, reference_beats, estimated_beats, **kwargs
    )

    # Goto
    scores["Goto"] = filter_kwargs(goto, reference_beats, estimated_beats, **kwargs)

    # P-Score
    scores["P-score"] = filter_kwargs(
        p_score, reference_beats, estimated_beats, **kwargs
    )

    # Continuity metrics
    (
        scores["Correct Metric Level Continuous"],
        scores["Correct Metric Level Total"],
        scores["Any Metric Level Continuous"],
        scores["Any Metric Level Total"],
    ) = filter_kwargs(continuity, reference_beats, estimated_beats, **kwargs)

    # Information gain
    scores["Information gain"] = filter_kwargs(
        information_gain, reference_beats, estimated_beats, **kwargs
    )

    return scores
