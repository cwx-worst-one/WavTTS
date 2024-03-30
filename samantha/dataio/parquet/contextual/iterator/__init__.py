import sys

from ..strategy import ContextualStrategy
from .contexture_iterator import ContextualIterator
from .dialogue_aba_iterator import DialogueABAIterator
from .greedy_iterator import GreedyIterator
from .non_overlapping_iterator import NonOverlappingIterator
from .same_speaker_iterator import SameSpeakerIterator


def get_iterator(sample_config: dict):
    if sample_config is None:
        sample_config = {}
    min_context_num = sample_config.get("min_context_num", 1)
    max_context_num = sample_config.get("max_context_num", 1)
    time_interval_threshold = sample_config.get("time_interval_threshold", 1)
    # for strategy DIALOGUE_ABA
    dialogue_aba_source_a_time_threshold = sample_config.get(
        "dialogue_aba_source_a_time_threshold", 30
    )
    dialogue_aba_source_b_time_threshold = sample_config.get(
        "dialogue_aba_source_b_time_threshold", 30
    )
    dialogue_aba_target_a_time_threshold = sample_config.get(
        "dialogue_aba_target_a_time_threshold", 30
    )
    main_speaker_round = sample_config.get("main_speaker_round", 2)
    # for strategy GREEDY_CONTEXT
    min_num_speakers = sample_config.get("min_num_speakers", 0)
    max_num_speakers = sample_config.get("max_num_speakers", sys.maxsize)
    max_duration = sample_config.get("max_duration", float("inf"))
    split_context = sample_config.get("split_context", False)

    strategy = sample_config.get(
        "contextual_strategy", ContextualStrategy.CONTINUOUS_CONTEXT
    )
    if strategy not in ContextualStrategy.values:
        raise ValueError(f"invalid contextual strategy: {strategy}")

    iterator_kwargs = {
        "split_context": split_context,
        "main_speaker_round": main_speaker_round,
        "min_context_num": min_context_num,
        "max_context_num": max_context_num,
        "min_num_speakers": min_num_speakers,
        "max_num_speakers": max_num_speakers,
        "max_duration": max_duration,
        "time_interval_threshold": time_interval_threshold,
        "dialogue_aba_source_a_time_threshold": dialogue_aba_source_a_time_threshold,
        "dialogue_aba_target_a_time_threshold": dialogue_aba_target_a_time_threshold,
        "dialogue_aba_source_b_time_threshold": dialogue_aba_source_b_time_threshold,
    }
    if strategy == ContextualStrategy.CONTINUOUS_CONTEXT:
        return ContextualIterator(**iterator_kwargs)
    if strategy == ContextualStrategy.SAME_SPEAKER:
        return SameSpeakerIterator(**iterator_kwargs)
    if strategy == ContextualStrategy.DIALOGUE_ABA:
        return DialogueABAIterator(**iterator_kwargs)
    if strategy == ContextualStrategy.NON_OVERLAPPING_CONTEXT:
        return NonOverlappingIterator(**iterator_kwargs)
    if strategy == ContextualStrategy.GREEDY_CONTEXT:
        return GreedyIterator(**iterator_kwargs)
    raise ValueError(f"Unknown strategy {strategy}")
