from enum import IntEnum


class ContextualStrategy(IntEnum):
    # find continuous samples according to a context length
    CONTINUOUS_CONTEXT = 1
    SAME_SPEAKER = 2
    DIALOGUE_ABA = 3
    # non-overlapping version of CONTINUOUS_CONTEXT, for inference mainly
    NON_OVERLAPPING_CONTEXT = 4
    # greedily find continuous same-speakers samples up to a speaker turn limit
    GREEDY_CONTEXT = 5

    @classmethod
    @property
    def values(cls):
        return list(map(lambda c: c.value, cls))
