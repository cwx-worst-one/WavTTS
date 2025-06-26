"""
Predict duration given lyrics and genre.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from .consts import DURATION_RANGE, SLB_DUR_DIST_CONFIG
from .utils import select_genre_in_list
from recipes.bigmusic.datasets.utils.zh_lyrics_proc import SongLyrics

logger = logging.getLogger(__file__)


SUPPORTED_GENRES = list(SLB_DUR_DIST_CONFIG.keys())

STD_SCALE = 0.3  # the spread of the distribution, a lower value means the distribution has a more focused spread
BOUND_SCALE = 2.0  # duration result will be clipped within [mean - bound_scale * std, mean + bound_scale * std]


class Distribution:
    def __init__(self, mean: Union[List, np.ndarray], cov: Union[List, np.ndarray]):
        self.mean = np.array(mean)
        self.cov = np.array(cov)

    def sample(
        self,
        x_value,
        std_scale: float,
        bound_scale: float,
        random_seed: Optional[int] = None,
    ) -> float:
        """
        Args:
            x_value: The value that the distribution is conditioned on.
            std_scale: The scale of the standard deviation of the distribution, controlling the spread of the distribution.
            bound_scale: The result will be clipped to [mean - bound_scale * std, mean + bound_scale * std].
            random_seed: The random seed for the sampling.
        Return:
            A sample from the distribution.
        """
        mean, std = self.cond_mean_std(x_value)
        result = _sample_normal(mean, std * std_scale, random_seed=random_seed)
        result = _clip(result, _bound_range(mean, std, bound_scale))
        return _clip(result, DURATION_RANGE)

    def cond_mean_std(self, x_value: float) -> Tuple[float, float]:
        # Regularize the covariance matrix to ensure it's positive definite
        cov = self.cov + np.eye(self.cov.shape[0]) * 1e-6
        # Split the mean and covariance matrix
        mu_x = self.mean[0]
        mu_y = self.mean[1]
        sigma_xx = cov[0, 0]
        sigma_yy = cov[1, 1]
        sigma_xy = cov[0, 1]
        sigma_yx = cov[1, 0]
        # Conditional mean of y given x
        mean = mu_y + sigma_yx / sigma_xx * (x_value - mu_x)
        # Conditional variance of y given x
        std = np.sqrt(sigma_yy - sigma_yx * sigma_xy / sigma_xx)
        return mean, std

    @classmethod
    def from_genre(cls, genre: Optional[Union[str, List[str]]]) -> "Distribution":
        genre = select_genre_in_list(genre, SUPPORTED_GENRES, logger)
        return cls(**SLB_DUR_DIST_CONFIG[genre])


def generate_duration(
    lyrics: str,
    genre: Optional[Union[str, List[str]]],
    mood: Optional[str] = None,
    prompt_duration: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> int:
    # TODO (hang): update duration predict logic: https://code.byted.org/data-speech/preprocess_py/blob/zh_vocal/v5_mir_api/preprocess/bigmusic/transforms/duration_v2.py
    # mood is not used for now
    dist = Distribution.from_genre(genre)
    song_lyrics = SongLyrics.parse(lyrics)
    if (
        song_lyrics[-1].section_tag == "inst"
    ):  # If the last section is inst, it is typicall a short song
        song_lyrics = song_lyrics[:-1]
    duration = dist.sample(
        song_lyrics.n_syllables,
        std_scale=STD_SCALE,
        bound_scale=BOUND_SCALE,
        random_seed=random_seed,
    )
    # logger.info("Sampled duration: {}".format(duration))
    n_non_vocal_sections = len(
        [paragraph for paragraph in song_lyrics if not paragraph.has_utterance]
    )
    duration = _adjust_duration(duration, n_non_vocal_sections, random_seed=random_seed)
    # logger.info("Adjusted duration: {}".format(duration))

    # Convert the duration from full song duration to the duration of the continuation part
    # Make sure the continuation part lasts at least DURATION_RANGE[0] seconds.
    prompt_duration = _reset_prompt_duration(prompt_duration)
    duration = max(DURATION_RANGE[0], duration - prompt_duration)

    return round(duration)


def transform_duration(
    duration: float,
    lyrics: str,
    genre: Optional[Union[str, List[str]]],
    prompt_duration: Optional[int],
) -> int:
    """Ensure the duration is in a proper range"""
    n_slbs = SongLyrics.parse(lyrics).n_syllables
    dist = Distribution.from_genre(genre)
    mean, std = dist.cond_mean_std(n_slbs)
    min_dur, max_dur = _bound_range(mean, std, BOUND_SCALE)
    dur_range = (max(min_dur, DURATION_RANGE[0]), min(max_dur, DURATION_RANGE[1]))
    # logger.info(
    #     "Duration range of the full song (with prompt audio): {}".format(dur_range)
    # )

    # Calculate the duration range of the generation part
    prompt_duration = _reset_prompt_duration(prompt_duration)
    dur_range = (
        max(0, dur_range[0] - prompt_duration),
        max(0, dur_range[1] - prompt_duration),
    )
    # logger.info(
    #     "Duration range of the generation part (without prompt audio): {}".format(
    #         dur_range
    #     )
    # )

    # Make sure the duration of the generation part is in range
    if not (dur_range[0] <= duration <= dur_range[1]):
        logger.info(
            "Duration {} is not in range {} for genre {}, clip".format(
                duration, dur_range, genre
            )
        )
    return round(_clip(duration, dur_range))


def _sample_normal(mean: float, std: float, random_seed: Optional[int] = None) -> float:
    rnd = np.random.RandomState(seed=random_seed)
    return rnd.normal(mean, std)


def _clip(val: float, val_range: Tuple[float, float]):
    min_val, max_val = val_range
    return max(min(val, max_val), min_val)


def _bound_range(mean: float, std: float, bound_scale: float) -> Tuple[float, float]:
    return (mean - bound_scale * std, mean + bound_scale * std)


def _adjust_duration(
    duration: float,
    n_non_vocal_sections: int,
    avg_inst_ratio: float = 0.2,
    random_seed: Optional[int] = None,
) -> float:
    # TODO (Yilin): The best way is to include structure in the stats
    # Since the previous sampling stage does not consider the instrumental section,
    # we assume that the duration value is an average value given the number of syallbles,
    # and the total duration of all instrumental parts is avg_inst_dur
    avg_inst_dur = duration * avg_inst_ratio
    # Roughly guess an actual instrumental duration based on the number of non-vocal sections
    inferred_inst_dur = (
        sum(
            _sample_normal(mean=10, std=3, random_seed=random_seed)
            for _ in range(n_non_vocal_sections)
        )
        if n_non_vocal_sections > 0
        else 0
    )
    return duration - avg_inst_dur + inferred_inst_dur


def _reset_prompt_duration(prompt_duration: Optional[float]) -> float:
    if prompt_duration is None:
        prompt_duration = 0
    if prompt_duration >= DURATION_RANGE[1]:
        raise ValueError(f"Prompt duration {prompt_duration} is too large")
    return prompt_duration
