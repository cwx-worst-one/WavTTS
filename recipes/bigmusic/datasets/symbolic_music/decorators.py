from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


def dfs_dict_section_info_extract(
    dfs_dict: Dict[str, Any],
    i: int,
    sec_row: pd.Series,
    reset_start_time=False,
):
    """Extract the i-th section from `dfs_dict`. It defines several
    coversion rules for each type of df and keep the original item
    if unseen key is encountered.

    When `reset_start_time` is True, returned `dfs_dict` will start
    from 0 in time. Otherwise will keep the original start time. Default
    is False.

    NOTE that we have different logic for chords and notes - for
        chords, we keep the start and end chords that are outside section
        boundary. But for notes, they have to be fully inside the section
        to be considered.
    """

    def df_beat_extractor(df_beat: pd.DataFrame):
        df_beat = df_beat.copy()
        df_beat_sec = df_beat[
            df_beat.time.apply(
                lambda x: x >= sec_row.start_time and x <= sec_row.end_time
            )
        ].reset_index(drop=True)
        if len(df_beat_sec) > 0 and reset_start_time:
            df_beat_sec["time"] = df_beat_sec.time - sec_row.start_time
        return df_beat_sec

    def df_chord_extractor(df_chord: pd.DataFrame):
        df_chord = df_chord.copy()
        sec_s, sec_e = sec_row.start_time, sec_row.end_time

        ## For chords, they just need to be overlapped with the section.
        df_chord_sec = df_chord[
            df_chord.apply(
                lambda x: x.end > sec_s and x.start < sec_e,
                axis=1,
            )
        ].reset_index(drop=True)

        if len(df_chord_sec) == 0:
            return df_chord_sec
        ## Clip chords at start and end boundary.
        if df_chord_sec["start"].iloc[0] < sec_s:
            df_chord_sec.at[0, "start"] = sec_s
        if df_chord_sec["end"].iloc[-1] > sec_e:
            df_chord_sec.at[len(df_chord_sec) - 1, "end"] = sec_e
        if reset_start_time:
            df_chord_sec["start"] = df_chord_sec.start - sec_s
            df_chord_sec["end"] = df_chord_sec.end - sec_s
        return df_chord_sec

    def df_note_extractor(df_note: pd.DataFrame):
        df_note = df_note.copy()

        ## For notes, they have to be fully inside the section.
        df_note_sec = df_note[
            df_note.apply(
                lambda x: x.start >= sec_row.start_time and x.end <= sec_row.end_time,
                axis=1,
            )
        ].reset_index(drop=True)
        if len(df_note_sec) > 0 and reset_start_time:
            df_note_sec["start"] = df_note_sec.start - sec_row.start_time
            df_note_sec["end"] = df_note_sec.end - sec_row.start_time
        return df_note_sec

    def df_key_extrator(df_key: pd.DataFrame):
        df_key = df_key.copy()
        df_key_sec = df_key[
            df_key.time.apply(
                lambda x: x >= sec_row.start_time and x <= sec_row.end_time
            )
        ].reset_index(drop=True)
        if len(df_key_sec) > 0 and reset_start_time:
            df_key_sec["time"] = df_key_sec.time - sec_row.start_time
        return df_key_sec

    def audio_sec_extractor(input: Tuple[np.ndarray, float]):
        audio, sr = input
        start_s = int(round(sec_row.start_time * sr))
        end_s = int(round(sec_row.end_time * sr))
        if len(audio.shape) == 1:
            return audio[start_s:end_s], sr
        elif len(audio.shape) == 2 and audio.shape[0] <= 2:
            return audio[:, start_s:end_s], sr
        elif len(audio.shape) == 2 and audio.shape[1] <= 2:
            return audio[start_s:end_s], sr
        else:
            raise ValueError(f"Audio shape not supported: {audio.shape}")

    def df_section_extractor(*args):
        df_sec = pd.DataFrame(sec_row).T.reset_index(drop=True)
        if reset_start_time:
            df_sec["start_time"] = 0
            df_sec["end_time"] = df_sec.end_time - sec_row.start_time
        return df_sec

    def df_lyrics_extractor(*args):
        raise NotImplementedError("Extractor for df_lyrics not implemented")

    def df_vocal2midi_extractor(*args):
        raise NotImplementedError("Extractor for df_lyrics not implemented")

    extractor_dict = {
        "key": lambda k: f"{k}.s{i}",
        "df_beat": df_beat_extractor,
        "df_chord": df_chord_extractor,
        "df_note": df_note_extractor,
        "df_key": df_key_extrator,
        "audio": audio_sec_extractor,
        "df_lyrics": df_lyrics_extractor,
        "df_vocal2midi": df_vocal2midi_extractor,
        "df_section": df_section_extractor,
    }

    result_dict = {}
    for k, v in dfs_dict.items():
        if k in extractor_dict:
            result_dict[k] = extractor_dict[k](v)
        else:
            result_dict[k] = v
            # raise ValueError(f"Key {k} not handled by section extraction rules")
    return result_dict


def split_section(
    selected_sec: List[str],
    reset_start_time: bool = False,
):
    """Convert a full song iter to section iter.

    This also assume there is a "index_dict" field in dfs_dict,
    keeping useful debug information for different level of index
    (shard index, original sample index, section index, etc.)

    selected_sec possible values: "intro", "verse", "chorus", "instrument",
    "bridge", "outro", "silence"
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            gen = func(*args, **kwargs)
            for dfs_dict in gen:
                df_section = dfs_dict["df_section"]
                df_section_selected = df_section[
                    df_section.function_name.apply(lambda x: x in selected_sec)
                ]
                for i, row in df_section_selected.iterrows():
                    dfs_dict_sec = dfs_dict_section_info_extract(
                        dfs_dict, i, row, reset_start_time=reset_start_time
                    )

                    yield dfs_dict_sec

        return wrapper

    return decorator


def until(num_samples: int):
    """Iterate until `num_samples` samples"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            gen = func(*args, **kwargs)
            for i, v in enumerate(gen):
                if num_samples > 0 and i >= num_samples:
                    break
                yield v

        return wrapper

    return decorator


def skip_by(skip_ratio: float, seed: Optional[float] = None):
    """Randomly skip iterator by a chance"""
    if seed is not None:
        np.random.seed(seed)

    def decorator(func):
        def wrapper(*args, **kwargs):
            gen = func(*args, **kwargs)
            for v in gen:
                random_num = np.random.rand()
                if random_num > (1 - skip_ratio):
                    continue
                yield v

        return wrapper

    return decorator


def skip_keys(keys: set = set()):
    """Skip a key from dfs_dict.
    NOTE that this decorator assume the iterator yields
    dfs_dict: Dict[str, Any] and dfs_dict contains a unique "key"
    field for each sample as its ID.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            gen = func(*args, **kwargs)
            for dfs_dict in gen:
                if len(keys) > 0 and dfs_dict["key"] in keys:
                    continue
                yield dfs_dict

        return wrapper

    return decorator

