"""
Intermediate meta data representation for Chinese vocal datasets with parsing and transforming functions
"""

from dataclasses import dataclass
from functools import partial, reduce
import operator
from typing import Any, Optional, List, Dict, Tuple, Union
import math
import random
from random import Random
import copy
import numpy as np

from recipes.bigmusic.datasets.mir_data_util import (
    ARTIST_ID_MAP_V2,
    SA_CAT_VOCAB,
    SA_TAGS_SPECIAL_MAP,
    AUDIO_CAT_VOCAB_V0,
    AUDIO_TAGS_GENRE_SPECIAL_MAP_V0,
    AUDIO_TAGS_MOOD_SPECIAL_MAP_V0,
    AUDIO_TAGS_SCENE_SPECIAL_MAP_V0,
    AUDIO_TAGS_GENDER_SPECIAL_MAP_V0,
    AUDIO_CAT_VOCAB_V1,
    AUDIO_TAGS_GENRE_SPECIAL_MAP_V1,
    AUDIO_TAGS_MOOD_SPECIAL_MAP_V1,
    AUDIO_TAGS_SCENE_SPECIAL_MAP_V1,
    AUDIO_TAGS_GENDER_SPECIAL_MAP_V1,
    AUDIO_CAT_VOCAB_V2,
    AUDIO_TAGS_GENRE_SPECIAL_MAP_V2,
    AUDIO_TAGS_MOOD_SPECIAL_MAP_V2,
    AUDIO_TAGS_SCENE_SPECIAL_MAP_V2,
    AUDIO_TAGS_GENDER_SPECIAL_MAP_V2,
    VOICE_THRESHOLDS,
    TEMPO_RANGE,
    tempo_to_label,
    TEMPO_LABEL_ID_MAP,
    KEYS,
    KEY_ID_MAP,
)
from recipes.datasets.mcc.sami_tokenizer import Phrase


@dataclass
class DeepChorus:
    tags: List
    confidence: float

@dataclass
class SongSlice:
    phrases: List[Phrase]

    @property
    def start(self) -> int:
        return self.phrases[0].start

    @property
    def end(self) -> int:
        return self.phrases[-1].end

    @property
    def duration(self) -> int:
        return self.end - self.start

    def is_time_span_valid(self, time_span: Tuple[int, int]) -> bool:
        start, end, duration = self.start, self.end, self.duration
        min_duration, max_duration = time_span
        return ((min_duration <= duration <= max_duration) and 
                start >= 0 and end >= 0 and start < end)

    def slice_audio(self, audio, sample_rate: int, extra_clip=None):
        slice_start, slice_end = self.start, self.end
        start = int(slice_start * sample_rate)
        end = int(slice_end * sample_rate)
        if extra_clip is not None:
            return audio[:, start:end], [x[:, start:end] for x in extra_clip]
        else:
            return audio[:, start:end], None

    @staticmethod
    def reformat_and_dropout(
        line_break_dropout_rate: float,
        section_tag_dropout_rate: float,
        phrases: List[Phrase]
    ) -> List[Phrase]:
        """Reformat the phrases to have single-line section tags. Dropout line breaks and section tags."""
        # The execution order matters
        phrases = [phrase for phrase in phrases if not phrase.is_empty]  # Remove empty phrases, which might be short inst phrases with section tags removed
        phrases = drop_out_line_breaks(phrases, line_break_dropout_rate)
        phrases = move_out_section_tags(phrases)  # Reformat the phrases to have single-line section tags
        phrases = remove_section_tag_counts(phrases)  # Remove the count number
        return drop_out_section_tags(phrases, section_tag_dropout_rate)

    def __add__(self, other):
        return self.__class__(phrases=self.phrases + other.phrases)


def drop_out_line_breaks(phrases: List[Phrase], rate: float, seed: Optional[int] = None) -> List[Phrase]:
    """Merge adjacent concatable phrases. The phrase list should NOT be reformatted by move_out_section_tags."""
    if not (0 <= rate <= 1):
        raise ValueError(f"Invalid dropout rate: {rate}")
    if len(phrases) <= 1 or rate == 0:
        return phrases
    mergeable_ind = [
        idx for idx, (curr_phrase, next_phrase) in enumerate(zip(phrases, phrases[1:])) 
        if Phrase.concatable(curr_phrase, next_phrase)
    ]
    if not mergeable_ind:
        return phrases
    rand_gen = Random(seed)
    phrases = phrases[:]  # shallow copy a new slice, Phrase is immutable so it's fine
    for idx in reversed(mergeable_ind):  # reverse it because the list shrinks
        if rand_gen.random() < rate:
            phrases[idx:idx+2] = [Phrase.concat(phrases[idx], phrases[idx+1])]
    return phrases


def move_out_section_tags(phrases: List[Phrase]) -> List[Phrase]:
    """Move section tags out of phrases as single phrases
    This function reformats a list of phrases to a format where section tags
    only appear once at the top of each section.
    """
    out_phrases = []
    for prev_phrase, phrase in zip([None] + phrases, phrases):
        if phrase.section_tag and (prev_phrase is None or phrase.section_tag != prev_phrase.section_tag):
            out_phrases.append(Phrase(section_tag=phrase.section_tag))
        if phrase.has_utterance:
            out_phrases.append(phrase._replace(section_tag=None))
    return out_phrases


def remove_section_tag_counts(phrases: List[Phrase]) -> List[Phrase]:
    """Remove the count number in the section_tags. Call this function AFTER move_out_section_tags."""
    return [
        phrase._replace(
            section_tag = None if phrase.section_tag is None else _remove_count_from_section_tag(phrase.section_tag)
        ) for phrase in phrases
    ]


def drop_out_section_tags(phrases: List[Phrase], rate: float, seed: Optional[int] = None) -> List[Phrase]:
    """Remove phrases with section tags. The phrase list SHOULD be reformatted by move_out_section_tags.
    - The function either drops out or keep all the section tags. Partial dropout could contaminate the
      training data by placing multiple sections under one section tag.
    - The function does not perform dropout if the given phrases do not have utterance, because empty
      phrases can't be tokenized.
    """
    if not (0 <= rate <= 1):
        raise ValueError(f"Invalid dropout rate: {rate}")
    if not any(phrase.has_utterance for phrase in phrases):
        return phrases
    rand_gen = Random(seed)
    if rand_gen.random() < rate:
        return [phrase for phrase in phrases if phrase.section_tag is None]
    return phrases


# ------------------------------------------

# The schema of the meta data is always changing, there is no way to define
# a fixed data structure. The current solution splits the parsing into two stages.
# The first stage parses the meta data into an intermediate representation,
# The second stage transforms it into SongSlices with filtering.
#
# For each dataset, create a subclass of `ZhSongMeta` and override the `parse` method
# with your specific parsing logic.
#
# If the data lacks the information that is intended to have, raise an error immediately, 
# unless the field is totally optional. It's the caller's responsibility to handle the error.


class ZhMetaError(Exception):
    pass


class ZhMetaParseError(ZhMetaError):
    """Data itself is invalid."""
    pass


class ZhMetaTransformError(ZhMetaError):
    """Data did not pass the filtering because of certain properties."""
    pass


def _get_value(_dict: Dict, key: Any, msg: Optional[str] = None) -> Any:
    """Try to get the value using the given key, raise a parser error if the key does not exist."""
    if key not in _dict:
        _msg = f"Key {key} not found" if msg is None else msg
        raise ZhMetaParseError(_msg)
    return _dict[key]


class ZhMetaLogger:
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg

    def __repr__(self) -> str:
        return f"<ZhMetaLogger: {self.msg}>"


@dataclass
class ZhMetaBase:
    utterances: List
    lyrics_confidence: Optional[float]
    structure_tags: Optional[DeepChorus]
    style_text: List[str]
    artist_id: int

    # misc
    unfamiliar_tags: Dict[str, str]
    is_sinking: bool

    # optional labels
    voice_tag: Optional[str] = None
    is_high_quality: Optional[bool] = None
    is_popular_potential: Optional[bool] = None
    source: Optional[str] = None

    # MIR labels
    tempo: Optional[int] = None
    key: Optional[str] = None

    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        # Parse and check the data here
        raise NotImplementedError()

    def transform(
        self,
        lyrics_confidence: Optional[float],
        segment_method: str,
        max_seg_per_track: int,
        duration_range: Tuple[int, int]
    ) -> Dict[str, Any]:
        self._validate(lyrics_confidence)
        _self = self._convert()
        logger, song_slices = _self._to_song_slices(segment_method, max_seg_per_track, duration_range)
        return {
            "logger": logger,  # ZhMetaLogger
            "song_slices": song_slices,  # List[SongSlices]
            "style_text": _self.style_text,  # List[str]
            "artist_id": _self.artist_id,  # int
            "lyrics_confidence": _self.lyrics_confidence,  # Optional[float]
            "structure_tags": _self.structure_tags,  # DeepChorus
            "tempo_label": _self._to_tempo_label_id(),  # int
            "key": _self._to_key_id(),  # int
        }

    def _validate(self, lyrics_confidence: Optional[float]):
        """Raise ZhMetaTransformError if the data is invalid"""
        validate_confidence_optional(lyrics_confidence, self.lyrics_confidence)
        validate_deepchorus_optional(self.structure_tags)
        validate_mir_tempo_optional(self.tempo)
        validate_mir_key_optional(self.key)

    def _convert(self):
        """Process the data and return a new data. No in-place operation."""
        _self = copy.deepcopy(self)
        convert_voice_tag(_self)
        return _self

    def _to_song_slices(
        self, 
        segment_method: str, 
        max_seg_per_track: int,
        duration_range: Tuple[int, int],
    ) -> Tuple[ZhMetaLogger, List[SongSlice]]:
        min_duration, max_duration = duration_range
        # The presence of structure_tags is decided by the parser. Therefore, we don't need to
        # check structure_tags here.
        if self.structure_tags is None:
            song_slices = transform_utts_to_song_slices_heuristic(
                self.utterances, min_duration, max_duration
            )
        else:
            song_slices = transform_utts_to_song_slices_structure(
                self.utterances,
                min_duration,
                max_duration,
                structure_tags=self.structure_tags.tags,
                complete_section=max_duration >= 60,  # auto-enable complete section grouping for dur >= 1m
            )
        song_slices, logger = filter_song_slices(song_slices, duration_range)
        song_slices = take_song_slices_by_method(song_slices, segment_method)
        song_slices = get_max_seg(song_slices, max_seg_per_track)
        return logger, song_slices

    def _to_tempo_label_id(self) -> int:
        return TEMPO_LABEL_ID_MAP[tempo_to_label(self.tempo)]

    def _to_key_id(self) -> int:
        k = "N" if self.key is None else self.key
        return KEY_ID_MAP[k]


# ------------------------------------------
#                 PARSERS
# ------------------------------------------

# ---------- utterance -------------

def parse_utterance_lyrics_gt(meta: Dict) -> List:
    """meta.lyrics_gt"""
    utterances = _get_value(meta, "lyrics_gt", "No lyrics_gt")
    if utterances is None or len(utterances) == 0:
        raise ZhMetaParseError("No utterances")
    return utterances


def parse_utterance_lyrics(meta: Dict) -> List:
    """meta.lyrics"""
    # Typically, utterances are in the "result" field. However, if the lyrics
    # are generated by ASR, it is possible to put the results under the "utterances" field.
    lyrics = _get_value(meta, "lyrics", "No lyrics")
    if not isinstance(lyrics, dict):
        raise ZhMetaParseError("Lyrics not in correct format")
    _result = lyrics.get("result")
    _asr_utterances = lyrics.get("utterances")
    if (  
        # empty or invalid result
        ((_result is None) or 
        (_result and len(_result) != 1))
        and
        # empty asr result
        _asr_utterances is None
        ):
        raise ZhMetaParseError("No lyrics.result")
    utterances = _result[0].get("utterances") if _result else _asr_utterances
    if utterances is None or len(utterances) == 0:
        raise ZhMetaParseError("No utterances")
    return utterances


def parse_utterance_mix(meta: Dict, lyrics_field: str) -> List:
    """meta.lyrics_gt or meta.lyrics"""
    if lyrics_field not in ["lyrics_gt", "lyrics"]:
        raise ValueError(f"Invalid lyrics_field {lyrics_field}")

    fn_map = {
        "lyrics_gt": parse_utterance_lyrics_gt,
        "lyrics": parse_utterance_lyrics,
    }
    if lyrics_field not in meta:  # switch lyrics_field
        _lyrics_field = "lyrics_gt" if lyrics_field == "lyrics" else "lyrics"
    else:
        _lyrics_field = lyrics_field
    utterances = fn_map[_lyrics_field](meta)
    return utterances


# ---------- structure_tags -------------
def get_deepchorus_score(deepchorus_tags):
    boundary = [b['start_prob'] for b in deepchorus_tags['segments']]
    function = [f['funct_prob'] for f in deepchorus_tags['segments']]
    return 0.7 * (sum(boundary)/len(boundary)) + 0.3 * (sum(function)/len(function))

_SECTION_TAG_SEP = "#"


def _get_deepchorus_score(deepchorus_tags):
    boundary = [b['start_prob'] for b in deepchorus_tags['segments']]
    function = [f['funct_prob'] for f in deepchorus_tags['segments']]
    return 0.7 * (sum(boundary)/len(boundary)) + 0.3 * (sum(function)/len(function))


def _add_count_to_section_tag(section_tag: str, count: int) -> str:
    if _SECTION_TAG_SEP in section_tag:
        return section_tag
    return f"{section_tag}{_SECTION_TAG_SEP}{str(count)}"


def _remove_count_from_section_tag(section_tag: str) -> str:
    return section_tag.split(_SECTION_TAG_SEP)[0]


def _format_deepchorus_structure_tags(deepchorus_tags: Dict) -> DeepChorus:
    """Convert the deepchorus field in metadata into the format of [{'tag': tag, 'start_time': sec, 'end_time': sec}].
    A number will be appended to the tag to differentiate adjacent tags with the same name.
    """
    structure_tags = []
    for count, segment in enumerate(deepchorus_tags['segments']):
        structure_tag = {
            "tag": _add_count_to_section_tag(segment["label"], count),
            "start_time": segment["interval"][0],
            "end_time": segment["interval"][1],
        }
        structure_tags.append(structure_tag)
    return DeepChorus(
        tags=structure_tags,
        confidence=_get_deepchorus_score(deepchorus_tags)
    )


def parse_structure_tags(meta: Dict) -> DeepChorus:
    """meta.deepchorus"""
    deepchorus_tags = _get_value(meta, "deepchorus", "No structure tags")
    return _format_deepchorus_structure_tags(deepchorus_tags)


def parse_structure_tags_optional(meta: Dict) -> Optional[DeepChorus]:
    try:
        return parse_structure_tags(meta)
    except ZhMetaParseError:
        return None


# ---------- lyrics_confidence -------------

def parse_lyrics_confidence_sa_asr(meta: Dict) -> float:
    """meta.lyrics.confidence"""
    get_value = partial(_get_value, msg="No confidence")
    return get_value(get_value(meta, "lyrics"), "confidence")


def parse_lyrics_confidence_force_alignment(
    meta: Dict,
    utterances: Optional[List] = None
) -> float:
    """meta.lyrics.confidence_avg or calculate average from utterances"""
    lyrics = _get_value(meta, "lyrics", "No lyrics")
    force_alignment_conf = lyrics.get("confidence_avg")
    if force_alignment_conf is not None:
        return force_alignment_conf

    if not utterances:
        raise ZhMetaParseError("No confidence found")
    
    # calculate avg confidence
    conf, num_utt = 0., 0.
    for utt in utterances:
        if utt['text'].strip():
            conf += float(utt.get("confidence", 1))
            num_utt += 1
    if num_utt == 0:
        return 0.0
    conf /= num_utt
    return conf


def parse_lyrics_confidence_optional(
    meta: Dict,
    utterances: Optional[List] = None
) -> Optional[float]:
    for parse_fn in [
        parse_lyrics_confidence_sa_asr, 
        partial(parse_lyrics_confidence_force_alignment, utterances=utterances),
    ]:
        try:
            return parse_fn(meta)
        except ZhMetaParseError:
            continue
    return None


# ---------- style_text -------------

def _parse_sa_music_tagging(music_tagging: Optional[Dict], sinking_threshold: float) -> Tuple[List[str], Dict[str, str], bool]:
    def parse_result(result: Union[List, str]) -> str:
        if isinstance(result, str):
            return result
        if len(result) == 0:
            return ""
        return result[0]

    def map_tag(tag: str) -> str:
        """Replace certain tags in the dataset"""
        return SA_TAGS_SPECIAL_MAP.get(tag, tag)

    # The order should match `mir_data_util`
    order = ["Genre20", "Mood", "Theme", "MusicLowQuality", "Language"]

    if music_tagging is None:
        return [""] * len(order), {}, False
    sinking_prob = music_tagging["MusicLowQuality"]["Sinking"]
    is_sinking = sinking_prob >= sinking_threshold
    quality = "Sinking" if is_sinking else "non-Sinking"
    tags = [quality if item == "MusicLowQuality" else map_tag(parse_result(music_tagging[item]["result"])) for item in order]
    unfamiliar_tags = {cat_name: tag for tag, cat_vocab_tags, cat_name in zip(tags, SA_CAT_VOCAB, order) if tag not in cat_vocab_tags}
    return [(tag if tag in cat_vocab_tags else "") for tag, cat_vocab_tags in zip(tags, SA_CAT_VOCAB)], unfamiliar_tags, is_sinking


def parse_style_text_sa(meta: Dict, sinking_threshold: float):
    return _parse_sa_music_tagging(_get_value(meta, "music_tagging", "No music_tagging"), sinking_threshold)


# ---------- audio_tags V0 -------------

def _parse_audio_tags_v0(audio_tags: Optional[Dict]) -> Tuple[List[str], Dict[str, str], bool]:
    def parse_result(result: Union[List, str]) -> str:
        if isinstance(result, str):
            return result
        if len(result) == 0:
            return ""
        return result[0]

    def map_tag(tag: str) -> str:
        """Replace certain tags in the dataset"""
        AUDIO_TAGS_SPECIAL_MAP = {}
        for k,v in AUDIO_TAGS_GENRE_SPECIAL_MAP_V0.items():
            AUDIO_TAGS_SPECIAL_MAP[k] = v
        for k,v in AUDIO_TAGS_MOOD_SPECIAL_MAP_V0.items():
            AUDIO_TAGS_SPECIAL_MAP[k] = v
        for k,v in AUDIO_TAGS_SCENE_SPECIAL_MAP_V0.items():
            AUDIO_TAGS_SPECIAL_MAP[k] = v
        for k,v in AUDIO_TAGS_GENDER_SPECIAL_MAP_V0.items():
            AUDIO_TAGS_SPECIAL_MAP[k] = v
        return AUDIO_TAGS_SPECIAL_MAP.get(tag, tag)

    # The order should match `mir_data_util`
    order = ["genre", "mood", "scene", "vocal_gender", "vocal_timbre"]
    is_sinking = False #'non-Sinking'

    if audio_tags is None:
        return [""] * len(order), {}, False
    tags = []
    unfamiliar_tags = {}
    for i in range(len(order)):
        item = order[i]
        cat_vocab_tags = AUDIO_CAT_VOCAB_V0[i]
        _tags = []
        for tag in audio_tags[item]:
            _tag = map_tag(tag)
            if _tag not in cat_vocab_tags:
                _tags.append("")
                if item not in unfamiliar_tags:
                    unfamiliar_tags[item] = []
                if _tag not in unfamiliar_tags[item]:
                    unfamiliar_tags[item].append(_tag)
            else:
                _tags.append(_tag)
        _tags = list(set(_tags))
        if i == 0 and len(_tags) > 2: # genre, ["Pop", "Rock", "Chinese Pop"] -> ["Rock"]
            _merged_tags = []
            for _t in _tags:
                if _t not in ['Pop', 'Chinese Pop']:
                    _merged_tags.append(_t)
            _tags = _merged_tags
        tags.append(_tags)
    return tags, unfamiliar_tags, is_sinking

def parse_style_text_audio_tags_v0(meta: Dict, sinking_threshold: float):
    return _parse_audio_tags_v0(_get_value(meta, "audio_tags", "No audio_tags"))


# ---------- audio_tags V1 -------------

def _parse_audio_tags_v1(music_tagging: Optional[Dict], audio_tags: Optional[Dict]) -> Tuple[List[str], Dict[str, str], bool]:
    def parse_result(result: Union[List, str]) -> str:
        if isinstance(result, str):
            return result
        if len(result) == 0:
            return ""
        return result[0]

    def map_tag(item: str, tag: str) -> str:
        """Replace certain tags in the dataset"""
        AUDIO_TAGS_SPECIAL_MAP = {
            'genre': AUDIO_TAGS_GENRE_SPECIAL_MAP_V1,
            'mood': AUDIO_TAGS_MOOD_SPECIAL_MAP_V1,
            'scene': AUDIO_TAGS_SCENE_SPECIAL_MAP_V1,
            'vocal_gender': AUDIO_TAGS_GENDER_SPECIAL_MAP_V1,
        }
        if item in AUDIO_TAGS_SPECIAL_MAP:
            return AUDIO_TAGS_SPECIAL_MAP[item].get(tag, tag)
        else:
            return tag

    # The order should match `mir_data_util`
    order = ["genre", "mood", "scene", "vocal_gender", "vocal_timbre"]
    is_sinking = False #'non-Sinking'

    if audio_tags is not None:
        tags = []
        unfamiliar_tags = {}
        for i in range(len(order)):
            item = order[i]
            cat_vocab_tags = AUDIO_CAT_VOCAB_V1[i]
            _tags = []
            for tag in audio_tags[item]:
                _tag = map_tag(item, tag)
                if _tag not in cat_vocab_tags:
                    _tags.append("")
                    if item not in unfamiliar_tags:
                        unfamiliar_tags[item] = []
                    if _tag not in unfamiliar_tags[item]:
                        unfamiliar_tags[item].append(_tag)
                else:
                    _tags.append(_tag)
            _tags = list(set(_tags))
            if i == 0 and len(_tags) > 2: # genre, ["Pop", "Rock", "Chinese Pop"] -> ["Rock"]
                _merged_tags = []
                for _t in _tags:
                    if _t not in ['Pop', 'Chinese Pop']:
                        _merged_tags.append(_t)
                _tags = _merged_tags
            tags.append(_tags)
        #print(tags, unfamiliar_tags, is_sinking)
        return tags, unfamiliar_tags, is_sinking
    elif music_tagging is not None:
        #print(music_tagging)
        order_map = {
            'genre': 'Genre20',
            'mood': 'Mood',
            'scene': 'Theme',
            'vocal_gender': 'sa_gender',
        }
        tags = []
        unfamiliar_tags = {}
        for i in range(len(order)):
            item = order[i]
            if item in order_map:
                _item = order_map[item]
                cat_vocab_tags = AUDIO_CAT_VOCAB_V1[i]
                _tags = []
                result = music_tagging[_item]['result']
                if isinstance(result, str):
                    result = result.split(',')
                for tag in result:
                    _tag = map_tag(item, tag)
                    if _tag not in cat_vocab_tags:
                        _tags.append("")
                        if item not in unfamiliar_tags:
                            unfamiliar_tags[item] = []
                        if _tag not in unfamiliar_tags[item]:
                            unfamiliar_tags[item].append(_tag)
                    else:
                        _tags.append(_tag)
                tags.append(_tags)
            else:
                tags.append([""])
        return tags, unfamiliar_tags, is_sinking
    else:
        return [""] * len(order), {}, False

def parse_style_text_audio_tags_v1(meta: Dict, sinking_threshold: float):
    #audio_tags = _get_value(meta, "audio_tags", "No audio_tags")
    #music_tagging = _get_value(meta, "music_tagging", "No music_tagging")
    audio_tags = meta.get('audio_tags', None)
    music_tagging = meta.get('music_tagging', None)
    if music_tagging is not None:
        music_tagging["sa_gender"] = meta.get("gender", {}).get("sa_gender", {"result": [""]})
    #print(audio_tags)
    #print(music_tagging)
    if audio_tags is None and music_tagging is None:
        raise ZhMetaParseError('No music_tagging and audio_tags')
    return _parse_audio_tags_v1(music_tagging, audio_tags)


# ---------- audio_tags V2 -------------

def _parse_audio_tags_v2(music_tagging: Optional[Dict], audio_tags: Optional[Dict]) -> Tuple[List[str], Dict[str, str], bool]:
    def parse_result(result: Union[List, str]) -> str:
        if isinstance(result, str):
            return result
        if len(result) == 0:
            return ""
        return result[0]

    def map_tag(item: str, tag: str) -> str:
        """Replace certain tags in the dataset"""
        AUDIO_TAGS_SPECIAL_MAP = {
            'genre': AUDIO_TAGS_GENRE_SPECIAL_MAP_V2,
            'mood': AUDIO_TAGS_MOOD_SPECIAL_MAP_V2,
            'scene': AUDIO_TAGS_SCENE_SPECIAL_MAP_V2,
            'vocal_gender': AUDIO_TAGS_GENDER_SPECIAL_MAP_V2,
        }
        if item in AUDIO_TAGS_SPECIAL_MAP:
            return AUDIO_TAGS_SPECIAL_MAP[item].get(tag, tag)
        else:
            return tag

    # The order should match `mir_data_util`
    order = ["genre", "mood", "scene", "vocal_gender", "vocal_timbre"]
    is_sinking = False #'non-Sinking'

    if audio_tags is not None:
        tags = []
        unfamiliar_tags = {}
        for i in range(len(order)):
            item = order[i]
            cat_vocab_tags = AUDIO_CAT_VOCAB_V2[i]
            _tags = []
            if i == 0: # process ['pop,rock'] -> ['pop', 'rock']
                new_item = []
                for v in audio_tags[item]:
                    new_item.extend(v.split(','))
                audio_tags[item] = new_item
            for tag in audio_tags[item]:
                _tag = map_tag(item, tag.strip())
                if _tag != '':
                    if _tag not in cat_vocab_tags:
                        _tags.append("")
                        if item not in unfamiliar_tags:
                            unfamiliar_tags[item] = []
                        if _tag not in unfamiliar_tags[item]:
                            unfamiliar_tags[item].append(_tag)
                    else:
                        _tags.append(_tag)
            _tags = list(set(_tags))
            if i == 0 and len(_tags) > 2: # genre, ["Pop", "Rock", "Chinese Pop"] -> ["Rock"]
                _merged_tags = []
                for _t in _tags:
                    if _t not in ['Pop', 'Chinese Pop']:
                        _merged_tags.append(_t)
                _tags = _merged_tags
            tags.append(_tags)
        for _genre in tags[0]: # if genre is Tuhai, is_sinking=True
            if _genre in ["MC", "DJ", "VinaHouse", "Vulgar Pop"]:
                is_sinking = True
        #print(tags, unfamiliar_tags, is_sinking)
        return tags, unfamiliar_tags, is_sinking
    elif music_tagging is not None:
        #print(music_tagging)
        order_map = {
            'genre': 'Genre20',
            'mood': 'Mood',
            'scene': 'Theme',
            'vocal_gender': 'sa_gender',
        }
        tags = []
        unfamiliar_tags = {}
        for i in range(len(order)):
            item = order[i]
            if item in order_map:
                _item = order_map[item]
                cat_vocab_tags = AUDIO_CAT_VOCAB_V2[i]
                _tags = []
                result = music_tagging[_item]['result']
                if isinstance(result, str):
                    result = result.split(',')
                for tag in result:
                    _tag = map_tag(item, tag)
                    if _tag not in cat_vocab_tags:
                        _tags.append("")
                        if item not in unfamiliar_tags:
                            unfamiliar_tags[item] = []
                        if _tag not in unfamiliar_tags[item]:
                            unfamiliar_tags[item].append(_tag)
                    else:
                        _tags.append(_tag)
                tags.append(_tags)
            else:
                tags.append([""])
        for _genre in tags[0]: # if genre is Tuhai, is_sinking=True
            if _genre in ["MC", "DJ", "VinaHouse", "Vulgar Pop"]:
                is_sinking = True
        return tags, unfamiliar_tags, is_sinking
    else:
        return [""] * len(order), {}, False

def parse_style_text_audio_tags_v2(meta: Dict, sinking_threshold: float):
    #audio_tags = _get_value(meta, "audio_tags", "No audio_tags")
    #music_tagging = _get_value(meta, "music_tagging", "No music_tagging")
    audio_tags = meta.get('audio_tags', None)
    music_tagging = meta.get('music_tagging', None)
    if music_tagging is not None:
        music_tagging["sa_gender"] = meta.get("gender", {}).get("sa_gender", {"result": [""]})
    #print(audio_tags)
    #print(music_tagging)
    if audio_tags is None and music_tagging is None:
        raise ZhMetaParseError('No music_tagging and audio_tags')
    return _parse_audio_tags_v2(music_tagging, audio_tags)

# ---------- artist_id -------------

def parse_artist_id(meta) -> int:
    """meta.artist_id, use zh_empty if it's empty"""
    return ARTIST_ID_MAP_V2[str(meta.get("artist_id", "zh_empty"))]


# ---------- filter_label -------------

def _parse_filter_label(filter_label: Optional[Dict[str, str]]) -> Tuple[bool, bool]:
    """Return (high_quality, popular_potential).
    Conservative filtering. Assume the song is high quality if it's not labeled.
    """
    if filter_label is None:
        return True, True
    return filter_label["high_quality"] == "yes", filter_label["popular_potential"] == "yes"


def parse_filter_label(meta: Dict) -> Tuple[bool, bool]:
    """meta.filter_label"""
    return _parse_filter_label(_get_value(meta, "filter_label", "No filter label"))


# ---------- voice_tag -------------

def _parse_voice_tag(voice_probs: Dict[str, float]) -> Optional[str]:
    """Return the voice tag based on probablity thresholds. 'adult' tag is not used."""
    is_child = voice_probs["child"] >= VOICE_THRESHOLDS["Child"]
    if is_child:
        return "Child"
    is_female = voice_probs["female"] >= VOICE_THRESHOLDS["Female"]
    is_male = voice_probs["male"] >= VOICE_THRESHOLDS["Male"]
    if is_female and not is_male:
        return "Female"
    if is_male and not is_female:
        return "Male"
    if is_female and is_male:
        if voice_probs["female"] >= voice_probs["male"]:
            return "Female"
        return "Male"
    return None


def _parse_voice_tag_sa(sa_gender: Dict) -> Optional[str]:
    """Solely rely on the result it provides"""
    for tag in sa_gender["result"]:
        if tag in ["child", "female", "male"]:
            return tag.capitalize()
    return None


def parse_voice_tag(meta: Dict) -> Optional[str]:
    """meta.gender"""
    return _parse_voice_tag(_get_value(meta, "gender", "No gender tag"))


def parse_voice_tag_sa(meta: Dict) -> Optional[str]:
    """meta.gender.sa_gender"""
    get_value = partial(_get_value, msg="No gender")
    return _parse_voice_tag_sa(get_value(get_value(meta, "gender"), "sa_gender"))


def parse_voice_tag_sa_optional(meta: Dict) -> Optional[str]:
    """meta.gender.sa_gender"""
    try:
        return parse_voice_tag_sa(meta)
    except ZhMetaParseError:
        return None

# ----------- source -----------

def parse_source_optional(meta: Dict) -> Optional[str]:
    """meta.source"""
    # source is optional
    return meta.get("source")


# ----------- MIR -----------

def parse_mir_tempo(meta: Dict) -> int:
    """meta.mir_service.beat.tempo"""
    get_value = partial(_get_value, msg="No tempo")
    return get_value(get_value(get_value(meta, "mir_service"), "beat"), "tempo")


def parse_mir_tempo_optional(meta: Dict) -> Optional[int]:
    """meta.mir_service.beat.tempo"""
    try:
        return parse_mir_tempo(meta)
    except ZhMetaParseError:
        return None


def parse_mir_key(meta: Dict) -> str:
    """meta.mir_service.key.song"""
    get_value = partial(_get_value, msg="No key")
    return get_value(get_value(get_value(meta, "mir_service"), "key"), "song")


def parse_mir_key_optional(meta: Dict) -> Optional[str]:
    """meta.mir_service.key.song"""
    try:
        return parse_mir_key(meta)
    except ZhMetaParseError:
        return None

# ------------------------------------------
#                 TRANSFORM
# ------------------------------------------

# ---------- converters -------------

# All the converters should do in-place operations on _self
# to avoid to many copies.

def convert_hqmy(_self):
        # Re-assign genre for HQMY songs
    if _self.source == "环球美音":
        _self.style_text[0] = "Chinese Tradition"


def convert_voice_tag(_self):
    # Re-assign gender tags
    if _self.voice_tag and (_self.artist_id == ARTIST_ID_MAP_V2["zh_empty"]):
        _self.artist_id = ARTIST_ID_MAP_V2[_self.voice_tag]


# ---------- validators -------------

def validate_confidence_optional(confidence_threshold: Optional[float], confidence: Optional[float]):
    if confidence_threshold is None or confidence is None:
        return
    if confidence < confidence_threshold:
        # Do not log the confidence value. The value is almost always different for each sample.
        # Adding the specific confidence value to the log will mess up the log messages that's
        # supposed to be aggregated and counted.
        raise ZhMetaTransformError("Low confidence")


def validate_style_text_sa(style_text: List[str], is_sinking: bool):
    _genre_tag, _, _, _, _lang_tag = style_text
    if _genre_tag in ["", "Chinese Opera", "Other genre"]:
        raise ZhMetaTransformError(f"Filter out genre {_genre_tag}")
    if (_genre_tag not in ["DJ", "MC"]) and is_sinking:
        raise ZhMetaTransformError(f"Filter out genre {_genre_tag} because of sinking")
    if not _lang_tag or (_lang_tag == "Chinese Dialects") or (_lang_tag == "Cantonese"):
        raise ZhMetaTransformError(f"Filter out lang {_lang_tag}")


def validate_quality(high_quality: bool):
    if not high_quality:
        raise ZhMetaTransformError(f"Filter out low quality")


def validate_deepchorus_optional(deepchorus: Optional[DeepChorus]):    
    if deepchorus is None:
        return
    if deepchorus.confidence < 0.45:
        raise ZhMetaTransformError(f"Low deepchorus confidence <0.45")


def validate_mir_tempo_optional(tempo: Optional[int]):
    if tempo is None:
        return
    l, h = TEMPO_RANGE
    if not (l < tempo <= h):
        raise ZhMetaTransformError("Tempo not in supported range")


def validate_mir_key_optional(key: Optional[str]):
    if key is None:
        return
    if key not in KEYS:
        raise ZhMetaTransformError(f"Unsupported key {key}")


# ---------- SongSlice -------------


def _format_utterances(utterances, time_in_sec=False):
    # This function cleans up the raw utterances data from metadata.
    # Extracts necessary information: [start_time_in_sec, end_time_in_sec, lyrics text with additional tags, precomputed phonemes]
    new_utterances = []    
    for i, u in enumerate(utterances):
        phone = u.get('phoneme', '')        
        phone = '' if not phone else phone

        if "lyrics" in u:
            u["text"] = u["lyrics"]
        
        utt_start = u.get('start_time', None)
        if utt_start is None:
            continue
        utt_start = math.floor(utt_start)   # Round down so we don't miss the first few phonemes
        utt_end = u.get('end_time', None)
        if utt_end is None:
            if i < (len(utterances) - 1):
                utt_end = max(math.floor(utterances[i+1].get('start_time', 0)), utt_start+1)
            else:
                continue
        utt_end = math.ceil(utt_end)        # Round up so we don't miss the last few phonemes
        if time_in_sec:
            new_utterances.append([utt_start, utt_end, u['text'], phone])
        else:
            # TODO (QQ) handle ms directly instead of converting to int.
            new_utterances.append([math.floor(utt_start/1000), math.ceil(utt_end/1000), u['text'], phone])


    # Sanity check utterances
    utterances = []
    for i, u in enumerate(new_utterances):
        if i > 0 and u[0] <= 0 and ":" in u[2]:
            # Handle the case of (0, 97, '合:对的人'), (-1, 97, '合:对的人')
            if i+1 < len(new_utterances) and u[1] > new_utterances[i+1][1]:
                # Handle the case of swap (0, 31, '合:色即是空 空即是色'), (25, 28, '选择就是选择无所谓对与错')
                u[0] = new_utterances[i+1][1]
                new_utterances[i] = new_utterances[i+1]
                new_utterances[i+1] = u            
            else:
                # Handle the case of (0, 25, '合:色即是空 空即是色'), (25, 28, '选择就是选择无所谓对与错')
                u[0] = min(new_utterances[max(i-1,0)][1], u[1]-1)
        if u[1] - u[0] > 0:
            # Only keep utterances that are longer than 1 sec.
            utterances.append(u)
    # Remove duplicate utterances, retain order.
    uniq_utt_idx = []
    for i in range(len(utterances)-1):
        if (utterances[i][0] == utterances[i+1][0]) and (utterances[i][1] == utterances[i+1][1]):
            continue
        uniq_utt_idx.append(i)
    utterances = [utterances[i] for i in uniq_utt_idx]
    
    if len(utterances) < 1:
        return []

    # Sanity check: utterances are non-overlapping.
    for i in range(len(utterances)-1):
        if utterances[i][1] > utterances[i+1][0] + 5:
            # logging.warning("utterances need to be non-overlapping")
            # TODO (QQ) need to handle time stamps of '合:色即是空 空即是色' more carefully.
            # print([x[:3] for x in utterances])
            return []
        
    return utterances


def _extract_seg_from_utts(i, utterances, min_duration, max_duration, new_line_token, sec_start_idx=None):
    # Given the start time (and end time), combine the utterances in the window into a segment
    # Output is in the format [seg_start_sec, seg_end_sec, lyrics_text, precomputed_phonemes, seg_tag]
    s, cur_seg = i, []
    seg_tag = sec_start_idx[i] if sec_start_idx else "None"
    while i < len(utterances) and (utterances[i][1] - utterances[s][0] < min_duration):
        cur_seg.append(utterances[i])
        i += 1
    j = i
    while j < len(utterances) and utterances[j][1] - utterances[s][0] <= max_duration:
        j += 1
    if j == i:
        i = s + 1
        return (i, None)
    else:
        # Extract a segment with target duration uniformly distributed between min/max duration.
        k = random.randint(i+1, j)
        cur_seg.extend([utterances[kk] for kk in range(i, k)])
        seg_start = cur_seg[0][0]
        seg_end = cur_seg[-1][1]
        lyrics_text = ""
        for kk in range(s, k):
            if sec_start_idx and (kk in sec_start_idx):
                seg_tag = sec_start_idx[kk]
                lyrics_text = lyrics_text + ("[" + seg_tag + "]" + ". ")                
            lyrics_text = lyrics_text + utterances[kk][2] + ". "
        phoneme_sequence = new_line_token.join([u[3] for u in cur_seg])    
        new_seg = [
            seg_start, 
            seg_end, 
            lyrics_text,  # lyrics text
            phoneme_sequence,   # precomputed lyrics phonemes
            seg_tag]
        i = k
        return (i, new_seg)


def _reset_section_tags_for_short_inst_phrases(song_slice: SongSlice) -> SongSlice:
    """Remove short instrumental phrases' section tags to avoid adding unnessary section tags."""
    def does_phrase_need_reset(phrase: Phrase) -> bool:
        return (
            not phrase.has_utterance and 
            phrase.section_tag is not None and 
            phrase.duration is not None and phrase.duration < 2  # hard-coded duration threshold
        )
    phrases=[
        phrase._replace(section_tag=None) if does_phrase_need_reset(phrase) else phrase
        for phrase in song_slice.phrases
    ]
    return SongSlice(phrases=phrases)


def transform_utts_to_song_slices_heuristic(
    utterances,
    min_duration,
    max_duration,
    time_in_sec=False,
    new_line_token='\n',
    infer_structure_tags=False
) -> List[SongSlice]:
    """Segment the full song into segments, each segment consists of multiple utterances. Refactored from group_utterances"""
    utterances = _format_utterances(utterances)
    if not utterances:
        return []
    sec_start_idx = {}        
    if infer_structure_tags:        
        # Infer sections based on separation of utterances.
        last_utt_end = -10
        MIN_NUM_CHAR = 3    # Minimum number of characters in an utterance to infer it's valid vocal.
        MIN_SEP_WITHIN_SEC = 4  # Max separation of two utterance to infer there is a section break.
        for i, utt in enumerate(utterances):
            if (utt[0] - last_utt_end >= MIN_SEP_WITHIN_SEC) and len(utt[2]) > MIN_NUM_CHAR:
                sec_start_idx[i] = "section"
            last_utt_end = utt[1]

    segs = []
    if sec_start_idx:
        # Extract segments starting from (inferred) starting points of sections.
        for sec_start_id, sec_tag in sec_start_idx.items():
            _, new_seg = _extract_seg_from_utts(
                sec_start_id, utterances, min_duration, max_duration, new_line_token, sec_start_idx)
            if new_seg:
                segs.append(new_seg)
    else:
        # Group utterances into random non-overlapping segments.
        i = 0
        while i < len(utterances):
            i, new_seg = _extract_seg_from_utts(i, utterances, min_duration, max_duration, new_line_token)
            if new_seg:
                segs.append(new_seg)

            # Move to the next vocal starting point
            while i < len(utterances):
                if len(utterances[i][2]) >= 2:
                    break
                i += 1
    return [
        SongSlice(phrases=[Phrase.parse(
            phonemes=phonemes,
            text=text,
            time_span=(start, end)
        )])
        for start, end, text, phonemes, _ in segs
    ]


def transform_utts_to_song_slices_structure(
    utterances: List[Dict],
    min_duration: int,
    max_duration: int,
    structure_tags: List[Dict],
    complete_section: bool = False,
) -> List[SongSlice]:
    """Segment the full song into a list of SongSlice, each consists of multiple utterances."""
    def format_utterances(utterances: List[Dict]) -> List[Phrase]:
        formatted_us = _format_utterances(utterances)
        return [
            Phrase.parse(text=nt, phonemes=phonemes, time_span=(start, end)) 
            for start, end, nt, phonemes in formatted_us
        ]

    def get_overlap(phrase_time_span: Tuple[int, int], section_time_span: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        left_overlap = max(phrase_time_span[0], section_time_span[0])
        right_overlap = min(phrase_time_span[1], section_time_span[1])
        if right_overlap <= left_overlap:
            return None
        return (left_overlap, right_overlap)

    def get_overlap_dur(phrase_time_span: Tuple[int, int], section_time_span: Tuple[float, float]) -> float:
        overlap = get_overlap(phrase_time_span, section_time_span)
        if overlap is None:
            return 0
        return overlap[1] - overlap[0]

    def get_section_span(structure_tag: Dict) -> Tuple[int, int]:
        return math.floor(structure_tag["start_time"]), math.ceil(structure_tag["end_time"])

    def add_section_tag(phrase: Phrase) -> Phrase:
        """Return a new phrase with a section tag based on the longest overlap"""
        if not structure_tags:
            return phrase
        overlaps = [
            get_overlap_dur(phrase.time_span, get_section_span(structure_tag))  # type: ignore 
            for structure_tag in structure_tags
        ]
        if not overlaps:
            return phrase
        idx = int(np.argmax(overlaps))
        return phrase._replace(section_tag=structure_tags[idx]["tag"])

    def get_inst_phrases(time_span: Tuple[int, int]) -> List[Phrase]:
        phrases = []
        for structure_tag in structure_tags:
            overlap = get_overlap(time_span, get_section_span(structure_tag))
            if overlap is None:
                continue
            phrases.append(Phrase(section_tag=structure_tag["tag"], time_span=overlap))
        return phrases

    def insert_inst_phrases(phrases: List[Phrase], song_time_span: Tuple[int, int]) -> List[Phrase]:
        """Insert instrument phrases if a gap presents between two adjacent vocal phrases."""
        song_start, song_end = song_time_span
        if not phrases:
            return get_inst_phrases(song_time_span)
        new_phrases = []
        for idx, (curr_phrase, next_phrase) in enumerate(zip(phrases, phrases[1:] + [None])):
            # the gap between song start and the first phrase's start
            if idx == 0 and curr_phrase.start - song_start > 0:  # first phrase
                _phrases = get_inst_phrases((song_start, curr_phrase.start))
                new_phrases.extend(_phrases)
            # add the current phrase
            new_phrases.append(curr_phrase)
            # the gap between the current phrase and the next phrase
            if idx < len(phrases) - 1 and next_phrase.start - curr_phrase.end > 0:
                _phrases = get_inst_phrases((curr_phrase.end, next_phrase.start))
                new_phrases.extend(_phrases)
            # the gap between the last phrase's end and the song end
            elif idx == len(phrases) - 1 and song_end - curr_phrase.end > 0:  # last phrase
                _phrases = get_inst_phrases((curr_phrase.end, song_end))
                new_phrases.extend(_phrases)
        return new_phrases

    def split_inst_phrase(phrase: Phrase, split_t: int) -> List[Phrase]:
        # exceptions should not be possible, but just in case
        if phrase.has_utterance:
            raise ValueError("Invalid inst phrase.")
        start, end = phrase.time_span
        if split_t >= end:
            raise ValueError("Invalid desired_t for phrase split.")
        return [
            Phrase(section_tag=phrase.section_tag, time_span=(start, split_t)),
            Phrase(section_tag=phrase.section_tag, time_span=(split_t, end)),
        ]

    def get_phrase_group_dur(phrase_group: List[Phrase]) -> int:
        if not phrase_group:
            return 0
        return phrase_group[-1].end - phrase_group[0].start

    def get_section_start_ind(phrases: List[Phrase]) -> List[int]:
        sec_tags = [p.section_tag for p in phrases]
        ind = [0]
        for idx, (curr_tag, next_tag) in enumerate(zip(sec_tags, sec_tags[1:])):
            if next_tag != curr_tag:
                ind.append(idx + 1)
        return ind

    def get_song_slices(phrases: List[Phrase]) -> List[SongSlice]:
        """Group phrases into song slices (might chunk instrument phrases).
        The logic assumes there are short overlaps (~1s) between phrases, or some phrases might
        have gaps inbetween. Therefore, the most accurate group duration is (end - start).
        """
        section_start_ind = get_section_start_ind(phrases)
        song_slices = []
        # Start from the beginning of each section, obtain a chunk
        for start_idx in section_start_ind:
            phrases_in_proc = copy.deepcopy(phrases)[start_idx:]
            # the length will grow, the last index helps insert the last group
            idx, group = 0, []
            while idx <= len(phrases_in_proc):
                phrase = phrases_in_proc[idx] if idx < len(phrases_in_proc) else None
                # add a new song slice if unable to include the current one
                if (phrase is None) or (get_phrase_group_dur(group + [phrase]) > max_duration):
                    if group:
                        song_slices.append(SongSlice(phrases=group))
                        break
                # add phrase to group
                if phrase.has_utterance:
                    # Unable to chunk utterance, add directly
                    group.append(phrase)
                else:
                    if get_phrase_group_dur(group + [phrase]) <= max_duration:
                        group.append(phrase)
                    else:
                        # The corner case is to have an inst phrase that has a gap
                        # between it and its previous phrase, and the phrase's start
                        # is after the split_t. We need to handle two cases separately.
                        _start = group[0].start if group else phrase.start
                        split_t = math.floor(_start + max_duration)
                        # split_t < phrase.end is always valid, otherwise it would go to the above if branch.
                        if split_t > phrase.start:
                            inst_phrases = split_inst_phrase(phrase, split_t)
                            # split the current inst phrase into two, insert into the current index
                            phrases_in_proc[idx:idx+1] = inst_phrases
                            group.append(phrases_in_proc[idx])
                        else:  # impossible to utilize the current phrase in the current group
                            if group:
                                song_slices.append(SongSlice(phrases=group))
                                break
                idx += 1
        return song_slices

    def get_song_slices_complete_section(phrases: List[Phrase]) -> List[SongSlice]:
        """Group phrases into song slices, but do not split any section."""
        if not phrases: return []
        section_start_ind = get_section_start_ind(phrases) + [len(phrases)]
        song_slices_by_sections = [
            SongSlice(phrases=phrases[start: end])
            for start, end in zip(section_start_ind, section_start_ind[1:])
        ]
        end_ts = np.array([song_slice.end for song_slice in song_slices_by_sections])
        song_slices = []
        for idx, song_slice in enumerate(song_slices_by_sections):
           rel_end_ts = end_ts[idx:] - song_slice.start  # section end times relative to the current song_slice's start
           # idx_inc indicates the number of song slices that should be combined
           idx_inc = max(1, int(np.searchsorted(rel_end_ts, max_duration, side="right")))
           song_slices.append(reduce(operator.add, song_slices_by_sections[idx: idx+idx_inc]))
        return song_slices

    song_time_span = (math.floor(structure_tags[0]["start_time"]), math.ceil(structure_tags[-1]["end_time"]))
    phrases = format_utterances(utterances)
    phrases = list(map(add_section_tag, phrases))
    phrases = insert_inst_phrases(phrases, song_time_span)
    song_slices = get_song_slices_complete_section(phrases) if complete_section else get_song_slices(phrases)
    # Remove short instrumental sections' section tags. After the removal, these phrases will become placeholders
    # for SongSlice to correctly calculate the start and end times, but will be completely ignored during tokenization.
    return list(map(_reset_section_tags_for_short_inst_phrases, song_slices))


def take_song_slices_by_method(song_slices: List[SongSlice], segment_method: str):
    if segment_method not in ["first", "verse_chorus_section", "random"]:
        raise ValueError("Invalid segment_method")

    if segment_method == "first":
        song_slices = song_slices[:1]
    elif segment_method == "verse_chorus_section":
        song_slices = [
            song_slice for song_slice in song_slices 
            if set(s.section_tag for s in song_slice.phrases).issubset(["verse", "chorus", "section"])
        ]

    return song_slices


def get_max_seg(song_slices: List[SongSlice], max_seg_per_track: int) -> List[SongSlice]:
    if max_seg_per_track <= 0:
        return song_slices
    random.shuffle(song_slices)
    return song_slices[:max_seg_per_track]


def filter_song_slices(song_slices: List[SongSlice], duration_range: Tuple[int, int]) -> Tuple[List[SongSlice], ZhMetaLogger]:
    n_slices_pre_filter = len(song_slices)
    song_slices = [ss for ss in song_slices if ss.is_time_span_valid(duration_range)]
    n_slices_post_filter = len(song_slices)
    n_filtered = n_slices_pre_filter-n_slices_post_filter

    if len(song_slices) == 0:
        raise ZhMetaTransformError(f"No valid song slice within duration range {duration_range}")

    if any(phrase.text and not phrase.phonemes for song_slice in song_slices for phrase in song_slice.phrases):
        raise ZhMetaTransformError("No phoneme found in all available song slices")

    logger = ZhMetaLogger(
        f"Removed song slices: {n_filtered}/{n_slices_pre_filter}" if n_filtered > 0 else ""
    )
    return song_slices, logger
