"""
Custom meta data parser definition and dataset registration.

To add a new dataset:
1. Check if there is already a meta data class that meets your requirement
   a. If so, go ahead to step 2
   b. If not, inherit ZhMetaBase to create a new meta data class
2. Register the dataset into the dataset registry table
"""

import copy
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple, TypeVar, Type

from .zh_meta import (
    ZhMetaBase,
    ZhMetaLogger,
    SongSlice,
    parse_freeform_text_everynoise_optional,
    parse_freeform_text_wyy_optional,
    parse_freeform_text_groupAB_optional,
    # Parsers
    parse_utterance_lyrics,
    parse_utterance_labeled_lyrics,
    parse_style_text_sa,
    parse_style_text_audio_tags_v3_or_music_tagging,
    parse_inst_style_text_audio_tags_v3_or_music_tagging,
    parse_freeform_text_sstk,
    parse_freeform_text_audio_tags,
    parse_vad_voice_proportion,
    parse_vad_voice_proportion_default0,
    parse_lyrics_confidence_force_align_legacy,
    parse_lyrics_confidence_force_align,
    parse_lyrics_confidence_sa_asr,
    parse_lyrics_confidence_labeled_lyrics,
    parse_lyrics_confidence_optional,
    parse_lyrics_confidence_sa_asr_or_force_align,
    parse_structure_tags,
    parse_structure_tags_optional,
    parse_structure_tags_custom_concat,
    parse_structure_tags_custom_concat_inst,
    parse_artist_id,
    parse_source_optional,
    parse_filter_label,
    parse_voice_tag,
    parse_voice_tag_sa,
    parse_voice_tag_sa_optional,
    parse_mir_tempo,
    parse_mir_tempo_optional,
    parse_mir_key,
    parse_mir_key_optional,
    parse_mir_tag_optional,
    parse_mir_vocal2midi_optional,
    parse_artist_tagging,
    parse_sstk_inst_tag,
    parse_freeform,
    parse_bpm,
    # Validators
    validate_style_text_sa,
    validate_style_text_sa_optional,
    validate_sstk_has_vocal,
    validate_mir_tempo_optional,
    validate_mir_key_optional,
    validate_deepchorus_optional,
    validate_vad_voice_proportion_optional,
    validate_filter_label_single_yes,
    # Converters
    convert_hqmy,
    convert_artist_id_from_voice_tag,
    convert_voice_tag_from_audio_tag,
    # Slicing
    transform_section_to_song_slices,
    sample_song_slices,
    # Infer
    parse_section_instruments_optional,
    add_section_instruments_to_structure_tags,
)
from recipes.bigmusic.datasets.mir_data_util import (
    tempo_to_label,
    parse_artist_id_by_lang_gender,
)
import random

ZhMetaBaseType = TypeVar("ZhMetaBaseType", bound=ZhMetaBase)

# ------------------------------------------
#                MetaTransform
# ------------------------------------------

@dataclass
class ZhMetaGeneral(ZhMetaBase):
    """The general meta transform that should cover most cases.
    Caveats:
    - It does not support fields/labels that are not common in most datasets.
    - Since the parsing is loose, some data missing issues could be unexpectedly ignored
    - Avoid using this transform whenever it is possible
    """
    @classmethod
    def parse(cls, meta, sinking_threshold: float, artist_tagging_confidence: float):
        utts = parse_utterance_lyrics(meta)
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=utts,
            lyrics_confidence=parse_lyrics_confidence_optional(meta, utts),
            structure_tags=parse_structure_tags_optional(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
        )

    def _validate(
        self,
        lyrics_confidence: Optional[float],
        deepchorus_confidence: Optional[float],
    ):
        super()._validate(lyrics_confidence, deepchorus_confidence)
        validate_style_text_sa(self.style_text, self.is_sinking)


class ZhMetaTransform:
    def __init__(
        self,
        zh_meta: Type[ZhMetaBaseType],
        sinking_threshold: float,
        lyrics_confidence: Optional[float],
        deepchorus_confidence: Optional[float],
        artist_tagging_confidence: Optional[float],
        segment_method: str,
        max_seg_per_track: int,
        duration_range: Tuple[int, int],
        lyrics_confidence_phrase: Optional[float],
        slice_mode: str = "section",
        mir_filters: Optional[List[str]] = None,
        multi_tasks: Optional[List[str]] = None,
    ):
        self.zh_meta = zh_meta
        self.sinking_threshold = sinking_threshold
        self.lyrics_confidence = lyrics_confidence
        self.deepchorus_confidence = deepchorus_confidence
        self.artist_tagging_confidence = artist_tagging_confidence
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.duration_range = duration_range
        self.lyrics_confidence_phrase = lyrics_confidence_phrase
        self.slice_mode = slice_mode
        self.mir_filters = mir_filters
        self.multi_tasks = multi_tasks

    @classmethod
    def from_data_id(cls, data_id: Optional[int], allow_general: bool = False, **kwargs):
        dataset_entry = ZH_DATASET_REGISTRY.get(data_id) if data_id is not None else None
        if not allow_general and dataset_entry is None:
            raise ValueError(f"data_id {data_id} is not registered")
        zh_meta = ZhMetaGeneral if dataset_entry is None else dataset_entry.parser
        return cls(zh_meta, **kwargs)

    def __call__(self, meta: Dict) -> Dict[str, Any]:
        """Parse the meta dict into an intermediate representation, then convert it into song slices.
        :return: (meta_logger, song_slices, style_text, artist_id, lyrics_confidence, deepchorus, tempo_label, key)
        """
        return self.zh_meta.parse(meta, self.sinking_threshold, self.artist_tagging_confidence, multi_tasks=self.multi_tasks).transform(
            lyrics_confidence=self.lyrics_confidence,
            lyrics_confidence_phrase=self.lyrics_confidence_phrase,
            deepchorus_confidence=self.deepchorus_confidence,
            segment_method=self.segment_method,
            max_seg_per_track=self.max_seg_per_track,
            duration_range=self.duration_range,
            slice_mode=self.slice_mode,
            mir_filters=self.mir_filters,
            multi_tasks=self.multi_tasks,
        )


# ------------------------------------------
#                Meta Data
# ------------------------------------------

# ----------------- Audio Tag Dataset Parsers ---------------

@dataclass
class ZhMetaAudioTagBase(ZhMetaBase):
    """Audio tag parser.
    Read and validate SA tags in addition to audio tags, convert voice tag from audio tag.
    """
    style_text_sa: list[str] = None

    def _validate(
        self,
        lyrics_confidence: Optional[float],
        deepchorus_confidence: Optional[float],
    ):
        super()._validate(lyrics_confidence, deepchorus_confidence)
        # Filter out unwanted songs even if they are human-labeled
        validate_style_text_sa_optional(self.style_text_sa, self.is_sinking)

    def _convert(self):
        _self = copy.deepcopy(self)
        convert_voice_tag_from_audio_tag(_self)
        convert_artist_id_from_voice_tag(_self)
        return _self


@dataclass
class ZhMetaSFTAudioTagsV4(ZhMetaAudioTagBase):
    """
    sa + audio tags in pt/sft/rl stage
    [genre, genre_extra, extra, mood, scene, gender, timbre, language, is_sinking]
    """
    @classmethod
    def parse(cls, meta, sinking_threshold: float, artist_tagging_confidence: float, multi_tasks: Optional[list]=None):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_audio_tags_v3_or_music_tagging(meta, sinking_threshold)
        # [genre, genre_extra, extra, mood, scene, gender, timbre, language, is_sinking, instruments, tempo, key, mode]

        if multi_tasks is not None and 'global_mir_control' in multi_tasks:
            # parse inst (9), tempo (10), key & mode (11, 12)
            inst_dict, bpm, key, tempo_text, key_text, mode_text = parse_mir_tag_optional(meta, None, style_text[9], inst_thre_ver="0.9")
            inst_text = list(set(inst_dict.keys()))
        else:
            key, bpm, inst_dict = [None] * 3
            tempo_text, key_text, mode_text = '', '', ''
            inst_text = []
        if len(inst_text) > 0:
            style_text[9] = inst_text
        if tempo_text:
            style_text[10] = [tempo_text]
        if key_text:
            style_text[11] = [key_text]
        if mode_text:
            style_text[12] = [mode_text]

        freeform_text = parse_freeform(meta, style_text)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags_custom_concat(meta),
            style_text=style_text,
            style_text_sa=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            freeform_text=freeform_text,
            artist_id=parse_artist_id_by_lang_gender(style_text[7], style_text[5]),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag_sa_optional(meta),
            language=style_text[7],
            tempo=bpm,
            key=key,
            instrument=inst_dict,
            vocal2midi=parse_mir_vocal2midi_optional(meta),
        )

    def _validate(
        self,
        lyrics_confidence: Optional[float],
        deepchorus_confidence: Optional[float],
    ):
        super()._validate(lyrics_confidence, deepchorus_confidence)
        # Filter out unwanted songs even if they are human-labeled
        validate_style_text_sa_optional(self.style_text_sa, self.is_sinking)


@dataclass
class ZhMetaSFTAudioTagsV4_Lyrics(ZhMetaAudioTagBase):
    """
    sa + audio tags in pt/sft/rl stage
    [genre, genre_extra, extra, mood, scene, gender, timbre, language, is_sinking]
    """
    @classmethod
    def parse(cls, meta, sinking_threshold: float, artist_tagging_confidence: float, multi_tasks: Optional[list]=None):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_audio_tags_v3_or_music_tagging(meta, sinking_threshold)
        # [genre, genre_extra, extra, mood, scene, gender, timbre, language, is_sinking, instruments, tempo, key, mode]

        if multi_tasks is not None and 'global_mir_control' in multi_tasks:
            # parse inst (9), tempo (10), key & mode (11, 12)
            inst_dict, bpm, key, tempo_text, key_text, mode_text = parse_mir_tag_optional(meta, None, style_text[9], inst_thre_ver="0.9")
            inst_text = list(set(inst_dict.keys()))
        else:
            key, bpm, inst_dict = [None] * 3
            tempo_text, key_text, mode_text = '', '', ''
            inst_text = []
        if len(inst_text) > 0:
            style_text[9] = inst_text
        if tempo_text:
            style_text[10] = [tempo_text]
        if key_text:
            style_text[11] = [key_text]
        if mode_text:
            style_text[12] = [mode_text]

        freeform_text = parse_freeform(meta, style_text)
        return cls(
            utterances=parse_utterance_labeled_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_labeled_lyrics(meta),
            structure_tags=parse_structure_tags_custom_concat(meta),
            style_text=style_text,
            style_text_sa=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            freeform_text=freeform_text,
            artist_id=parse_artist_id_by_lang_gender(style_text[7], style_text[5]),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag_sa_optional(meta),
            language=style_text[7],
            tempo=bpm,
            key=key,
            instrument=inst_dict,
            vocal2midi=parse_mir_vocal2midi_optional(meta),
        )

    def _validate(
        self,
        lyrics_confidence: Optional[float],
        deepchorus_confidence: Optional[float],
    ):
        super()._validate(lyrics_confidence, deepchorus_confidence)
        # Filter out unwanted songs even if they are human-labeled
        validate_style_text_sa_optional(self.style_text_sa, self.is_sinking)


@dataclass
class ZhMetaInst(ZhMetaBase):
    """
    parse general instrumental datasets, e.g., everynoise, wyy, or any data with sa tags
    it tries to load all possible keywords into freeform_text
    - no lyrics
    - only rely on chorus to segment song slices
    """
    @classmethod
    def parse(cls, meta, sinking_threshold: float, artist_tagging_confidence: float, multi_tasks: Optional[list]=None):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=[],
            lyrics_confidence=1,
            structure_tags=parse_structure_tags_custom_concat_inst(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            freeform_text=parse_freeform(meta),
            artist_id=parse_artist_id(meta),
            vad_voice_proportion=0,
        )
    
    def _validate(
        self,
        lyrics_confidence: Optional[float],      # not used for instrumental
        deepchorus_confidence: Optional[float],  # not used for instrumental
    ):
        # do not validate deepchorus confidence, since the averaged section label prob is not reliable for instruments
        # instead, only validate has_vocal. If it is a vocal song, then disgard it since there is no corresponding lyrics
        # validate_sstk_has_vocal(self.has_vocal)
        validate_vad_voice_proportion_optional(self.vad_voice_proportion, vad_threshold=0.5)
        validate_mir_tempo_optional(self.tempo)
        validate_mir_key_optional(self.key)

    def _to_song_slices(
        self,
        segment_method: str,
        max_seg_per_track: int,
        duration_range: Tuple[int, int],
        lyrics_confidence_phrase: Optional[float],  # not used for instrumental
        slice_mode: str = "section",
    ) -> Tuple[ZhMetaLogger, List[SongSlice]]:
        min_duration, max_duration = duration_range
        song_slices = transform_section_to_song_slices(
            min_duration,
            max_duration,
            structure_tags=self.structure_tags.tags,
            slice_mode=slice_mode,
            chorus_only=True,  # guarantee each slice to contain chorus (assumption: deepchorus for instrumental is only reliable for chorus)
        )
        song_slices = sample_song_slices(song_slices, segment_method, max_seg_per_track)
        return ZhMetaLogger(""), song_slices


@dataclass
class ZhMetaInstV4(ZhMetaInst):
    """
    parse general instrumental datasets, e.g., everynoise, wyy, or any data with sa/audio tags
    it tries to load all possible keywords into freeform_text
    - no lyrics
    - only rely on chorus to segment song slices
    """
    @classmethod
    def parse(cls, meta, sinking_threshold: float, artist_tagging_confidence: float, multi_tasks: Optional[list]=None):
        style_text, unfamiliar_tags, is_sinking = parse_inst_style_text_audio_tags_v3_or_music_tagging(meta, sinking_threshold)
        # [genre, genre_extra, extra, mood, scene, gender, timbre, lang, is_sinking, instruments, tempo, key, mode]

        if multi_tasks is not None and 'global_mir_control' in multi_tasks:
            # parse inst (9), tempo (10), key & mode (11, 12)
            inst_dict, bpm, key, tempo_text, key_text, mode_text = parse_mir_tag_optional(meta, None, style_text[9], inst_thre_ver="0.9")
            inst_text = list(set(inst_dict.keys()))
        else:
            key, bpm, inst_dict = [None] * 3
            tempo_text, key_text, mode_text = '', '', ''
            inst_text = []
        if len(inst_text) > 0:
            style_text[9] = inst_text
        if tempo_text:
            style_text[10] = [tempo_text]
        if key_text:
            style_text[11] = [key_text]
        if mode_text:
            style_text[12] = [mode_text]
        freeform_text = parse_freeform(meta, style_text)
        structure_tags=parse_structure_tags_custom_concat_inst(meta)
        section_instruments = parse_section_instruments_optional(meta)
        structure_tags = add_section_instruments_to_structure_tags(structure_tags, section_instruments)
        return cls(
            utterances=[],
            lyrics_confidence=1,
            structure_tags=structure_tags,
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            freeform_text=freeform_text,
            artist_id=parse_artist_id_by_lang_gender(style_text[7], style_text[5]),
            vad_voice_proportion=parse_vad_voice_proportion_default0(meta),
            tempo=bpm,
            key=key,
            instrument=inst_dict,
        )

    def _validate(
        self,
        lyrics_confidence: Optional[float],      # not used for instrumental
        deepchorus_confidence: Optional[float],  # not used for instrumental
    ):
        # do not validate deepchorus confidence, since the averaged section label prob is not reliable for instruments
        # instead, only validate has_vocal. If it is a vocal song, then disgard it since there is no corresponding lyrics
        # validate_sstk_has_vocal(self.has_vocal)
        validate_vad_voice_proportion_optional(self.vad_voice_proportion, vad_threshold=0.5)
        validate_mir_tempo_optional(self.tempo)
        validate_mir_key_optional(self.key)

    def _to_song_slices(
        self,
        segment_method: str,
        max_seg_per_track: int,
        duration_range: Tuple[int, int],
        lyrics_confidence_phrase: Optional[float],  # not used for instrumental
        slice_mode: str = "section",
        mir_filters: Optional[list[str]] = None,
        multi_tasks: Optional[list[str]] = None,
    ) -> Tuple[ZhMetaLogger, List[SongSlice]]:
        min_duration, max_duration = duration_range
        song_slices = transform_section_to_song_slices(
            min_duration,
            max_duration,
            structure_tags=self.structure_tags.tags,
            slice_mode=slice_mode,
            chorus_only=True,  # guarantee each slice to contain chorus (assumption: deepchorus for instrumental is only reliable for chorus)
        )
        song_slices = sample_song_slices(song_slices, segment_method, max_seg_per_track)

        if multi_tasks is not None and 'global_mir_control' in multi_tasks:
            verified_song_slices = []
            for song_slice in song_slices:
                verified_song_slice = song_slice.select_and_add_mir_info(self.key, self.tempo, self.instrument, mir_filters)
                if mir_filters is not None and \
                    (("unstable_key" in mir_filters and verified_song_slice.mir_info.key is None) or \
                    ("unstable_tempo" in mir_filters and verified_song_slice.mir_info.tempo is None)):
                    continue
                verified_song_slices.append(verified_song_slice)
            song_slices = verified_song_slices

        return ZhMetaLogger(""), song_slices


# ------------------------------------------
#          Dataset-parser Mapping
# ------------------------------------------


@dataclass
class ZhDatasetEntry:
    parser: Type[ZhMetaBaseType]
    desc: str

    is_copyright_cleared: bool
    is_releasable: bool
    is_for_pretrain: bool
    is_for_sft: bool

    is_validation_set: bool = False
    misc: Optional[Dict] = None

    @classmethod
    def new_pretrain_lowrisk(
        cls,
        parser: Type[ZhMetaBaseType],
        desc: str,
        is_validation_set: bool = False,
        misc: Optional[Dict] = None
    ):
        return cls(
            parser=parser,
            desc=desc,
            is_copyright_cleared=False,
            is_releasable=True,
            is_for_pretrain=True,
            is_for_sft=False,
            is_validation_set=is_validation_set,
            misc=misc,
        )

    @classmethod
    def new_pretrain_copyright_cleared(
        cls,
        parser: Type[ZhMetaBaseType],
        desc: str,
        is_validation_set: bool = False,
        misc: Optional[Dict] = None
    ):
        return cls(
            parser=parser,
            desc=desc,
            is_copyright_cleared=True,
            is_releasable=True,
            is_for_pretrain=True,
            is_for_sft=False,
            is_validation_set=is_validation_set,
            misc=misc,
        )

    @classmethod
    def new_pretrain_no_copyright(
        cls,
        parser: Type[ZhMetaBaseType],
        desc: str,
        is_validation_set: bool = False,
        misc: Optional[Dict] = None
    ):
        return cls(
            parser=parser,
            desc=desc,
            is_copyright_cleared=False,
            is_releasable=False,
            is_for_pretrain=True,
            is_for_sft=False,
            is_validation_set=is_validation_set,
            misc=misc,
        )

    @classmethod
    def new_sft_copyright_cleared(
        cls,
        parser: Type[ZhMetaBaseType],
        desc: str,
        is_validation_set: bool = False,
        misc: Optional[Dict] = None,
    ):
        return cls(
            parser=parser,
            desc=desc,
            is_copyright_cleared=True,
            is_releasable=True,
            is_for_pretrain=False,
            is_for_sft=True,
            is_validation_set=is_validation_set,
            misc=misc,
        )
    
    @classmethod
    def new_sft_lowrisk(
        cls,
        parser: Type[ZhMetaBaseType],
        desc: str,
        is_validation_set: bool = False,
        misc: Optional[Dict] = None,
    ):
        return cls(
            parser=parser,
            desc=desc,
            is_copyright_cleared=False,
            is_releasable=True,
            is_for_pretrain=False,
            is_for_sft=True,
            is_validation_set=is_validation_set,
            misc=misc,
        )

    @classmethod
    def new_sft_no_copyright(
        cls,
        parser: Type[ZhMetaBaseType],
        desc: str,
        is_validation_set: bool = False,
        misc: Optional[Dict] = None,
    ):
        return cls(
            parser=parser,
            desc=desc,
            is_copyright_cleared=False,
            is_releasable=False,
            is_for_pretrain=False,
            is_for_sft=True,
            is_validation_set=is_validation_set,
            misc=misc,
        )


ZH_DATASET_REGISTRY = {
    5488: ZhDatasetEntry.new_pretrain_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="sstk 1.08M, with sa tags"
    ),
    4105: ZhDatasetEntry.new_sft_copyright_cleared(
        # parser=ZhMetaInstV4_1,
        parser=ZhMetaInstV4,
        desc="3k, sstk, SFT, double yes, human labels"
    ),
    5253: ZhDatasetEntry.new_sft_copyright_cleared(
        # parser=ZhMetaInstV4_1,
        parser=ZhMetaInstV4,
        desc="20k, sstk, all double yes with fine-grained label",
    ),
    6239: ZhDatasetEntry.new_sft_copyright_cleared(
        # parser=ZhMetaInstV4_1,
        parser=ZhMetaInstV4,
        desc="67k, inst sft",
    ),
    6288: ZhDatasetEntry.new_sft_copyright_cleared(
        # parser=ZhMetaInstV4_1,
        parser=ZhMetaInstV4,
        desc="6239 + section instruments",
    ),
    5148: ZhDatasetEntry.new_pretrain_copyright_cleared(
        # parser=ZhMetaInstV4_1,
        parser=ZhMetaInstV4,
        desc="sstk 1k for validation"       # 2024-10-10 又创建了一个打包数据集，看看这个数据集会不会有问题
    ),
    6245: ZhDatasetEntry.new_pretrain_copyright_cleared(
        # parser=ZhMetaInstV4_1,
        parser=ZhMetaInstV4,
        desc="sstk 1k for validation, 5148 + sa tags"
    ),
    6289: ZhDatasetEntry.new_pretrain_copyright_cleared(
        # parser=ZhMetaInstV4_1,
        parser=ZhMetaInstV4,
        desc="sstk 1k for validation, 6245 + section instruments"
    ),
    6687: ZhDatasetEntry.new_pretrain_copyright_cleared(
        # parser=ZhMetaInstV4_1,
        parser=ZhMetaInstV4,
        desc="sstk 1.08M, with sa tags, with llm_tags"
    ),
    6688: ZhDatasetEntry.new_pretrain_copyright_cleared(
        # parser=ZhMetaInstV4_1,
        parser=ZhMetaInstV4,
        desc="different from 6244: add llm_tags to sstk 1.08M"
    ),
    7660: ZhDatasetEntry.new_pretrain_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="sstk 1.24M, part 1 (with llm_tags) + part 2"
    ),
    3708: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="wyy, instrumental only, subset of 3659, high quality, 113k"
    ),
    7636: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="wyy, instrumental only, 532k"
    ),
    7634: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="Everynoise 11.8k Instrumental"
    ),
    3520: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="multitags debug or valid",
        is_validation_set=True
    ),
    4791: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="1306k inst",
    ),
    5075: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="1655k vocal",
    ),
    5803: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="185k vocal for sft stage1, copy cantonese",
    ),
    5805: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="1655k vocal, copy cantonese",
    ),
    5806: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="2138k vocal",
    ),
    5821: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="2146k vocal",
    ),
    5827: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="501k inst, bpm and inst not empty",
    ),
    5828: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="1807k inst, double bpm and inst not empty",
    ),
    6244: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="1456k inst (no double bpm/inst), add second part of sstk 150k",
    ),
    5904: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,    
        desc="2146k vocal, based on 5821, pt stage, musicfm+ key/tempo/structure/chord/instrument tagging",
    ),
    6000: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,    
        desc="2414k groupB, based on 4776, pt stage, musicfm+ key/tempo/structure/chord/instrument tagging",
    ),
    5910: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,    
        desc="140k, based on 5239, sft stage1 (zh/en/cant), asr auto+86, musicfm+ key/tempo/structure/chord/instrument tagging",
    ),
    5911: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="119k, based on 5238, Inst+SSTK, sft stage1, musicfm+ key/tempo/structure/chord/instrument tagging",
    ),
    6244: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="1456k inst (no double bpm/inst), add second part of sstk 150k",
    ),
    6296: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="debug everynoise",
    ),
    6297: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="debug wy",
    ),
    6315: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="13k vocal sft for v5(cantonese)",
    ),
    6320: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="70k inst sft for v5",
    ),
    7862: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="272k vocal sft for v5 with key/tempo",
    ),
    7863: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="70k inst sft for v5 with key/tempo",
    ),
    6321: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="370k vocal sft for v5",
    ),
    7823: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="370k vocal sft for v5 with key/tempo/instrument",
    ),
    7964: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="haimian domain specific 300 pop+sorrow/sad",
    ),
    6432: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="112k ja vocal for v5",
    ),
    6443: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="17k vocal sft for v5(cantonese)",
    ),
    6508: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="17.5k vocal sft for v5(not one gender)",
    ),
    6590: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="2005k vocal pt for v5(added japanese)",
    ),
    6902: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="??? vocal pt for v5(zh/en/cant/ja)",
    ),
    6902: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="2298 vocal pt for v5(zh/en/cant/ja)",
    ),
    6881: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="3165 inst pt for v5",
    ),
    7637: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="APM, instrumental, 1.06M",
    ),
    7638: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaInstV4,
        desc="RYM, instrumental, 80k",
    ),
    6332: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="6321+key/tempo vocal for v5 sft",
    ),
    6333: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="6320+key/tempo inst for v5 sft",
    ),
    4423: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="validation for v5",
    ),
    7124: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="6321+key/tempo vocal for v5 sft",
    ),
    7125: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="6320+key/tempo inst for v5 sft",
    ),
    5004: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="185k vocal for sft stage1",
    ),
    5127: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="svs",
    ),
    7982: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="2398k, groupB for pt",
    ),
    8125: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="2640k, vocal for pt",
    ),
    8126: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="1737k, vocal for pt",
    ),
    8123: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="3424k, inst for pt",
    ),
    8130: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="115k, inst for sft",
    ),
    8431: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="115k, inst for sft",
    ),
    4893: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="18k inst for sft stage1",
    ),
    8132: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="631k, vocal for sft(-UT plus SP)",
    ),
    8176: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="508k, vocal for sft(-UT plus public)",
    ),
    8203: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="573k, vocal for sft(-UT plus SP)(rm low quality)",
    ),
    8441: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="573k, vocal for sft(-UT plus SP)(rm low quality) + musicfm section",
    ),
    9799: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="7k, 9609 filtered out songs without lyrics, vocal for sft(-UT plus SP)(rm low quality) + human label section",
    ),
    8440: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="8k, vocal with human label MIR for sft(-UT plus SP)(rm low quality)",
    ),    
    8204: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="448k, vocal for sft(-UT plus public)(rm low quality)",
    ),
    8283: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="21k, vocal smaller for sft(-UT SP)(cant, date>2000)",
    ),
    8415: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="483k, vocal for sft(-UT SP)(date>2000)(keep raw genres)",
    ),
    8426: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaInstV4,
        desc="35k, inst for sft(rm sstk)",
    ),
    8604: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="[fix phoneme lost]407k, vocal for sft(-UT plus)(date>2000)(keep raw genres)",
    ),
    8605: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="[fix phoneme lost]482k, vocal for sft(-UT SP)(date>2000)(keep raw genres)",
    ),
    8606: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="[fix phoneme lost]573k, vocal for sft(-UT SP)(rm low quality)",
    ),
    8622: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4_Lyrics,
        desc="12k, zh labeled lyrics",
    ),
    8639: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4_Lyrics,
        desc="[fix]14k, zh labeled lyrics",
    ),
    9048: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="44k, vocal for sft(-UT plus)(date>2000)(keep raw genres)(niche subgenre)",
    ),
    9049: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="52k, vocal for sft(-UT plus)(date>2000)(keep raw genres)(SP artists)",
    ),
    9447: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="35k, vocal for sft(-UT plus)(date>2000)(keep raw genres)(SP artists)(filter start prob)",
    ),
    9454: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4_Lyrics,
        desc="[fix]10k, zh labeled lyrics(filter start prob)",
    ),
    9461: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="18k, vocal for sft(-UT plus)(date>2000)(keep raw genres)(balanced subgenre)(filter start prob)",
    ),
    9564: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="56k, vocal for sft(-UT SP)(wyy top+sa)",
    ),
    10188: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="7k, human label section (need further clarification)",
    ),
    10290: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="473k, SFT (8604) + deepchorus 2",
    ),
    10288: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV4,
        desc="407k, vocal sft stage1+deepchorus 2",
    ),
}
