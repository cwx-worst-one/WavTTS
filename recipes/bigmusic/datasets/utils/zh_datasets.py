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
from typing import Any, Optional, Dict, Tuple, TypeVar, Type

from .zh_meta import (
    ZhMetaBase,
    # Parsers
    parse_utterance_lyrics,
    parse_utterance_lyrics_force_align,
    parse_style_text_sa,
    parse_style_text_audio_tags_v0,
    parse_style_text_audio_tags_v1,
    parse_style_text_audio_tags_v2,
    parse_lyrics_confidence_force_align_legacy,
    parse_lyrics_confidence_force_align,
    parse_lyrics_confidence_sa_asr,
    parse_lyrics_confidence_optional,
    parse_structure_tags,
    parse_structure_tags_optional,
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
    # Validators
    validate_style_text_sa,
    validate_deepchorus_optional,
    # Converters
    convert_hqmy,
    convert_artist_id_from_voice_tag,
    convert_voice_tag_from_audio_tag,
)

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
    def parse(cls, meta, sinking_threshold: float):
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

    def _validate(self, lyrics_confidence: Optional[float]):
        super()._validate(lyrics_confidence)
        validate_style_text_sa(self.style_text, self.is_sinking)


class ZhMetaTransform:
    def __init__(
        self,
        zh_meta: Type[ZhMetaBaseType],
        sinking_threshold: float,
        lyrics_confidence: Optional[float],
        segment_method: str,
        max_seg_per_track: int,
        duration_range: Tuple[int, int],
        lyrics_confidence_phrase: Optional[float],
    ):
        self.zh_meta = zh_meta
        self.sinking_threshold = sinking_threshold
        self.lyrics_confidence = lyrics_confidence
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.duration_range = duration_range
        self.lyrics_confidence_phrase = lyrics_confidence_phrase

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
        return self.zh_meta.parse(meta, self.sinking_threshold).transform(
            lyrics_confidence=self.lyrics_confidence,
            segment_method=self.segment_method,
            max_seg_per_track=self.max_seg_per_track,
            duration_range=self.duration_range,
            lyrics_confidence_phrase=self.lyrics_confidence_phrase,
        )


# ------------------------------------------
#                Meta Data
# ------------------------------------------

# Add custom data processing here

@dataclass
class ZhMetaLowRisk(ZhMetaBase):
    """music_tagging (SA), deepchorus, force_alignment lyrics"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        utts = parse_utterance_lyrics(meta)
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=utts,
            lyrics_confidence=parse_lyrics_confidence_force_align_legacy(meta, utts),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
        )

    def _validate(self, lyrics_confidence: Optional[float]):
        super()._validate(lyrics_confidence)
        validate_style_text_sa(self.style_text, self.is_sinking)


@dataclass
class ZhMetaLowRiskVoiceTag(ZhMetaLowRisk):
    """music_tagging (SA), deepchorus, force_alignment lyrics, gender"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        utts = parse_utterance_lyrics(meta)
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=utts,
            lyrics_confidence=parse_lyrics_confidence_force_align_legacy(meta, utts),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag(meta),
        )


@dataclass
class ZhMetaLowRiskSA(ZhMetaLowRisk):
    """music_tagging (SA), deepchorus, asr lyrics (SA), sa_gender"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag_sa(meta),
        )

@dataclass
class ZhMetaLowRiskMIR(ZhMetaLowRisk):
    """music_tagging (SA), deepchorus, asr lyrics (SA), sa_gender, tempo, key"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag_sa(meta),
            tempo=parse_mir_tempo(meta),
            key=parse_mir_key(meta),
        )


@dataclass
class ZhMetaMixLangVoiceTag(ZhMetaLowRisk):
    """music_tagging (SA), deepchorus, asr lyrics (SA), sa_gender"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag_sa(meta),
        )


@dataclass
class ZhMetaMixLangOpt(ZhMetaLowRisk):
    """music_tagging (SA), deepchorus, asr lyrics (SA), sa_gender (optional), tempo (optional), key (optional)"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag_sa_optional(meta),
            tempo=parse_mir_tempo_optional(meta),
            key=parse_mir_key_optional(meta),
        )


@dataclass
class ZhMetaForceAlignVoiceTag(ZhMetaLowRisk):
    """music_tagging (SA), deepchorus, lyrics_force_align, gender"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        utts = parse_utterance_lyrics_force_align(meta)
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=utts,
            lyrics_confidence=parse_lyrics_confidence_force_align(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag_sa(meta),
        )


@dataclass
class ZhMetaSFTBase(ZhMetaBase):
    """music_tagging (SA), deepchorus, asr lyrics (SA). Deepchorus confidence validation."""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
        )

    def _validate(self, lyrics_confidence: Optional[float]):
        super()._validate(lyrics_confidence)
        validate_style_text_sa(self.style_text, self.is_sinking)
        validate_deepchorus_optional(self.structure_tags)

    def _convert(self):
        _self = super()._convert()
        convert_hqmy(_self)
        return _self


@dataclass
class ZhMetaSFTVoiceTagSA(ZhMetaSFTBase):
    """music_tagging (SA), deepchorus, asr lyrics (SA), sa_gender"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag_sa(meta),
        )


@dataclass
class ZhMetaSFTMIRVoiceTagSA(ZhMetaSFTBase):
    """music_tagging (SA), deepchorus, asr lyrics (SA), tempo, key"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=parse_voice_tag_sa(meta),
            tempo=parse_mir_tempo(meta),
            key=parse_mir_key(meta),
        )


@dataclass
class ZhMetaSFTFilterLabel(ZhMetaSFTBase):
    """music_tagging (SA), deepchorus, asr lyrics (SA), filter_label"""
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        is_high_quality, is_popular_potential = parse_filter_label(meta)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            is_high_quality=is_high_quality,
            is_popular_potential=is_popular_potential,
        )

@dataclass
class ZhMetaSFTAudioTagsV0(ZhMetaBase):
    # only support read audio_tags
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_audio_tags_v0(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=1.0,
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
        )

    def _validate(self, lyrics_confidence: Optional[float]):
        pass

@dataclass
class ZhMetaSFTAudioTagsV1(ZhMetaBase):
    # support read audio_tags and music_tagging
    """
    meta.audio_tags.genre/meta.music_tagging.Genre20,
    meta.audio_tags.mood/meta.music_tagging.Mood,
    meta.audio_tags.scene/meta.music_tagging.Theme,
    meta.audio_tags.vocal_gender/meta.gender.sa_gender,
    meta.audio_tags.vocal_timbre,
    meta.deepchorus,
    meta.lyrics,
    meta.mir_service.beat.tempo,
    meta.mir_service.key.song
    """
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_audio_tags_v1(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            #tempo=parse_mir_tempo_optional(meta),
            #key=parse_mir_key_optional(meta),
        )

    def _validate(self, lyrics_confidence: Optional[float]):
        super()._validate(lyrics_confidence)
        #pass


@dataclass
class ZhMetaSFTAudioTagsV1NoConf(ZhMetaSFTAudioTagsV1):
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_audio_tags_v1(meta, sinking_threshold)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=None,  # No confidence
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
        )


@dataclass
class ZhMetaSFTAudioTagsV2(ZhMetaBase):
    """
    Support read audio_tags and music_tagging.
    meta.audio_tags.genre/meta.music_tagging.Genre20,
    meta.audio_tags.mood/meta.music_tagging.Mood,
    meta.audio_tags.scene/meta.music_tagging.Theme,
    meta.audio_tags.vocal_gender/meta.gender.sa_gender,
    meta.audio_tags.vocal_timbre,
    meta.deepchorus,
    meta.lyrics,
    meta.mir_service.beat.tempo,
    meta.mir_service.key.song
    """
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_audio_tags_v2(meta, sinking_threshold)
        voice_tag = parse_voice_tag_sa_optional(meta)
        return cls(
            utterances=parse_utterance_lyrics(meta),
            lyrics_confidence=parse_lyrics_confidence_sa_asr(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=voice_tag,
        )

    def _convert(self):
        _self = copy.deepcopy(self)
        convert_voice_tag_from_audio_tag(_self)
        convert_artist_id_from_voice_tag(_self)
        return _self


@dataclass
class ZhMetaSFTForceAlignConfAudioTagsV2(ZhMetaBase):
    """
    Support read audio_tags and music_tagging, only read GT lyrics_force_align. Filter by confidence.
    meta.audio_tags.genre/meta.music_tagging.Genre20,
    meta.audio_tags.mood/meta.music_tagging.Mood,
    meta.audio_tags.scene/meta.music_tagging.Theme,
    meta.audio_tags.vocal_gender/meta.gender.sa_gender,
    meta.audio_tags.vocal_timbre,
    meta.deepchorus,
    meta.lyrics,
    """
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        style_text, unfamiliar_tags, is_sinking = parse_style_text_audio_tags_v2(meta, sinking_threshold)
        voice_tag = parse_voice_tag_sa_optional(meta)
        return cls(
            utterances=parse_utterance_lyrics_force_align(meta),
            lyrics_confidence=parse_lyrics_confidence_force_align(meta),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source_optional(meta),
            voice_tag=voice_tag,
        )

    def _convert(self):
        _self = copy.deepcopy(self)
        convert_voice_tag_from_audio_tag(_self)
        convert_artist_id_from_voice_tag(_self)
        return _self


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
    # low risk
    1527: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset, 712k",
    ),
    1528: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk validation dataset",
        is_validation_set=True,
    ),
    1973: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset, 450k",
    ),
    1974: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset, 712k + 450k",
    ),
    2117: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset with filtering, 476k + 329k",
    ),
    2192: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset with mir service(V41.join(V33))--Train set",
        is_validation_set=True
    ),
    2191: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset with mir service(V41.join(V33))--Test set",
        is_validation_set=True
    ),
    1725: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRiskVoiceTag,
        desc="Chinese low risk dataset with gender tag",
    ),
    2226: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRiskSA,
        desc="New Chinese low risk dataset (mcc, dq, wyy) with aggressive filtering, 141k + 520k + 339k",
    ),
    2228: ZhDatasetEntry.new_pretrain_no_copyright(
        parser=ZhMetaMixLangOpt,
        desc="English Group A (key, tempo) + Billboard (key, tempo, no gender tag) + New Chinese low risk dataset (mcc, dq, wyy) with aggressive filtering, 283k + 100k + 141k + 520k + 339k",
    ),
    2287: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaLowRiskMIR,
        desc="New Chinese low risk dataset (mcc, dq, wyy) with MIR tags and aggressive filtering, 141k + 519k + 337k",
    ),
    2350: ZhDatasetEntry.new_pretrain_no_copyright(
        parser=ZhMetaMixLangOpt,
        desc="English Group A (key, tempo) + Billboard (key, tempo, no gender) + New Chinese low risk dataset (mcc, dq, wyy) with aggressive filtering, 283k + 100k + 141k + 520k + 339k + 440k high risk, MIR",
    ),
    2512: ZhDatasetEntry.new_pretrain_no_copyright(
        parser=ZhMetaMixLangVoiceTag,
        desc="English Group A (key, tempo), 283k + Billboard (key, tempo, gender), 101k + new Chinese high risk with MIR, 432k ",
    ), 
    2504: ZhDatasetEntry.new_pretrain_lowrisk(
        parser=ZhMetaForceAlignVoiceTag,
        desc="New Chinese low risk dataset (mcc, dq, wyy) with force alignment lyrics, 809k",
    ),
    # SFT, internal
    1879: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="Everynoise, 4k",
    ),    
    1662: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="Chinese artist, 7k",
    ),
    1932: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="Everynoise, 4k + Chinese artist, 7k",
    ),
    1934: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="Everynoise, 4k + Chinese artist, 7k (+MSS)",
    ),
    2049: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="Everynoise, 4k + Chinese artist, 7k + Playlist3k5 + Artist2k",
    ),
    2074: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="Everynoise, 4k + Chinese artist, 7k + Playlist3k5 + Artist2k, deduplication",
    ),
    2140: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="200k high risk qq/wyy, fix phone",
    ),
    2168: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="400k high risk qq/wyy",
    ),
    2118: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="110 Billboard",
    ),
    2120: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="3x(Everynoise, 4k + Chinese artist, 7k + Playlist3k5 + Artist2k) + Billboard 110k + Authorized 110k",
    ),
    2144: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="3x(Everynoise, 4k + Chinese artist, 7k + Playlist3k5 + Artist2k) + 2x High risk 200k + Billboard 110k",
    ),
    2189: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="3x(Everynoise, 4k + Chinese artist, 7k + Playlist3k5 + Artist2k) + High risk 440k + Billboard 110k + 20 X 7k lightweight internal",
    ),
    2249: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="7k5 authorized lightweight filtered from 110k without fine-grained labels",
    ),
    2343: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV0,
        desc="high risk 1k7, from xiaohong",
    ),
    2408: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 3.5k+multitags 6k(from wy)+low risk 5k+high risk 1k7 from xiaohong, rerun sa lyrics, remove valid, update phoneme",
    ),
    2407: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 3.5k+multitags 6k(from mcc)+low risk 5k+high risk 1k7 from xiaohong, rerun sa lyrics, remove valid, update phoneme",
    ),
    # SFT, releasable
    1935: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTBase,
        desc="HQMY + StarNation, 20k",
    ),
    1837: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTBase,
        desc="First batch authorized, 35k",
    ),
    1945: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTBase,
        desc="HQMY + StarNation, 20k + First batch authorized, 35k",
    ),
    1976: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTBase,
        desc="HQMY + StarNation, 20k + First batch authorized, 35k (+MSS)",
    ),
    2122: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTFilterLabel,
        desc="lightweight labelling, 30k",
    ),
    2123: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTFilterLabel,
        desc="lightweight labelling, high quality + popular potential, 7k",
    ),
    # SFT, with gender tags
    1939: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTVoiceTagSA,
        desc="ASR lyrics + phonemes + gender tagging, 35k",
    ),
    2095: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTVoiceTagSA,
        desc="Same as 1939 (todo: remove it)"
    ),
    2119: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTMIRVoiceTagSA,  # parse additional MIR info
        desc="Authorized data combined, 110k",
    ),
    2256: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTMIRVoiceTagSA,
        desc="Authorized data combined, aggressive filtering, 60k",
    ),
    2193: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV0,
        desc="multitags 3.5k",
    ),
    # 2194: ZhDatasetEntry.new_sft_copyright_cleared(
    #     parser=ZhMetaSFTAudioTags,
    #     desc="multitags 10 for valid or debug",
    #     is_validation_set=True,
    # ),
    2301: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="multitags 3.5k, add mir and update SA asr lyrics",
    ),
    2307: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="multitags 3.5k + double-yes 7.5k, fix sa lyrics",
    ),
    2289: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="multitags 3.5k + double-yes 7.5k",
    ),
    2293: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1NoConf,
        desc="double-yes 7.5k",
    ),
    2301: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="multitags 3.5k, add mir and update SA asr lyrics",
    ),
    2307: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="multitags 3.5k + double-yes 7.5k, fix sa lyrics",
    ),
    2310: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="dq low risk from xiaohong 2.5k, chinese",
    ),
    2311: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="multitags 3.5k, fix sa lyrics",
    ),
    2379: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="multitags 7k + 3.5k, low risk 5k",
    ),
    2332: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="wyy low risk from xiaohong 2k, chinese",
    ),
    2333: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="mcc low risk from xiaohong 0.5k, chinese",
    ),
    2364: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="multitags 7.5k",
    ),
    2367: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags debug or valid",
    ),
    2378: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 6k, labeled from wy",
    ),
    2379: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 3.5k+multitags 6k(from wy)+low risk from xiaohong 5k, fix sa lyrics, remove valid",
    ),
    2383: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV1,
        desc="Above 3, mcc low risk from xiaohong 5k, chinese",
    ),
    2384: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 3.5k+multitags 6k(from wy)+low risk from xiaohong 5k, rerun sa lyrics, remove valid",
    ),
    2385: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 3.5k+multitags 6k(from mcc)+low risk from xiaohong 5k, rerun sa lyrics, remove valid",
    ),
    2405: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 3.5k+multitags 6k(from wy), update phoneme",
    ),
    2406: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 3.5k+multitags 6k(from mcc), update phoneme",
    ),
    2401: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 6k(from wy)",
    ),
    2454: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="multitags 6k",
    ),
    2456: ZhDatasetEntry.new_sft_lowrisk(
        parser=ZhMetaSFTMIRVoiceTagSA,
        desc="lowrisk taobao 4k",
    ),
    2476: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTForceAlignConfAudioTagsV2,
        desc="6k fine-grained from 110K authorized (GT lyrics) + 3.5K fine-grained from initial 35K authorized (GT lyrics)",
    ),
    2477: ZhDatasetEntry.new_sft_lowrisk(
        parser=ZhMetaSFTForceAlignConfAudioTagsV2,
        desc="5k taobao with supplimentary genre V1 from 1.47 low risk (GT lyrics)",
    ),
    2490: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTForceAlignConfAudioTagsV2,
        desc="lowrisk taobao 4k, gt lyrics (NOTE: Only a small portion of data has gt_lyrics!!)",
    ),
    2558: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTForceAlignConfAudioTagsV2,
        desc="lowrisk taobao 4k, gt lyrics",
    ),
    2596: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTAudioTagsV2,
        desc="lowrisk taobao 5k -> fine-grained 3.5k",
    ),
    2674: ZhDatasetEntry.new_sft_copyright_cleared(
        parser=ZhMetaSFTForceAlignConfAudioTagsV2,
        desc="lowrisk taobao 5k -> fine-grained 3.5k, GT lyrics",
    ),
}
