"""
Custom meta data parser definition and dataset registration.

To add a new dataset:
1. Check if there is already a meta data class that meets your requirement
   a. If so, go ahead to step 2
   b. If not, inherit ZhMetaBase to create a new meta data class
2. Register the dataset into the dataset registry table
"""


from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, TypeVar, Type

from .zh_meta import (
    SongSlice,
    ZhMetaBase,
    ZhMetaLogger,
    # Parsers
    parse_utterance_lyrics,
    parse_style_text_sa,
    parse_lyrics_confidence_force_alignment,
    parse_lyrics_confidence_sa_asr,
    parse_lyrics_confidence_optional,
    parse_structure_tags,
    parse_structure_tags_optional,
    parse_artist_id,
    parse_source,
    parse_filter_label,
    parse_voice_tag,
    parse_voice_tag_sa,
    # Validators
    validate_quality,
    validate_style_text_sa,
    # Converters
    convert_hqmy,
    convert_voice_tag,
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
    ):
        self.zh_meta = zh_meta
        self.sinking_threshold = sinking_threshold
        self.lyrics_confidence = lyrics_confidence
        self.segment_method = segment_method
        self.max_seg_per_track = max_seg_per_track
        self.duration_range = duration_range

    @classmethod
    def from_data_id(cls, data_id: Optional[int], allow_general: bool = False, **kwargs):
        dataset_entry = ZH_DATASET_REGISTRY.get(data_id) if data_id is not None else None
        if not allow_general and dataset_entry is None:
            raise ValueError(f"data_id {data_id} is not registered")
        zh_meta = ZhMetaGeneral if dataset_entry is None else dataset_entry.parser
        return cls(zh_meta, **kwargs)

    def __call__(self, meta: Dict) -> Tuple[ZhMetaLogger, List[SongSlice], List[str], int]:
        """Parse the meta dict into an intermediate representation, then convert it into song slices.
        :return: (meta_logger, song_slices, style_text, artist_id)
        """
        return self.zh_meta.parse(meta, self.sinking_threshold).transform(
            lyrics_confidence=self.lyrics_confidence,
            segment_method=self.segment_method,
            max_seg_per_track=self.max_seg_per_track,
            duration_range=self.duration_range,
        )


# ------------------------------------------
#                Meta Data
# ------------------------------------------

# Add custom data processing here

@dataclass
class ZhMetaLowRisk(ZhMetaBase):
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        utts = parse_utterance_lyrics(meta)
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=utts,
            lyrics_confidence=parse_lyrics_confidence_force_alignment(meta, utts),
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
    @classmethod
    def parse(cls, meta, sinking_threshold: float):
        utts = parse_utterance_lyrics(meta)
        style_text, unfamiliar_tags, is_sinking = parse_style_text_sa(meta, sinking_threshold)
        return cls(
            utterances=utts,
            lyrics_confidence=parse_lyrics_confidence_force_alignment(meta, utts),
            structure_tags=parse_structure_tags(meta),
            style_text=style_text,
            unfamiliar_tags=unfamiliar_tags,
            is_sinking=is_sinking,
            artist_id=parse_artist_id(meta),
            source=parse_source(meta),
            voice_tag=parse_voice_tag(meta),
        )


@dataclass
class ZhMetaSFTBase(ZhMetaBase):
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
            source=parse_source(meta),
        )

    def _validate(self, lyrics_confidence: Optional[float]):
        super()._validate(lyrics_confidence)
        validate_style_text_sa(self.style_text, self.is_sinking)

    def _convert(self):
        _self = super()._convert()
        convert_hqmy(_self)
        convert_voice_tag(_self)
        return _self


@dataclass
class ZhMetaSFTVoiceTagSA(ZhMetaSFTBase):
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
            source=parse_source(meta),
            voice_tag=parse_voice_tag_sa(meta),
        )


@dataclass
class ZhMetaSFTFilterLabel(ZhMetaSFTBase):
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
            source=parse_source(meta),
            is_high_quality=is_high_quality,
            is_popular_potential=is_popular_potential,
        )

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
    def new_sft_copyright_cleared(
        cls,
        parser: Type[ZhMetaBaseType],
        desc: str,
        misc: Optional[Dict] = None,
    ):
        return cls(
            parser=parser,
            desc=desc,
            is_copyright_cleared=True,
            is_releasable=True,
            is_for_pretrain=False,
            is_for_sft=True,
            misc=misc,
        )

    @classmethod
    def new_sft_no_copyright(
        cls,
        parser: Type[ZhMetaBaseType],
        desc: str,
        misc: Optional[Dict] = None,
    ):
        return cls(
            parser=parser,
            desc=desc,
            is_copyright_cleared=False,
            is_releasable=False,
            is_for_pretrain=False,
            is_for_sft=True,
            misc=misc,
        )


ZH_DATASET_REGISTRY = {
    # low risk
    1527: ZhDatasetEntry(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset, 712k",
        is_copyright_cleared=False,
        is_releasable=True,
        is_for_pretrain=True,
        is_for_sft=False,
    ),
    1528: ZhDatasetEntry(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk validation dataset",
        is_copyright_cleared=False,
        is_releasable=True,
        is_for_pretrain=True,
        is_for_sft=False,
        is_validation_set=True
    ),
    1973: ZhDatasetEntry(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset, 450k",
        is_copyright_cleared=False,
        is_releasable=True,
        is_for_pretrain=True,
        is_for_sft=False,
        is_validation_set=True
    ),
    1974: ZhDatasetEntry(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset, 712k + 450k",
        is_copyright_cleared=False,
        is_releasable=True,
        is_for_pretrain=True,
        is_for_sft=False,
        is_validation_set=True
    ),
    2117: ZhDatasetEntry(
        parser=ZhMetaLowRisk,
        desc="Chinese low risk dataset with filtering, 476k + 329k",
        is_copyright_cleared=False,
        is_releasable=True,
        is_for_pretrain=True,
        is_for_sft=False,
        is_validation_set=True
    ),
    1725: ZhDatasetEntry(
        parser=ZhMetaLowRiskVoiceTag,
        desc="Chinese low risk dataset with gender tag",
        is_copyright_cleared=False,
        is_releasable=True,
        is_for_pretrain=True,
        is_for_sft=False,
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
    2115: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="200k high risk qq/wyy",
    ),
    2118: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="110 Billboard",
    ),
    2120: ZhDatasetEntry.new_sft_no_copyright(
        parser=ZhMetaSFTBase,
        desc="3x(Everynoise, 4k + Chinese artist, 7k + Playlist3k5 + Artist2k) + Billboard 110k + Authorized 110k",
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
        parser=ZhMetaSFTVoiceTagSA,
        desc="Authorized data combined, 110k",
    ),
}
