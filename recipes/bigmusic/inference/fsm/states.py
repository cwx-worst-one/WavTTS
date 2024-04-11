from __future__ import annotations
from typing import List, Optional

from bidict import bidict

from recipes.bigmusic.datasets.symbolic_music.consts import (
    BPM_VALUES,
    STEM_LABELS,
    CHORD_LABELS,
    SECTION_LABELS,
    GENRE_TAGS,
)


class State:
    def __init__(self, name: str):
        self.name = name
    
    @property
    def matched_tokens(self) -> List[str]:
        raise NotImplementedError()

    def matches(self, token: str) -> bool:
        return token in self.matched_tokens
    
    def __repr__(self):
        return self.name


class NullState(State):
    """Empty state that does not match any token.
    """
    def __init__(self):
        super().__init__("NULL")
    
    @property
    def matched_tokens(self) -> List[str]:
        return []


class SingleTokenState(State):
    """Match only a single token.
    """
    def __init__(self, matched_token: str, name: Optional[str] = None):
        if name is None:
            super().__init__(f"SINGLE_TOKEN {matched_token}")
        else:
            super().__init__(name)
        self.matched_token = matched_token
    
    @property
    def matched_tokens(self) -> List[str]:
        return [self.matched_token]
    
    def matches(self, token: str) -> bool:
        return token == self.matched_token


class AnyState(State):
    """Special state that matches any token. Requires `rule_set` to
    contain `indexer` member to get all possible tokens.
    """
    def __init__(self, indexer: bidict[str, int]):
        super().__init__("ANY")
        self.indexer = indexer
    
    @property
    def matched_tokens(self) -> List[str]:
        return list(self.indexer.keys())
    
    def matches(self, token: str) -> bool:
        return True


class BeginState(SingleTokenState):
    def __init__(self):
        super().__init__(matched_token="eos", name="BEGIN")


class BarState(SingleTokenState):
    def __init__(self):
        super().__init__(matched_token="bar", name="BAR")


class PositionState(State):
    prefix = "position_"

    # class Category(IntEnum):
    #     Chord = auto()
    #     InstStem = auto()
    #     DrumStem = auto()

    def __init__(
        self,
        # category: PositionState.Category,
        min_pos: Optional[int] = None,
        max_pos: Optional[int] = None,
    ):
        super().__init__("POSITION")
        # self.category = category
        self.min_pos = min_pos or 0
        self.max_pos = max_pos or 16
        self.pos = None
    
    @property
    def matched_tokens(self) -> List[str]:
        return [f"{self.prefix}{p}" for p in range(self.min_pos, self.max_pos)]

    def matches(self, token: str) -> bool:
        if not token.startswith(self.prefix):
            return False
        pos = int(token[len(self.prefix):])
        if pos >= self.min_pos and pos < self.max_pos:
            self.pos = pos
            return True
        return False
    

class ChordState(State):
    prefix = "chord_"
    def __init__(
        self,
        # ref_position_state: PositionState,
        valid_chords: Optional[List[str]] = None
    ):
        super().__init__("CHORD")
        # self.ref_position_state = ref_position_state
        self.valid_chords = valid_chords
        if valid_chords is None:
            self.valid_chord_indexes = list(range(len(CHORD_LABELS)))
        else:
            self.valid_chord_indexes = [
                i for i, c in enumerate(CHORD_LABELS) if c in valid_chords
            ]

    @property
    def matched_tokens(self) -> List[str]:
        return [f"{self.prefix}{i}" for i in self.valid_chord_indexes]


class DrumStemState(SingleTokenState):
    def __init__(self):
        super().__init__(matched_token="stem_5", name="DRUM_STEM")


class InstStemState(State):
    prefix = "stem_"
    def __init__(
        self,
        valid_stems: Optional[List[str]] = None
    ):
        """
        valid_stems:
            a list of valid stems in ["vocal", "piano", "guitar", "bass", "drums"]
            Note that it's function name, not event string!!!
        """
        super().__init__("INST_STEM")
        self.valid_stems = valid_stems
        if valid_stems is None:
            self.valid_stem_indexes = list(range(len(STEM_LABELS) - 1))
        else:
            self.valid_stem_indexes = [
                i for i, c in enumerate(STEM_LABELS) if c in valid_stems
            ]

    @property
    def matched_tokens(self) -> List[str]:
        return [f"{self.prefix}{i}" for i in self.valid_stem_indexes]


class NoteOnState(State):
    prefix = "note_on_"
    def __init__(
        self,
        # ref_position_state: PositionState,
        min_pitch: Optional[int] = None,
        max_pitch: Optional[int] = None,
    ):
        super().__init__("NOTE_ON")
        # self.ref_position_state = ref_position_state
        self.min_pitch = min_pitch or 0
        self.max_pitch = max_pitch or 128
    
    @property
    def matched_tokens(self) -> List[str]:
        return [f"{self.prefix}{p}" for p in range(self.min_pitch, self.max_pitch)]


class NoteDurationState(State):
    prefix = "note_duration_"
    def __init__(
        self,
        # ref_position_state: PositionState,
        valid_durations: Optional[List[int]] = None
    ):
        super().__init__("NOTE_DURATION")
        # self.ref_position_state = ref_position_state
        self.valid_durations = valid_durations or list(range(1, 33))

    @property
    def matched_tokens(self) -> List[str]:
        return [f"{self.prefix}{d}" for d in self.valid_durations]


class DrumState(State):
    prefix = "drum_"
    def __init__(
        self,
        # ref_position_state: PositionState,
    ):
        super().__init__("DRUM")
        # self.ref_position_state = ref_position_state

    @property
    def matched_tokens(self) -> List[str]:
        return [f"{self.prefix}{i}" for i in range(35, 82)]


class SectionState(State):
    prefix = "sec_"
    def __init__(
        self,
        valid_sections: Optional[List[str]] = None
    ):
        super().__init__("SECTION")
        self.valid_sections = valid_sections
        if valid_sections is None:
            self.valid_section_indexes = list(range(len(SECTION_LABELS)))
        else:
            self.valid_section_indexes = [
                i for i, c in enumerate(SECTION_LABELS) if c in valid_sections
            ]

    @property
    def matched_tokens(self) -> List[str]:
        return [f"{self.prefix}{i}" for i in self.valid_section_indexes]


class BPMLevelState(State):
    prefix = "bpm_level_"
    def __init__(
        self,
        # valid_sections: Optional[List[str]] = None
    ):
        super().__init__("BPM_LEVEL")
        self.valid_bpm_indexes = list(range(len(BPM_VALUES)))
        # self.valid_sections = valid_sections
        # if valid_sections is None:
        # else:
        #     self.valid_section_indexes = [
        #         i for i, c in enumerate(SECTION_LABELS) if c in valid_sections
        #     ]

    @property
    def matched_tokens(self) -> List[str]:
        return [f"{self.prefix}{i}" for i in self.valid_bpm_indexes]


class GenreState(State):
    prefix = "genre_"
    def __init__(
        self,
        valid_genres: Optional[List[str]] = None
    ):
        """valid_genres is a list of items in GENRE_TAGS
        """
        super().__init__("GENRE")
        if valid_genres is None:
            self.valid_genre_indexes = list(range(len(GENRE_TAGS)))
        else:
            self.valid_section_indexes = [
                i for i, c in enumerate(GENRE_TAGS) if c in valid_genres
            ]

    @property
    def matched_tokens(self) -> List[str]:
        return [f"{self.prefix}{i}" for i in self.valid_genre_indexes]
