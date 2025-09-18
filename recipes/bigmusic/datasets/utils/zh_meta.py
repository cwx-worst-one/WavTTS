"""
Intermediate meta data representation for Chinese vocal datasets with parsing and transforming functions
"""

from dataclasses import dataclass, asdict
from functools import partial, reduce
import operator
from typing import Any, Optional, List, Dict, Tuple, Union
import math
from collections import Counter
import random
from random import Random
import copy
import numpy as np
import json
import ast
import logging
logger = logging.getLogger(__file__)

from recipes.bigmusic.datasets.mir_data_util import (
    ARTIST_ID_MAP_V2,
    SA_CAT_VOCAB,
    SA_TAGS_MOOD_SPECIAL_MAP,
    SA_TAGS_SPECIAL_MAP,
    AUDIO_TAGS_GENRE_SPECIAL_MAP_V2,
    AUDIO_TAGS_MOOD_SPECIAL_MAP_V2,
    AUDIO_TAGS_SCENE_SPECIAL_MAP_V2,
    AUDIO_TAGS_GENDER_SPECIAL_MAP_V2,
    AUDIO_TAGS_TIMBRE_SPECIAL_MAP,  # dedup
    AUDIO_CAT_VOCAB_V3,
    AUDIO_TAGS_LANG_SPECIAL_MAP,
    AUDIO_TAGS_EXTRA_SPECIAL_MAP,
    AUDIO_TAGS_INST_SPECIAL_MAP,
    AUDIO_TAGS_TEMPO_SPECIAL_MAP,
    AUDIO_TAGS_MODE_SPECIAL_MAP,
    AUDIO_TAGS_KEY_SPECIAL_MAP,
    VOICE_THRESHOLDS,
    TEMPO_RANGE, 
    tempo_to_label,
    tempo_to_coarse_label,
    TEMPO_LABEL_ID_MAP,
    KEYS, ROOTS, ROOT_ID_MAP, MODE_ID_MAP, ID_ROOT_MAP,
    KEY_ID_MAP, VOCAB2ID_INST_V1,
    TIME_SIGNATURE_ID_MAP
)
from recipes.bigmusic.datasets.combo_audio_tags import COMBO_GENRE_TO_INSTRUMENTS_V4
from recipes.bigmusic.datasets.raw_meta_tags import (
    EVERYNOISE_MATCHED_TAG_TO_CATEGORY,
    EVERYNOISE_NON_MATCHED_TAG_TO_CATEGORY,
    SSTK_INST_TO_VOCAB,
    KARAOKE_INST_270_TO_38,
    KARAOKE_INST_38_TO_VOCAB,
    WYY_TAG_TO_EXTRA,
    WYY_TAG_TO_FREEFORM,
    WYY_TAG_TO_GENRE_EXTRA,
    WYY_TAG_TO_GENRE,
    WYY_TAG_TO_INST,
    WYY_TAG_TO_LANG,
    WYY_TAG_TO_MOOD,
    WYY_TAG_TO_SCENE,
    APM_MATCHED_TAG_TO_GENRE,
    APM_MATCHED_TAG_TO_MOOD,
    APM_MATCHED_TAG_TO_SCENE,
    APM_MATCHED_TAG_TO_INST,
    APM_NON_MATCHED_TAG_TO_GENRE,
    APM_NON_MATCHED_TAG_TO_MOOD,
    APM_NON_MATCHED_TAG_TO_SCENE,
    APM_NON_MATCHED_TAG_TO_INST,
    APM_TAG_TO_VOCAL,
    APM_TAG_TO_EXTRA,
    APM_TAG_TO_TEMPO,
    APM_TAG_TO_FREEFORM,
    APM_TAG_TO_REMOVE,
    RYM_MATCHED_TAG_TO_GENRE,
    RYM_NON_MATCHED_TAG_TO_GENRE,
    RYM_MATCHED_TAG_TO_MOOD,
    RYM_NON_MATCHED_TAG_TO_MOOD,
    RYM_MATCHED_TAG_TO_SCENE,
    RYM_NON_MATCHED_TAG_TO_SCENE,
    RYM_MATCHED_TAG_TO_INST,
    RYM_NON_MATCHED_TAG_TO_INST,
    RYM_TAG_TO_VOCAL,
    RYM_TAG_TO_EXTRA,
    SSTK_MATCHED_TAG_TO_GENRE,
    SSTK_NON_MATCHED_TAG_TO_GENRE,
    SSTK_MATCHED_TAG_TO_MOOD,
    SSTK_NON_MATCHED_TAG_TO_MOOD,
    SSTK_MOOD_TAG_TO_TYPE,
    SSTK_MATCHED_TAG_TO_SCENE,
    SSTK_NON_MATCHED_TAG_TO_SCENE,
    SSTK_MATCHED_TAG_TO_INST,
    SSTK_NON_MATCHED_TAG_TO_INST,
    # SSTK_TAG_TO_VOCAL,
    SSTK_TAG_TO_EXTRA,
)
from recipes.bigmusic.datasets.transforms.lyrics_section_alignment import merge_raw_song_structure_by_start_prob
from recipes.datasets.mcc.sami_tokenizer import Phrase
from recipes.bigmusic.datasets.transforms.leadsheet import fetch_notes_from_phones_v3
from recipes.bigmusic.datasets.transforms.sami_phoneme import transform_phonemes_by_language
from recipes.bigmusic.datasets.transforms.inst import inst_38_to_tagging, tagging_inst_to_38, get_inst_family, is_valid_inst
from recipes.bigmusic.datasets.utils.sami_parser import NO_VOCAL_SECTION_TAGS

@dataclass
class DeepChorus:
    tags: List
    confidence: float


@dataclass
class Note:
    pitch: int
    time_span: Tuple[float, float]

    @classmethod
    def from_dict(cls, d: Dict) -> "Note":
        def get_time_span(d: Dict) -> Optional[Tuple[float, float]]:
            if "time_span" in d:
                return d["time_span"]
            elif "start" in d and "end" in d:
                return (d["start"], d["end"])
            return None
        return cls(
            pitch=d["pitch"],
            time_span=get_time_span(d),
        )

    def to_leadsheet_note(self) -> Dict:
        if not self.time_span:
            raise ValueError("time_span is not set")
        return {
            "pitch": self.pitch,
            "start": self.time_span[0],
            "end": self.time_span[1],
        }

    @classmethod
    def from_vocal2midi(cls, d: Dict) -> List["Note"]:
        return [cls.from_dict(_d) for _d in d["notes"]]
    
    def transpose(self, semitones: int) -> "Note":
        """For augmentation"""
        return self.__class__(**{**asdict(self), "pitch": self.pitch + semitones})

@dataclass
class MIRInfo:
    # --- musicfm+ output format --- #
    key_list: Optional[List] = None
    beat_list: Optional[List] = None

    # --- key / mode / tempo value and LOCAL ID ---
    key: Optional[str] = None
    key_id: Optional[int] = None
    key_root_id: Optional[int] = None   
    key_mode_id: Optional[int] = None   
    tempo: Optional[int] = None 
    tempo_id: Optional[int] = None 
    main_inst_id: Optional[int] = None   
    insts_id: Optional[List[int]] = None
    time_signature: Optional[int] = None

    # --- style text | audio tag ---
    key_text: Optional[str] = None  # "C"
    mode_text: Optional[str] = None     # "Major"
    tempo_text: Optional[str] = None    # 
    main_inst_text: Optional[str] = None    # Acoustic_Piano
    insts_text: Optional[Union[list, str]] = None

    @staticmethod
    def key_conversion(key, convert="root_convert_to_maj"):
        if convert is None or key == "X":
            return key
        
        root, mode = key.split(":") 
        if mode == "Min":
            major_root = (ROOT_ID_MAP[root] + 3) % 12
            root = ID_ROOT_MAP[major_root if major_root != 0 else 12]
            if "key_convert_to_maj" in convert:
                key = f"{root}:Maj"
            elif "root_convert_to_maj" in convert:  
                key = f"{root}:{mode}"   # keep the original mode
        return key
    
    @staticmethod
    def key_id_conversion(key_id, convert="root_convert_to_maj"):
        if convert is None or key_id == KEY_ID_MAP["X"] or \
            (not any([x in convert for x in ["root_convert_to_maj", "major_convert_to_maj"]])):
            return key_id
        if key_id % 2 == 0: # even number -> 'minor'
            root_id = key_id // 2
            major_root_id = (root_id + 3) % 12
            root_id = major_root_id if major_root_id != 0 else 12
            key_id = root_id * 2
        return key_id
        
    @staticmethod
    def get_time_signature_from_beat_idx(beat_idx_list):
        length = len(set(beat_idx_list))
        if length in [2, 3, 4]: 
            return f"{length}/4"
        else:
            return "X"

    @staticmethod
    def get_tempo(beat_list=None, tempo=None, mir_filters=None) -> int:
        # 1) global/gt tempo is given
        time_signature = "X"
        tempo_text = ''

        if tempo is not None:   
            tempo_value, tempo_text = round(tempo), tempo_to_label(tempo)
            return tempo_value, TEMPO_LABEL_ID_MAP[tempo_text], tempo_text, TIME_SIGNATURE_ID_MAP[time_signature]

        # 2) both global/gt tempo and beat list are not available
        tempo = -1
        if beat_list is None or len(beat_list) <= 8:    # empty tempo: "", empty ts: "X"
            return tempo, TEMPO_LABEL_ID_MAP[tempo_text], tempo_text, TIME_SIGNATURE_ID_MAP[time_signature]
        
        # 3) beat list can be used to infer tempo
        beat_list = np.array(beat_list)
        beat_time_list, beat_idx_list = beat_list[:, 0], beat_list[:, 1]
        
        if mir_filters is not None and 'unstable_tempo' in mir_filters:
            beat_diff = np.diff(beat_time_list, n=1)
            mean, std = beat_diff.mean(), beat_diff.std()
            if std / mean >= 0.15:   # TODO (qinxin): adjust the ratio according to slice duration
                return None, None, None, None
        
        time_signature = MIRInfo.get_time_signature_from_beat_idx(beat_idx_list)
            
        avg_beat_interval = np.mean(np.diff(beat_time_list, n=1))
        tempo = 60 / avg_beat_interval
        if avg_beat_interval < 0.001 or math.isnan(tempo):
            tempo, time_signature = -1, "X"
        tempo_text = tempo_to_label(tempo)
        return round(tempo), TEMPO_LABEL_ID_MAP[tempo_text], tempo_text, TIME_SIGNATURE_ID_MAP[time_signature]

    
    @staticmethod
    def get_key(key_list=None, key=None, mir_filters=None) -> int:
        if key is not None and key in KEY_ID_MAP: # X:Maj/Min
            if any([x in mir_filters for x in ["root_convert_to_maj", "key_convert_to_maj"]]):
                key = MIRInfo.key_conversion(key, mir_filters)
            root, mode = key.split(":")
            return key, root, mode, KEY_ID_MAP[key], ROOT_ID_MAP[root], MODE_ID_MAP[mode]
        
        key, root, mode = "X", "X", "X"
        if key_list is None or len(key_list) <= 0:  # empty key/root/mode: "X"
            return key, root, mode, KEY_ID_MAP[key], ROOT_ID_MAP[root], MODE_ID_MAP[mode]

        key_list = [key_type for (key_time, key_type) in key_list]

        if mir_filters is not None:
            if any([x in mir_filters for x in ["root_convert_to_maj", "key_convert_to_maj"]]):
                new_key_list = [MIRInfo.key_conversion(key, mir_filters) for key in key_list]
                key_list = new_key_list
            if "unstable_key" in mir_filters:
                key, freq = Counter(key_list).most_common(1)[0]
                if freq / len(key_list) < 0.75:
                    # print("[get_key] unstable key", Counter(key_list))
                    return [None] * 6
                
        key, freq = Counter(key_list).most_common(1)[0]
        if key == "X":
            root, mode = "X", "X"
        else:
            root, mode = key.split(":")
        return key, root, mode, KEY_ID_MAP[key], ROOT_ID_MAP[root], MODE_ID_MAP[mode]
    
    @staticmethod
    def get_inst(inst_dict=None, main_instrument=None, instruments=None, mir_filters=None):
        main_inst, insts = None, None
        main_inst_id, insts_id = None, None
        inst_vocab = VOCAB2ID_INST_V1.to_dict()
        if main_instrument is not None or instruments is not None: # human-labelled data processing
            if main_instrument is not None and main_instrument in inst_vocab:
                main_inst = main_instrument
                main_inst_id = inst_vocab[main_instrument]
            if instruments is not None:
                insts = [inst for inst in instruments if inst in inst_vocab]
                insts_id = [inst_vocab[inst] for inst in instruments if inst in inst_vocab]

        elif inst_dict is not None and len(inst_dict) > 0:  # musicfm tagging processing
            # sort ALL instruments based on their probabilities (including main inst)
            insts = sorted(inst_dict, key=inst_dict.get, reverse=True)
            # assume main instrument is the inst with the maximum probability
            main_inst = insts[0]
            if mir_filters is not None:
                candidate_insts = copy.deepcopy(insts)
                for _inst in ["Vocal", "Drums", "Drum_Set", "Bass", "Choir_and_Voice"]:
                    if f'main_inst_exclude_{_inst}' in mir_filters and main_inst == _inst:
                        candidate_insts.remove(_inst)
                        main_inst = None if len(candidate_insts) == 0 else candidate_insts[0]
            main_inst_id = inst_vocab[main_inst] if main_inst is not None and main_inst in inst_vocab else inst_vocab['None']
            insts_id = list(set([inst_vocab[_inst] for _inst in insts if _inst in inst_vocab]))
        return main_inst, insts, main_inst_id, insts_id


    @classmethod
    def new(cls, key_list, beat_list, inst_dict, key, tempo, mir_filters):
        key, root, mode, key_id, root_id, mode_id = cls.get_key(key_list, key, mir_filters)
        tempo_value, tempo_id, tempo_text, time_signature = cls.get_tempo(beat_list, tempo, mir_filters)
        main_inst, insts, main_inst_id, insts_id = cls.get_inst(inst_dict, mir_filters=mir_filters)
        return cls(
            key_list=key_list,
            beat_list=beat_list,
            tempo=tempo_value,
            tempo_id=tempo_id,
            tempo_text=tempo_text,
            time_signature=time_signature,
            key=key, key_id=key_id, 
            key_root_id=root_id, key_mode_id=mode_id, 
            key_text=root if root != 'X' else '', mode_text=mode if mode != 'X' else '',
            main_inst_text=main_inst, main_inst_id=main_inst_id,
            insts_text=insts, insts_id=insts_id,
        )
    
    def rewrite_style_text(self, ori_style_text, mir_dropout_rate=0.0,
                            ori_freeform_text=None, rewrite_freeform_text=False, 
                            seed=None):
        # Note: mir_dropout_rate only applies to key, tempo, mode (no inst)
        rnd = Random(seed)
        new_style_text = copy.deepcopy(ori_style_text)
        new_freeform_text = copy.deepcopy(ori_freeform_text)

        if self.insts_text is not None:
            new_style_text[9] = self.insts_text
            if rewrite_freeform_text and ori_freeform_text is not None:
                instruments = self.insts_text
                instruments_text_list = []
                for inst in instruments:
                    instruments_text_list.extend([item.replace("_", " ") for item in get_inst_family(inst) if item])
                instruments_text_list = list(set([inst.replace("_", " ") for inst in instruments]))
                # Do not shuffle sorted instruments; If instrument list is too long, freeform text may be overwhelmed with instrument info
                # Also main instrument info will be used in priority 
                if len(instruments_text_list) > 0:    # insert into a random position of freeform_text_short
                    fft_list = new_freeform_text.split(", ")
                    if rnd.random() < 0.5:   
                        rnd.shuffle(instruments_text_list)
                    instruments_description = ", ".join(instruments_text_list[:min(rnd.randint(0, len(instruments_text_list)),10)])
                    # if song_slice.mir_info.main_instrument is not None and random.random() < 0.5:
                    #     main_inst = song_slice.mir_info.main_instrument
                    #     instruments_description = instruments_description + f", {main_inst}"
                    fft_list.insert(rnd.choice(range(len(fft_list))), instruments_description)
                    new_freeform_text = ", ".join(fft_list)
            
        if self.tempo_text is not None:
            if rnd.random() < mir_dropout_rate:
                new_style_text[10] = ['empty tempo']
            else:
                new_style_text[10] = [self.tempo_text]

                if rewrite_freeform_text and new_freeform_text is not None:
                    candidate_speed_description = []
                    if self.tempo is not None and self.tempo > 0:
                        tempo_value = self.tempo
                        bpm_value_description = rnd.choice([f"bpm is {tempo_value}", f"BPM {tempo_value}", f"BPM: {tempo_value}"])
                        speed_tag, coarse_speed_tag = tempo_to_label(tempo_value), tempo_to_coarse_label(tempo_value) 
                        candidate_speed_description= [speed_tag, coarse_speed_tag, bpm_value_description]
                    elif self.tempo_text is not None:
                        candidate_speed_description = [self.tempo_text]
                    if len(candidate_speed_description) > 0:
                        rnd.shuffle(candidate_speed_description)
                        speed_description = ", ".join(candidate_speed_description[:rnd.randint(0, len(candidate_speed_description))])
                        if new_freeform_text is not None and speed_description != '':    # insert into a random position of freeform_text_short
                            fft_list = new_freeform_text.split(", ")
                            fft_list.insert(rnd.choice(range(len(fft_list))), speed_description)
                            new_freeform_text = ", ".join(fft_list)

        if self.key_text is not None:
            if rnd.random() < mir_dropout_rate:
                new_style_text[11] = ['empty key'] 
            else:
                new_style_text[11] = [self.key_text] if self.key_text != 'X' else ['empty key']

        if self.mode_text is not None:
            if rnd.random() < mir_dropout_rate:
                new_style_text[12] = ['empty mode']
            else:
                new_style_text[12] = [self.mode_text] if self.mode_text != 'X' else ['empty mode']

        return new_style_text, new_freeform_text

    @classmethod
    def merge(cls, mir_info, mir_filters=None):
        if mir_info is None:
            return cls
        
        # (TODO: qinxin): merge song slices (currently this function will not be called)
        return cls
        
        def majority(lst):
            return Counter(lst).most_common(1)[0][0]
        
        def get_unique_values(lst):
            all_lst = []
            for _lst in lst:
                all_lst.extend(_lst)
            return list(set(all_lst))
        
        def merge_dict(dics):
            new_dict = {}
            for d in dics:
                new_dict.update(d)
            return new_dict

        def merge_value(lst):
            valid_values = [x for x in lst if x is not None]
            return None if len(valid_values) <= 0 else round(np.mean(valid_values))

        # merge tempo
        tempo = merge_value([x.tempo for x in [cls, mir_info]])
        if tempo is not None:
            tempo_text = tempo_to_label(tempo)
            tempo_id = TEMPO_LABEL_ID_MAP[tempo_text]
        
        key = majority([MIRInfo.key_id_conversion(mir_info.key, mir_filters) for mir_info in mir_infos])
        key_root = (key - 1) // (len(MODES) - 1) + 1 if key != KEY_ID_MAP["X"] else ROOT_ID_MAP["X"] # exclude "X"
        key_mode = (key - 1) % (len(MODES) - 1) + 1 if key != KEY_ID_MAP["X"] else MODE_ID_MAP["X"]
        merged = MIRInfo(key=key, key_root=key_root, key_mode=key_mode,
                    tempo=merge_value([mir_info.tempo for mir_info in mir_infos]),
                    tempo_label=majority([mir_info.tempo_label for mir_info in mir_infos]),
                    time_signature=majority([mir_info.time_signature for mir_info in mir_infos]),
                    main_inst_text=majority([mir_info.main_inst_text for mir_info in mir_infos]),
                    insts_text=get_unique_values([mir_info.insts_text for mir_info in mir_infos]), 
                    main_inst_id=majority([m.main_inst_id for m in mir_infos]),
                    insts_id=get_unique_values([m.insts_id for m in mir_infos]),
                )


@dataclass
class SongSlice:
    phrases: List[Phrase]
    notes: Optional[List[Note]] = None
    mir_info: Optional[MIRInfo] = None

    # Do not set is_reformatted to True unless you are sure
    # the phrases are already reformatted. If not, use reformat_and_dropout
    # to reformat it.
    is_reformatted: bool = False
    # attributes to override during inference
    duration_at_inference: Optional[float] = None
    section_durations_at_inference: Optional[List[float]] = None

    @property
    def start(self) -> float:
        return self.phrases[0].start

    @property
    def end(self) -> float:
        return self.phrases[-1].end

    @property
    def duration(self) -> float:
        if self.duration_at_inference is not None:
            return self.duration_at_inference
        return self.end - self.start
    
    @property
    def has_utterance(self) -> bool:
        if not self.phrases:
            return False
        return any(phrase.has_utterance for phrase in self.phrases)

    def select_and_add_notes(self, notes: List[Note]) -> "SongSlice":
        # phoneme_sequence = []
        # for phrase in self.phrases:
        #     if phrase.phoneme_timestamps:
        #         phoneme_sequence += phrase.phoneme_timestamps
        note_sequence = [note.to_leadsheet_note() for note in notes]
        sliced_notes, _ = fetch_notes_from_phones_v3([{"start": self.start, "end": self.end, "phone": ""}], note_sequence)
        sliced_notes = [Note.from_dict(note) for note in sliced_notes]
        new_self = copy.deepcopy(self)
        new_self.notes = sliced_notes
        return new_self

    def select_and_add_mir_info(self, key_list_or_key, beat_list_or_tempo, inst_dict, mir_filters=None) -> "SongSlice":
        slice_beat_list, slice_key_list = None, None
        key, tempo = None, None
        
        if len(self.phrases) > 0:
            st, et = self.phrases[0].time_span[0], self.phrases[-1].time_span[-1]
            if key_list_or_key is not None and isinstance(key_list_or_key, list):
                slice_key_list = [[k_time, k] for (k_time, k) in key_list_or_key if st <= k_time <= et]
            if beat_list_or_tempo is not None and isinstance(beat_list_or_tempo, list):
                slice_beat_list = [[b_time, beat_idx] for (b_time, beat_idx) in beat_list_or_tempo if st <= b_time <= et]
            
        if isinstance(key_list_or_key, str):
            key = key_list_or_key
        if isinstance(beat_list_or_tempo, int) or isinstance(beat_list_or_tempo, float):
            tempo = beat_list_or_tempo

        mir_info = MIRInfo.new(
            key_list=slice_key_list, 
            beat_list=slice_beat_list, 
            inst_dict=inst_dict,
            key=key, tempo=tempo,
            mir_filters=mir_filters)
        new_self = copy.deepcopy(self)
        new_self.mir_info = mir_info
        return new_self

    def is_time_span_valid(self, time_span: Tuple[float, float]) -> bool:
        start, end, duration = self.start, self.end, self.duration
        min_duration, max_duration = time_span
        return ((min_duration <= duration <= max_duration) and 
                start >= 0 and end >= 0 and start < end)
    
    def is_lyrics_confidence_phrase_valid(self, lyrics_confidence: float) -> bool:
        return all((
            (phrase.lyrics_confidence is not None and phrase.lyrics_confidence >= lyrics_confidence)
             or not phrase.has_utterance)
            for phrase in self.phrases
        )

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
        phrases = move_out_singer_tags(phrases)  # Reformat the phrases to have single-line singer tags
        phrases = remove_section_tag_counts(phrases)  # Remove the count number
        return drop_out_section_tags(phrases, section_tag_dropout_rate)

    @staticmethod
    def get_section_durations(phrases: List[Phrase]) -> List[float]:
        """This should be always called AFTER `reformat_and_dropout`"""
        section_tag_phrases = [phrase for phrase in phrases if phrase.section_tag]
        section_durations = [phrase.duration for phrase in section_tag_phrases]
        if all(dur is None for dur in section_durations):
            return []
        return section_durations

    def reformat_and_dropout_(
        self,
        line_break_dropout_rate: float,
        section_tag_dropout_rate: float,
    ) -> "SongSlice":
        # TODO: The static method should be removed
        if self.is_reformatted:
            return self
        self.duration_at_inference = self.duration
        song_slice = copy.deepcopy(self)
        song_slice.phrases = SongSlice.reformat_and_dropout(
            line_break_dropout_rate,
            section_tag_dropout_rate,
            self.phrases
        )
        song_slice.is_reformatted = True
        return song_slice

    def get_section_durations_(self) -> List[float]:
        # TODO: The static method should be removed
        if self.section_durations_at_inference is not None:
            return self.section_durations_at_inference
        if self.is_reformatted:
            return SongSlice.get_section_durations(self.phrases)
        raise ZhMetaTransformError("SongSlice is not reformatted, unable to calculate section durations")

    def __add__(self, other):
        notes = None if (self.notes is None or other.notes is None) else self.notes + other.notes 
        duration_at_inference = None if (self.duration_at_inference is None or other.duration_at_inference is None) else self.duration_at_inference + other.duration_at_inference 
        section_durations_at_inference = None if (self.section_durations_at_inference is None or other.section_durations_at_inference is None) else self.section_durations_at_inference + other.section_durations_at_inference 
        if self.is_reformatted and other.is_reformatted:
            is_reformatted = True
        elif not self.is_reformatted and not other.is_reformatted:
            is_reformatted = False
        else:
            raise ZhMetaTransformError("Unable to concat two SongSlices of different formats")
        return self.__class__(
            phrases=self.phrases + other.phrases,
            notes=notes,
            duration_at_inference=duration_at_inference,
            section_durations_at_inference=section_durations_at_inference,
            is_reformatted=is_reformatted,
        )
    
    def _group_notes_based_on_phrases(self) -> List[List[Note]]:
        def get_overlap(time_span_a: Tuple[int, int], time_span_b: Tuple[float, float]) -> float:
            left_overlap = max(time_span_a[0], time_span_b[0])
            right_overlap = min(time_span_a[1], time_span_b[1])
            if right_overlap <= left_overlap:
                return 0.0
            return right_overlap - left_overlap
        if self.notes is None:
            raise ZhMetaTransformError("Notes are not available")
        if self.phrases is None:
            raise ZhMetaTransformError("Phrases are not available")
        note_groups = [[] for _ in range(len(self.phrases))]
        phrase_ts = [phrase.time_span if phrase.has_utterance else () for phrase in self.phrases]
        for note in self.notes:
            overlaps = []
            for phrase_time_span in phrase_ts:
                if not phrase_time_span:
                    overlaps.append(0)
                else:
                    overlaps.append(get_overlap(phrase_time_span, note.time_span))
            if sum(overlaps) == 0:
                continue  # skip the note
            note_groups[overlaps.index(max(overlaps))].append(note)
        return note_groups

    def to_dicts(
        self,
        mode: str = "phoneme",
        drop_target: Optional[Union[str, list[str]]] = None,
        offset_start_time: bool = True
    ) -> List[Dict]:
        """Convert the SongSlice to a list of dicts that can be fed into the tokenizer.
        Arg:
            mode:
            - "phoneme": Encode special tags and phoneme symbols
            - "phoneme_note_concat": Encode special tags, phoneme symbols with time stamps, \
                and notes with time stamps. Notes are appeneded to the end of the phoneme sequence.
            drop_target:
            - "slice_duration": "slice_duration" only
            - "section_duration": "section_duration" only
            - "section_instruments": "section_instruments" only
            - "duration": both "slice_duration" and "section_duration"
            - "section": "section_duration" and section tag symbols
            - "all": "slice_duration", "section_duration", "line_break", and section tag symbols
            - "lyrics_all": all lyrics tokens and components listed above
            use_phoneme_time: Whether to encode phoneme time
            offset_start_time: Whether to set the first start time to zero
        """
        def get_start_time_offset() -> float:
            if not self.phrases or not offset_start_time:
                return 0
            phrase_start = 0 if self.start is None else self.start
            if self.notes:
                note_start = self.notes[0].time_span[0]
                return min(phrase_start, note_start)
            return phrase_start

        if mode not in ["phoneme", "phoneme_note_concat"]:
            raise ZhMetaTransformError(f"Unsupported mode for SongSlice.to_dicts {mode}")
        if mode == "phoneme_note_concat":
            if not self.notes:
                raise ZhMetaTransformError(f"The SongSlice does not have notes for mode {mode}")

        drop_target_map = {
            "slice_duration": ["slice_duration"],
            "section_duration": ["section_duration"],
            "section_instruments": ["section_instruments"],
            "duration": ["slice_duration", "section_duration"],
            "section": ["section_duration", "section_tag", "section_instruments"],
            "line_break": ["line_break"],
            "all": ["slice_duration", "section_duration", "section_tag", "singer_tag", "line_break"],
            "lyrics_all": ["slice_duration", "section_duration", "section_tag", "singer_tag", "line_break", "lyrics_tokens"],
        }
        drop_items = drop_target_map.get(drop_target, []) if isinstance(drop_target, str) else drop_target

        if not self.is_reformatted:
            raise ZhMetaTransformError("SongSlice.to_dicts only works on a reformatted SongSlice")
        # if not self.phrases:
        #     raise ZhMetaTransformError("Reformatted SongSlice has empty phrases")
        offset = get_start_time_offset()
        
        if mode == "phoneme_note_concat":
            note_groups = self._group_notes_based_on_phrases()
            dicts = []
            for idx, (phrase, notes) in enumerate(zip(self.phrases, note_groups)):
                sub_song_slice = copy.deepcopy(self)
                sub_song_slice.phrases = [phrase]
                # Skip the checking
                # if phrase.has_utterance and not notes:
                #     raise ZhMetaTransformError(f"Phrase has no notes")
                sub_song_slice.notes = notes
                if idx != 0:
                    _drop_items = drop_items + ["slice_duration"]
                else:
                    _drop_items = drop_items
                dicts.extend(sub_song_slice._to_dicts_concat(
                    mode=mode,
                    drop_items=_drop_items,
                    offset=offset,
                ))
            return dicts
        return self._to_dicts_concat(
            mode=mode,
            drop_items=drop_items,
            offset=offset,
        )

    def _to_dicts_concat(
        self,
        mode: str = "phoneme",
        drop_items: Optional[List[str]] = None,
        offset: float = 0.0,
    ) -> List[Dict]:
        if drop_items is None:
            drop_items = []
        seq = []
        if "slice_duration" not in drop_items:
            if self.duration <= 0:
                raise ZhMetaTransformError("Non-positive slice_duration")
            seq.append({"slice_duration": self.duration})
        if "line_break" in drop_items:
            #print('before linebreak: ', self.phrases)
            phrases = drop_out_line_breaks(self.phrases, rate=1.0)
        else:
            phrases = self.phrases
            #print('after linebreak: ', self.phrases)
        for phrase in phrases:
            if "section_tag" not in drop_items and phrase.section_tag:
                seq.append({"symbol": phrase.section_tag})
                if "section_duration" not in drop_items:
                    section_duration = None
                    if phrase.time_span:
                        section_duration = phrase.time_span[1] - phrase.time_span[0]
                    if phrase.duration_val:
                        section_duration = phrase.duration_val
                    if section_duration is not None:
                        if section_duration <= 0:
                            raise ZhMetaTransformError("Non-positive section_duration")
                        seq.append({"section_duration": section_duration})
                if "section_instruments" not in drop_items and phrase.instruments:
                    section_instruments = phrase.instruments
                    for section_instrument in section_instruments:
                        seq.append({"symbol": section_instrument})
            elif "singer_tag" not in drop_items and phrase.singer_tag:
                seq.append({"symbol": phrase.singer_tag})
            elif phrase.phonemes and "lyrics_tokens" not in drop_items:
                if mode == "phoneme_note_concat":
                    if not phrase.time_span:
                        raise ZhMetaTransformError("Phrase with phonemes does not have time_span")
                    seq.append({
                        "phonemes": phrase.phonemes,
                        "time_span": (phrase.time_span[0] - offset, phrase.time_span[1] - offset),
                    })
                else:  # mode == "phoneme"
                    seq.append({"phonemes": phrase.phonemes})
        if mode == "phoneme_note_concat" and "lyrics_tokens" not in drop_items and self.notes:
            seq.extend([{
                "pitch": note.pitch,
                "time_span": (note.time_span[0] - offset, note.time_span[1] - offset),
            } for note in self.notes])
        return seq


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

    2024/06/25 update:
    The function also calculates section-wise time spans, assign them to section-tag phrases,
    which makes it easier to calculate section durations after reformatting.
    The phrases may have incomplete section or section-wise time_span coverage.
    """
    section_duration_dict = {}
    for prev_phrase, phrase in zip([None] + phrases, phrases):
        if not phrase.section_tag:
            continue
        if prev_phrase is None or phrase.section_tag != prev_phrase.section_tag:
            # Insert a new record whenever a new section is found
            section_duration_dict[phrase.section_tag] = phrase.time_span
        else:
            # For a phrase in the same section, extend the end time of the time_span
            if section_duration_dict[phrase.section_tag]:
                section_duration_dict[phrase.section_tag] = (section_duration_dict[phrase.section_tag][0], phrase.end)

    out_phrases = []
    last_section_tag_phrase_idx = -1
    for prev_phrase, phrase in zip([None] + phrases, phrases):
        if phrase.section_tag and (prev_phrase is None or phrase.section_tag != prev_phrase.section_tag):
            # Insert section-tag phrase with time_span calculated by the previous step
            out_phrases.append(Phrase(section_tag=phrase.section_tag, time_span=section_duration_dict.get(phrase.section_tag), instruments=phrase.instruments))
            last_section_tag_phrase_idx = len(out_phrases) - 1
        if phrase.has_utterance:
            # Remove section tag
            out_phrases.append(phrase._replace(section_tag=None))
            # Phrase is not covered by deepchorus (it should not happen most of the time), extend the section end time
            if phrase.section_tag is None and last_section_tag_phrase_idx >= 0:
                last_section_tag_phrase = out_phrases[last_section_tag_phrase_idx]
                section_time_span = last_section_tag_phrase.time_span
                if section_time_span is not None:
                    out_phrases[last_section_tag_phrase_idx] = last_section_tag_phrase._replace(
                        time_span=(section_time_span[0], phrase.end)
                    )
    return out_phrases


def remove_section_tag_counts(phrases: List[Phrase]) -> List[Phrase]:
    """Remove the count number in the section_tags. Call this function AFTER move_out_section_tags."""
    return [
        phrase._replace(
            section_tag = None if phrase.section_tag is None else _remove_count_from_section_tag(phrase.section_tag)
        ) for phrase in phrases
    ]

def drop_out_mir_info(type: str, orig_id: int,
                      rate: float, seed: Optional[int] = None) -> int:
    """replace MIR info (tempo/key/insturments) with empty index with the given dropout rate."""
    if not (0 <= rate <= 1):
        raise ValueError(f"Invalid dropout rate: {rate}")
    
    rand_gen = Random(seed)
    if rand_gen.random() < rate:
        if type == 'tempo':
            return TEMPO_LABEL_ID_MAP[""]
        elif type == 'key':
            return KEY_ID_MAP["X"]  # same ID for ROOT_ID_MAP & MODE_ID_MAP & TIME_SIGNATURE_MAP
        elif type == 'inst':
            return []
    return orig_id

def move_out_singer_tags(phrases: List[Phrase]) -> List[Phrase]:
    """This function should be called AFTER move_out_section_tags"""
    out_phrases = []
    for prev_phrase, phrase in zip([None] + phrases, phrases):
        if phrase.singer_tag and (prev_phrase is None or phrase.singer_tag != prev_phrase.singer_tag):
            out_phrases.append(Phrase(singer_tag=phrase.singer_tag))
        if phrase.has_utterance or phrase.section_tag:
            out_phrases.append(phrase._replace(singer_tag=None))
    return out_phrases


def drop_out_section_tags(phrases: List[Phrase], rate: float, seed: Optional[int] = None) -> List[Phrase]:
    """Remove phrases with section tags. The phrase list SHOULD be reformatted by move_out_section_tags.
    - The function either drops out or keeps all the section tags. Partial dropout could contaminate the
      training data by placing multiple sections under one section tag.
    - The function does not perform dropout if the given phrases do not have utterance, because empty
      phrases can't be tokenized.
    """
    if not (0 <= rate <= 1):
        raise ValueError(f"Invalid dropout rate: {rate}")
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
    instrument: Optional[Dict] = None
    vocal2midi: Optional[Dict] = None

    # added for instrumental
    has_vocal: bool = True
    freeform_text: Optional[str] = None  
    vad_voice_proportion: Optional[float] = None  

    # add language for offset phoneme
    language: Optional[str] = None

    @classmethod
    def parse(cls, meta, sinking_threshold: float, artist_tagging_confidence: float = 0.5):
        # Parse and check the data here
        raise NotImplementedError()

    def transform(
        self,
        lyrics_confidence: Optional[float],
        lyrics_confidence_phrase: Optional[float],
        deepchorus_confidence: Optional[float],
        segment_method: str,
        max_seg_per_track: int,
        duration_range: Tuple[int, int],
        slice_mode: str,
        mir_filters: Optional[list] = None,
        multi_tasks: Optional[list] = None,
    ) -> Dict[str, Any]:
        self._validate(lyrics_confidence, deepchorus_confidence)
        _self = self._convert()
        logger, song_slices = _self._to_song_slices(
            segment_method=segment_method,
            max_seg_per_track=max_seg_per_track,
            duration_range=duration_range,
            lyrics_confidence_phrase=lyrics_confidence_phrase,
            slice_mode=slice_mode,
            multi_tasks=multi_tasks,
            mir_filters=mir_filters,
        )

        return {
            "logger": logger,  # ZhMetaLogger
            "song_slices": song_slices,  # List[SongSlices]
            "style_text": _self.style_text,  # List[str]
            "freeform_text": _self.freeform_text,    # str, processed by randomly pick a key from the freeform_text_raw
            "artist_id": _self.artist_id,  # int
            "lyrics_confidence": _self.lyrics_confidence,  # Optional[float]
            "structure_tags": _self.structure_tags,  # DeepChorus
            "utterances": _self.utterances,
        }

    def _validate(
        self,
        lyrics_confidence: Optional[float],
        deepchorus_confidence: Optional[float],
    ):
        """Raise ZhMetaTransformError if the data is invalid"""
        validate_lyrics_confidence_optional(lyrics_confidence, self.lyrics_confidence)
        validate_deepchorus_optional(self.structure_tags, deepchorus_confidence)
        validate_mir_tempo_optional(self.tempo)
        validate_mir_key_optional(self.key)

    def _convert(self):
        """Process the data and return a new data. No in-place operation."""
        _self = copy.deepcopy(self)
        convert_artist_id_from_voice_tag(_self)
        return _self

    def _to_song_slices(
        self, 
        segment_method: str, 
        max_seg_per_track: int,
        duration_range: Tuple[int, int],
        lyrics_confidence_phrase: Optional[float],
        slice_mode: str,
        mir_filters: Optional[List[str]] = None,
        multi_tasks: Optional[List[str]] = None,
    ) -> Tuple[ZhMetaLogger, List[SongSlice]]:
        min_duration, max_duration = duration_range
        # The presence of structure_tags is decided by the parser. Therefore, we don't need to
        # check structure_tags here.
        if self.structure_tags is None:
            song_slices = transform_utts_to_song_slices_heuristic(
                self.utterances, 
                min_duration,
                max_duration,
                language=self.language,
            )
        else:
            song_slices = transform_utts_to_song_slices_structure(
                self.utterances,
                min_duration,
                max_duration,
                structure_tags=self.structure_tags.tags,
                language=self.language,
                slice_mode=slice_mode,
            )
        song_slices, logger = filter_song_slices(song_slices, duration_range, lyrics_confidence_phrase)
        song_slices = sample_song_slices(song_slices, segment_method, max_seg_per_track)

        # Add mir info (inst, key, tempo) to song_slices
        if multi_tasks is not None and "global_mir_control" in multi_tasks:
            song_slices = filter_and_process_song_slices_with_mir_info(
                song_slices, self.key, self.tempo, self.instrument, mir_filters
            )

        # Add notes to song_slices
        if self.vocal2midi:
            notes = Note.from_vocal2midi(self.vocal2midi)
            song_slices = [
                song_slice.select_and_add_notes(notes)
                for song_slice in song_slices
            ]
        return logger, song_slices

    def _to_tempo_label_id(self) -> int:
        return TEMPO_LABEL_ID_MAP[tempo_to_label(self.tempo)]

    def _to_key_id(self) -> int:
        k = "N" if self.key is None else self.key
        return KEY_ID_MAP[k]


# ------------------------------------------
#                 PARSERS
# ------------------------------------------

def sample_pct(keywords, dropout=0.5, min_examples=1):
    random.shuffle(keywords)
    if len(keywords) * (1 - dropout) <= min_examples:
        return keywords
    return [a for a in keywords if random.random() >= dropout]

# ---------- utterance -------------
def parse_utterance_labeled_lyrics(meta: Dict) -> List:
    discard = meta.get("lyrics_labeled", {"discard": "yes"}).get("discard", "no")
    if discard == "yes":
        raise ZhMetaParseError("No lyrics_labeled")
    lyrics = _get_value(meta, "lyrics_force_align", "No lyrics")
    utterances = []
    for item in lyrics:
        utt = {
            "attribute": {
                "confidence": 1.0,
                "event": "singing",
            },
            "start_time": item.get("start_time"),
            "end_time": item.get("end_time"),
            "text": item.get("text"),
            "phoneme_v86": item.get("phoneme_v86"),
            "phoneme_v86_tn": item.get("phoneme_v86_tn"),
        }
        utterances.append(utt)
    if len(utterances) == 0:
        raise ZhMetaParseError("No utterances")
    return utterances


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


def parse_utterance_lyrics_force_align(meta: Dict) -> List:
    """meta.lyrics_force_align"""
    utterances = _get_value(meta, "lyrics_force_align")
    if utterances is None or len(utterances) == 0:
        raise ZhMetaParseError("No utterances")
    return adaption_utterance_lyrics_force_align(utterances)


def adaption_utterance_lyrics_force_align(utterances: List) -> List:
    # force align service will return special timestamp "-2" for all punctuation
    # the function here is to remove all special timestamp

    cleaned_utterances = []
    for utterance in utterances:
        if utterance["start_time"] < 0:
            print(f"negative start_time: {utterance}")
            continue
        if utterance["end_time"] > 0:
            cleaned_utterances.append(utterance)
        else:
            # find the last word which is not punctuation
            words_num = len(utterance["words"])
            idx = words_num - 1
            while(idx >= 0):
                if utterance["words"][idx]["end_time"] > 0:
                    break
                idx -= 1
            
            new_utterance = copy.deepcopy(utterance)
            last_word_no_punc = utterance["words"][idx]
            new_utterance["end_time"] = last_word_no_punc["end_time"]
            # print(f'{idx=} {new_utterance["end_time"]=} {utterance["end_time"]=}')
            cleaned_utterances.append(new_utterance)
            # TODO is it safe to keep special timestamp "-2" still in words?
            # TODO should we replace all punctuation to a given symbol?
    return cleaned_utterances


def parse_utterance_lyrics_sa_asr_or_force_align(meta: Dict) -> List:
    if 'lyrics_force_align' in meta:
        return parse_utterance_lyrics_force_align(meta)
    elif 'lyrics' in meta:
        return parse_utterance_lyrics(meta)
    else:
        raise ZhMetaParseError("No utterances")


# ---------- structure_tags -------------

def get_deepchorus_score(deepchorus_tags):
    boundary = [b['start_prob'] for b in deepchorus_tags['segments']]
    function = [f['funct_prob'] for f in deepchorus_tags['segments']]
    return 0.7 * (sum(boundary)/len(boundary)) + 0.3 * (sum(function)/len(function))

_SECTION_TAG_SEP = "#"


def _get_deepchorus_score(segments):
    if len(segments) == 0:
        print("empty deepchorus segments")
        return 0
    boundary = [b['start_prob'] for b in segments]
    function = [f['funct_prob'] for f in segments]
    return 0.7 * (sum(boundary)/len(boundary)) + 0.3 * (sum(function)/len(function))


def _add_count_to_section_tag(section_tag: str, count: int) -> str:
    if _SECTION_TAG_SEP in section_tag:
        return section_tag
    return f"{section_tag}{_SECTION_TAG_SEP}{str(count)}"


def _remove_count_from_section_tag(section_tag: str) -> str:
    return section_tag.split(_SECTION_TAG_SEP)[0]


def _format_deepchorus_structure_tags(segments: List[Dict]) -> DeepChorus:
    """Convert the deepchorus field in metadata into the format of [{'tag': tag, 'start_time': sec, 'end_time': sec}].
    A number will be appended to the tag to differentiate adjacent tags with the same name.
    """
    structure_tags = []
    for count, segment in enumerate(segments):
        structure_tag = {
            "tag": _add_count_to_section_tag(segment["label"], count),
            "start_time": segment["interval"][0],
            "end_time": segment["interval"][1],
        }
        structure_tags.append(structure_tag)
    return DeepChorus(
        tags=structure_tags,
        confidence=_get_deepchorus_score(segments)
    )

def add_section_instruments_to_structure_tags(structure_tags: DeepChorus, section_instruments: Optional[list] = None) -> DeepChorus:
    if section_instruments is None:
        section_instruments = []
    for item in structure_tags.tags:
        item['instruments'] = next((sec['instruments'] for sec in section_instruments 
                            if sec['start'] <= item['start_time'] < sec['end']), [])
    return structure_tags


def parse_structure_tags(meta: Dict) -> DeepChorus:
    """meta.deepchorus.segments"""
    deepchorus_tags = _get_value(meta, "deepchorus", "No structure tags")
    segments = _get_value(deepchorus_tags, "segments", "No segments")
    return _format_deepchorus_structure_tags(segments)


def parse_structure_tags_optional(meta: Dict) -> Optional[DeepChorus]:
    """meta.deepchorus.segments"""
    try:
        return parse_structure_tags(meta)
    except ZhMetaParseError:
        return None


def convert_human_label_into_segments(human_label_segments):
    segments = []
    for human_segment in human_label_segments:
        segment = {"interval":[human_segment["start_time"], human_segment["end_time"]], 
                   "label": human_segment["function"], 
                   "start_prob": 1.0, "funct_prob": 1.0}
        segments.append(segment)
    return segments


def parse_structure_tags_custom_concat(meta: Dict) -> DeepChorus:
    """meta.deepchorus.raw_segments"""
    if "music_structure_labeled" in meta:        
        human_label_segments = _get_value(meta["music_structure_labeled"], "struct_result", "No struct_result in music_structure_labeled")
        segments = convert_human_label_into_segments(human_label_segments)
    elif "musicfm_structure" in meta:
        segments = _get_value(meta, "musicfm_structure", "No musicfm_structure")
        segments = merge_raw_song_structure_by_start_prob(segments, 'vocal')
    else:
        deepchorus_tags = _get_value(meta, "deepchorus", "No structure tags")
        segments = _get_value(deepchorus_tags, "raw_segments", "No raw_segments")
        segments = merge_raw_song_structure_by_start_prob(segments, 'vocal')
    return _format_deepchorus_structure_tags(segments)


def parse_structure_tags_custom_concat_inst(meta: Dict) -> DeepChorus:
    """meta.deepchorus.raw_segments"""
    deepchorus_tags = _get_value(meta, "deepchorus", "No structure tags")
    segments = _get_value(deepchorus_tags, "raw_segments", "No raw_segments")
    segments = merge_raw_song_structure_by_start_prob(segments, 'inst')
    return _format_deepchorus_structure_tags(segments)


# ---------- lyrics_confidence -------------

def parse_lyrics_confidence_sa_asr(meta: Dict) -> float:
    """meta.lyrics.confidence"""
    get_value = partial(_get_value, msg="No confidence")
    return get_value(get_value(meta, "lyrics"), "confidence")

def parse_lyrics_confidence_labeled_lyrics(meta: Dict) -> float:
    discard = meta.get("lyrics_labeled", {"discard": "yes"}).get("discard", "no")
    if discard == "yes":
        raise ZhMetaParseError("No lyrics_labeled")
    lyrics = _get_value(meta, "lyrics_force_align", "No lyrics")
    return 1.0

def parse_lyrics_confidence_force_align_legacy(
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


def parse_lyrics_confidence_force_align(meta: Dict) -> float:
    lyrics_force_align = _get_value(meta, "lyrics_force_align", "No lyrics_force_align")
    if lyrics_force_align is None or len(lyrics_force_align) == 0:
        raise ZhMetaParseError("No lyrics_force_align")
    global_confidence = 0.0
    num = 0
    for item in lyrics_force_align:
        if 'confidence' in item: 
            global_confidence += item['confidence']
            num += 1
    if num == 0:
        return 0.0
    else:
        return global_confidence / num


def parse_lyrics_confidence_optional(
    meta: Dict,
    utterances: Optional[List] = None
) -> Optional[float]:
    for parse_fn in [
        parse_lyrics_confidence_force_align,
        parse_lyrics_confidence_sa_asr, 
        partial(parse_lyrics_confidence_force_align_legacy, utterances=utterances),
    ]:
        try:
            return parse_fn(meta)
        except ZhMetaParseError:
            continue
    return None


def parse_lyrics_confidence_sa_asr_or_force_align(meta: Dict) -> float:
    if 'lyrics_force_align' in meta:
        return parse_lyrics_confidence_force_align(meta)
    elif 'lyrics' in meta:
        return parse_lyrics_confidence_sa_asr(meta)
    else:
        raise ZhMetaParseError("No lyrics confidence")
        

# ---------- style_text -------------

def _parse_sa_music_tagging(music_tagging: Optional[Dict], sinking_threshold: float) -> Tuple[List[str], Dict[str, str], bool]:
    def parse_result(result: Union[List, str], category: str) -> str:
        if isinstance(result, str):
            return result
        if len(result) == 0:
            return ""
        if category != "Language":
            return result[0]
        # For Language category
        if "Cantonese" in result:
            return "Cantonese"
        if "Chinese Dialects" in result:
            return "Chinese Dialects"
        if not set(result).issubset(["Chinese", "English"]):
            return "Other"
        # So far, it can only be Chinese or English or Chinese-English mixture
        if "Chinese" in result:
            return "Chinese"
        return "English"  # English only


    def map_tag(tag: str, category: str) -> str:
        """Replace certain tags in the dataset"""
        if category == "Mood":
            return SA_TAGS_MOOD_SPECIAL_MAP.get(tag, tag)
        return SA_TAGS_SPECIAL_MAP.get(tag, tag)

    # The order should match `mir_data_util`
    order = ["Genre20", "Mood", "Theme", "MusicLowQuality", "Language"]

    if music_tagging is None:
        return [""] * len(order), {}, False
    sinking_prob = music_tagging["MusicLowQuality"]["Sinking"]
    is_sinking = sinking_prob >= sinking_threshold
    quality = "Sinking" if is_sinking else "non-Sinking"
    tags = [quality if item == "MusicLowQuality" else map_tag(parse_result(music_tagging[item]["result"], item), item) for item in order]
    unfamiliar_tags = {cat_name: tag for tag, cat_vocab_tags, cat_name in zip(tags, SA_CAT_VOCAB, order) if tag not in cat_vocab_tags}
    return [(tag if tag in cat_vocab_tags else "") for tag, cat_vocab_tags in zip(tags, SA_CAT_VOCAB)], unfamiliar_tags, is_sinking


def parse_style_text_sa(meta: Dict, sinking_threshold: float) -> Tuple[List[str], Dict[str, str], bool]:
    return _parse_sa_music_tagging(_get_value(meta, "music_tagging", "No music_tagging"), sinking_threshold)


def parse_style_text_sa_optional(meta: Dict, sinking_threshold: float) -> Optional[Tuple[List[str], Dict[str, str], bool]]:
    try:
        return parse_style_text_sa(meta, sinking_threshold)
    except ZhMetaParseError:
        return None

# ---------- audio_tags -------------

def _parse_human_label(meta: Dict) -> Optional[Dict]:
    """
    Parse the human_label field in metadata. Possibly an old annotation format.
    """
    human_label = meta.get('human_label', None)
    if human_label is None:
        return None
    
    genre = human_label.get('label_genre') or []        # sometimes this could be None from meta ...
    if isinstance(genre, str):
        genre = genre.split(',')
    subgenre = human_label.get('label_subgenre') or []
    if isinstance(subgenre, str):
        subgenre = subgenre.split(',')
    genre.extend(subgenre)
    genre_extra = human_label.get('genre_extra') or []
    if isinstance(genre_extra, str):
        genre_extra = genre_extra.split(',')
    extra = human_label.get('extra') or []
    if isinstance(extra, str):
        extra = extra.split(',')
    mood = human_label.get('label_mood') or []
    if isinstance(mood, str):
        mood = mood.split(',')
    scene = human_label.get('label_theme') or []
    if isinstance(scene, str):
        scene = scene.split(',')
    instrument = human_label.get('label_instrument') or []
    if isinstance(instrument, str):
        instrument = instrument.split(',')

    audio_tags = {}
    audio_tags['genre'] = genre
    audio_tags['genre_extra'] = genre_extra
    audio_tags['extra'] = extra
    audio_tags['mood'] = mood
    audio_tags['scene'] = scene
    audio_tags['instrument'] = instrument

    return audio_tags


def _parse_audio_tags_or_audio_tag_or_music_tagging(meta: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    audio_tags = meta.get('audio_tags', meta.get('audio_tag', {}))  # NOTE: This is a hack, only read "audio_tags" after the dataset issue gets fixed
    if not audio_tags:
        audio_tags = _parse_human_label(meta)
    if not audio_tags:      # add this to read llm_tags (for sstk 1.08M) as a second choice.
        audio_tags = meta.get('llm_tags', {})
    music_tagging = meta.get('music_tagging', {})
    if audio_tags:
        if 'language' not in audio_tags:
            audio_tags['language'] = music_tagging['Language']['result'] if music_tagging else [""]
    elif music_tagging:
        music_tagging["sa_gender"] = meta.get("gender", {}).get("sa_gender", {"result": [""]})
    else:
        raise ZhMetaParseError('No music_tagging and audio_tags')
    return audio_tags, music_tagging

# ---------- audio_tags V3 -------------

def _mapping_genre_extra(tags):
    genre, genre_extra = tags[0], tags[1]
    # when genre_extra is empty, will replaced by genre
    if len(genre) > 0:
        if len(genre_extra) == 0 or \
            genre_extra == [''] or \
            genre_extra == ['empty genre']:
            genre_extra = genre
    tags[1] = genre_extra
    return tags

def _mapping_extra(tags):
    genre, extra, scene, lang = tags[0], tags[2], tags[4], tags[7]
    # when genre only include tuhai(old version), will convert to Grassroots
    if 'Tuhai_AUDIO_EXTRA' in extra and \
        'Grassroots' not in extra:
        extra.remove('Tuhai_AUDIO_EXTRA')
        extra.append('Grassroots')
    # when genre include DJ/MC, will add Grassroots/Tuhai
    if 'DJ' in genre or \
        'MC' in genre:
        extra.extend(['Grassroots', 'Tuhai_AUDIO_EXTRA'])
    # when lang is non-vocal and extra do not include Canned Music, will add non-canned music (only for sft)
    if lang == ['Instrumental/Non-vocal']:
        if 'Canned Music' not in extra:
            extra.append('non-canned music')
    # when extra include Fashionable, will add non-nostalgic
    if 'Fashionable' in extra:
        extra.append('non-nostalgic')
    # when genre include sone special genre, will add non-nostalgic
    if 'Indie Pop' in genre or \
        'Indie Folk' in genre or \
        'Indie Rock' in genre or \
        'Alternative/Indie' in genre or \
        'Electropop' in genre or \
        'Hip Hop' in genre or \
        'Contemporary R&B' in genre or \
        'Neo Soul' in genre or \
        'Neo Funk' in genre:
        extra.append('non-nostalgic')
    # when scene include sone special scene, will add non-nostalgic
    if 'Vlog/DailyLife' in scene or \
        'Beauty/Fashion' in scene or \
        'Transition' in scene or \
        'Nightclub' in scene or \
        'Marketplace' in scene or \
        'Sport' in scene or \
        'Game' in scene or \
        'Running' in scene or \
        'Danceable' in scene:
        extra.append('non-nostalgic')

    if len(extra) > 1 and 'empty extra' in extra:
        extra.remove('empty extra')
    if len(extra) == 0:
        extra = ['empty extra']
    tags[2] = extra
    return tags

def _mapping_mood(tags):
    mood = tags[3]
    if len(mood) == 0:
        mood = ['empty mood']
    tags[3] = mood
    return tags

def _mapping_scene(tags):
    scene = tags[4]
    if len(scene) == 0:
        scene = ['empty scene']
    tags[4] = scene
    return tags

def _mapping_gender(tags):
    gender, lang = tags[5], tags[7]
    # when gender include adult and female/male at same time, will delete adult
    if 'Adult' in gender:
        if 'Female' in gender or 'Male' in gender:
            gender.remove('Adult')
    # when lang is non-vocal and gender is empty, will convert nonvocal nongender
    if lang == ['Non-vocal'] or \
        lang == ['Instrumental/Non-vocal']:
        if gender == ['empty gender']:
            gender = ['nonvocal nongender']
    if len(gender) == 0:
        gender = ['empty gender']
    tags[5] = gender
    return tags

def _mapping_timbre(tags):
    timbre, lang = tags[6], tags[7]
    # when lang is non-vocal and timbre is empty, will convert to nonvocal nontimbre
    if lang == ['Non-vocal'] or \
        lang == ['Instrumental/Non-vocal']:
        if timbre == ['empty timbre']:
            timbre = ['nonvocal nontimbre']
    if len(timbre) == 0:
        timbre = ['empty timbre']
    tags[6] = timbre
    return tags

def _mapping_is_sinking(tags):
    genre, extra, is_sinking = tags[0], tags[2], tags[8]
    # when genre include DJ/MC, will convert to Sinking
    if 'DJ' in genre or \
        'MC' in genre:
        is_sinking = ['Sinking']
    # when extra include Grassroots, will convert to Sinking
    if 'Grassroots' in extra:
        is_sinking = ['Sinking']
    tags[8] = is_sinking
    return tags

def _mapping_instrument(tags):
    lang, instrument = tags[7], tags[9]
    # when lang is inst and instrument do not include Vocal, will add nonvocal
    if lang == ['Non-vocal'] or \
        lang == ['Instrumental/Non-vocal']:
        if 'Vocal' not in instrument:
            instrument.append('nonvocal')
    if len(instrument) > 1 and 'empty instrument' in instrument:
        instrument.remove('empty instrument')
    if len(instrument) == 0:
        instrument = ['empty instrument']
    tags[9] = instrument
    return tags

def process_tags(tags):
    tags = _mapping_genre_extra(tags)
    tags = _mapping_extra(tags)
    tags = _mapping_mood(tags)
    tags = _mapping_scene(tags)
    tags = _mapping_gender(tags)
    tags = _mapping_timbre(tags)
    tags = _mapping_is_sinking(tags)
    tags = _mapping_instrument(tags)
    return tags

def _parse_audio_tags_v3(music_tagging: Optional[Dict], audio_tags: Optional[Dict], sinking_threshold: float, mapping_tag: Optional[Dict]) -> Tuple[List[str], Dict[str, str], bool]:
    def map_tag(item: str, tag: str) -> str:
        """Replace certain tags in the dataset"""
        AUDIO_TAGS_SPECIAL_MAP = {
            'genre': AUDIO_TAGS_GENRE_SPECIAL_MAP_V2,
            'genre_extra': AUDIO_TAGS_GENRE_SPECIAL_MAP_V2,
            'extra': AUDIO_TAGS_EXTRA_SPECIAL_MAP,
            'mood': AUDIO_TAGS_MOOD_SPECIAL_MAP_V2,
            'scene': AUDIO_TAGS_SCENE_SPECIAL_MAP_V2,
            'vocal_gender': AUDIO_TAGS_GENDER_SPECIAL_MAP_V2,
            'vocal_timbre': AUDIO_TAGS_TIMBRE_SPECIAL_MAP,
            'language': AUDIO_TAGS_LANG_SPECIAL_MAP,
            'is_sinking': {},
            'instrument': AUDIO_TAGS_INST_SPECIAL_MAP,
            'tempo': AUDIO_TAGS_TEMPO_SPECIAL_MAP,
            'key': AUDIO_TAGS_KEY_SPECIAL_MAP,
            'mode': AUDIO_TAGS_MODE_SPECIAL_MAP,
        }
        if item in AUDIO_TAGS_SPECIAL_MAP:
            return AUDIO_TAGS_SPECIAL_MAP[item].get(tag, tag)
        else:
            return tag

    def uniq_tags(tags):
        _tags = []
        for item in tags:
            temp_tags = list(set(item))
            if len(temp_tags) > 1 and '' in temp_tags:
                temp_tags.remove('')
            _tags.append(temp_tags)
        return _tags

    is_sinking = False # 'non-Sinking'

    # The order should match `mir_data_util`
    # order = ['genre', 'genre_extra', 'extra', 'mood', 'scene', 'vocal_gender', 'vocal_timbre', 'language', 'is_sinking', 'instrument', 'tempo', 'mode', 'key']
    tags = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    unfamiliar_tags = {}
    if audio_tags is not None:
        order = [
            {
                'AUDIO_CAT_VOCAB_IDX': 0,
                'META_KEY': 'genre',
                'ORDER_KEY': 'genre',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 1,
                'META_KEY': 'genre_extra',
                'ORDER_KEY': 'genre_extra',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 2,
                'META_KEY': 'extra',
                'ORDER_KEY': 'extra',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 3,
                'META_KEY': 'mood',
                'ORDER_KEY': 'mood',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 4,
                'META_KEY': 'scene',
                'ORDER_KEY': 'scene',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 5,
                'META_KEY': 'vocal_gender',
                'ORDER_KEY': 'vocal_gender',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 6,
                'META_KEY': 'vocal_timbre',
                'ORDER_KEY': 'vocal_timbre',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 7,
                'META_KEY': 'language',
                'ORDER_KEY': 'language',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 8,
                'META_KEY': None,
                'ORDER_KEY': 'is_sinking',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 9,
                'META_KEY': 'instrument',
                'ORDER_KEY': 'instrument',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 10,
                'META_KEY': None,
                'ORDER_KEY': 'tempo',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 11,
                'META_KEY': None,
                'ORDER_KEY': 'mode',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 12,
                'META_KEY': None,
                'ORDER_KEY': 'key',
            },
        ]
        for order_map in order:
            idx = order_map['AUDIO_CAT_VOCAB_IDX']
            meta_key = order_map['META_KEY']
            order_key = order_map['ORDER_KEY']
            cat_vocab_tags = AUDIO_CAT_VOCAB_V3
            if meta_key not in audio_tags and music_tagging is not None:
                if meta_key == 'language':
                    audio_tags['language'] = music_tagging.get('Language', {}).get('result', '')
                if meta_key == 'vocal_gender':
                    audio_tags['vocal_gender'] = music_tagging.get('sa_gender', {}).get('result', '')

            if order_key == 'is_sinking':
                tags[idx].append(map_tag(order_key, 'non-Sinking'))
            else:
                result = audio_tags.get(meta_key, [])
                if isinstance(result, str):
                    result = result.split(',')
                if result is None or len(result) == 0:
                    tags[idx].append(map_tag(order_key, ""))
                else:
                    for tag in result:
                        tag = tag.strip()
                        _tag = map_tag(order_key, tag)
                        if _tag not in cat_vocab_tags:
                            tags[idx].append(map_tag(order_key, ""))
                            if order_key not in unfamiliar_tags:
                                unfamiliar_tags[order_key] = []
                            if _tag not in unfamiliar_tags[order_key]:
                                unfamiliar_tags[order_key].append(_tag)
                        else:
                            tags[idx].append(_tag)

        #print('before tags: ', tags)
        tags = process_tags(tags)
        #print('after tags: ', tags)
        is_sinking = False
        if 'Sinking' in tags[8]:
            is_sinking = True
        return uniq_tags(tags), unfamiliar_tags, is_sinking

    elif music_tagging is not None:
        order = [
            {
                'AUDIO_CAT_VOCAB_IDX': 0,
                'META_KEY': 'Genre20',
                'ORDER_KEY': 'genre',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 1,
                'META_KEY': None,
                'ORDER_KEY': 'genre_extra',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 2,
                'META_KEY': None,
                'ORDER_KEY': 'extra',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 3,
                'META_KEY': 'Mood',
                'ORDER_KEY': 'mood',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 4,
                'META_KEY': 'Theme',
                'ORDER_KEY': 'scene',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 5,
                'META_KEY': 'sa_gender',
                'ORDER_KEY': 'vocal_gender',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 6,
                'META_KEY': None,
                'ORDER_KEY': 'vocal_timbre',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 7,
                'META_KEY': 'Language',
                'ORDER_KEY': 'language',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 8,
                'META_KEY': None,
                'ORDER_KEY': 'is_sinking',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 9,
                'META_KEY': None,
                'ORDER_KEY': 'instrument',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 10,
                'META_KEY': None,
                'ORDER_KEY': 'tempo',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 11,
                'META_KEY': None,
                'ORDER_KEY': 'mode',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 12,
                'META_KEY': None,
                'ORDER_KEY': 'key',
            },
        ]
        for order_map in order:
            idx = order_map['AUDIO_CAT_VOCAB_IDX']
            meta_key = order_map['META_KEY']
            order_key = order_map['ORDER_KEY']
            cat_vocab_tags = AUDIO_CAT_VOCAB_V3

            if order_key == 'is_sinking':
                tags[idx].append(map_tag(order_key, 'non-Sinking'))
            else:
                result = music_tagging.get(meta_key, {}).get('result', [])
                if mapping_tag.get(order_key, ''):
                    result = mapping_tag[order_key]
                if isinstance(result, str):
                    result = result.split(',')
                if result is None or len(result) == 0:
                    tags[idx].append(map_tag(order_key, ""))
                else:
                    for tag in result:
                        _tag = map_tag(order_key, tag)
                        if _tag not in cat_vocab_tags:
                            tags[idx].append(map_tag(order_key, ""))
                            if order_key not in unfamiliar_tags:
                                unfamiliar_tags[order_key] = []
                            if _tag not in unfamiliar_tags[order_key]:
                                unfamiliar_tags[order_key].append(_tag)
                        else:
                            tags[idx].append(_tag)

        is_sinking = False
        if ('MusicLowQuality' in music_tagging) and \
            (music_tagging["MusicLowQuality"]["Sinking"] > sinking_threshold):
            tags[8].append('Sinking')
        else:
            tags[8].append('non-Sinking')

        #print('before tags: ', tags)
        tags = process_tags(tags)
        #print('after tags: ', tags)
        if 'Sinking' in tags[8]:
            is_sinking = True
        ## only drop sinking when use vocal pt data
        #if is_sinking and \
        #    'DJ' not in tags[0] and \
        #    'MC' not in tags[0]:
        #    raise ZhMetaTransformError(f"Filter out sinking of genre \"{tags[0]}\"")
        return uniq_tags(tags), unfamiliar_tags, is_sinking
    else:
        return [""] * 13, {}, False


def parse_style_text_audio_tags_v3_or_music_tagging(meta: Dict, sinking_threshold: float):
    audio_tags, music_tagging = _parse_audio_tags_or_music_tagging_with_lang_filt(meta)
    mapping_tag = mapping_tag_from_freeform(meta)
    return _parse_audio_tags_v3(music_tagging, audio_tags, sinking_threshold, mapping_tag)


def _parse_audio_tags_or_music_tagging_with_lang_filt(meta: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    audio_tags, music_tagging = _parse_audio_tags_or_audio_tag_or_music_tagging(meta)
    languages = []
    if audio_tags is not None:
        if 'satisfy_filter_standard' in audio_tags and audio_tags['satisfy_filter_standard'] != 'yes':
            raise ZhMetaParseError('audio_tags: satisfy_filter_standard is not yes')
        if audio_tags.get('language', ''):
            languages = audio_tags.get('language', '')
        else:
            languages = music_tagging.get('Language', {'result': ['Other']})['result']
        if languages is None:
            languages = ''
        if isinstance(languages, str):
            languages = languages.split(',')
        audio_tags['language'] = languages
        if not audio_tags.get('genre', ''):
            genres = music_tagging.get('Genre20', {'result': ['']})['result']
            if genres is None:
                genres = ''
            if isinstance(genres, str):
                genres = genres.split(',')
            audio_tags['genre'] = genres
        if not audio_tags.get('vocal_gender', ''):
            gender = music_tagging.get('sa_gender', {'result': ['']})['result']
            audio_tags['vocal_gender'] = gender
        if not audio_tags.get('mood', ''):
            mood = music_tagging.get('Mood', {'result': ['']})['result']
            if mood is None:
                mood = ''
            if isinstance(mood, str):
                mood = mood.split(',')
            audio_tags['mood'] = mood
        if not audio_tags.get('scene', ''):
            scene = music_tagging.get('Theme', {'result': ['']})['result']
            if scene is None:
                scne = ''
            if isinstance(scene, str):
                scene = scene.split(',')
            audio_tags['scene'] = scene
    elif music_tagging is not None:
        languages = music_tagging.get('Language', {'result': ['Other']})['result']
        if languages is None:
            languages = ''
        if isinstance(languages, str):
            languages = languages.split(',')
    if not set(languages).issubset(['Chinese', 'English', 'Cantonese', 'Japanese', 'Sichuanese']):
        raise ZhMetaParseError('audio_tags: Not Chinese or English or Cantonese or Japanese or Sichuanese')
    if 'Cantonese' in languages and 'Chinese' in languages:
        raise ZhMetaParseError('audio_tagging: Can not keep Chinese and Cantonese at same time')
    return audio_tags, music_tagging


def infer_freeform_text_from_style_text(
    style_text: List[Union[List[str], str]], freeform_dropout: float=0.0, is_infer: bool=False, shuffle=True, seed=None
) -> str:
    # print(f"[infer_freeform_text_from_style_text] {shuffle=} {seed=}")
    rnd = random.Random(seed)
    style_text = copy.deepcopy(style_text)
    if len(style_text) > 9: # Acoustic_Piano -> Acoustic Piano
        style_text[9] = [item.replace("_", " ") for item in style_text[9]]
        # merge mode & key for freeform text parsing
        if style_text[11] and style_text[12] and style_text[11] != ['empty key'] and style_text[12] != ['empty mode']:
            style_text[12] = [f'{style_text[11][0]} {style_text[12][0]}']
            style_text[11] = ['empty key']

    # Remove empty categories
    style_text = [[item] if isinstance(item, str) else item for item in style_text if item]
    freeform_text = []
    inst = None
    # there is 80% chance to select each dimension. (remove Adult, since this seems useless)
    if is_infer:
        # [genre, genre_extra, extra, mood, scene, gender, timbre, lang, sinking]
        # when use vocal, will add instruments ramdomly by main genre
        if not set(style_text[7]).issubset(['Non-vocal', 'Instrumental/Non-vocal']):
            if isinstance(style_text[0], str):
                genres = style_text[0].split(',')
            else:
                genres = style_text[0]
            for mg in genres:
                if mg in COMBO_GENRE_TO_INSTRUMENTS_V4:
                    inst = rnd.choice(COMBO_GENRE_TO_INSTRUMENTS_V4[mg])
        idx = 0
        for category_list in style_text:
            tmp_tag = []
            for tag in category_list:
                if idx == 3 and tag in ['Nostalgic/Memory']:
                    continue
                elif idx == 5 or \
                  idx == 6 or \
                  idx == 7 or \
                  idx == 8:
                    continue
                elif idx == 4:
                    # add scene to freeform_text exlucde some genres
                    for mg in ['Pop', 'Reggae', 'Electronic', 'Hip Hop/Rap']:
                        if mg not in style_text[0]:
                            tmp_tag.append(tag)
                else:
                    tmp_tag.append(tag)
            tmp_tag = sorted(list(set(tmp_tag)))
            freeform_text.append(', '.join(tmp_tag))
            idx += 1
    else:
        not_include_list = [
            'Adult',
            'non-Sinking',
            'Sinking',
            'Chinese',
            'English',
            'Cantonese',
            'Japanese',
            'Sichuanese',
            'Instrumental/Non-vocal',
            'Non-vocal',
            'Nostalgic/Memory', # Ambiguity
            'Chinese Folk', # Ambiguity
            'Other',
            'Others',
            'Other genre',
            'Other Genre',
            'Other_AUDIO_GENRE',
            'Other scene',
            'Other mood',
            'Other Mood',
            'Other_AUDIO_MOOD',
            'Other Lang',
            'Other Inst',
            'empty genre',
            'empty extra',
            'empty scene',
            'empty mood',
            'empty lang',
            'empty timbre',
            'empty instrument',
            'empty tempo',
            'empty mode',
            'empty key',
            'empty mode empty key',
            'Unkonwn',
            'Unknown',
            'No Mood',
            'nonvocal nongender',
            'nonvocal nontimbre',
            'non-canned music',
            'non-nostalgic',
            'nonvocal',
            'empty gender',
        ]
        freeform_text = [
            ', '.join([tag for tag in category_list if (tag and (tag not in not_include_list))])
            for category_list in style_text if rnd.random() < (1.0-freeform_dropout)
        ]
    freeform_text = list(set(freeform_text))
    if shuffle:
        rnd.shuffle(freeform_text)
    # Remove empty strings
    freeform_text = ', '.join([text for text in freeform_text if text])
    # in case all dimensions are not selected, just select everything
    if not freeform_text:     
        freeform_text = ', '.join(sub_item for item in style_text for sub_item in item if (sub_item and (sub_item not in not_include_list)))
    if inst:
        freeform_text += ', ' + inst
    # print(f"[infer_freeform_text_from_style_text] {freeform_text=}")
    return freeform_text

# ----------- musicfm+ mir tags (key/tempo/instrument) ------
def parse_mir_tag_optional(meta, sstk_info=None, inst_text=None, inst_thre_ver="0.9"):
    """
    inst_dict: dictionary {'inst_name': inst_confidence}
    bpm: int value or beat list
    key: str or key list (root + mode)
    tempo_text, key_text, mode_text: for style_text[10:13] (registered tag)
    """
    try:
        bpm, tempo_text = None, ''
        inst_dict = dict()
        key, key_text, mode_text = None, '', ''
        
        if sstk_info: # field
            if sstk_info[0] and sstk_info[0][0] not in ['', '\\N']:
                bpm = int(sstk_info[0][0])
                tempo_text = tempo_to_label(bpm)
            if sstk_info[1]:    # use sstk inst info by default
                if isinstance(sstk_info, list):
                    inst_dict = {inst.replace(" ", "_"): 1.0 for inst in sstk_info[1]} # instrument str/list
                else:   # str
                    inst_dict = {sstk_info[1].replace(" ", "_"): 1.0}

        if "bpm" in meta and meta["bpm"] not in ['\\N', '']:
            bpm = int(meta["bpm"])
            tempo_text = tempo_to_label(bpm)
        if "instruments" in meta and meta["instruments"] not in ['\\N', ''] and inst_text == ['empty instrument']:
            instruments = meta["instruments"].split(", ")
            for inst in instruments:
                if tagging_inst_to_38(inst) is not None:
                    inst_dict[inst.replace(" ", "_")] = 1.0 # instrument str/list

        # fine-grained key/beat tagging by musicfm+
        if "musicfm_plus" in meta:    
            get_value = partial(_get_value, msg="No musicfm+ processed data")
            if bpm is None: # (TODO: qinxin) use tagged info by chance
                try:
                    bpm = get_value(meta["musicfm_plus"], "beat")    # list
                except ZhMetaParseError:
                    bpm = None
            
            try:
                key = get_value(meta["musicfm_plus"], "key")    # list
            except ZhMetaParseError:
                key = None

        # song-level instrument tagging by musicfm+
        # if "musicfm_tagging" in meta and len(inst_dict) <= 0:
        #     inst_prob_dict = meta["musicfm_tagging"]
        #     for k, v in inst_prob_dict.items():
        #         k_38cls = tagging_inst_to_38(k.replace(" ", "_")).replace(" ", "_")
        #         if is_valid_inst(k_38cls, v, version=inst_thre_ver):
        #             inst_dict[k_38cls] = v

        # parse song-level tags
        if key is not None:
            if isinstance(key, list):
                key_list = [_key[1] for _key in key] # key name
                key_counter = Counter(key_list)
                song_key = key_counter.most_common(1)[0][0]
                if song_key != "X":
                    key_text, mode_text = song_key.split(":")
            elif isinstance(key, str) and ':' in key:
                key_text, mode_text = key.split(":")
            if len(mode_text) == 3:
                mode_text = mode_text + "or"    # Maj -> Major

        if bpm is not None and tempo_text == '':
            if isinstance(bpm, list):
                avg_beat_interval = np.mean(np.diff(np.array(bpm)[:, 0], n=1))
                tempo = 60 / avg_beat_interval
                if avg_beat_interval > 0.001 and not math.isnan(tempo):
                    tempo_text = tempo_to_label(tempo)

        # try to get everynoise/spotify analysis tags
        if "raw" in meta and "audio_analysis" in meta["raw"] and \
            meta["raw"]["audio_analysis"] is not None and meta["raw"]["audio_analysis"] != 'null':
            audio_analysis = meta["raw"]["audio_analysis"]
            if "track" in audio_analysis:   # here ignore the confidence since it is not quite reliable
                root, mode, tempo = None, None, None
                if "key" in audio_analysis["track"]:
                    root = ID_ROOT_MAP[audio_analysis["track"]["key"]+1]
                if "mode" in audio_analysis["track"]:
                    mode = "Maj" if audio_analysis["track"]["mode"] == 1 else "Min"
                if "tempo" in audio_analysis["track"]:
                    tempo = round(audio_analysis["track"]["tempo"])

                # (TODO: qinxin): replace key/tempo by add_key/add_tempo by chance 
                add_key = ":".join([root, mode]) if root is not None and mode is not None else None
                add_bpm = tempo

        return inst_dict, bpm, key, tempo_text, key_text, mode_text
    except Exception as e:
        return dict(), None, None, '', '', '' 


def parse_vad_voice_proportion(meta: Dict) -> Optional[float]:
    get_value = partial(_get_value, msg="No voice proportion")
    return get_value(get_value(get_value(meta, "vad"), "extra"), "voice_proportion")


def parse_vad_voice_proportion_default0(meta: Dict) -> Optional[float]:
    try:
        voice_proportation = parse_vad_voice_proportion(meta)
    except:
        voice_proportation = 0
    return voice_proportation


# ---------- freeform_text -------------

def parse_freeform_text_audio_tags(meta: Dict) -> Optional[str]:
    """
    "audio_tags": {
        "extra": [],
        "genre": [],
        "genre_extra": [],
        "instrument": [],
        "language": ["Instrumental/Non-vocal"],
        "mood": [],
        "remark": "",
        "satisfy_filter_standard": "yes",
        "scene": []
    }
    
    """
    get_value = partial(_get_value, msg="No audio_tags")
    keywords = []
    audio_tags = get_value(meta, "audio_tags")
    for key in ['extra', 'genre', 'genre_extra', 'instrument', 'mood', 'scene']:
        tags = audio_tags.get(key, [])
        if tags:
            random.shuffle(tags)
            if random.random() < 0.5:   # sometimes 'instrument' can have many tags, try to make dimensions like this not too override
                tags = tags[:random.randint(1, 5)]
            keywords.extend(tags)
    keywords = list(set(keywords))
    random.shuffle(keywords)
    if random.random() < 0.2:
        keywords = keywords[:random.randint(1, 5)]
    if random.random() < 0.5:
        freeform_text = ", ".join(keywords)
    else:
        freeform_text = " ".join(keywords)
    return freeform_text


def parse_freeform_text_groupAB_optional(meta: Dict) -> Optional[str]:
    """meta.raw.v1 & v2"""
    try:
        # if platform
        platform = meta.get("raw", {}).get("platform", "")
        if platform in ['dq', 'wyy', 'mcc']:
            return None
        library = meta.get("library", "")
        if library in ['pond5', 'shutterstock']:
            return None

        v1 = meta.get("raw", {}).get("v1", {})
        v2 = meta.get("raw", {}).get("v2", {})
        meta_song_genre = v1.get("meta_song_genre", "")
        if meta_song_genre:
            if '(' in meta_song_genre or ')' in meta_song_genre:
                meta_song_genre = [meta_song_genre]
            else:
                meta_song_genre = meta_song_genre.strip().replace(' / ', '\t').replace(' & ', '\t').replace(' - ', '\t').replace(',', '\t')
                meta_song_genre = meta_song_genre.split('\t')
        else:
            meta_song_genre = []
        genres = ast.literal_eval(v1.get("genres", "[]"))
        if genres and isinstance(genres, str):
            genres = [genres]
        final_genre = v2.get("final_genre", "")
        if final_genre and isinstance(final_genre, str):
            final_genre = [final_genre]
        keywords = []
        for item in [meta_song_genre, genres, final_genre]:
            for x in item:
                if x:
                    keywords.append(x)
        keywords = list(set(keywords))
        random.shuffle(keywords)
        freeform_text = ", ".join(keywords)
        return freeform_text if freeform_text else None
    except Exception as e:
        logging.warning("error in parse_freeform_text_groupAB: ", e)
        return None


def extract_keywords_sstk(meta: Dict) -> Tuple[List[str], Optional[str], Optional[float]]:
    # if platform
    platform = meta.get("raw", {}).get("platform", "")
    if platform in ['dq', 'wyy', 'mcc']:
        return [], None, None
    # library = meta.get("library", "")
    # if library not in ['pond5', 'shutterstock']:
    #     return [], None, None
    
    instruments = meta.get('instruments', '')
    if not instruments:
        instruments = meta.get('raw', {}).get('instruments', '')
    instruments = instruments.split(', ') if instruments else []
    instruments = [x for x in instruments if x != '\\N']
    keywords = meta.get('keywords', '')
    if not keywords:
        keywords = meta.get('raw', {}).get('keywords', '')
    keywords = keywords.split(', ') if keywords else []
    all_tags = [tag for tag in instruments + keywords if tag]
    keywords = list(set(all_tags))

    description = meta.get('description')
    tempo = meta.get('bpm')
    try:
        tempo = float(tempo)
    except (ValueError, TypeError):
        tempo = None

    return keywords, description, tempo

def extract_keywords_everynoise(meta: Dict) -> List[str]:
    """meta.raw.genres, meta.everynoise_genre, meta.everynoise_trending.vantage/genre"""
    # if platform
    platform = meta.get("raw", {}).get("platform", "")
    if platform in ['dq', 'wyy', 'mcc']:
        return []
    library = meta.get("library", "")
    if library in ['pond5', 'shutterstock']:
        return []
    raw_genres = meta.get("raw", {}).get("genres", [])
    everynoise_genre = meta.get("everynoise_genre", "")
    everynoise_trending_vantage = meta.get('everynoise_trending', {}).get('vantage', '')    # sometimes 'everynoise_trending' exists but is None
    everynoise_trending_genre = meta.get('everynoise_trending', {}).get('genre', '')
    all_tags = [tag for tag in [raw_genres, everynoise_genre, everynoise_trending_genre, everynoise_trending_vantage] if tag]
    flat_list = [tag for sublist in all_tags for tag in (sublist if isinstance(sublist, list) else [sublist])]
    keywords = list(set(flat_list))
    return keywords


def extract_keywords_wyy(meta: Dict) -> List[str]:
    # if platform
    platform = meta.get("raw", {}).get("platform", "")
    if platform in ['dq', 'wyy', 'mcc']:
        return []
    library = meta.get("library", "")
    if library in ['pond5', 'shutterstock']:
        return []
    tags = ast.literal_eval(meta.get("raw", {}).get("tags", "[]"))
    category = meta.get("raw", {}).get("category", "")
    song_tag = meta.get("raw", {}).get("song_tag", "")
    song_biz_tag = meta.get("raw", {}).get("song_biz_tag", "")
    if song_tag:
        song_tag = song_tag.split('-', 1)
    else:
        song_tag = []
    if song_biz_tag:
        song_biz_tag = song_biz_tag.split(',')
    else:
        song_biz_tag = []
    if category:
        category = category.split(',')
    else:
        category = []
    keywords = []
    keywords.extend(tags)
    keywords.extend(category)
    keywords.extend(song_tag)
    keywords.extend(song_biz_tag)
    keywords = list(set(keywords))
    return keywords


def extract_keywords_apm(meta: Dict) -> Tuple[List[str], Optional[str]]:
    # if platform
    raw = meta.get('raw', {})
    platform = meta.get("raw", {}).get("platform", "")
    if platform in ['dq', 'wyy', 'mcc']:
        return [], None
    library = meta.get("library", "")
    if library in ['pond5', 'shutterstock']:
        return [], None
    facets_list = json.loads(raw.get("facets_list", "[]"))
    keywords = [item for d in facets_list for value in d.values() for item in value if item is not None]
    term = raw.get("term")
    if term:
        keywords.extend(term.split(','))
    keywords = [x for x in keywords if x not in APM_TAG_TO_REMOVE]
    description = raw.get('description')
    return keywords, description


def extract_keywords_rym(meta: Dict) -> List[str]:
    # if platform
    platform = meta.get("raw", {}).get("platform", "")
    if platform in ['dq', 'wyy', 'mcc']:
        return []
    library = meta.get("library", "")
    if library in ['pond5', 'shutterstock']:
        return []

    track_genres = meta.get("raw", {}).get("track_genres", [])        
    album_genres = meta.get("raw", {}).get("album_genres", [])
    album_genres = [x for item in album_genres for x in item]
    album_descriptors = meta.get("raw", {}).get("album_descriptors", [])
    keywords = []
    keywords.extend(track_genres)
    keywords.extend(album_genres)
    keywords.extend(album_descriptors)
    keywords = list(set(keywords))
    return keywords


def parse_freeform_text_sstk(
        meta: Dict, 
        freeform_long_rate=0.0,
        keyword_reduce_rate = 0.2,
        tempo_dropout=0.5, 
    ) -> Tuple[Optional[str], Optional[str]]:
    # output freeform_text_short or freeform_text_long
    # why differentiate them? freeform_text_long should not shuffle by comma in later steps
    try:
        keywords, description, tempo = extract_keywords_sstk(meta)
        if random.random() < freeform_long_rate and description:
            return None, description

        if random.random() < keyword_reduce_rate:
            keywords = keywords[:random.randint(1, 10)]
        if random.random() < 0.5:
            keywords = [k.lower() for k in keywords]
        else:
            keywords = [k.capitalize() for k in keywords]

        if tempo and random.random() > tempo_dropout:
            if random.random() < 0.5:
                tempo_label = tempo_to_label(float(tempo))
            else:
                tempo_label = tempo_to_coarse_label(float(tempo))
            if tempo_label:
                keywords.append(random.choice([
                    'bpm is %s' % tempo,
                    'BPM: %s' % tempo,
                    '%s bpm' % tempo,
                    'BPM %s' % tempo,
                    'tempo is %s' % tempo_label,
                    '%s' % tempo_label,
                ]))
        random.shuffle(keywords)
        freeform_text = ", ".join(keywords)
        return freeform_text, None
    
    except Exception as e:
        logging.warning("error in parse_freeform_text_sstk: ", e)
        return None, None


def parse_freeform_text_everynoise_optional(meta: Dict) -> Optional[str]:
    try:
        keywords = extract_keywords_everynoise(meta)
        freeform_list = []
        for x in keywords:
            if x:
                freeform_list.append(x)
                if x in EVERYNOISE_NON_MATCHED_TAG_TO_CATEGORY:
                    freeform_list.extend(EVERYNOISE_NON_MATCHED_TAG_TO_CATEGORY[x])
        freeform_list = list(set(freeform_list))
        random.shuffle(freeform_list)
        freeform_text = ', '.join(freeform_list) if freeform_list else None
        return freeform_text
    except Exception as e:
        logging.warning("error in parse_freeform_text_everynoise: ", e)
        return None


def parse_freeform_text_wyy_optional(meta: Dict) -> Optional[str]:    
    try:
        keywords = extract_keywords_wyy(meta)
        freeform_list = []
        for x in keywords:
            if x in WYY_TAG_TO_FREEFORM:
                freeform_list.append(WYY_TAG_TO_FREEFORM[x])
            else:
                pass
                # print ('!!!!! keyword not found: ', x)
        freeform_list = list(set(freeform_list))
        random.shuffle(freeform_list)
        freeform_text = ', '.join(freeform_list) if freeform_list else None
        return freeform_text

    except Exception as e:
        logging.warning("error in parse_freeform_text_wyy: ", e)
        return None


def parse_freeform_text_apm(
        meta: Dict, 
        freeform_long_rate=0.0,     # whether output freeform_text_short or freeform_text_long
        keyword_reduce_rate=0.2     # if output freeform_text_short, use this rate to reduce the number of keywords
    ) -> Tuple[Optional[str], Optional[str]]:
    # output freeform_text_short or freeform_text_long
    # why differentiate them? freeform_text_long should not shuffle by comma in later steps
    try:
        keywords, description = extract_keywords_apm(meta)
        if random.random() < freeform_long_rate and description:
            return None, description

        freeform_list = []
        for x in keywords:
            if x:
                freeform_list.append(x)
                # if x in APM_NON_MATCHED_TAG_TO_GENRE:
                #     freeform_list.extend(APM_NON_MATCHED_TAG_TO_GENRE[x])
                # if x in APM_NON_MATCHED_TAG_TO_MOOD:
                #     freeform_list.extend(APM_NON_MATCHED_TAG_TO_MOOD[x])
                # if x in APM_NON_MATCHED_TAG_TO_SCENE:
                #     freeform_list.extend(APM_NON_MATCHED_TAG_TO_SCENE[x])
                # if x in APM_NON_MATCHED_TAG_TO_INST:
                #     freeform_list.extend(APM_NON_MATCHED_TAG_TO_INST[x])
                # if x in APM_TAG_TO_TEMPO:
                #     freeform_list.append(APM_TAG_TO_TEMPO[x])
                # if x in APM_TAG_TO_FREEFORM:
                #     freeform_list.append(APM_TAG_TO_FREEFORM[x])

        freeform_list = list(set(freeform_list))
        random.shuffle(freeform_list)
        if random.random() < keyword_reduce_rate:
            freeform_list = freeform_list[:random.randint(1, 10)]
        if random.random() < 0.5:
            freeform_list = [k.lower() for k in freeform_list]
        else:
            freeform_list = [k.capitalize() for k in freeform_list]

        freeform_text = ', '.join(freeform_list) if freeform_list else None
        return freeform_text, None

    except Exception as e:
        logging.warning("error in parse_freeform_text_apm: ", e)
        return None


def parse_freeform_text_rym(
        meta: Dict, 
        keyword_reduce_rate=0.2     # use this rate to reduce the number of keywords
    ) -> Optional[str]:

    try:
        keywords = extract_keywords_rym(meta)
        freeform_list = []
        for x in keywords:
            if x:
                freeform_list.append(x)
                # if x in RYM_NON_MATCHED_TAG_TO_GENRE:
                #     freeform_list.extend(RYM_NON_MATCHED_TAG_TO_GENRE[x])
                # if x in RYM_NON_MATCHED_TAG_TO_MOOD:
                #     freeform_list.extend(RYM_NON_MATCHED_TAG_TO_MOOD[x])
                # if x in RYM_NON_MATCHED_TAG_TO_SCENE:
                #     freeform_list.extend(RYM_NON_MATCHED_TAG_TO_SCENE[x])
                # if x in RYM_NON_MATCHED_TAG_TO_INST:
                #     freeform_list.extend(RYM_NON_MATCHED_TAG_TO_INST[x])
                # if x in RYM_TAG_TO_EXTRA:
                #     freeform_list.extend(RYM_TAG_TO_EXTRA[x])

        freeform_list = list(set(freeform_list))
        random.shuffle(freeform_list)
        if random.random() < keyword_reduce_rate:
            freeform_list = freeform_list[:random.randint(1, 10)]
        if random.random() < 0.5:
            freeform_list = [k.lower() for k in freeform_list]
        else:
            freeform_list = [k.capitalize() for k in freeform_list]

        freeform_text = ', '.join(freeform_list) if freeform_list else None
        return freeform_text
    
    except Exception as e:
        logging.warning("error in parse_freeform_text_rym: ", e)
        return None


# ---------- section instruments -------------

def parse_section_instruments_optional(meta: Dict, threshold=0.85) -> Optional[list]:
    
    try:    
        section_instruments = meta['musicfm_tagging']['section_based_instrument']
    except:
        return []
    section_instruments = [{'start': x['interval'][0], 'end': x['interval'][1], 'instruments': x['instruments']} for x in section_instruments]
    for item in section_instruments:
        item['instruments'] = [x[0] for x in item['instruments'].items() if x[1] > threshold]
        item['instruments'] = [KARAOKE_INST_270_TO_38[x] for x in item['instruments'] if x in KARAOKE_INST_270_TO_38]
        item['instruments'] = [x for sublist in item['instruments'] for x in sublist]
        item['instruments'] = [KARAOKE_INST_38_TO_VOCAB[x] for x in item['instruments'] if x in KARAOKE_INST_38_TO_VOCAB]
        item['instruments'] = [x for sublist in item['instruments'] for x in sublist]
        item['instruments'] = list(set(item['instruments']))

    return section_instruments


# ---------- artist_id -------------

def parse_artist_id(meta) -> int:
    """meta.artist_id, use zh_empty if it's empty"""
    return ARTIST_ID_MAP_V2[str(meta.get("artist_id", "zh_empty"))]

def parse_artist_id_by_lang_gender(lang, gender) -> int:
    if 'Chinese' in lang:
        artist_id = ARTIST_ID_MAP_V2['zh_empty']
    elif 'Cantonese' in lang:
        artist_id = ARTIST_ID_MAP_V2['cant_empty']
    elif 'English' in lang:
        artist_id = ARTIST_ID_MAP_V2['en_empty']
    else:
        artist_id = ARTIST_ID_MAP_V2['inst_empty']

    if 'Neutral' in gender:
        artist_id = ARTIST_ID_MAP_V2['Neutral']
    elif 'Multiple' in gender:
        artist_id = ARTIST_ID_MAP_V2['Multiple']
    elif 'Chorus' in gender:
        artist_id = ARTIST_ID_MAP_V2['Chorus']
    elif 'Child' in gender:
        artist_id = ARTIST_ID_MAP_V2['Child']
    elif 'Male' in gender and 'Female' in gender:
        artist_id = ARTIST_ID_MAP_V2['Male_Female']
    elif 'Male' in gender:
        artist_id = ARTIST_ID_MAP_V2['Male']
    elif 'Female' in gender:
        artist_id = ARTIST_ID_MAP_V2['Female']
    return artist_id

def parse_artist_tagging(meta: Dict, threshold: float = 0.5) -> Optional[List[str]]:
    get_value = partial(_get_value, msg="No artist tagging")
    artist_tagging = get_value(get_value(get_value(meta, "musicfm_tagging"), "artist"), "artist")
    """
    "{'李荣浩': 0.3328, '孙燕姿': 0.11053, 's.h.e': 0.04617, '汪苏泷': 0.04077, '林俊杰': 0.03476}"
    """
    if artist_tagging is None:
        return None
    if isinstance(artist_tagging, str):
        artist_tagging = ast.literal_eval(artist_tagging)
    artists = [artist for artist, probability in artist_tagging.items() if probability > threshold]
    return artists

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


def parse_mir_vocal2midi(meta: Dict) -> Dict:
    """meta.mir_service.vocal2midi"""
    get_value = partial(_get_value, msg="No vocal2midi")
    return get_value(get_value(meta, "mir_service"), "vocal2midi")


def parse_mir_vocal2midi_optional(meta: Dict) -> Optional[Dict]:
    """meta.mir_service.vocal2midi"""
    try:
        return parse_mir_vocal2midi(meta)
    except ZhMetaParseError:
        return None


# ------------------------------------------
#                 TRANSFORM
# ------------------------------------------

# ---------- converters -------------

# All the converters should do in-place operations on _self
# to avoid to many copies.

def convert_hqmy(_self):
    """Re-assign genre for HQMY songs"""
    if _self.source == "环球美音":
        _self.style_text[0] = "Chinese Tradition"


def convert_artist_id_from_voice_tag(_self):
    """Re-assign gender tags"""
    if _self.voice_tag and (_self.artist_id == ARTIST_ID_MAP_V2["zh_empty"]):
        _self.artist_id = ARTIST_ID_MAP_V2[_self.voice_tag]


def convert_voice_tag_from_audio_tag(_self):
    """Override voice_tag if Male or Female is in audio tags"""
    voice_tags = _self.style_text[3]
    if 'Male' in voice_tags:
        voice_tag = 'Male'
    elif 'Female' in voice_tags:
        voice_tag = 'Female'
    else:
        return
    _self.voice_tag = voice_tag


def convert_zh_to_cant(_self):
    pass


# ---------- validators -------------

def validate_lyrics_confidence_optional(confidence_threshold: Optional[float], confidence: Optional[float]):
    if confidence_threshold is None or confidence is None:
        return
    if confidence < confidence_threshold:
        # Do not log the confidence value. The value is almost always different for each sample.
        # Adding the specific confidence value to the log will mess up the log messages that's
        # supposed to be aggregated and counted.
        raise ZhMetaTransformError("Low lyrics confidence")


def validate_style_text_sa(style_text: List[str], is_sinking: bool):
    _genre_tag, _, _, _, _lang_tag = style_text
    if _genre_tag in ["", "Chinese Opera", "Other genre"]:
        raise ZhMetaTransformError(f"Filter out genre \"{_genre_tag}\"")
    if (_genre_tag not in ["DJ", "MC"]) and is_sinking:
        raise ZhMetaTransformError(f"Filter out sinking of genre \"{_genre_tag}\"")
    if not _lang_tag or (_lang_tag == "Chinese Dialects") or (_lang_tag == "Cantonese") or (_lang_tag == "Other"):
        raise ZhMetaTransformError(f"Filter out lang \"{_lang_tag}\"")


def validate_style_text_sa_optional(style_text: Optional[List[str]], is_sinking: bool):
    if style_text is None:
        return
    if len(style_text) == 5:
        validate_style_text_sa(style_text, is_sinking)
    elif len(style_text) == 9:
        genre, genre_extra, extra, mood, scene, gender, vocal, language, is_sinking = style_text
        if not set(language).issubset(['Chinese', 'English', 'Cantonese']):
            raise ZhMetaTransformError(f"In 9slots, Filter out lang \"{language}\"")
        if "Chinese Opera" in genre or "Other genre" in genre:
            raise ZhMetaTransformError(f"In 9slots, Filter out genre \"{genre}\"")
        if ("DJ" not in genre and "MC" not in genre) and is_sinking == 'Sinking':
            raise ZhMetaTransformError(f"In 9slots, Filter out sinking of genre \"{genre}\" and \"{is_sinking}\"")
    elif len(style_text) == 13:
        genre, genre_extra, extra, mood, scene, gender, timbre, language, is_sinking, instruments, tempo, key, mode = style_text
        if not set(language).issubset(['Chinese', 'English', 'Cantonese', 'Japanese', 'Sichuanese']):
            raise ZhMetaTransformError(f"In 13slots, Filter out lang \"{language}\"")
        if "Chinese Opera" in genre:
            raise ZhMetaTransformError(f"In 13slots, Filter out genre \"{genre}\"")
    else:
        raise ZhMetaTransformError(f"Unvalid length \"{style_text}\"")


def validate_quality(high_quality: bool):
    if not high_quality:
        raise ZhMetaTransformError(f"Filter out low quality")


def validate_deepchorus_optional(deepchorus: Optional[DeepChorus], confidence: Optional[float] = None):    
    if deepchorus is None or confidence is None:
        return
    if deepchorus.confidence < confidence:
        raise ZhMetaTransformError(f"Low deepchorus confidence")


def validate_mir_tempo_optional(tempo: Optional[int]):
    if tempo is None or isinstance(tempo, list):
        return
    l, h = TEMPO_RANGE
    if not (l < tempo <= h):
        raise ZhMetaTransformError("Tempo not in supported range: %s" % tempo)


def validate_mir_key_optional(key: Optional[str]):
    if key is None or isinstance(key, list):
        return
    if key is not None:
        if isinstance(key, list):
            key_list = [_key[1] for _key in key]
            if any([_key not in KEYS for _key in key_list]):
                raise ZhMetaTransformError(f"Unsupported key {key_list}")
        elif isinstance(key, str):
            if key not in ROOTS:
                raise ZhMetaTransformError(f"Unsupported key {key}")


def validate_sstk_has_vocal(has_vocal: bool):
    if has_vocal:
        raise ZhMetaTransformError(f"Exclude vocal song from sstk for now, assumng no lyrics yet")


def validate_vad_voice_proportion_optional(vad_vocal_proportion: Optional[float], vad_threshold: Optional[float]):
    if vad_threshold is None or vad_vocal_proportion is None:
        return
    if vad_vocal_proportion > vad_threshold:
        raise ZhMetaTransformError("VAD vocal proportion too high")

def validate_filter_label_single_yes(is_high_quality: bool, is_popular_potential: bool):
    if is_high_quality or is_popular_potential:
        return
    else:
        raise ZhMetaTransformError("filter_label of is_high_quality and is_popular_potential are both False")

# ---------- SongSlice -------------

def _format_utterances(utterances, time_in_sec: bool = False):
    # This function cleans up the raw utterances data from metadata.
    # Extracts necessary information: [start_time_in_sec, end_time_in_sec, lyrics text with additional tags, precomputed phonemes]
    new_utterances = []    
    for i, u in enumerate(utterances):
        phone_v86 = u.get('phoneme_v86')  # phoneme string tagged by tts frontend v86, use it whenever it's possible
        phone = phone_v86 if phone_v86 else u.get('phoneme', '')

        # NOTE: This operation tries to obtain confidence from 2 sources:
        # - For force align lyrics: utterance.confidence
        # - For ASR lyrics: utterance.attribute.confidence
        # This is not ideal, because we want to separate the parsing methods, and make each one specific.
        # But it might require more code change. 
        # For now, it is only up to the validation step to make sure the format is correct.
        # Since the given utterance is either from ASR or force alignment, it not very likely to go wrong.
        confidence = u.get('confidence', u.get('attribute', {}).get('confidence'))

        if "lyrics" in u:
            u["text"] = u["lyrics"]
        
        utt_start = u.get('start_time', None)
        if utt_start is None:
            continue
        utt_end = u.get('end_time', None)
        if utt_end is None:
            if i < (len(utterances) - 1):
                utt_end = max(utterances[i+1].get('start_time', 0), utt_start+1)
            else:
                continue
        utt_end = math.ceil(utt_end)        # Round up so we don't miss the last few phonemes
        if time_in_sec:
            new_utterances.append([utt_start, utt_end, u['text'], phone])
        else:
            # TODO (QQ) handle ms directly instead of converting to int.
            new_utterances.append([utt_start/1000, utt_end/1000, u['text'], phone, confidence])


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
        if u[1] < 0 or u[1] - u[0] > 0:
            # Only keep utterances that are longer than 1 sec.
            utterances.append(u)
    # Remove duplicate utterances, retain order.
    uniq_utt_idx = []
    for i in range(len(utterances)):
        if (i < len(utterances) - 1) and (utterances[i][0] == utterances[i+1][0]) and (utterances[i][1] == utterances[i+1][1]):
            continue
        uniq_utt_idx.append(i)
    
    utterances = [utterances[i] for i in uniq_utt_idx]
    
    if len(utterances) < 1:
        return []

    # Sanity check: utterances are non-overlapping.
    for i in range(len(utterances)-1):
        # skip negative end time
        if utterances[i][1] < 0:
            continue
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
    phrases = [phrase for phrase in phrases if not phrase.is_empty]
    return SongSlice(phrases=phrases)


def transform_utts_to_song_slices_heuristic(
    utterances,
    min_duration,
    max_duration,
    time_in_sec=False,
    new_line_token='\n',
    infer_structure_tags=False,
    language: Optional[str] = None,
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
            phonemes=transform_phonemes_by_language(phonemes, language),
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
    slice_mode: str,
    language: Optional[str] = None,
) -> List[SongSlice]:
    """Segment the full song into a list of SongSlice, each consists of multiple utterances.
    Args:
        slice_mode:
        - "max": Try to reach the max duration for every slice.
          It always preserves phrase boundary, but the result might contain incomplete sections.
        - "section": Return all the possible slices within max_duration with complete sections.
        - "full": Return the entire song
    """
    def format_utterances(utterances: List[Dict]) -> List[Phrase]:
        formatted_us = _format_utterances(utterances)
        return [
            Phrase.parse(
                text=nt,
                phonemes=transform_phonemes_by_language(phonemes, language),
                time_span=(start, end),
                lyrics_confidence=conf,
            ) for start, end, nt, phonemes, conf in formatted_us
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

    def get_section_span(structure_tag: Dict) -> Tuple[float, float]:
        return structure_tag["start_time"], structure_tag["end_time"]

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

    def remove_long_inst_intro(phrases: List[Phrase], max_intro_dur: int = 20) -> List[Phrase]:
        """Remove instrumental intro phrases whose lengths are longer max_intro_dur.
        `max_intro_dur` is hard-coded for now. Ideally it should change according to the full song length.
        """
        return [phrase for phrase in phrases if not (
            phrase.section_tag is not None and
            _remove_count_from_section_tag(phrase.section_tag) == "intro" and
            not phrase.has_utterance and
            phrase.duration > max_intro_dur
        )]

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
                        split_t = _start + max_duration
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

    def infer_missing_singer_tags(phrases: List[Phrase]) -> List[Phrase]:
        """2024/06/26: This function is for handling the data missing issue in the current dataset.
        Sometimes the first a few phrases in a song do not have singer tags
        """
        if not any(phrase.singer_tag for phrase in phrases):
            return phrases[:]
        utt_phrase_ind = [idx for idx, phrase in enumerate(phrases) if phrase.has_utterance]
        utt_singer_tags = [phrase.singer_tag for phrase in phrases if phrase.has_utterance]
        first_singer_tag = next(singer_tag for singer_tag in utt_singer_tags if singer_tag)
        inferred_singer_tag = {"男": "女", "女": "男"}.get(first_singer_tag)  # it's almost always this pattern
        if inferred_singer_tag is None:
            return phrases[:]
        no_singer_tag_phrase_ind = utt_phrase_ind[:utt_singer_tags.index(first_singer_tag)]
        return [
            (phrase._replace(singer_tag=inferred_singer_tag) if idx in no_singer_tag_phrase_ind else phrase)
            for idx, phrase in enumerate(phrases)
        ]

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
            # idx_inc indicates the max number of song slices that can be combined
            idx_inc = max(1, int(np.searchsorted(rel_end_ts, max_duration, side="right")))
            # NOTE: The next step pushes every possible lengths into the list starting from idx.
            # If we simply push [idx: idx+idx_inc], we are enforcing the slice to reach max_duration
            # as much as possible, which is not necessarily what we want if we need to handle 
            # short lyrics and generate short songs.
            for _inc in range(1, idx_inc+1):
                song_slices.append(reduce(operator.add, song_slices_by_sections[idx: idx+_inc]))
        return song_slices
    
    def get_song_slices_full(phrases: List[Phrase]) -> List[SongSlice]:
        if not phrases: return []
        return [SongSlice(phrases=phrases)]
    
    def remove_phrases_in_no_vocal_sections(phrases: List[Phrase]) -> List[Phrase]:
        """Remove phrases in no vocal sections."""
        phrases = [
            phrase for phrase in phrases \
                if _remove_count_from_section_tag(phrase.section_tag) not in NO_VOCAL_SECTION_TAGS
        ]
        return phrases

    slice_fn = {
        "max": get_song_slices,
        "section": get_song_slices_complete_section,
        "full": get_song_slices_full,
    }[slice_mode]
    song_time_span = (structure_tags[0]["start_time"], structure_tags[-1]["end_time"])
    phrases = format_utterances(utterances)
    phrases = list(map(add_section_tag, phrases))  # TODO: prevent phrase in instrumental section. Just throw them away
    phrases = remove_phrases_in_no_vocal_sections(phrases)
    phrases = insert_inst_phrases(phrases, song_time_span)
    
    # NOTE: 2025-03-19 Remove this since we are doing much longer audio generation, and we do want some long intro
    # if slice_mode != "full":  # long intro can be included under "full" mode
    #     phrases = remove_long_inst_intro(phrases)
    
    # NOTE: 2025-03-19 This is not needed anymore.
    # phrases = infer_missing_singer_tags(phrases)
    
    
    song_slices = slice_fn(phrases)
    # Remove short instrumental sections' section tags. After the removal, these phrases will become placeholders
    # for SongSlice to correctly calculate the start and end times, but will be completely ignored during tokenization.
    song_slices = list(map(_reset_section_tags_for_short_inst_phrases, song_slices))
    return [song_slice for song_slice in song_slices if song_slice.phrases]


def transform_section_to_song_slices(
    min_duration: int,
    max_duration: int,
    structure_tags: List[Dict],
    slice_mode: str = 'section',
    chorus_only: bool = True,
) -> List[SongSlice]:

    def generate_dummy_phrases(structure_tags: List[Dict]):
        phrases = [
            Phrase(
                    time_span=(section['start_time'], section['end_time']),
                    section_tag=section['tag'],
                    instruments=section['instruments'] if 'instruments' in section else None,
                )
            for section in structure_tags
        ]
        return phrases
    
    def generate_interval_candidates(phrases: list, min_duration: float, max_duration: float) -> List[Tuple]:
        """Generate all candidate intervals that contains at least one chorus"""
        if not phrases:
            return []
        # chorus_intervals = [(x.time_span[0], x.time_span[1]) for x in phrases if 'chorus' in x.section_tag]  # take chorus only
        chorus_intervals = [(x.time_span[0], x.time_span[1]) for x in phrases]
        boundaries = [x.time_span[0] for x in phrases] + [phrases[-1].time_span[1]]
        interval_candidates = set()

        for start, end in chorus_intervals:
            for i in range(len(boundaries)):
                for j in range(i + 1, len(boundaries)):
                    boundary_start = boundaries[i]
                    boundary_end = boundaries[j]
                    length = boundary_end - boundary_start
                    if min_duration <= length <= max_duration and boundary_start <= start and boundary_end >= end:
                        interval_candidates.add((boundary_start, boundary_end))

        return list(interval_candidates)

    def generate_song_slices(phrases, interval_candidates):
        song_slices = []
        for boundary_start, boundary_end in interval_candidates:
            current_phrases = [
                x for x in phrases if x.time_span[0] >= boundary_start and x.time_span[1] <= boundary_end
            ]
            song_slices.append(SongSlice(current_phrases))
        return song_slices
    
    def get_song_slices_full(phrases: List[Phrase]) -> List[SongSlice]:
        if not phrases: return []
        return [SongSlice(phrases=phrases)]
    
    phrases = generate_dummy_phrases(structure_tags)

    if slice_mode == 'full':
        song_slices = get_song_slices_full(phrases)
    elif slice_mode in ['section', 'max']:
        # if not chorus_only:     # 先不管这个
        #     raise NotImplementedError("chorus_only must be True")
        interval_candidates = generate_interval_candidates(phrases, min_duration, max_duration)
        song_slices = generate_song_slices(phrases, interval_candidates)
    else:
        raise NotImplementedError(f"Not supported slice_mode: {slice_mode}")

    if len(song_slices) == 0:
        raise ZhMetaTransformError("No valid instrumental song slice")

    return song_slices


def sample_song_slices(song_slices: List[SongSlice], segment_method: str, max_seg_per_track: int) -> List[SongSlice]:
    def method_first(song_slices: List[SongSlice], max_seg_per_track: int) -> List[SongSlice]:
        return song_slices[:1]
    
    def method_random(song_slices: List[SongSlice], max_seg_per_track: int) -> List[SongSlice]:
        song_slices = song_slices[:]  # make a shallow copy
        random.shuffle(song_slices)
        return song_slices[:max_seg_per_track]

    def method_uniform(song_slices: List[SongSlice], max_seg_per_track: int) -> List[SongSlice]:
        """
        This function does not guarantee the dataloader always produces an uniform distribution of slice durations.
        A song with a majority of slices in mid and low duration ranges is more likely to have more selected
        slices in these ranges, unless we set max_seg_per_track to 1. However, even if max_seg_per_track
        is 1, the actual distribution also depends on the dataset. A dataset that contains mostly short songs can
        skew the distribution of slice duration towards the short side.
        """
        song_slices = song_slices[:]  # make a shallow copy
        random.shuffle(song_slices)
        durs = [song_slice.duration for song_slice in song_slices]
        ind = sample_n_uniform(durs, max_seg_per_track)
        return [song_slices[idx] for idx in ind]

    def sample_uniform(lst: List[float], leeway: float = 5.0) -> int:
        """Sample an item from a list to ensure the sampled value is under an uniform distribution.
        leeway makes samples at the edges, such as 0 or 10 in range [0, 10], easier to be sampled.
        """
        val = random.uniform(min(lst) - leeway, max(lst) + leeway)
        return np.argmin(np.abs(np.array(lst) - val))  # index of the selected item

    def sample_n_uniform(lst: List[float], n: int) -> List[int]:
        """Sample n items from a list of floats (the result does not necessarily follows a uniform distribution)"""
        if n <= 0:
            raise ValueError("n must be a positive integer")
        n = min(len(lst), n)
        if n == 1:
            return [sample_uniform(lst)]
        idx = sample_uniform(lst)
        new_lst = lst[:idx] + lst[idx + 1:]
        ind = sample_n_uniform(new_lst, n - 1)
        ind = [(i if i < idx else (i + 1)) for i in ind]
        return sorted([idx] + ind)

    if segment_method not in ["first", "random", "uniform"]:
        raise ValueError("Invalid segment_method")
    
    if max_seg_per_track <= 0:
        max_seg_per_track = len(song_slices)
    
    try:
        seg_fn = {
            "first": method_first,
            "random": method_random,
            "uniform": method_uniform,
        }[segment_method]
    except KeyError:
        raise ValueError(f"Invalid segment_method: {segment_method}")
    segs = seg_fn(song_slices, max_seg_per_track)

    # TODO: pick as long as possible more frerquently
    # if random < 0.1:
    #     segs = pick the longest one from song_slices
        
    return segs


def filter_and_process_song_slices_with_mir_info(
    song_slices: List[SongSlice],
    key: Optional[str],
    tempo: Optional[int],
    instrument: Optional[Dict],
    mir_filters: Optional[List],
) -> List[SongSlice]:
    verified_song_slices = []
    for song_slice in song_slices:
        verified_song_slice = song_slice.select_and_add_mir_info(key, tempo, instrument, mir_filters)
 #       if mir_filters is not None and \
 #           (("unstable_key" in mir_filters and verified_song_slice.mir_info.key is None) or \
 #           ("unstable_tempo" in mir_filters and verified_song_slice.mir_info.tempo is None)):
 #           continue
        verified_song_slices.append(verified_song_slice)
    if not verified_song_slices:
        raise ZhMetaTransformError(f"No valid song slice passed mir_filters {mir_filters}")
    return verified_song_slices


def filter_song_slices(
    song_slices: List[SongSlice],
    duration_range: Tuple[int, int],
    lyrics_confidence_phrase: Optional[float],
) -> Tuple[List[SongSlice], ZhMetaLogger]:
    n_slices_pre_filter = len(song_slices)
    song_slices = [ss for ss in song_slices if ss.is_time_span_valid(duration_range)]
    if lyrics_confidence_phrase is not None:
        # Only filter by phrase level confidence if threshold is given
        # Drop the entire slice if any phrase in the slice has a low confidence
        song_slices = [ss for ss in song_slices if ss.is_lyrics_confidence_phrase_valid(lyrics_confidence_phrase)]
    n_slices_post_filter = len(song_slices)
    n_filtered = n_slices_pre_filter-n_slices_post_filter

    if len(song_slices) == 0:
        raise ZhMetaTransformError(f"No valid song slice within duration range {duration_range} and above phrase-level confidence {lyrics_confidence_phrase}")

    if any(phrase.text and not phrase.phonemes for song_slice in song_slices for phrase in song_slice.phrases):
        raise ZhMetaTransformError("No phoneme found in all available song slices")

    logger = ZhMetaLogger(
        f"Removed song slices: {n_filtered}/{n_slices_pre_filter}" if n_filtered > 0 else ""
    )
    return song_slices, logger


def _parse_inst_audio_tags_v3(music_tagging: Optional[Dict], audio_tags: Optional[Dict], sinking_threshold: float, mapping_tag: Optional[Dict]) -> Tuple[List[str], Dict[str, str], bool]:
    def map_tag(item: str, tag: str) -> str:
        """Replace certain tags in the dataset"""
        AUDIO_TAGS_SPECIAL_MAP = {
            'genre': AUDIO_TAGS_GENRE_SPECIAL_MAP_V2,
            'genre_extra': AUDIO_TAGS_GENRE_SPECIAL_MAP_V2,
            'extra': AUDIO_TAGS_EXTRA_SPECIAL_MAP,
            'mood': AUDIO_TAGS_MOOD_SPECIAL_MAP_V2,
            'scene': AUDIO_TAGS_SCENE_SPECIAL_MAP_V2,
            'vocal_gender': AUDIO_TAGS_GENDER_SPECIAL_MAP_V2,
            'vocal_timbre': AUDIO_TAGS_TIMBRE_SPECIAL_MAP,
            'language': AUDIO_TAGS_LANG_SPECIAL_MAP,
            'is_sinking': {},
            'instrument': AUDIO_TAGS_INST_SPECIAL_MAP,
            'tempo': AUDIO_TAGS_TEMPO_SPECIAL_MAP,
            'key': AUDIO_TAGS_KEY_SPECIAL_MAP,
            'mode': AUDIO_TAGS_MODE_SPECIAL_MAP,
        }
        if item in AUDIO_TAGS_SPECIAL_MAP:
            return AUDIO_TAGS_SPECIAL_MAP[item].get(tag, tag)
        else:
            return tag

    def uniq_tags(tags):
        _tags = []
        for item in tags:
            temp_tags = list(set(item))
            if len(temp_tags) > 1 and '' in temp_tags:
                temp_tags.remove('')
            _tags.append(temp_tags)
        return _tags

    # The order should match `mir_data_util`
    # order = ['genre', 'genre_extra', 'extra', 'mood', 'scene', 'vocal_gender', 'vocal_timbre', 'language', 'is_sinking', 'instrument', 'tempo', 'mode', 'key']
    tags = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    unfamiliar_tags = {}
    if audio_tags is not None:
        order = [
            {
                'AUDIO_CAT_VOCAB_IDX': 0,
                'META_KEY': 'genre',
                'ORDER_KEY': 'genre',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 1,
                'META_KEY': 'genre_extra',
                'ORDER_KEY': 'genre_extra',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 2,
                'META_KEY': 'extra',
                'ORDER_KEY': 'extra',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 3,
                'META_KEY': 'mood',
                'ORDER_KEY': 'mood',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 4,
                'META_KEY': 'scene',
                'ORDER_KEY': 'scene',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 5,
                'META_KEY': None,
                'ORDER_KEY': 'vocal_gender',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 6,
                'META_KEY': None,
                'ORDER_KEY': 'vocal_timbre',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 7,
                'META_KEY': 'language',
                'ORDER_KEY': 'language',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 8,
                'META_KEY': None,
                'ORDER_KEY': 'is_sinking',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 9,
                'META_KEY': 'instrument',
                'ORDER_KEY': 'instrument',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 10,
                'META_KEY': None,
                'ORDER_KEY': 'tempo',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 11,
                'META_KEY': None,
                'ORDER_KEY': 'key',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 12,
                'META_KEY': None,
                'ORDER_KEY': 'mode',
            },
        ]
        for order_map in order:
            idx = order_map['AUDIO_CAT_VOCAB_IDX']
            meta_key = order_map['META_KEY']
            order_key = order_map['ORDER_KEY']
            cat_vocab_tags = AUDIO_CAT_VOCAB_V3

            if order_key == 'is_sinking':
                tags[idx].append(map_tag(order_key, 'non-Sinking'))
            else:
                result = audio_tags.get(meta_key, [])
                if isinstance(result, str):
                    result = result.split(',')
                if isinstance(result, float) and math.isnan(result):
                    result = None
                if result is None or len(result) == 0:
                    tags[idx].append(map_tag(order_key, ""))
                else:
                    for tag in result:
                        tag = tag.strip()
                        _tag = map_tag(order_key, tag)
                        if _tag not in cat_vocab_tags:
                            tags[idx].append(map_tag(order_key, ""))
                            if order_key not in unfamiliar_tags:
                                unfamiliar_tags[order_key] = []
                            if _tag not in unfamiliar_tags[order_key]:
                                unfamiliar_tags[order_key].append(_tag)
                        else:
                            tags[idx].append(_tag)

        # print('before tags: ', tags)
        tags = process_tags(tags)
        # print('after tags: ', tags)
        is_sinking = False
        if 'Sinking' in tags[8]:
            is_sinking = True
        return uniq_tags(tags), unfamiliar_tags, is_sinking

    elif music_tagging is not None:
        order = [
            {
                'AUDIO_CAT_VOCAB_IDX': 0,
                'META_KEY': 'Genre20',
                'ORDER_KEY': 'genre',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 1,
                'META_KEY': None,
                'ORDER_KEY': 'genre_extra',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 2,
                'META_KEY': None,
                'ORDER_KEY': 'extra',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 3,
                'META_KEY': 'Mood',
                'ORDER_KEY': 'mood',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 4,
                'META_KEY': 'Theme',
                'ORDER_KEY': 'scene',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 5,
                'META_KEY': None,
                'ORDER_KEY': 'vocal_gender',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 6,
                'META_KEY': None,
                'ORDER_KEY': 'vocal_timbre',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 7,
                'META_KEY': 'Language',
                'ORDER_KEY': 'language',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 8,
                'META_KEY': None,
                'ORDER_KEY': 'is_sinking',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 9,
                'META_KEY': None,
                'ORDER_KEY': 'instrument',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 10,
                'META_KEY': None,
                'ORDER_KEY': 'tempo',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 11,
                'META_KEY': None,
                'ORDER_KEY': 'key',
            },
            {
                'AUDIO_CAT_VOCAB_IDX': 12,
                'META_KEY': None,
                'ORDER_KEY': 'mode',
            },
        ]
        for order_map in order:
            idx = order_map['AUDIO_CAT_VOCAB_IDX']
            meta_key = order_map['META_KEY']
            order_key = order_map['ORDER_KEY']
            cat_vocab_tags = AUDIO_CAT_VOCAB_V3

            if order_key == 'is_sinking':
                tags[idx].append(map_tag(order_key, 'non-Sinking'))
            else:
                result = music_tagging.get(meta_key, {}).get('result', [])
                if mapping_tag.get(order_key, ''):
                    result = mapping_tag[order_key]
                if isinstance(result, str):
                    result = result.split(',')
                if result is None or len(result) == 0:
                    tags[idx].append(map_tag(order_key, ""))
                else:
                    for tag in result:
                        _tag = map_tag(order_key, tag)
                        if _tag not in cat_vocab_tags:
                            tags[idx].append(map_tag(order_key, ""))
                            if order_key not in unfamiliar_tags:
                                unfamiliar_tags[order_key] = []
                            if _tag not in unfamiliar_tags[order_key]:
                                unfamiliar_tags[order_key].append(_tag)
                        else:
                            tags[idx].append(_tag)

        is_sinking = False
        if ('MusicLowQuality' in music_tagging) and \
            (music_tagging["MusicLowQuality"]["Sinking"] > sinking_threshold):
            tags[8].append('Sinking')
        else:
            tags[8].append('non-Sinking')

        # print('before tags: ', tags)
        tags = process_tags(tags)
        # print('after tags: ', tags)
        if 'Sinking' in tags[8]:
            is_sinking = True

        return uniq_tags(tags), unfamiliar_tags, is_sinking
    else:
        return [""] * 13, {}, False


def _parse_inst_audio_tags_or_music_tagging_with_lang_filt(meta: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    audio_tags, music_tagging = _parse_audio_tags_or_audio_tag_or_music_tagging(meta)
    languages = []
    if audio_tags is not None:
        if 'satisfy_filter_standard' in audio_tags and audio_tags['satisfy_filter_standard'] != 'yes':
            raise ZhMetaParseError('audio_tags: satisfy_filter_standard is not yes')
        if audio_tags.get('language', ''):
            languages = audio_tags.get('language', '')
        else:
            languages = music_tagging.get('Language', {'result': ['Other']})['result']
        if languages is None:
            languages = ''
        if isinstance(languages, str):
            languages = languages.split(',')
        audio_tags['language'] = languages
        if not audio_tags.get('genre', ''):
            genres = music_tagging.get('Genre20', {'result': ['']})['result']
            if genres is None:
                genres = ''
            if isinstance(genres, str):
                genres = genres.split(',')
            audio_tags['genre'] = genres
        if not audio_tags.get('mood', ''):
            mood = music_tagging.get('Mood', {'result': ['']})['result']
            if mood is None:
                mood = ''
            if isinstance(mood, str):
                mood = mood.split(',')
            audio_tags['mood'] = mood
        if not audio_tags.get('scene', ''):
            scene = music_tagging.get('Theme', {'result': ['']})['result']
            if scene is None:
                scne = ''
            if isinstance(scene, str):
                scene = scene.split(',')
            audio_tags['scene'] = scene
    elif music_tagging is not None:
        languages = music_tagging.get('Language', {'result': ['Other']})['result']
        if languages is None:
            languages = ''
        if isinstance(languages, str):
            languages = languages.split(',')
    if not set(languages).issubset(['Non-vocal', 'Instrumental/Non-vocal']):
        raise ZhMetaParseError('audio_tags: Not Instrumental/Non-vocal')
    return audio_tags, music_tagging


def parse_inst_style_text_audio_tags_v3_or_music_tagging(meta: Dict, sinking_threshold: float):
    audio_tags, music_tagging = _parse_inst_audio_tags_or_music_tagging_with_lang_filt(meta)
    mapping_tag = mapping_tag_from_freeform(meta)
    return _parse_inst_audio_tags_v3(music_tagging, audio_tags, sinking_threshold, mapping_tag)


def parse_sstk_inst_tag(meta: Dict):
    instruments = meta.get('instruments', '')
    if not instruments or instruments == '\\N':
        return [""]
    instruments = instruments.split(',')
    instruments = [x.strip(',') for x in instruments]
    instruments = [x.strip() for x in instruments]
    instruments = [x.lower() for x in instruments]
    inst_tags = []
    for instrument in instruments:
        inst_tags.extend(SSTK_INST_TO_VOCAB.get(instrument, []))
    return inst_tags


def mapping_wyy_tag_optional(meta):
    """meta.raw.tags, meta.raw.category"""
    try:
        keywords = extract_keywords_wyy(meta)
        if not keywords:
            return {}
        tag_mapping = {
            "genre": [],
            "genre_extra": [],
            "extra": [],
            "mood": [],
            "scene": [],
            "instrument": [],
        }
        for x in keywords:
            if x in WYY_TAG_TO_GENRE:
                tag_mapping["genre"].extend(WYY_TAG_TO_GENRE[x])
            if x in WYY_TAG_TO_GENRE_EXTRA:
                tag_mapping["genre_extra"].extend(WYY_TAG_TO_GENRE_EXTRA[x])
            if x in WYY_TAG_TO_EXTRA:
                tag_mapping["extra"].extend(WYY_TAG_TO_EXTRA[x])
            if x in WYY_TAG_TO_MOOD:
                tag_mapping["mood"].extend(WYY_TAG_TO_MOOD[x])
            if x in WYY_TAG_TO_SCENE:
                tag_mapping["scene"].extend(WYY_TAG_TO_SCENE[x])
            if x in WYY_TAG_TO_INST:
                tag_mapping["instrument"].extend(WYY_TAG_TO_INST[x])
        return tag_mapping
    except Exception as e:
        logging.warning("error in mapping_wyy_tag_optional: ", e)
        return {}


def mapping_sstk_tag_optional(meta, parse_tempo=True):
    try:
        keywords, _, tempo = extract_keywords_sstk(meta)
        tag_mapping = {
            'genre': [],
            'genre_extra': [],
            'mood': [],
            'scene': [],
            'instrument': [],
            # 'vocal': [],
            'extra': [],
            'tempo': [],
        }
        for x in keywords:
            if x:
                if x in SSTK_MATCHED_TAG_TO_GENRE:
                    if not tag_mapping['genre']:
                        tag_mapping['genre'].extend(SSTK_MATCHED_TAG_TO_GENRE[x])
                    else:
                        tag_mapping['genre_extra'].extend(SSTK_MATCHED_TAG_TO_GENRE[x])
                if x in SSTK_MATCHED_TAG_TO_MOOD:
                    tag_mapping['mood'].extend(SSTK_MATCHED_TAG_TO_MOOD[x])
                if x in SSTK_MATCHED_TAG_TO_SCENE:
                    tag_mapping['scene'].extend(SSTK_MATCHED_TAG_TO_SCENE[x])
                if x in SSTK_MATCHED_TAG_TO_INST:
                    tag_mapping['instrument'].extend(SSTK_MATCHED_TAG_TO_INST[x])
                # if x in SSTK_TAG_TO_VOCAL:
                #     tag_mapping['vocal'].extend(SSTK_TAG_TO_VOCAL[x])
                if x in SSTK_TAG_TO_EXTRA:
                    tag_mapping['extra'].extend(SSTK_TAG_TO_EXTRA[x])
        for x in keywords:
            if x:
                if x in SSTK_NON_MATCHED_TAG_TO_GENRE:
                    if not tag_mapping['genre']:
                        tag_mapping['genre'].extend(SSTK_NON_MATCHED_TAG_TO_GENRE[x])
                    else:
                        if not tag_mapping['genre_extra']:
                            tag_mapping['genre_extra'].extend(SSTK_NON_MATCHED_TAG_TO_GENRE[x])
                if x in SSTK_NON_MATCHED_TAG_TO_MOOD and not tag_mapping['mood']:
                    tag_mapping['mood'].extend(SSTK_NON_MATCHED_TAG_TO_MOOD[x])
                if x in SSTK_NON_MATCHED_TAG_TO_SCENE and not tag_mapping['scene']:
                    tag_mapping['scene'].extend(SSTK_NON_MATCHED_TAG_TO_SCENE[x])
                if x in SSTK_NON_MATCHED_TAG_TO_INST and not tag_mapping['instrument']:
                    tag_mapping['instrument'].extend(SSTK_NON_MATCHED_TAG_TO_INST[x])

        if tempo and parse_tempo:
            tag_mapping["tempo"].append(tempo_to_label(float(tempo)))
        return tag_mapping

    except Exception as e:
        logging.warning("error in mapping_sstk_tag_optional: ", e)
        return {}


def mapping_everynoise_tag_optional(meta):
    """meta.raw.genres, meta.everynoise_genre, meta.everynoise_trending.vantage/genre"""
    try:
        keywords = extract_keywords_everynoise(meta)
        if not keywords:
            return {}
        tag_mapping = {
            'genre': [],
            'genre_extra': [],
        }
        for x in keywords:
            if x and x in EVERYNOISE_MATCHED_TAG_TO_CATEGORY:
                if not tag_mapping['genre']:
                    tag_mapping['genre'].extend(EVERYNOISE_MATCHED_TAG_TO_CATEGORY[x])
                else:
                    tag_mapping['genre_extra'].extend(EVERYNOISE_MATCHED_TAG_TO_CATEGORY[x])
        for x in keywords:
            if x and x in EVERYNOISE_NON_MATCHED_TAG_TO_CATEGORY:
                if not tag_mapping['genre']:
                    tag_mapping['genre'].extend(EVERYNOISE_NON_MATCHED_TAG_TO_CATEGORY[x])
                else:
                    if not tag_mapping['genre_extra']:
                        tag_mapping['genre_extra'].extend(EVERYNOISE_NON_MATCHED_TAG_TO_CATEGORY[x])
        return tag_mapping
    except Exception as e:
        logging.warning("error in mapping_everynoise_tag_optional: ", e)
        return {}


def mapping_apm_tag_optional(meta):
    try:
        keywords, _ = extract_keywords_apm(meta)
        if not keywords:
            return {}
        tag_mapping = {
            'genre': [],
            'genre_extra': [],
            'mood': [],
            'scene': [],
            'instrument': [],
            'gender': [],
            'extra': [],
        }
        for x in keywords:
            if x:
                if x in APM_MATCHED_TAG_TO_GENRE:
                    if not tag_mapping['genre']:
                        tag_mapping['genre'].extend(APM_MATCHED_TAG_TO_GENRE[x])
                    else:
                        tag_mapping['genre_extra'].extend(APM_MATCHED_TAG_TO_GENRE[x])
                if x in APM_MATCHED_TAG_TO_MOOD:
                    tag_mapping['mood'].extend(APM_MATCHED_TAG_TO_MOOD[x])
                if x in APM_MATCHED_TAG_TO_SCENE:
                    tag_mapping['scene'].extend(APM_MATCHED_TAG_TO_SCENE[x])
                if x in APM_MATCHED_TAG_TO_INST:
                    tag_mapping['instrument'].extend(APM_MATCHED_TAG_TO_INST[x])
                if x in APM_TAG_TO_VOCAL:
                    tag_mapping['gender'].extend(APM_TAG_TO_VOCAL[x])
                if x in APM_TAG_TO_EXTRA:
                    tag_mapping['extra'].extend(APM_TAG_TO_EXTRA[x])
        for x in keywords:
            if x:
                if x in APM_NON_MATCHED_TAG_TO_GENRE:
                    if not tag_mapping['genre']:
                        tag_mapping['genre'].extend(APM_NON_MATCHED_TAG_TO_GENRE[x])
                    else:
                        if not tag_mapping['genre_extra']:
                            tag_mapping['genre_extra'].extend(APM_NON_MATCHED_TAG_TO_GENRE[x])
                if x in APM_NON_MATCHED_TAG_TO_MOOD and not tag_mapping['mood']:
                    tag_mapping['mood'].extend(APM_NON_MATCHED_TAG_TO_MOOD[x])
                if x in APM_NON_MATCHED_TAG_TO_SCENE and not tag_mapping['scene']:
                    tag_mapping['scene'].extend(APM_NON_MATCHED_TAG_TO_SCENE[x])
                if x in APM_NON_MATCHED_TAG_TO_INST and not tag_mapping['instrument']:
                    tag_mapping['instrument'].extend(APM_NON_MATCHED_TAG_TO_INST[x])
        return tag_mapping
    except Exception as e:
        logging.warning("error in mapping_apm_tag_optional: ", e)
        return {}


def mapping_rym_tag_optional(meta):
    try:
        keywords = extract_keywords_rym(meta)
        if not keywords:
            return {}
        tag_mapping = {
            'genre': [],
            'genre_extra': [],
            'mood': [],
            'scene': [],
            'instrument': [],
            'gender': [],
            'extra': [],
        }
        for x in keywords:
            if x:
                if x in RYM_MATCHED_TAG_TO_GENRE:
                    if not tag_mapping['genre']:
                        tag_mapping['genre'].extend(RYM_MATCHED_TAG_TO_GENRE[x])
                    else:
                        tag_mapping['genre_extra'].extend(RYM_MATCHED_TAG_TO_GENRE[x])
                elif x in EVERYNOISE_MATCHED_TAG_TO_CATEGORY:   # current rym mapping list is in addition to everynoise mapping
                    if not tag_mapping['genre']:
                        tag_mapping['genre'].extend(EVERYNOISE_MATCHED_TAG_TO_CATEGORY[x])
                    else:
                        tag_mapping['genre_extra'].extend(EVERYNOISE_MATCHED_TAG_TO_CATEGORY[x])
                if x in RYM_MATCHED_TAG_TO_MOOD:
                    tag_mapping['mood'].extend(RYM_MATCHED_TAG_TO_MOOD[x])
                if x in RYM_MATCHED_TAG_TO_SCENE:
                    tag_mapping['scene'].extend(RYM_MATCHED_TAG_TO_SCENE[x])
                if x in RYM_MATCHED_TAG_TO_INST:
                    tag_mapping['instrument'].extend(RYM_MATCHED_TAG_TO_INST[x])
                if x in RYM_TAG_TO_VOCAL:
                    tag_mapping['gender'].extend(RYM_TAG_TO_VOCAL[x])
                if x in RYM_TAG_TO_EXTRA:
                    tag_mapping['extra'].extend(RYM_TAG_TO_EXTRA[x])
        for x in keywords:
            if x:
                if x in RYM_NON_MATCHED_TAG_TO_GENRE:
                    if not tag_mapping['genre']:
                        tag_mapping['genre'].extend(RYM_NON_MATCHED_TAG_TO_GENRE[x])
                    else:
                        if not tag_mapping['genre_extra']:
                            tag_mapping['genre_extra'].extend(RYM_NON_MATCHED_TAG_TO_GENRE[x])
                if x in EVERYNOISE_NON_MATCHED_TAG_TO_CATEGORY:
                    if not tag_mapping['genre']:
                        tag_mapping['genre'].extend(EVERYNOISE_NON_MATCHED_TAG_TO_CATEGORY[x])
                    else:
                        if not tag_mapping['genre_extra']:
                            tag_mapping['genre_extra'].extend(EVERYNOISE_NON_MATCHED_TAG_TO_CATEGORY[x])
                if x in RYM_NON_MATCHED_TAG_TO_MOOD and not tag_mapping['mood']:
                    tag_mapping['mood'].extend(RYM_NON_MATCHED_TAG_TO_MOOD[x])
                if x in RYM_NON_MATCHED_TAG_TO_SCENE and not tag_mapping['scene']:
                    tag_mapping['scene'].extend(RYM_NON_MATCHED_TAG_TO_SCENE[x])
                if x in RYM_NON_MATCHED_TAG_TO_INST and not tag_mapping['instrument']:
                    tag_mapping['instrument'].extend(RYM_NON_MATCHED_TAG_TO_INST[x])
        return tag_mapping
    except Exception as e:
        logging.warning("error in mapping_rym_tag_optional: ", e)
        return {}


def mapping_tag_from_freeform(meta):
    mapping_tag = {}

    def mapping_freeform_meta_optional(fn):
        for k, v in fn(meta).items():
            if k not in mapping_tag:
                mapping_tag[k] = v
            else:
                mapping_tag[k].extend(v)

    mapping_freeform_meta_optional(mapping_wyy_tag_optional)
    mapping_freeform_meta_optional(mapping_sstk_tag_optional)
    mapping_freeform_meta_optional(mapping_everynoise_tag_optional)
    mapping_freeform_meta_optional(mapping_apm_tag_optional)
    mapping_freeform_meta_optional(mapping_rym_tag_optional)

    for k in mapping_tag:
        v = list(set(mapping_tag[k]))
        mapping_tag[k] = v
    return mapping_tag


def parse_freeform(
        meta, 
        style_text, 
        keyword_dropout=0.2, 
        style_text_rate=0.5
    ):
    """
    keyword_dropout: dropout rate for keywords
    style_text_rate: rate to include freeform from style_text or not
    """
    freeform_text_style_text = infer_freeform_text_from_style_text(style_text)
    #print('freeform_text_style_text: ', freeform_text_style_text)
    freeform_text_wyy = parse_freeform_text_wyy_optional(meta)
    #print('freeform_text_wyy: ', freeform_text_wyy)
    freeform_text_everynoise = parse_freeform_text_everynoise_optional(meta)
    #print('freeform_text_everynoise: ', freeform_text_everynoise)
    freeform_text_groupAB = parse_freeform_text_groupAB_optional(meta)
    #print('freeform_text_groupAB: ', freeform_text_groupAB)
    freeform_text_sstk_short, freeform_text_sstk_long = parse_freeform_text_sstk(meta)
    #print('freeform_text_sstk: ', freeform_text_sstk)
    freeform_text_apm_short, freeform_text_apm_long = parse_freeform_text_apm(meta)
    #print('freeform_text_apm: ', freeform_text_apm)
    freeform_text_rym = parse_freeform_text_rym(meta)
    #print('freeform_text_rym: ', freeform_text_rym)

    # if any long description is extracted, do not append style_text keywords and shuffle
    if freeform_text_sstk_long or freeform_text_apm_long:
        freeform_text = freeform_text_sstk_long if freeform_text_sstk_long else freeform_text_apm_long
        return freeform_text

    freeform_list = []
    for f in [
        freeform_text_wyy, 
        freeform_text_everynoise, 
        freeform_text_groupAB, 
        freeform_text_sstk_short,
        freeform_text_apm_short,
        freeform_text_rym,
        ]:
        if f:
            for x in f.split(','):
                freeform_list.append(x.strip())
    if random.random() < style_text_rate:
        if freeform_text_style_text:
            for x in freeform_text_style_text.split(','):
                freeform_list.append(x.strip())
            if random.random() < 0.2:       # in a small chance, only use style_text in freeform
                freeform_list = freeform_text_style_text.split(',')
    freeform_list = list(set(freeform_list))
    freeform_list = sample_pct(freeform_list, dropout=keyword_dropout)
    freeform_text = ', '.join(freeform_list) if freeform_list else None

    return freeform_text


def parse_bpm(meta):
    try:
        # if platform
        platform = meta.get("raw", {}).get("platform", "")
        if platform in ['dq', 'wyy', 'mcc']:
            return None
        #library = meta.get("library", "")
        #if library not in ['pond5', 'shutterstock']: # only read bpm from sstk now
        #    return None

        bpm = meta.get('bpm', None)
        if not bpm:
            bpm = meta.get('BPM', None)
        if bpm and bpm != '\\N':
            return float(bpm)
    except Exception as e:
        logging.warning("error in parse_bpm: ", e)
        return None
