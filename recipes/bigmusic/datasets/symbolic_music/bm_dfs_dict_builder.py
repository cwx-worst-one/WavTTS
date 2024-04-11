from typing import List, Dict, Any
import json

import numpy as np
import pandas as pd

from torchaudio_augmentations import Compose
from samantha.transforms.audio import (
    SetAudioDimensions,
    ToTensor,
)

from recipes.bigmusic.utils.audio_utils import audio_bytes_to_array
from recipes.bigmusic.utils.common_utils import colorful_sequence


class BMDfsDictBuilder:
    def __init__(self, item):
        self.item = item
        self.output_dict = {}
        self.error_dict = {}
        self.audio_transforms = Compose(
            [ToTensor(), SetAudioDimensions()]
        )

    def pre_load_meta(self):
        if isinstance(self.item["meta"], str):
            self.item["meta"] = json.loads(self.item["meta"])
        return self

    def add_key(self):
        self.output_dict["key"] = self.item["uttid"]
        return self

    def add_df_lyrics(self):
        try:
            df_lyrics = pd.DataFrame(
                self.item["meta"]["lyrics"]["result"][0]["utterances"]
            )
            df_lyrics = df_lyrics[df_lyrics.phoneme.apply(
                lambda x: x is not None
            )].reset_index(drop=True)
            self.output_dict["df_lyrics"] = df_lyrics
        except Exception as e:
            self.output_dict["df_lyrics"] = None
            self.error_dict["df_lyrics"] = repr(e)
        return self
    
    def add_df_vocal2midi(self):
        try:
            df_vocal2midi = pd.DataFrame(self.item["meta"]["midi"])
            df_vocal2midi["pitch"] = df_vocal2midi["pitch"].astype(int)
            self.output_dict["df_vocal2midi"] = df_vocal2midi
        except Exception as e:
            self.output_dict["df_vocal2midi"] = None
            self.error_dict["df_vocal2midi"] = repr(e)
        return self

    def add_df_beat(self):
        try:
            df_beat = pd.DataFrame(
                self.item["meta"]["mir_service"]["beat"]["beat"],
                columns=["time", "beat"],
            )
            df_beat["beat"] = df_beat.beat.astype(float).astype(int)
            # self.output_dict["df_beat"] = fix_double_beat_tempo(df_beat)
            self.output_dict["df_beat"] = df_beat
        except Exception as e:
            self.output_dict["df_beat"] = None
            self.error_dict["df_beat"] = repr(e)
        return self

    def add_df_chord(self):
        try:
            intervals = self.item["meta"]["mir_service"]["chord"]["intervals"]
            chords = self.item["meta"]["mir_service"]["chord"]["chords"]
            assert len(intervals) == len(chords), "chords and intervals are not the same length"
            df_chord = pd.DataFrame(
                intervals,
                columns=["start", "end"],
            )
            df_chord["chord"] = chords
            self.output_dict["df_chord"] = df_chord
        except Exception as e:
            self.output_dict["df_chord"] = None
            self.error_dict["df_chord"] = repr(e)
        return self

    def add_df_note(self, subsets: List[str] = ["vocal"]):
        """Possible subsets: 'vocal', 'bass', 'drums', 'guitar', 'piano'"""
        try:
            trans_5stem_res = self.item["meta"]["mir_service"]["trans_5stem"]
            dfs = []
            for s in subsets:
                df_n = pd.DataFrame(trans_5stem_res["notes"][s])
                if len(df_n) > 0:
                    df_n["stem"] = s
                    dfs.append(df_n)
            if len(dfs) == 0:
                raise ValueError("No note detected")
            df_note = pd.concat(dfs)
            df_note["pitch"] = df_note["pitch"].astype(int)
            self.output_dict["df_note"] = df_note
        except Exception as e:
            self.output_dict["df_note"] = None
            self.error_dict["df_note"] = repr(e)
        return self
        
    def add_audio(self, audio_key="wav"):
        try:
            audio_ori = self.audio_transforms(self.item[audio_key]).numpy()
            sr = self.item['src_sample_rate']
            self.output_dict["audio"] = (audio_ori, sr)
        except Exception as e:
            self.output_dict["audio"] = None
            self.error_dict["audio"] = repr(e)
        return self

    def add_df_section(self):
        try:
            df_section = pd.DataFrame(
                self.item["meta"]["mir_service"]["structure"]["merged"],
                columns=["interval", "function_name"],
            )
            df_section["start_time"] = df_section.interval.apply(lambda x: x[0])
            df_section["end_time"] = df_section.interval.apply(lambda x: x[1])
            df_section = df_section.drop(columns=["interval"])
            if len(df_section) == 0:
                raise ValueError("No section detected")
            self.output_dict["df_section"] = df_section
        except Exception as e:
            self.output_dict["df_section"] = None
            self.error_dict["df_section"] = repr(e)
        return self

    def add_df_key(self):
        try:
            df_key = pd.DataFrame(
                self.item["meta"]["mir_service"]["key"]["time"], columns=["time", "key"]
            )
            self.output_dict["df_key"] = df_key
        except Exception as e:
            self.output_dict["df_key"] = None
            self.error_dict["df_key"] = repr(e)
        return self
    
    def quantize_chord_to_beat(self):
        """This will add a "chord" column to df_beat"""
        self._check_error()
        try:
            df_beat, df_chord = (
                self.output_dict["df_beat"],
                self.output_dict["df_chord"],
            )
            assert len(df_beat) >= 2, "Need at least 2 beats"
            assert len(df_chord) >= 1, "Need at least 1 chord"
            us = [*df_beat.time, 2 * df_beat.iloc[-1].time - df_beat.iloc[-2].time]
            cs = [*df_chord.start, df_chord.iloc[-1].end]
            labels = np.array([*df_chord.chord, "N"])
            df_beat["chord"] = labels[colorful_sequence(cs, us)]
            self.output_dict["df_beat"] = df_beat
        except Exception as e:
            self.error_dict["quantize_chord_to_beat"] = repr(e)
        return self

    def quantize_key_to_beat(self):
        """This will add a "key" column to df_beat"""
        self._check_error()
        try:
            df_beat, df_key = (
                self.output_dict["df_beat"],
                self.output_dict["df_key"],
            )
            assert len(df_beat) >= 2, "Need at least 2 beats"
            assert len(df_key) >= 1, "Need at least 1 key"
            us = [*df_beat.time, 2 * df_beat.iloc[-1].time - df_beat.iloc[-2].time]
            cs = df_key.time.to_list()
            labels = np.array([*df_key.key, "N"])
            df_beat["key"] = labels[colorful_sequence(cs, us)]
            self.output_dict["df_beat"] = df_beat
        except Exception as e:
            self.error_dict["quantize_key_to_beat"] = repr(e)
        return self

    def quantize_key_to_section(self):
        """This will add a "key" column to df_section, use majority vote
        This is recommended v.s. quantize_key_to_beat since
        key info is more global.
        """
        self._check_error()
        try:
            df_section, df_key = (
                self.output_dict["df_section"],
                self.output_dict["df_key"],
            )
            assert len(df_section) >= 1, "Need at least 1 section"
            section_keys = []
            for i, row in df_section.iterrows():
                key_list: pd.Series = df_key[df_key.time.apply(
                    lambda x: x >= row.start_time and x <= row.end_time
                )].key
                if len(key_list) > 0:
                    section_keys.append(key_list.value_counts().index[0])
                else:
                    section_keys.append("N")
            df_section["key"] = section_keys
            self.output_dict["df_section"] = df_section
        except Exception as e:
            self.error_dict["quantize_key_to_section"] = repr(e)
        return self

    def quantize_section_to_downbeat(self):
        """Quantize section boundary to downbeat.
        This function will anchor section boundary to the closest downbeat,
        and remove any section that has 0 duration afterwards.
        """
        self._check_error()
        try:
            df_section, df_beat = (
                self.output_dict["df_section"],
                self.output_dict["df_beat"],
            )
            downbeat_times = df_beat[df_beat.beat == 1].time.values
            idx = 0
            query_times = [*df_section.start_time, df_section.end_time.iloc[-1]]
            anchor_times = []
            for qt in query_times:
                while idx + 1 < len(downbeat_times) and downbeat_times[idx + 1] < qt:
                    idx += 1
                # Check if the current or next number is closer to the target
                if idx + 1 < len(downbeat_times) and abs(downbeat_times[idx + 1] - qt) < abs(downbeat_times[idx] - qt):
                    anchor_times.append(downbeat_times[idx + 1])
                else:
                    anchor_times.append(downbeat_times[idx])
            while len(anchor_times) < len(query_times):
                anchor_times.append(downbeat_times[idx])
            df_section["start_time"] = anchor_times[:-1]
            df_section["end_time"] = anchor_times[1:]
            df_section = df_section[df_section.apply(
                lambda x: x.end_time - x.start_time > 0, axis=1
            )].reset_index(drop=True)
            self.output_dict["df_section"] = df_section
        except Exception as e:
            self.error_dict["quantize_section_to_downbeat"] = repr(e)
        return self
    
    def _check_error(self):
        error_str = "; ".join(
            [f"{k}: {v}" for k, v in self.error_dict.items() if len(v) > 0]
        )
        assert len(error_str) == 0, error_str

    def create_output(self) -> Dict[str, Any]:
        self._check_error()
        return self.output_dict
