import os
import numpy as np
import torch
import webdataset as wds
from typing import Callable
from recipes.bigmusic.datasets.utils.zh_meta import _get_value
from recipes.bigmusic.datasets.transforms.lyrics import pad_crop

CHORD_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_TYPES = ["maj", "min", "aug", "dim", "sus4", "sus2"]
CHORD_NAMES = ["N"] + [r+":"+t for t in CHORD_TYPES for r in CHORD_ROOTS ]


def get_beat_chord(metadata, beat_resolution=2):
    try:
        intervals = _get_value(_get_value(_get_value(metadata, 'mir_service'), 'chord'), 'intervals')
        chords = _get_value(_get_value(_get_value(metadata, 'mir_service'), 'chord'), 'chords')
        beat = _get_value(_get_value(_get_value(metadata, 'mir_service'), 'beat'), 'beat')
    except:
        return None

    beat_chord = []
    idx_ch = 0
    if beat_resolution == 1:
        idx_beat_anchor = range(len(beat))
    if beat_resolution == 2:
        idx_beat_anchor =  [i for i, b in enumerate(beat) if b[1] in (1, 3)]
    for idx_b in idx_beat_anchor:
        t_cur = beat[idx_b][0]
        t_next = beat[idx_b+1][0] if idx_b < len(beat)-1 else t_cur
        while idx_ch < len(intervals) and intervals[idx_ch][1] <= t_cur:
            idx_ch += 1
        if idx_ch == len(intervals):     # beat approaches end, no chord labels
            label = "N"
        else:
            label = chords[idx_ch] if (intervals[idx_ch][1] - t_cur) > (t_next - intervals[idx_ch][1]) else chords[idx_ch+1]
        beat_chord.append([t_cur, beat[idx_b][1], label])

    return beat_chord


def crop_beat_chord(beat_chord, start, end):

    beat_chord = [x for x in beat_chord if start<=x[0]<end]
    return beat_chord


def crop_beat_chord_at_downbeat(beat_chord):
    
    beats = [x[1] for x in beat_chord]
    indices_downbeat = [i for i, beat in enumerate(beats) if beat == 1]
    idx_first_downbeat = indices_downbeat[0] if indices_downbeat else None
    idx_last_downbeat = indices_downbeat[-1] if indices_downbeat else None
    try:
        beat_chord = beat_chord[idx_first_downbeat: idx_last_downbeat]
    except:
        beat_chord = []
    return beat_chord


def extract_chord_seq(metadata, start, end, mode="on_downbeat"):
    
    try:
        beat_chord = _get_value(metadata, 'beat_chord')
    except:
        return []

    beat_chord = crop_beat_chord(beat_chord, start, end)
    if mode == "on_downbeat":
        beat_chord = crop_beat_chord_at_downbeat(beat_chord)
    return [x[2] for x in beat_chord]


class ChordSeqTokenTransform():

    def __init__(self, pad_id=0, chord_max_seq_len=50, item_key="chord_seq", chord_mode="raw_sequence", handler:Callable=wds.ignore_and_continue):
        
        self.pad_id = pad_id
        self.chord_max_seq_len = chord_max_seq_len
        self.item_key = item_key
        self.chord_mode = chord_mode
        assert chord_mode in ["raw_sequence"]
        self.handler = handler

    def chord2id(self, chord_seq):
        return [CHORD_NAMES.index(x) if x in CHORD_NAMES else "N" for x in chord_seq]
    
    def __call__(self, item):
        
        try:
            chord_seq = item[self.item_key]
            chord_seq_tokens = torch.tensor(self.chord2id(chord_seq), dtype=torch.long)
            chord_seq_lengths = len(chord_seq_tokens)
            chord_seq_tokens = pad_crop(
                chord_seq_tokens, 
                self.chord_max_seq_len, 
                chord_seq_tokens.dtype, 
                padding_value=0,
                )
            return {**item, "chord_seq_tokens": chord_seq_tokens, "chord_seq_lengths": chord_seq_lengths}
        except Exception as e:
            self.handler(e)
            return None