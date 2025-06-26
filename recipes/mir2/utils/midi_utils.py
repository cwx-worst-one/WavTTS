import logging
import os
import time
import math
import json
import pretty_midi
from copy import deepcopy
from pathlib import Path
from functools import partial
from itertools import accumulate
from statistics import median, mean
from collections import namedtuple

from music21 import converter, stream, note, tempo, exceptions21
from mido import MidiFile, MidiTrack, MetaMessage, Message, bpm2tempo


logger = logging.getLogger(__name__)


Note = namedtuple('Note', ('pitch', 'start', 'duration', 'velocity'))  # start and duration are in "beats"


def make_parent_dir(fp):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)


def iter_midi_in_dir(dir):
    for root, _, fps in os.walk(dir):
        for fp in fps:
            if fp.endswith(".mid"):
                yield Path(root) / fp


def load_stream(score_fp):
    return converter.parse(score_fp)


def load_midi(score_fp):
    return MidiFile(score_fp)


def save_midi(score, out_fp):
    """
    :param score: mido.MidiFile object
    :param out_fp: The output path
    """
    make_parent_dir(out_fp)
    score.save(out_fp)


def save_stream(score, out_fp):
    """
    :param score: A music21.stream.Stream object
    :param out_fp: The output path
    """
    fmt = os.path.splitext(out_fp)[-1][1:]
    make_parent_dir(out_fp)
    try:
        score.write(fmt, out_fp)
    except exceptions21.StreamException:  # TODO: this is a hacky workaround
        part_score = score.elements[0]
        part_score.write(fmt, out_fp)


def load_json(json_fp):
    with open(json_fp, 'r') as f:
        j = json.load(f)
    return j


def dump_json(data, json_fp):
    make_parent_dir(json_fp)
    with open(json_fp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def extract_lyrics(score):
    """
    Extract lyrics from the given score. The lyrics are separated by measure
    :param score: A music21.stream.Stream object
    :return: A list of lyrics, each element is the lyrics of a measure
    """
    lyrics = []
    for element in score.recurse():
        if isinstance(element, stream.Measure):
            phrase = ''.join(n.lyric for n in element if isinstance(n, note.GeneralNote) and n.lyric)
            lyrics.append(phrase)
    return lyrics


def generate_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def set_stream_bpm(score, bpm):
    mark = tempo.MetronomeMark(number=bpm)
    for i in score.recurse():
        if isinstance(i, stream.Measure):
            has_mark = False
            for j in i:
                if isinstance(j, tempo.MetronomeMark):
                    j.number = bpm
                    has_mark = True
                    break
            if not has_mark:
                i.insert(0, mark)
            break
    # Fallback case - if no measures detected, insert BPM at front. (fix for music21 <6.7.1)
    has_metronome = any([isinstance(i, tempo.MetronomeMark) for i in score.recurse()])
    if not has_metronome:
        score.insert(0, mark)

def parse_score_file(score_fp, fmt='xml', part=None, stack_tracks=False):
    track_names = ('melody', 'chords')

    def get_part(score, part_name):
        if isinstance(part_name, int):
            return stream.Stream(score.parts[part_name])
        else:
            return stream.Stream(next(filter(lambda p: p.partName == part_name, score.parts)))

    def get_track(midi, track_name):
        def get_track_by_name(midi, track_name):
            return next(filter(lambda t: t.name == track_name, midi.tracks))

        def get_track_by_id(midi, track_id):
            return midi.tracks[track_id]

        if isinstance(track_name, int):
            track = get_track_by_id(midi, track_name)
        else:
            track = get_track_by_name(midi, track_name)
        meta_track = get_track_by_name(midi, '')
        new_midi = MidiFile()
        new_midi.tracks = [meta_track, track]
        return new_midi

    if fmt == 'xml':
        score = load_stream(score_fp)
        if part is None and stack_tracks:
            return score
        elif part is None and not stack_tracks:
            return list(map(partial(get_part, score), track_names))
        else:
            return get_part(score, part)
    elif fmt in ('mid', 'midi'):
        midi_file = MidiFile(score_fp)
        if part is None and stack_tracks:
            return midi_file
        elif part is None and not stack_tracks:
            return list(map(partial(get_track, midi_file), track_names))
        else:
            return get_track(midi_file, part)


# ---- MIDI ----


def is_beg_meta(msg):
    return isinstance(msg, MetaMessage) and (msg.type in ('time_signature', 'key_signature', 'set_tempo'))


def is_note(msg):
    return isinstance(msg, Message) and (msg.type in ('note_on', 'note_off'))


def is_note_on(msg):
    return isinstance(msg, Message) and msg.type == 'note_on' and msg.velocity != 0


def is_note_off(msg):
    if isinstance(msg, Message) and msg.type == 'note_off':
        return True
    elif isinstance(msg, Message) and msg.type == 'note_on' and msg.velocity == 0:
        return True
    return False


def is_end_meta(msg):
    return isinstance(msg, MetaMessage) and (msg.type == 'end_of_track')


def lfilter_beg_track(track):
    return list(filter(lambda msg: is_beg_meta(msg) or is_note(msg), track))


def lfilter_mid_track(track):
    return list(filter(is_note, track))


def lfilter_end_track(track):
    return list(filter(lambda msg: is_note(msg) or is_end_meta(msg), track))


def set_midi_bpm(midi_file, bpm):
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                msg.tempo = bpm2tempo(bpm)
                break
        else:  # if no "set_tempo" message, insert one after 'track_name'
            track_name_msg_ind = [msg_idx for msg_idx, msg in enumerate(track) if msg.type == 'track_name']
            idx_to_insert = min(track_name_msg_ind) if track_name_msg_ind else 0
            track.insert(idx_to_insert, MetaMessage('set_tempo', tempo=bpm2tempo(bpm), time=0))


def check_ticks_per_beat(midi_files):
    tpbs = [mf.ticks_per_beat for mf in midi_files]
    if tpbs.count(tpbs[0]) != len(tpbs):
        raise ValueError('MidiFiles have different ticks per beat values')


def check_num_tracks(midi_files):
    num_tracks = [len(mf.tracks) for mf in midi_files]
    for x in num_tracks[1:]:
        if x != num_tracks[0]:
            raise ValueError('MIDI files have different number of tracks')


def find_not_complete_notes(track):
    """
    :return: not_started_notes, not_ended_notes
    """
    not_started_notes = []
    not_ended_notes = []
    for pitch in range(128):
        note_msgs = [msg for msg in track if is_note(msg) and msg.note == pitch]
        if note_msgs:
            if is_note_off(note_msgs[0]):
                not_started_notes.append(pitch)
            if is_note_on(note_msgs[-1]):
                not_ended_notes.append(pitch)

    return not_started_notes, not_ended_notes


def check_note_on_off_balanced(track):
    a, b = find_not_complete_notes(track)
    if a or b:
        raise ValueError('Note ons and offs are not balanced ')


def get_abs_time_stamps(track, msg_type=None):
    msg_abs_t = accumulate(msg.time for msg in track)
    msg_abs_tps = [(abs_t, msg.type) for abs_t, msg in zip(msg_abs_t, track)]
    filter_fn = lambda msg_tp: msg_tp[1] == msg_type if msg_type else lambda _: True
    return [msg_tp[0] for msg_tp in msg_abs_tps if filter_fn(msg_tp)]


def get_length_in_ticks(track):
    return sum(msg.time for msg in track)


def get_onsets_time_stamps(track):
    return get_abs_time_stamps(track, 'note_on')


def get_offsets_time_stamps(track):
    return get_abs_time_stamps(track, 'note_off')


def get_num_notes_in_track(track):
    return len([msg for msg in track if is_note_on(msg)])


def get_num_bars(track, ticks_per_beat, beats_per_bar=4):
    return get_length_in_ticks(track) / ticks_per_beat / beats_per_bar


def is_end_bars_mergeable(midi_file_a, midi_file_b, num_bars=1):
    """ Check if midi_file_b's first bar can be merged into midi_file_a's last bar """
    def get_empty_ticks_beg(track):
        tss = get_onsets_time_stamps(track)
        return tss[0] if tss else expected_overlapped_ticks

    def get_empty_ticks_end(track):
        track_length_in_bars = math.ceil(get_length_in_ticks(track) / ticks_per_bar)
        tss = get_offsets_time_stamps(track)
        return track_length_in_bars * ticks_per_bar - tss[-1] if tss else expected_overlapped_ticks

    check_ticks_per_beat([midi_file_a, midi_file_b])
    check_num_tracks([midi_file_a, midi_file_b])

    beats_per_bar = 4
    ticks_per_beat = midi_file_a.ticks_per_beat
    ticks_per_bar = ticks_per_beat * beats_per_bar
    expected_overlapped_ticks = ticks_per_bar * num_bars

    min_end_ticks = min(get_empty_ticks_end(track) for track in midi_file_a.tracks)
    min_beg_ticks = min(get_empty_ticks_beg(track) for track in midi_file_b.tracks)
    if min_end_ticks + min_beg_ticks < expected_overlapped_ticks:
        return False
    return True


def merge_end_bars(midi_file_a, midi_file_b, num_bars=1):
    def change_time_delta_to_abs(midi_file):
        for track in midi_file.tracks:
            for msg, abs_t in zip(track, get_abs_time_stamps(track)):
                msg.time = abs_t

    def change_time_abs_to_delta(midi_file):
        for track in midi_file.tracks:
            prev_t = 0
            for msg in track:
                now_t = msg.time
                msg.time -= prev_t
                prev_t = now_t

    def shift_tick_for_abs_time(midi_file, ticks):
        for track in midi_file.tracks:
            for msg in track:
                msg.time += ticks

    if not is_end_bars_mergeable(midi_file_a, midi_file_b):
        raise ValueError("The given MIDI files are not mergeable")
    beats_per_bar = 4
    ticks_per_beat = midi_file_a.ticks_per_beat
    ticks_per_bar = ticks_per_beat * beats_per_bar
    expected_overlapped_ticks = ticks_per_bar * num_bars

    midi_a_num_bars = math.ceil(max(get_num_bars(track, ticks_per_beat, beats_per_bar) for track in midi_file_a.tracks))
    midi_b_tick_shift = (midi_a_num_bars - 1) * expected_overlapped_ticks
    num_tracks = len(midi_file_a.tracks)
    track_names = [track.name for track in midi_file_a.tracks]

    midi_file_a = deepcopy(midi_file_a)
    midi_file_b = deepcopy(midi_file_b)
    for i in range(num_tracks):
        midi_file_a.tracks[i] = lfilter_beg_track(midi_file_a.tracks[i])
    for i in range(num_tracks):
        midi_file_b.tracks[i] = lfilter_end_track(midi_file_b.tracks[i])
    change_time_delta_to_abs(midi_file_a)
    change_time_delta_to_abs(midi_file_b)
    shift_tick_for_abs_time(midi_file_b, midi_b_tick_shift)

    merged_midi_file = MidiFile()
    merged_midi_file.ticks_per_beat = ticks_per_beat
    for track_a, track_b, track_name in zip(midi_file_a.tracks, midi_file_b.tracks, track_names):
        new_track = MidiTrack()
        new_track.name = track_name
        new_track.extend(track_a + track_b)
        merged_midi_file.tracks.append(new_track)
    change_time_abs_to_delta(merged_midi_file)

    return merged_midi_file


def stitch_midi_files(midi_files):
    """
    :param midi_files: a list of MidiFile objects
    :return: a MidiFile object combined with all the input midi files
    """

    def get_missed_ticks(midi_file, prev_missed_ticks):
        num_ticks = []
        for track, prev_missed_tick in zip(midi_file.tracks, prev_missed_ticks):
            note_times = [msg.time for msg in track if is_note(msg)]
            num_ticks.append(sum(note_times) if note_times else -prev_missed_tick)
        num_supposed_tick = max(get_length_in_ticks(track) for track in midi_file.tracks)
        missed_ticks = [num_supposed_tick - nt for nt in num_ticks]
        return missed_ticks

    check_ticks_per_beat(midi_files)
    check_num_tracks(midi_files)

    midi_files = deepcopy(midi_files)
    merged_mid = MidiFile()
    merged_mid.ticks_per_beat = midi_files[0].ticks_per_beat
    merged_mid.tracks = []
    for track in midi_files[0].tracks:
        new_track = MidiTrack()
        new_track.name = track.name
        merged_mid.tracks.append(new_track)

    # combine messages
    missed_ticks = [0] * len(midi_files[0].tracks)
    for i, midi_file in enumerate(midi_files):
        if i == 0:
            filtered_tracks = list(map(lfilter_beg_track, midi_file.tracks))
        elif i == len(midi_files) - 1:
            filtered_tracks = list(map(lfilter_end_track, midi_file.tracks))
        else:
            filtered_tracks = list(map(lfilter_mid_track, midi_file.tracks))

        for j, (m_track, f_track) in enumerate(zip(merged_mid.tracks, filtered_tracks)):
            for msg_idx, msg in enumerate(f_track):
                msg = deepcopy(msg)
                if msg_idx == 0:
                    msg.time += missed_ticks[j]
                m_track.append(msg)

        missed_ticks = get_missed_ticks(midi_file, missed_ticks)
    return merged_mid


def stack_tracks_in_files(midi_files):
    check_ticks_per_beat(midi_files)
    stacked_midi = MidiFile()
    stacked_midi.ticks_per_beat = midi_files[0].ticks_per_beat
    stacked_midi.tracks = [track for midi_file in midi_files for track in midi_file.tracks]
    return stacked_midi


def chunk_midi(midi_file, bar_shift, num_bars, split_tie=True):
    def find_start_end(track_ts, start_tick, end_tick):
        start_idx = None
        for i in range(len(track_ts)):
            if track_ts[i] >= start_tick:
                start_idx = i
                break
        end_idx = None
        for i in reversed(range(len(track_ts))):
            if track_ts[i] <= end_tick:
                end_idx = i
                break
        if None in [start_idx, end_idx]:
            return slice(0)
        return slice(start_idx, end_idx+1)

    def make_note(note_type, pitch, delta_time):
        note_msg = Message(note_type)
        note_msg.note = pitch
        note_msg.time = delta_time
        return note_msg

    def add_msg(track, msg, time_stamp, start_tick, tick_shift):
        msg.time = time_stamp - start_tick - tick_shift
        tick_shift += msg.time
        track.append(msg)
        return tick_shift

    def get_section(section, time_stamps, start_tick):
        section = deepcopy(section)
        for msg, time_stamp in zip(section, time_stamps):
            if time_stamp == start_tick:
                if is_note_on(msg):  # only note_on will be chosen at start tick
                    msg.time = time_stamp - start_tick
                    break
            else:
                if is_note(msg):
                    msg.time = time_stamp - start_tick
                    break
        return section

    def add_notes_into_track(new_track, section, section_ts, start_tick, end_tick, tick_shift):
        for msg, time_stamp in zip(section, section_ts):
            if is_note(msg):
                if (time_stamp == start_tick and is_note_off(msg)) or \
                        (time_stamp == end_tick and is_note_on(msg)):
                    continue
                tick_shift = add_msg(new_track, msg, time_stamp, start_tick, tick_shift)
        return tick_shift

    def add_notes_to_split_tie(track, start_tick, end_tick, tick_shift):
        # FIXME: for now this algorithm doesn't take "channel" into account
        not_started_notes, not_ended_notes = find_not_complete_notes(track)
        # add note_ons at the beginning
        if not_started_notes:
            idx = next(idx for idx, msg in enumerate(track) if is_note(msg))
            for note in not_started_notes:
                note_on_msg = make_note('note_on', note, 0)
                track.insert(idx, note_on_msg)
        # add note_offs at the end
        delta_time = (end_tick - start_tick) - tick_shift
        for idx, note in enumerate(not_ended_notes):
            note_off_msg = make_note('note_off', note, delta_time if idx == 0 else 0)
            tick_shift += note_off_msg.time
            track.append(note_off_msg)
        return tick_shift

    def add_beg_meta(track, beg_meta):
        beg_meta = deepcopy(beg_meta)
        for m in beg_meta:
            m.time = 0
        track.extend(beg_meta)

    def add_end_meta(track, end_meta, section_len, tick_shift):
        end_meta = deepcopy(end_meta)
        end_meta.time = section_len - tick_shift
        track.append(end_meta)

    midi_file = deepcopy(midi_file)

    beg_meta_msgs = [list(filter(is_beg_meta, track)) for track in midi_file.tracks]
    end_meta_msgs = [next(filter(is_end_meta, track)) for track in midi_file.tracks]

    # num_sections, ticks_per_section, abs_time_stamps = get_time_info(midi_file)
    beats_per_bar = 4
    ticks_per_bar = midi_file.ticks_per_beat * beats_per_bar
    start_tick = ticks_per_bar * bar_shift
    end_tick = ticks_per_bar * (bar_shift + num_bars)
    abs_time_stamps = [get_abs_time_stamps(track) for track in midi_file.tracks]

    new_midi = MidiFile()
    new_midi.ticks_per_beat = midi_file.ticks_per_beat
    new_midi.tracks = [MidiTrack() for _ in midi_file.tracks]

    for track, new_track, ts, beg_meta, end_meta in zip(midi_file.tracks, new_midi.tracks, abs_time_stamps,
                                                        beg_meta_msgs, end_meta_msgs):
        new_track.name = track.name
        ind_slice = find_start_end(ts, start_tick, end_tick)
        section_ts = ts[ind_slice]
        section = get_section(track[ind_slice], section_ts, start_tick)
        tick_shift = 0

        add_beg_meta(new_track, beg_meta)
        tick_shift = add_notes_into_track(new_track, section, section_ts, start_tick, end_tick, tick_shift)
        if split_tie:
            tick_shift = add_notes_to_split_tie(new_track, start_tick, end_tick, tick_shift)
        add_end_meta(new_track, end_meta, end_tick-start_tick, tick_shift)

    return new_midi


def split_midi_by_bars(midi_file, num_bars_to_split=4, split_tie=True, bar_shift=0):
    num_bars = math.ceil(max(get_num_bars(track, midi_file.ticks_per_beat) for track in midi_file.tracks))
    num_sections = num_bars // num_bars_to_split
    split_midi_files = []
    for i in range(num_sections):
        split_midi_files.append(chunk_midi(midi_file, num_bars_to_split*i+bar_shift, num_bars_to_split, split_tie))
    return split_midi_files


def make_bars(num_bars, note_list=None, track_name='', ticks_per_beat=960):
    def beats_to_ticks(num_beats):
        return round(ticks_per_beat * num_beats)

    def get_note_msgs(note_list):
        note_msgs = []
        for n in note_list:
            note_msgs.extend([  # use abs time for now
                Message('note_on', note=n.pitch, time=beats_to_ticks(n.start), velocity=n.velocity),
                Message('note_off', note=n.pitch, time=beats_to_ticks(n.start + n.duration),
                        velocity=n.velocity)
            ])
        note_msgs.sort(key=lambda n: n.time)
        prev_t = 0
        for n in note_msgs:  # convert back to relative time
            abs_t = n.time
            n.time -= prev_t
            prev_t = abs_t
        return note_msgs

    beats_per_bar = 4
    num_ticks = beats_to_ticks(num_bars * beats_per_bar)
    note_msgs = get_note_msgs(note_list) if note_list else []

    midi_file = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    track.name = track_name
    
    track.extend(note_msgs)
    end_delta_t = num_ticks - sum([msg.time for msg in track])
    if end_delta_t < 0:
        raise ValueError('The MIDI length is too short to contain the notes')
    track.append(MetaMessage('end_of_track', time=end_delta_t))
    midi_file.tracks = [track]
    return midi_file


def get_notes_from_track(track, ticks_per_beat=None):
    def ticks_to_beats(n_ticks): return n_ticks / ticks_per_beat

    check_note_on_off_balanced(track)
    track = deepcopy(track)
    abs_time_stamps = get_abs_time_stamps(track)
    note_msgs = []
    for msg, t in zip(track, abs_time_stamps):
        if is_note(msg):
            msg.time = t
            note_msgs.append(msg)
    note_msgs.sort(key=lambda n: n.time)

    start_ts = [0] * 128
    vels = [0] * 128
    note_list = []
    for msg in note_msgs:
        if is_note_on(msg):
            start_ts[msg.note] = msg.time
            vels[msg.note] = msg.velocity
        else:
            start = start_ts[msg.note]
            duration = msg.time - start
            if ticks_per_beat:
                start, duration = ticks_to_beats(start), ticks_to_beats(duration)
            note_list.append(Note(msg.note, start, duration, vels[msg.note]))

    return note_list


def is_track_monophonic(track):
    notes = get_notes_from_track(track)
    prev_end_t = 0
    for n in notes:
        if n.start < prev_end_t:
            return False
        prev_end_t = n.start + n.duration
    return True


def change_ticks_per_beat(midi_file, target_ticks_per_beat):
    """ In-place operation """
    orig_ticks_per_beat = midi_file.ticks_per_beat
    ratio = target_ticks_per_beat / orig_ticks_per_beat

    midi_file.ticks_per_beat = target_ticks_per_beat

    for track in midi_file.tracks:
        for msg in track:
            msg.time = int(msg.time * ratio)


def add_missing_end_of_track(midi_file, length_in_ticks):
    for track in midi_file.tracks:
        missing_ticks = length_in_ticks - get_length_in_ticks(track)
        if track[-1].type == 'end_of_track':
            track[-1].time += missing_ticks
        else:
            track.append(MetaMessage('end_of_track', time=missing_ticks))


def get_rest_positions(note_list):
    rest_pos = []
    prev_end_t = 0
    for i, n in enumerate(note_list):
        if n.start - prev_end_t > 0.001:
            rest_pos.append((i - 1, i))
        prev_end_t = n.start + n.duration
    return rest_pos


# def note_list_to_m21(note_list, bpm=120):
#     ds = DivaScore.from_note_list(note_list, bpm)
#     return DivaScore.to_m21_score(ds)


# ---- pitch range and transposition ----


def shift_to_c3_c5(pitch):
    c3_pitch = 48
    c5_pitch = 72
    if pitch < c3_pitch:
        while pitch < c3_pitch:
            pitch += 12
    elif pitch > c5_pitch:
        while pitch > c5_pitch:
            pitch -= 12
    return pitch


def limit_track_pitch_range(midi_track):
    """ In-place operation. """
    for msg in midi_track:
        if is_note(msg):
            msg.note = shift_to_c3_c5(msg.note)


def transpose_track(midi_track, delta):
    """ In-place operation. """
    for msg in midi_track:
        if is_note(msg):
            msg.note += delta


def transpose_midi_file(midi_file, delta):
    """ In-place operation. """
    for track in midi_file.tracks:
        transpose_track(track, delta)


def mean_pitch(track):
    return mean([msg.note for msg in track if is_note_on(msg)])


def median_pitch(track):
    return median([msg.note for msg in track if is_note_on(msg)])


def mid_pitch(track):
    return (max(msg.note for msg in track if is_note_on(msg)) + min(msg.note for msg in track if is_note_on(msg))) / 2


def auto_transpose_midi_file(midi_file, center_pitch):
    """ In-place operation. Track #0 is melody. """
    melody_track = midi_file.tracks[0]
    transpose_midi_file(midi_file, int(center_pitch - mid_pitch(melody_track)))


# ---- pitch classes and chords ----


PITCH_CLASS_TABLE = {
    'c': 0, 'c#': 1, 'd': 2, 'd#': 3, 'e': 4, 'f': 5, 'f#': 6, 'g': 7, 'g#': 8, 'a': 9, 'a#': 10, 'b': 11
}


def pitch_class_names_to_nums(pc_names): return [PITCH_CLASS_TABLE[n] for n in pc_names]


def pitch_to_class(pitch): return pitch % 12


def is_pitch_in_class(pitch, pitch_class_name):
    return pitch_to_class(pitch) == PITCH_CLASS_TABLE[pitch_class_name]


def get_closest_pitch_in_class(pitch, pitch_class):
    input_pitch_class = pitch_to_class(pitch)
    delta_a = pitch_class - input_pitch_class
    delta_b = delta_a % -12 if delta_a >= 0 else delta_a % 12
    delta = delta_a if abs(delta_a) < abs(delta_b) else delta_b
    return pitch + delta


INTERVALS = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
}


def _get_chord_table():
    chord_table = {}
    for pitch_class_name, pitch_class in PITCH_CLASS_TABLE.items():
        for key_type, itvs in INTERVALS.items():
            chord_table[f'{pitch_class_name}_{key_type}'] = [pitch_to_class(pitch_class + itv) for itv in itvs]
    return chord_table


CHORD_TABLE = _get_chord_table()


def get_chord_name(pitch_class_set):
    chord_name = None
    for name, pitch_classes in CHORD_TABLE.items():
        if set(pitch_classes).issubset(pitch_class_set):
            chord_name = name
    return chord_name


INTERVAL_TO_STEP = {  
    '1'  : 0,
    'b2' : 1,
    '2'  : 2,
    'b3' : 3,
    '3'  : 4,
    '4'  : 5, 
    '#4' : 6,
    'b5' : 6,
    '5'  : 7,
    'b6' : 8,
    '6'  : 9,
    'b7' : 10,
    '7'  : 11,
    'b9' : 13,
    '9'  : 14,
    'b11': 16,
    '11' : 17,
    'b13': 20,
    '13' : 21
}

NOTE_TO_OFFSET = {
    'Ab': 8,
    'A': 9,
    'A#': 10,
    'Bb': 10,
    'B': 11,
    'Cb': 11,
    'C': 0,
    'C#': 1,
    'Db': 1,
    'D': 2,
    'D#': 3,
    'Eb': 3,
    'E': 4,
    'E#': 5,  
    'F': 5,
    'F#': 6,
    'Gb': 6,
    'G': 7,
    'G#': 8,
}

pitch_n = {
            'C': 0,
            'C#': 1,
            'Db': 1,
            'D': 2,
            'D#': 3,
            'Eb': 3,
            'E': 4,
            'Fb':4,
            'E#': 5, 
            'F': 5,
            'F#': 6,
            'Gb':6,
            'G': 7,
            'G#': 8,
            'Ab': 8,
            'A': 9,
            'A#': 10,
            'Bb':10,
            'B': 11,
            'Cb':11,
            'N': None,
            'end': None
        } 

def cut_notes(chord_notes, quality):
    if len(quality) <= 3 or quality[3] == '/' or quality[3] == '(':
        return chord_notes[:3]
    
    if quality[3] == '7':
        return chord_notes[:4]
    elif quality[3] == '9':
        return chord_notes[:5]
    elif quality[3:5] == '11':
        return chord_notes[:6]
    elif quality[3:5] == '13':
        return chord_notes[:7]
    else:
        return ValueError(f"Unknown Chord Type: {quality[3]}")        

def string_to_chord(chord_name_string):

    key, quality = chord_name_string.split(":")
    root = 0
    chord_notes = []

    if 'minmaj7' in quality:
        chord_notes = [root, root + 3, root + 7, root + 11]
    elif 'hdim7' in quality: 
        chord_notes = [root, root + 3, root + 6, root + 10]
    elif 'maj6' in quality:
        chord_notes = [root, root + 4, root + 7, root + 9]
    elif 'min6' in quality:
        chord_notes = [root, root + 3, root + 7, root + 8]
    elif 'maj' in quality:
        chord_notes = [root, root + 4, root + 7, root + 11, root + 14, root + 17]
        chord_notes = cut_notes(chord_notes, quality)
    elif 'min' in quality:
        chord_notes = [root, root + 3, root + 7, root + 10, root + 14, root + 17]
        chord_notes = cut_notes(chord_notes, quality)
    elif 'dim' in quality:
        chord_notes = [root, root + 3, root + 6, root + 9]
        chord_notes = cut_notes(chord_notes, quality)
    elif 'aug' in quality:
        chord_notes = [root, root + 4, root + 8, root + 10]
        chord_notes = cut_notes(chord_notes, quality)
    elif 'sus' in quality: 
        sus = int(quality[3])
        if sus == 2:
            chord_notes = [root, root + 2, root + 7]
        elif sus == 4:
            chord_notes = [root, root + 5, root + 7]
        else:
            raise ValueError(f"Unknown suspension: {sus}")
    elif '7' in quality:
        chord_notes = [root, root + 4, root + 7, root + 10]
    elif '9' in quality:
        chord_notes = [root, root + 4, root + 7, root + 11, root + 14]
    elif '11' in quality: 
        chord_notes = [root, root + 4, root + 7, root + 11, root + 14, root + 17]
    else:
        raise ValueError("Unknown Chord")

    if '(' in quality: 
        adds = quality[quality.find('(') + 1 : quality.find(')')]
        adds = adds.split(",")
        for add in adds:
            chord_notes.append(INTERVAL_TO_STEP[add])
    
    chord_notes = sorted(chord_notes)

    if "/" in quality:
        new_bass = quality.split("/")[1]
        chord_notes = [INTERVAL_TO_STEP[new_bass] % 12 - 12] + chord_notes 
    
    pitch_offset = pitch_n[key]

    chord_notes = [note + pitch_offset for note in chord_notes]

    return chord_notes


def wrapping_chord(chord_events):
    lyrics = []
    chord_symbols = chord_events['chords']
    timings = chord_events['intervals']

    chord_track = pretty_midi.Instrument(program=0)

    for i in range(len(chord_symbols)):
        chord = chord_symbols[i]
        
        if chord is None or chord == 'N':
            continue
        
        try:
            root, _ = chord.split(':')
        except:
            root = None         
                
        if not root:
            continue
        
        comp = string_to_chord(chord)
        # event_on/off
        start = float(timings[i][0])
        end = float(timings[i][1])

        lyrics.append(pretty_midi.Lyric(chord, start))

        for note in comp:
            note = pretty_midi.Note(velocity=100, pitch=int(48 + note), start=start, end=end)
            chord_track.notes.append(note)

    return chord_track, lyrics

def chord_to_midi(
        chord_events,
        midi_name,
        key='C',
        bpm=120,
        beats_in_measure=4,
        ) -> bytearray:

    if bpm == 0.0 or bpm == None:
        bpm = 120
        
    bpm = float(bpm)

    beats_in_measure = int(beats_in_measure)
    lead_sheet = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    beats_sec = 60.0 / bpm

    chord_track, chord_symbols = wrapping_chord(chord_events)
    ts = pretty_midi.TimeSignature(beats_in_measure, 4, 0)
    ks = pretty_midi.KeySignature(NOTE_TO_OFFSET[key], 0)

    lead_sheet.time_signature_changes.append(ts)
    lead_sheet.key_signature_changes.append(ks)

    lead_sheet.instruments.append(chord_track)
    lead_sheet.lyrics = chord_symbols

    lead_sheet.write(midi_name)



Note = namedtuple('Note', ('pitch', 'start', 'duration', 'velocity'))  # start and duration are in "beats"
BEATS_PER_BAR = 4

def time_to_beat(time, bpm):
    return time * (bpm / 60)

def parse_note_dicts(note_dicts, track_name='', ticks_per_beat=480):
    bpm = 120
    _time_to_beat = partial(time_to_beat, bpm=bpm)

    note_list = []
    for note_dict in note_dicts:
        note_list.append(Note(
            start=_time_to_beat(note_dict['start']),
            duration=_time_to_beat(note_dict['end']-note_dict['start']),
            pitch=note_dict['pitch'],
            velocity=100,
        ))

    if note_dicts:
        max_end = _time_to_beat(max(nd['end'] for nd in note_dicts))
    else:
        max_end = 0
    num_bars = math.ceil(max_end / BEATS_PER_BAR)
    return make_bars(num_bars, note_list, track_name=track_name, ticks_per_beat=ticks_per_beat)


def process_transcription_result(result, out_fp):
    stem_names = {
        'vocal': 52,
        'bass': 34,
        'brass': 62,
        'chromatic_percussion': 9,
        'drums': -1,
        'guitar': 26,
        'organ': 19,
        'piano': 1,
        'pipe': 73,
        'reed': 66,
        'strings': 41,
        'synth_lead': 81,
        'synth_pad': 89
    }
    notes = result['notes']
    default_bpm = 120

    if isinstance(out_fp, dict):
        for stem_name in stem_names:
            if stem_name in out_fp:
                stem_notes = notes[stem_name]
                _out_fp = out_fp[stem_name]
                mf = parse_note_dicts(stem_notes, track_name=stem_name)
                set_midi_bpm(mf, default_bpm)
                save_midi(mf, _out_fp)
    else:
        mfs = []
        for stem_name in stem_names:
            if stem_name not in notes:
                continue
            stem_notes = notes[stem_name]
            if stem_notes:
                mf = parse_note_dicts(stem_notes, track_name=stem_name)
                set_midi_bpm(mf, default_bpm)
                mfs.append(mf)

        mf = stack_tracks_in_files(mfs)
        save_midi(mf, out_fp)

def note2midi(note_list, end_time, midi_name, program=1) -> bytearray:
    # Create a PrettyMIDI object
    output = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(program=program)

    if not note_list:
        note = pretty_midi.Note(velocity=0, pitch=0, start=0, end=end_time)
        track.notes.append(note)
    else:
        for note in note_list:
            start, end, pitch = note['start'], note['end'], note['pitch']
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
            track.notes.append(note)

    output.instruments.append(track)
    output.write(midi_name)