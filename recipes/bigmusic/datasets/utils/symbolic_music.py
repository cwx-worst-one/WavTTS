import io

import pretty_midi


def pretty_midi_obj_to_midi_bytes(
    midi_obj: pretty_midi.PrettyMIDI,
) -> bytes:
    with io.BytesIO() as out_io:
        midi_obj.write(out_io)
        out_io.seek(0)
        midi_data = out_io.read()
    return midi_data


# Dictionary that maps note names to MIDI note numbers
note_name_to_number = {
    'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 'E': 64, 'F': 65,
    'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
}

# Function to return the MIDI note numbers for a given chord
def chord_name_to_notes(chord_name):
    if chord_name == "N":
        return []
    root_name, tone = chord_name.split(":")
    # root_name = chord_name[:-1] if chord_name[-1] in ['m', 'M'] else chord_name
    root_note = note_name_to_number[root_name] - 24

    # Assuming standard tuning and equal temperament
    # Define intervals for various chord types
    tone_interval_dict = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "aug": [0, 4, 8],
        "dim": [0, 3, 6],
    }
    # Determine chord type based on suffix and assign respective intervals
    intervals = tone_interval_dict[tone]
    notes = [root_note + interval for interval in intervals]
    return notes