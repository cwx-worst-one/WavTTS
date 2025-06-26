import copy
from typing import Optional

import torch


class LeadSheetTokenizer:
    """
    YILIN NOTE: This is a refactored and simplified version of the original LeadSheetTokenizerV2.

    This class expands and tokenizes a lead sheet into a 1D list of tokens.
    It builds timestamp_tokenizer, note tokenizer and phoneme tokenizer accordingly.

    Input should be a lead sheet, which is a list of tuples (note, [phonemes], start_time, end_time):
    For example,
    leadsheet = [('Rest', ['sil'],           0.0,   0.5),
                 ('B3',   ['C0d', 'C0eng'],  0.5,   0.975),
                 ('Db4',  ['C0g', 'C0uang'], 0.975, 1.4)]

    For each note, we expand as:
    note_token + phone_tokens + '[START_OF_TIME]' +  duration_tokens + '[END_OF_NOTE]' + [next_note_token...
    output = [414, 60, 390, 0, 416, 35, 415, 99, 376, 165, 35, 416, 39, 27, 15, 415, 103, 303, 274, 39, 27, 15, 416, 41,
      34, 415, 417]
    """

    def __init__(
        self,
        token_to_id: dict[str, int],
        time_start_token: str = "[TIME_START]",
        time_duration_token: str = "[TIME_DURATION]",
        end_of_note_token: str = "[END_OF_NOTE]",
        special_tokens: Optional[list] = None,
    ):

        self.time_start_token = time_start_token
        self.time_duration_token = time_duration_token
        self.end_of_note_token = end_of_note_token
        self.special_tokens = [] if not special_tokens else special_tokens

        # build note vocab
        offset = max(token_to_id.values()) + 1
        note_token_to_id = {
            note: i + offset
            for i, note in enumerate(self._generate_note_vocab())
            if note not in token_to_id
        }
        self.note_token_to_id = note_token_to_id
        self.note_id_to_token = {v: k for k, v in note_token_to_id.items()}
        token_to_id = {**token_to_id, **note_token_to_id}

        # build special token vocab
        offset = max(token_to_id.values()) + 1
        spec_token_to_id = {
            sp: i + offset
            for i, sp in enumerate(
                [time_start_token, time_duration_token, end_of_note_token]
                + self.special_tokens
            )
            if sp not in token_to_id
        }
        token_to_id = {**token_to_id, **spec_token_to_id}

        self.vocab_size = len(token_to_id)
        self.token_to_id = token_to_id

    def normalize_sym(self, sym, pitch_shift=0):
        if isinstance(sym, int):
            pitch_id = sym + pitch_shift
            assert (
                pitch_id <= 127 and pitch_id >= 0
            )  # inside midi pitch id [0,127] + rest
            sym = self.pitchid2sym[pitch_id]
        if sym in self.note_token_to_id.keys():
            return self.note_token_to_id[sym]
        # sym must in self.sym2pitchid
        if ("#" not in sym) and ("b" not in sym):
            return self.note_token_to_id[sym]

        if "#" in sym:
            shift_note = "#"
            n_alter = sym.count(shift_note)
        else:  # 'b' in sym
            shift_note = "b"
            n_alter = -sym.count(shift_note)

        assert abs(n_alter) <= 2  # only allow like C#4/C##4/Cb4/Cbb4, not allow C###4

        sym_base = sym.replace(shift_note * abs(n_alter), "")
        altered_id = self.note_token_to_id[sym_base] + n_alter + pitch_shift
        assert altered_id in self.note_id_to_token.keys()
        return altered_id

    def residual_time(self, x, mod=2):
        residual = x % mod
        return residual

    def _generate_note_vocab(self):
        # https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
        # Generate a list of notes according to full MIDI note (128 note)
        # With middle C defined as C4 :
        sym = [
            f"{note}{octave}"
            for octave in range(-1, 10)
            for note in [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]
        ]
        sym = sym[:-4] + ["Rest"]  # extended MIDI range (128 note + rest)
        self.pitchid2sym = {i: s for i, s in enumerate(sym)}
        self.sym2pitchid = {s: i for i, s in enumerate(sym)}
        return sym

    def _tokenize_note(self, notes, pitch_shift=0):
        if not isinstance(notes, list):
            notes = [notes]

        return [self.normalize_sym(note, pitch_shift=pitch_shift) for note in notes]

    def _tokenize_phones(self, phone_ids):
        flattened = []
        for sublist in phone_ids:
            if isinstance(sublist, list):
                flattened.extend(sublist)
            else:
                flattened.append(sublist)
        return [self.token_to_id[phone] for phone in flattened]

    def _tokenize_time(self, start_time, end_time):
        time_float = [start_time, end_time - start_time]
        time_tokens = [
            self.token_to_id[self.time_start_token],
            self.token_to_id[self.time_duration_token],
        ]
        return time_tokens, time_float

    def _tokenize(self, leadsheet: list, pitch_shift=0) -> list:
        # Tokenize the entire lead sheet.
        input_ids = []
        phoneme_input_ids = []
        note_input_ids = []
        coffs = []
        for item in leadsheet:
            note = item["pitch"]
            phone_ids = item["phone"]
            start_time = item["start"]
            end_time = item["end"]
            note_token = self._tokenize_note(note, pitch_shift=pitch_shift)
            phoneme_token = self._tokenize_phones(phone_ids)
            time_tokens, time_float = self._tokenize_time(start_time, end_time)

            # For each note, we expand as:
            # note_token + phoneme_token + '[START_OF_TIME]' +  duration_tokens
            note = (
                note_token
                + phoneme_token
                + time_tokens
                + [self.token_to_id[self.end_of_note_token]]
            )
            token_coff_this = (
                len(note_token) * [1] + len(phoneme_token) * [1] + time_float + [1]
            )
            phoneme_input_ids.extend(phoneme_token)
            note_input_ids.extend(note_token)
            input_ids.extend(note)
            coffs.extend(token_coff_this)
        return input_ids, phoneme_input_ids, note_input_ids, coffs

    def __call__(
        self, leadsheet: list, return_dict=True, pitch_shift=0
    ) -> torch.Tensor:
        if return_dict == False:
            raise NotImplementedError
        input_ids, phoneme_input_ids, note_input_ids, coffs = self._tokenize(
            leadsheet, pitch_shift
        )
        return {
            "input_ids": torch.tensor(input_ids),
            "phoneme_input_ids": torch.tensor(phoneme_input_ids),
            "note_input_ids": torch.tensor(note_input_ids),
            "coffs": torch.tensor(coffs),
        }


def fetch_notes_from_phones_v3(phoneme_sequence, note_sequence):
    global_start = phoneme_sequence[0]["start"]
    global_end = phoneme_sequence[-1]["end"]
    # global_start = max(phoneme_sequence[0]['start'], note_sequence[0]['start'])
    # global_end = min(phoneme_sequence[-1]['end'], note_sequence[-1]['end'])
    # phoneset = set(all_voiced_phones)
    note_sequence = copy.deepcopy(note_sequence)
    phoneme_sequence = copy.deepcopy(phoneme_sequence)

    new_note_seq = []
    for idx, note in enumerate(note_sequence):
        if note["start"] > global_start and note["end"] < global_end:
            note["phone"] = []
            new_note_seq.append((note, idx))

    if len(new_note_seq) == 0:
        # phoneme and note are totally not matched
        return [], []

    start_idx, end_idx = new_note_seq[0][1], new_note_seq[-1][1]
    if start_idx - 1 >= 0 and note_sequence[start_idx - 1]["end"] > global_start:
        new_note_seq.insert(
            0,
            (
                {
                    "start": global_start,
                    "end": note_sequence[start_idx - 1]["end"],
                    "phone": [],
                    "pitch": note_sequence[start_idx - 1]["pitch"],
                },
                start_idx - 1,
            ),
        )
    elif start_idx - 1 >= 0 and note_sequence[start_idx - 1]["end"] <= global_start:
        new_note_seq[0][0]["start"] = global_start

    if (
        end_idx + 1 < len(note_sequence)
        and note_sequence[end_idx + 1]["start"] < global_end
    ):
        new_note_seq.append(
            (
                {
                    "start": note_sequence[end_idx + 1]["start"],
                    "end": global_end,
                    "phone": [],
                    "pitch": note_sequence[end_idx + 1]["pitch"],
                },
                end_idx + 1,
            )
        )
    elif (
        end_idx + 1 < len(note_sequence)
        and note_sequence[end_idx + 1]["start"] >= global_end
    ):
        new_note_seq[-1][0]["end"] = global_end

    new_phoneme_sequence = []
    for phone in phoneme_sequence:
        if phone["start"] >= global_start and phone["end"] <= global_end:
            new_phones = []
            for i in phone["phone"]:
                # YILIN NOTE: use all phones, do not distinguish voiced and unvoiced phones
                # if i in phoneset:
                new_phones.append(i)
            if len(new_phones) > 0:
                phone["pitch"] = []
                phone["phone"] = new_phones
                new_phoneme_sequence.append(phone)

    return [i[0] for i in new_note_seq], new_phoneme_sequence
