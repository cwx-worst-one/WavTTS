import json
import copy
import torch
import numpy as np
from recipes.datasets.mcc.sami_tokenizer import convert_labels_to_text_id, EN_vowel, EN_consonant, ZH_consonant, ZH_vowel, sep_strs, all_phones

all_phones = all_phones + ["ext"]
all_voiced_phones = (EN_consonant + EN_vowel + ZH_consonant + ZH_vowel + ["ext"])

sym = [f"{note}{octave}" for octave in range(-1, 10) for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]]
# sym = sym[9:-11] + ["Rest"] # piano MIDI range (88 note + rest)
sym = sym[:-4] + ["Rest"] # extended MIDI range (128 note + rest)
pitchid2sym = {i:s for i, s in enumerate(sym)}
sym2pitchid = {s:i for i, s in enumerate(sym)}

class PhoneType:
    SIL = 0
    VOWEL = 1
    LEFT_CONS = 2  # cons before vowel
    RIGHT_CONS = 3  # cons after vowel
    # (for those in speech data)
    UNK_CONS = 4  # cons don't know whether is left or right


def is_vowel(phn):
    # return phn in vowel_set
    return phn in ZH_vowel or phn in EN_vowel


def is_cons(phn):
    # return phn in cons_set
    return phn in ZH_consonant or phn in EN_consonant

def mark_english_phone_type(itvs):
    class PhoneTypeConnector:
        def __init__(self, phone) -> None:
            self.phone = phone 
            self.phone_type = None

    _itvs = [PhoneTypeConnector(i) for i in itvs.get('phone')]
    # phonetypes = [-1] * len(itvs.get('phone')) 
    # consonant before vowel
    while not is_vowel(_itvs[0].phone):
        _itvs[0].phone_type = PhoneType.LEFT_CONS
        _itvs = _itvs[1:]
    # consonant after vowel
    while not is_vowel(_itvs[-1].phone):
        _itvs[-1].phone_type = PhoneType.RIGHT_CONS
        _itvs = _itvs[:-1]
    # the rest should all be the SAME VOWEL
    for v in _itvs:
        v.phone_type = PhoneType.VOWEL

    return itvs

def mark_vowel_phone_type(word):
    return [ 1 if is_vowel(i) else 0  for i  in word]


def is_left(int):
    return (int in [1,2])
def is_right(int):
    return (int in [1,3])

def is_end_of_note(left, right):
    if is_right(left) and is_left(right):
        return True
    else:
        return False


def collect_phone_by_note(notes, phonemes, phone_types, start_times, end_times):
    note_seq_list = []
    score_seq_list = []

    for i in range(len(notes)):
        collected_phones = []
        phone_type = phone_types[i]
        if phone_type == 0:
            note = notes[i]
            collected_phones = [phonemes[i]]
            start_time = start_times[i]
            end_time = end_times[i]
            note_seq_list.append((note, start_time, end_time))
            score_seq_list.append((note, collected_phones, start_time, end_time))
            continue
        
        if phone_types[i] == 1:
            note = notes[i]
            right_boundary = left_boundary = i
            for left in range(i-1, 0, -1):
                if phone_types[left] == 2:
                    left_boundary = left
                    continue
                else:
                    left_boundary = left+1
                    break

            for right in range(i+1, len(notes)):
                if phone_types[right] == 3:
                    right_boundary = right
                    continue
                else:
                    right_boundary = right-1
                    break

            collected_phones = [phonemes[left_boundary:right_boundary+1]]
            start_time = start_times[left_boundary]
            end_time = end_times[right_boundary]
            note_seq_list.append((note, start_time, end_time))
            score_seq_list.append((note, collected_phones, start_time, end_time))
    return note_seq_list, score_seq_list

def offset_to_accu(offset_list):
    accumulation_result = []
    total = 0
    for offset in offset_list:
        total += offset
        accumulation_result.append(total)
    return accumulation_result

def clean_phoneme(ph):
    if ph in ['ext', 'sil']:
        return ph
    if ph[-1] in set(map(str, range(0, 9))):
        ph = ph[:-1]
    if ph[0].isupper():
        return "E0"+ph.lower()
    else:
        return "C0"+ph.lower()


def leadsheet_offset(sequence, offset):
    _sequence = []
    for note, phone, start, end in sequence:
        _sequence.append((note, phone, start+offset, end+offset))
    return _sequence

def seq_offset(sequence, offset):
    _sequence = copy.deepcopy(sequence)
    for seq in _sequence:
        seq['start'] += offset
        seq['end'] += offset
    return _sequence

def extract_phoneme_sequence(utterances):
    words = []
    for utt in utterances:
        words.extend(utt["words"])

    phoneme_sequence = []
    for word in words:
        start_time=float(word['start_time'])/1000.0
        end_time=float(word['end_time'])/1000.0
        phoneme=word['phoneme']
        if phoneme == None:
            continue
        phone_tone_ids, phones, tones = convert_labels_to_text_id(phoneme.split("\n"))
        data = {"start":start_time, "end":end_time, "phone": phones}
        phoneme_sequence.append(data)
    return phoneme_sequence

def extract_note_sequence(note_sequence):
    for note in note_sequence:
        note["start"] = float(note["start"])
        note["end"] = float(note["end"])
    return note_sequence

def match_vowels_to_notes(phoneme_sequence, note_sequence):
    phoneset = set(all_voiced_phones)
    vowelset = set(EN_vowel) | set(ZH_vowel)

    # Initialize empty list of phonemes for each note
    for note in note_sequence:
        note['phone'] = []

    # Helper function to uniformly split the time boundaries
    def uniform_split_intervals(start, end, parts):
        if parts == 0:
            return []
        interval_length = (end - start) / parts
        return [(start + i * interval_length, start + (i + 1) * interval_length, start, end) for i in range(parts)]

    # Helper function to find the closest note start for a given time
    def find_closest_note_start(vow_start, notes, tolerance=1):
        closest = None # maybe list for one syall, multiple-note

        if vow_start > notes[-1]['start']:
            return None


        min_difftime = 99999
        for note in notes:
            difftime = note['start'] - vow_start
            abs_difftime = abs(difftime)

            # global closest
            if abs_difftime < min_difftime and abs_difftime < tolerance: 
                closest = note
                min_difftime = abs_difftime
            
        return closest

    # Extract vowels from phoneme sequence and split their times
    vowels_with_intervals = []
    vowels_with_cons_expanded = []
    vowels_with_cons = []
    for word in phoneme_sequence:
        vowels = []
        for ph in word['phone']:
            if ph in phoneset:
                vowels_with_cons.append(ph)
            if ph in vowelset:
                vowels.append(ph)
                vowels_with_cons_expanded.append(vowels_with_cons)
                vowels_with_cons = []

        intervals = uniform_split_intervals(word['start'], word['end'], len(vowels))
        vowels_with_intervals.extend(zip(vowels, intervals))

    # Adjust vowel intervals to the closest note start times
    vowels_with_intervals_v2 = []
    bindex = 0 
    for vowels_with_interval, vowels_with_cons  in zip(vowels_with_intervals, vowels_with_cons_expanded):
        vowel, (vow_start, _, word_start, word_end) = vowels_with_interval
        closest_note = find_closest_note_start(vow_start, note_sequence)
        if closest_note != None:
            # closest_note["matched"] = True
            closest_note_start = closest_note['start']
            closest_note['phone'].extend(vowels_with_cons)
            # Update the interval start for the vowel
            index = next(i for i, v in enumerate(vowels_with_intervals) if v[0] == vowel and v[1][0] == vow_start)
            vowels_with_intervals_v2.append((vowel, vowels_with_cons, (closest_note_start, vowels_with_intervals[index][1][1])))
            bindex += 1

    return note_sequence

def trim_unphoned(leadsheet):
    start_index = 0
    end_index = len(leadsheet)
    for i in leadsheet:
        start_index += 1
        if i['phone'] != []:
            break
    for i in reversed(leadsheet):
        end_index -= 1
        if i['phone'] != []:
            break
    return leadsheet[start_index:end_index]


def make_leadsheet_from_note_and_utterances(note_sequence, utterances=None, phoneme_sequence=None, align_mode="v1", boundaries=[1.0, 1.0]):
    if phoneme_sequence == None:
        phoneme_sequence = extract_phoneme_sequence(utterances)
    note_sequence = extract_note_sequence(note_sequence)
    if align_mode == "forced":
        matched_notes = match_vowels_to_notes(phoneme_sequence, note_sequence)
        matched_notes = trim_unphoned(matched_notes)
        return matched_notes
    elif align_mode == "concat":
        new_note_seq, new_phoneme_sequence = match_vowels_to_notes_v2(phoneme_sequence, note_sequence, boundaries=boundaries)
        return new_note_seq, new_phoneme_sequence
    elif align_mode == "sort_by_starttime":
        event_dict = {}
        new_note_seq, new_phoneme_sequence = match_vowels_to_notes_v2(phoneme_sequence, note_sequence, boundaries=boundaries)
        for event in new_note_seq + new_phoneme_sequence:
            start = event['start']
        # ...
    else:
        raise NotImplementedError


def make_leadsheet_from_note_and_utterances_v2(note_sequence, utterances=None, phoneme_sequence=None, align_mode="v1", boundaries=None):
    if phoneme_sequence == None:
        phoneme_sequence = extract_phoneme_sequence(utterances)
    note_sequence = extract_note_sequence(note_sequence)
    if align_mode == "forced":
        matched_notes = match_vowels_to_notes(phoneme_sequence, note_sequence)
        matched_notes = trim_unphoned(matched_notes)
        return matched_notes
    elif align_mode == "concat" or align_mode == 'interleave':
        new_note_seq, new_phoneme_sequence = fetch_notes_from_phones_v3(phoneme_sequence, note_sequence)
        return new_note_seq, new_phoneme_sequence
    elif align_mode == "sort_by_starttime":
        event_dict = {}
        new_note_seq, new_phoneme_sequence = fetch_notes_from_phones_v3(phoneme_sequence, note_sequence)
        for event in new_note_seq + new_phoneme_sequence:
            start = event['start']
        # ...
    else:
        raise NotImplementedError




def fetch_notes_from_phones_v3(phoneme_sequence, note_sequence):
    global_start = phoneme_sequence[0]['start']
    global_end = phoneme_sequence[-1]['end']
    # global_start = max(phoneme_sequence[0]['start'], note_sequence[0]['start'])
    # global_end = min(phoneme_sequence[-1]['end'], note_sequence[-1]['end'])
    phoneset = set(all_voiced_phones)
    note_sequence = copy.deepcopy(note_sequence)
    phoneme_sequence = copy.deepcopy(phoneme_sequence)

    new_note_seq = []
    for idx, note in enumerate(note_sequence):
        if note['start'] > global_start and note['end'] < global_end:
            note['phone'] = []
            new_note_seq.append((note, idx))

    if len(new_note_seq) == 0:
        # phoneme and note are totally not matched
        return [], []

    start_idx, end_idx = new_note_seq[0][1], new_note_seq[-1][1]
    if start_idx - 1 >= 0 and note_sequence[start_idx - 1]['end'] > global_start:
        new_note_seq.insert(0, ({'start': global_start,
                                 'end': note_sequence[start_idx - 1]['end'],
                                 'phone': [],
                                 'pitch':note_sequence[start_idx - 1]['pitch']
                                 },
                                start_idx - 1))
    elif start_idx - 1 >= 0 and note_sequence[start_idx - 1]['end'] <= global_start:
        new_note_seq[0][0]['start'] = global_start


    if end_idx + 1 < len(note_sequence) and note_sequence[end_idx + 1]['start'] < global_end:
        new_note_seq.append(({'start': note_sequence[end_idx + 1]['start'],
                                 'end': global_end,
                                 'phone': [],
                                 'pitch':note_sequence[end_idx + 1]['pitch']
                                 },
                                end_idx + 1))
    elif end_idx + 1 < len(note_sequence) and note_sequence[end_idx + 1]['start'] >= global_end:
        new_note_seq[-1][0]['end'] = global_end

        
    new_phoneme_sequence = []
    for phone in phoneme_sequence:
        if phone['start'] >= global_start and phone['end'] <= global_end:
            new_phones = []
            for i in phone['phone']:
                if i in phoneset:
                    new_phones.append(i)
            if len(new_phones) > 0:
                phone['pitch'] = []
                phone['phone'] = new_phones
                new_phoneme_sequence.append(phone)
    
    
    return [i[0] for i in new_note_seq], new_phoneme_sequence



def match_vowels_to_notes_v2(phoneme_sequence, note_sequence, boundaries=[1.0, 1.0]):
    global_start = max(phoneme_sequence[0]['start'], note_sequence[0]['start']) + boundaries[0]
    global_end = min(phoneme_sequence[-1]['end'], note_sequence[-1]['end']) + boundaries[1]
    phoneset = set(all_voiced_phones)
    note_sequence = copy.deepcopy(note_sequence)
    phoneme_sequence = copy.deepcopy(phoneme_sequence)

    new_note_seq = []
    for note in note_sequence:
        if note['start'] > global_start and note['end'] < global_end:
            note['phone'] = []
            new_note_seq.append(note)
    
    new_phoneme_sequence = []
    for phone in phoneme_sequence:
        if phone['start'] > global_start and phone['end'] < global_end:
            new_phones = []
            for i in phone['phone']:
                if i in phoneset:
                    new_phones.append(i)
            if len(new_phones) > 0:
                phone['pitch'] = []
                phone['phone'] = new_phones
                new_phoneme_sequence.append(phone)
    return new_note_seq, new_phoneme_sequence

def split_leadsheet2note_and_phone(leadsheet):
    sliced_notes = []
    sliced_phones = []
    for note, phone_ids, start_time, end_time in leadsheet:
        sliced_notes.append({
                                "start":start_time, 
                                "end":end_time, 
                                "phone":[], 
                                "pitch":note, 
                                })
        if isinstance(phone_ids[0], list):
            phone_ids = phone_ids[0]
        if len(phone_ids) == 0:
            continue
        sliced_phones.append({
                                "start":start_time, 
                                "end":end_time, 
                                "phone":phone_ids, 
                                "pitch":[], 
                                })
    return sliced_notes, sliced_phones


# def split_leadsheet2note_and_phone(leadsheet):
#     sliced_notes = []
#     sliced_phones = []
#     for note, phone_ids, start_time, end_time in leadsheet:
        # remove rest notes in SFT dataset, because not exists in Pretrain
        # if note == "Rest" or note == ["Rest"]:
        #     continue
#         sliced_notes.append({
#                                 "start":start_time, 
#                                 "end":end_time, 
#                                 "phone":[], 
#                                 "pitch":note, 
#                                 })
#         if isinstance(phone_ids[0], list):
#             phone_ids = phone_ids[0]

#         # remove ext and sil phoneme in SFT dataset, because not exists in Pretrain
#         # new_phone_ids = []
#         # for phone_id in phone_ids:
#         #     if phone_id != "sil": #phone_id != "ext" and 
#         #         new_phone_ids.append(phone_id)
#         # if len(new_phone_ids) == 0:
#         #     continue
#         sliced_phones.append({
#                                 "start":start_time, 
#                                 "end":end_time, 
#                                 "phone":new_phone_ids, 
#                                 "pitch":[], 
#                                 })
#     return sliced_notes, sliced_phones

def read_token_txt(token_path):
    # Read note and phoneme tokens
    with open(token_path, 'r') as f:
        data = f.readlines()
        
    notes = [i.strip().split('\t')[1] for i in data]
    phonemes = [i.strip().split('\t')[0] for i in data]
    phone_types = [i.strip().split('\t')[3] for i in data]
    notes = [i.strip().split('\t')[1] for i in data]
    phonemes = [clean_phoneme(i) for i in phonemes]
    # phone_types = token_data["phone_type"].split()
    phone_types = list(map(int, phone_types))

    time_stamp = [i.strip().split('\t')[2] for i in data]
    # time_stamp = token_data["frm_duration_pr"].split()
    time_stamp = list(map(float, time_stamp))
    start_time = copy.deepcopy(time_stamp)
    end_time = copy.deepcopy(time_stamp)

    start_time.insert(0, 0)
    start_time.pop(-1)

    start_time = offset_to_accu(start_time)
    end_time = offset_to_accu(end_time)

    start_times = [round(i* 0.0125, 4) for i in start_time]
    end_times = [round(i* 0.0125, 4) for i in end_time]
    return notes, phonemes, phone_types, start_times, end_times

def get_score_from_token_txt(token_path):
    notes, phonemes, phone_types, start_times, end_times = read_token_txt(token_path)
    for phoneme in phonemes:
        if phoneme not in all_phones:
            print("score", phoneme, token_path)
    note_seq_list, score_seq_list = collect_phone_by_note(notes, phonemes, phone_types, start_times, end_times)
    phoneme_start_end_time = list(zip(start_times, end_times))
    return score_seq_list


def clean_up_compact_leadsheet(leadsheet):
    # leadsheet: list of (note_str, phone_list, start_time_float, end_time_float)
    # remove duration < 1ms
    new_sequence = []
    for note in leadsheet:
        duration = note[3] - note[2]
        if duration > 0.001:
            new_sequence.append(note)
        elif duration < 0:
            print("illegal leadsheet, note=", note)
        else:
            pass
    leadsheet = new_sequence 


    # merging connected Rest notes
    new_sequence = []
    buffer = []
    for this_note in leadsheet:
        pitch = this_note[0]
        if pitch != "Rest":
            if buffer != []:
                rest_note = ("Rest", ['sil'], buffer[0][2], buffer[-1][3])
                new_sequence.append(rest_note)
                buffer = []
            new_sequence.append(this_note)
        else:
            buffer.append(this_note)

    if buffer != []:
        rest_note = ("Rest", ['sil'], buffer[0][2], buffer[-1][3])
        new_sequence.append(rest_note)
        buffer = []
    leadsheet = new_sequence 

    # merging connected Rest notes
    new_sequence = []
    buffer = []
    for this_note in leadsheet:
        phone = this_note[1]
        if phone != ['sil']:
            if buffer != []:
                rest_phone = ("Rest", ['sil'], buffer[0][2], buffer[-1][3])
                new_sequence.append(rest_phone)
                buffer = []
            new_sequence.append(this_note)
        else:
            buffer.append(this_note)

    if buffer != []:
        rest_phone = ("Rest", ['sil'], buffer[0][2], buffer[-1][3])
        new_sequence.append(rest_phone)
        buffer = []
    leadsheet = new_sequence 
    
    return leadsheet

def clean_up_leadsheet(note_sequence):
    # remove duration < 1ms
    new_sequence = []
    for note in note_sequence:
        duration = note['end'] - note['start']
        if duration > 0.001:
            new_sequence.append(note)
        elif duration < 0:
            print("illegal leadsheet, note=", note)
        else:
            pass
    note_sequence = new_sequence 

    # merging connected Rest notes
    new_sequence = []
    buffer = []
    for this_note in note_sequence:
        pitch = this_note['pitch']
        if pitch != "Rest":
            if buffer != []:
                merged_note = {"start":buffer[0]['start'], "end":buffer[-1]['end'], "phone":[], "pitch":"Rest"}
                new_sequence.append(merged_note)
                buffer = []
            new_sequence.append(this_note)
        else:
            buffer.append(this_note)

    if buffer != []:
        merged_note = {"start":buffer[0]['start'], "end":buffer[-1]['end'], "phone":[], "pitch":"Rest"}
        new_sequence.append(merged_note)
        buffer = []
    note_sequence = new_sequence 

    # merging connected Rest notes
    new_sequence = []
    buffer = []
    for this_note in note_sequence:
        phone = this_note['phone']
        if phone != ['sil']:
            if buffer != []:
                merged_note = {"start":buffer[0]['start'], "end":buffer[-1]['end'], "phone":['sil'], "pitch":[]}
                new_sequence.append(merged_note)
                buffer = []
            new_sequence.append(this_note)
        else:
            buffer.append(this_note)

    if buffer != []:
        merged_note = {"start":buffer[0]['start'], "end":buffer[-1]['end'], "phone":['sil'], "pitch":[]}
        new_sequence.append(merged_note)
        buffer = []
    note_sequence = new_sequence 
    
    return note_sequence

def concat_alignment(leadsheet_tokenizer, notes, phones, pitch_shift=0):
    notes = clean_up_leadsheet(notes)
    phones = clean_up_leadsheet(phones)
    note_tokens_dict = leadsheet_tokenizer(notes, return_dict=True, pitch_shift=pitch_shift)
    note_timing_tokens = note_tokens_dict['leadsheet_tokens']
    note_tokens = note_tokens_dict['note_tokens']
    note_tokens_coff = note_tokens_dict['token_coff']

    phone_tokens_dict = leadsheet_tokenizer(phones, return_dict=True, pitch_shift=pitch_shift)
    phoneme_timing_tokens = phone_tokens_dict['leadsheet_tokens']
    phoneme_tokens = phone_tokens_dict['phoneme_tokens']
    phone_tokens_coff = phone_tokens_dict['token_coff']

    leadsheet_tokens =  torch.cat([phoneme_timing_tokens, note_timing_tokens], 0)
    leadsheet_tokens_coff =  torch.cat([phone_tokens_coff, note_tokens_coff], 0)
    return leadsheet_tokens, leadsheet_tokens_coff, note_tokens, phoneme_tokens


def interleave_alignment(leadsheet_tokenizer, notes, phones, pitch_shift=0):
    notes = clean_up_leadsheet(notes)
    phones = clean_up_leadsheet(phones)
    
    leadsheet, flag = merge_leadsheet(notes, phones)
    if flag:
        leadsheet_token_dict = leadsheet_tokenizer(leadsheet, return_dict=True, pitch_shift=pitch_shift)

        leadsheet_tokens = leadsheet_token_dict['leadsheet_tokens']
        note_tokens = leadsheet_token_dict['note_tokens']
        phoneme_tokens = leadsheet_token_dict['phoneme_tokens']
        leadsheet_tokens_coff = leadsheet_token_dict['token_coff']

        return leadsheet_tokens, leadsheet_tokens_coff, note_tokens, phoneme_tokens
    
    else:
        return False, None, None, None

def merge_leadsheet(notes, phones):
    for note_dict, phone_dict in zip(notes, phones):
        if note_dict['start'] == phone_dict['start'] and note_dict['end'] == phone_dict['end']:
            note_dict['phone'] = phone_dict['phone']
            # note_dict['phoneme_ids'] = phone_dict['phoneme_ids']
        else:
            return [], False

    return notes, True





def dump_leadsheet(index, audio, sliced_notes, sliced_phones, dir="assets/debug"):
    import os
    import torchaudio
    # idx = int(meta_song_id_str.split("-")[1].replace(".wav", "")[-1])
    # if idx == 1:
    # if speaker_id not in self.dump_cache:
    #     self.dump_cache[speaker_id] = 0
    # if self.dump_cache[speaker_id] < 10: # dump 10 example for every spkr would enough
        # self.dump_cache[speaker_id] += 1
    save_dir = os.path.join(dir, str(index))
    print("save_dir "+save_dir)
    os.makedirs(save_dir, exist_ok=True)
    torchaudio.save(f"{save_dir}/audio.wav", audio.view(1, -1), 24000)
    with open(f"{save_dir}/note.json", "w") as f:
        json.dump(sliced_notes, f, indent=4)
    with open(f"{save_dir}/phones.json", "w") as f:
        json.dump(sliced_phones, f, indent=4)


if __name__ == "__main__":
    # midi = "vocal_midi/mnt/bn/audio-diffusion/wtl/vocal_midi/0/6684775854397982721_tmp_vocal.json"
    # midi = json.load(open(midi, "r"))

    # transription = "metadata_ 6705046105425446914.json"
    # utterances = json.load(open(transription, "r"))["lyrics"]["result"][0]["utterances"]
    # # Example usage:
    # phoneme_sequence = [
    #     {'start': 7.392, 'end': 7.8, 'phone': ['sil', 'E0s', 'E0ey', 'en_word_sep', '。', 'en_word_sep']},
    #     {'start': 7.8, 'end': 8.073, 'phone': ['sil', 'E0ah', 'en_word_sep', '。', 'en_word_sep']},
    #     {'start': 8.073, 'end': 8.339, 'phone': ['sil', 'E0l', 'E0ih', 'E0t', 'E0ah', 'E0l', 'en_word_sep', '。', 'en_word_sep']},
    #     # ... (other words)
    # ]

    # note_sequence = [
    #     {'start': 7.66, 'end': 7.86, 'pitch': 71},
    #     {'start': 7.9, 'end': 8.040000000000001, 'pitch': 71},
    #     {'start': 8.120000000000001, 'end': 8.36, 'pitch': 71},
    #     {'start': 8.36, 'end': 8.540000000000001, 'pitch': 72},
    #     {'start': 8.6, 'end': 8.94, 'pitch': 69},
    #     # ... (other notes)
    # ]

    # # Define EN_vowel and EN_consonant outside of this snippet as they are long lists.
    # EN_vowel = [
    #     "E0aa",
    #     "E0ae",
    #     "E0ah",
    #     # ... add other vowels
    # ]

    # Call the function
    # Rap
    # test_6684795195810138114
    # test_6694675719970097154
    # test_6694688147281807362
    # test_id = "6715296497715382273" # vocal_test
    test_id = "6715296497715382273"

    note_sequence = json.load(open(f"./score_test//notes_{test_id}.json", "r"))
    metadata = json.load(open(f"./score_test//metadata_{test_id}.json", "r"))
    utterances = []
    for result in metadata["lyrics"]["result"]:
        utterances.extend(result["utterances"])
    leadsheet_seq = make_leadsheet_from_note_and_utterances(note_sequence, utterances = utterances)
    print(len(leadsheet_seq))

    utterances = utterances[:2]
    leadsheet_seq = make_leadsheet_from_note_and_utterances(note_sequence, utterances = utterances)
    print(len(leadsheet_seq))
