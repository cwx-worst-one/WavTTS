import torch 
import os 
import copy 
import samantha.utils.hdfs_helper as hh
from recipes.musiclm.utils.dist import local_zero_first
from recipes.umm.models import rmvpe
from recipes.bigmusic.datasets.transforms.leadsheet import mark_english_phone_type, mark_vowel_phone_type
import numpy as np 
MAX_PITCH = 127
RMVPE_SAMPLE_RATE = 16000
HOP_SIZE = 160
FRAME_PER_SECOND = HOP_SIZE / RMVPE_SAMPLE_RATE
hz = int (1 / FRAME_PER_SECOND)

def init_rmvpe(hpath, cache_dir='/opt/tiger/samantha/.module_cache/bigmusic'):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    
    with local_zero_first():
        if hpath.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(hpath)}"
            if not os.path.exists(local_path):
                if not hh.get(hpath, local_path):
                    raise ConnectionError(f"Cannot retrieve file from {hpath}.")
        else:
            local_path = hpath
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))
    
        model = rmvpe.RMVPE()
        model.load_and_eval(state_dict)
        return model
    

def f0ToPitch(f0):
    mask = (f0 != 0)
    pitch = torch.zeros_like(f0)
    pitch[mask] = torch.log2(f0[mask] / 27.5) * 12 + 21
    return pitch



def find_max_voiced_range(uv_list, all_range_start, all_range_len):
    '''return max (voiced_len, voiced_start_i)'''
    voiced_start_i = -1
    voiced_len = 0
    voiced_range_list = []
    for i in range(all_range_start, all_range_start + all_range_len):
        if voiced_start_i < 0:
            if uv_list[i] > 0:
                voiced_start_i = i
                voiced_len = 1
        else:
            if uv_list[i] > 0:
                voiced_len += 1
            else:
                voiced_range_list.append((voiced_len, voiced_start_i))
                if voiced_len > all_range_start + all_range_len - i:
                    break
                voiced_start_i = -1
                voiced_len = 0
    
    if voiced_start_i >= 0:
        voiced_range_list.append((voiced_len, voiced_start_i))

    if len(voiced_range_list) <= 0:
        return None

    return max(voiced_range_list)



def get_vuv_range(nums):
    # return unvoiced range
    positions = []
    start = None
    for i in range(len(nums)):
        if nums[i] == 0:
            if start is None:
                start = i
        else:
            if start is not None:
                positions.append([start, i-1])
                start = None
    if start is not None:
        positions.append([start, len(nums)-1])
    
    return positions


def similarity_check_v0(asr_sliced_phones, midi_sliced_notes, tail_sec, thres=0.75):
    # choose the similar one directly, others are abandoned
    # v1 can choose the similar one after legal phone detection
    # new_sliced_phones = [] 
    used_notes = []
    similarities = []
    for word in asr_sliced_phones:
        st, ed = word['start'], word['end']
        related_notes = []
        for note in midi_sliced_notes:
            # if note['start'] <= ed and note['start'] >= st:
            if st <= note['start'] < ed:
                overlap = max(min(note['end'], word['end']) - max(note['start'], word['start']), 0.01) / (ed - st)
                if note['pitch'] == 'Rest' and note['start'] == ed:
                    raise ProcessNoteException("phone: {} note: {}".format(' '.join([str(i) for i in asr_sliced_phones]), ' '.join([str(i) for i in sliced_notes])))
                    # continue
                related_notes += [(note, overlap)]
            elif st == note['start'] and ed == note['end'] and note['pitch']  == 'Rest':
                related_notes += [(note, 1)] 
        if len(related_notes) > 0:
            single_note_sim = sum([i[1] for i in related_notes])
            used_notes.extend([i[0] for i in related_notes])
            similarities.append(single_note_sim)
    
    # if np.array(similarities).mean() < sim_thres:
        # return False

    unmatched_notes =  [i for i in midi_sliced_notes if i not in used_notes]
    unmatched_time = sum([i.get('end') - i.get('start') for i in unmatched_notes])
    out_range_rate = unmatched_time / tail_sec
    sim_score = np.array(similarities).mean()
    confidence = (1 - out_range_rate) * 0.4 + sim_score * 0.6
    # print("confidence {} overlap mean {}  out range rate {}".format(confidence, sim_score, out_range_rate))
    if confidence < thres: return False
    # if unmatched_time / tail_sec > oth_thres:
        # return False

    return True


def force_alignment_v0(asr_sliced_phones, midi_sliced_notes, vuv, f0, tail_sec):
    try:
        pitch = f0ToPitch(f0)
        sil_range = vuv_from_frame_to_second(vuv.squeeze(),phoneme_timestamp=asr_sliced_phones, thres=15)
        new_sliced_notes, new_sliced_phones, flag, = add_sil_in_notes_and_phones(f0, sliced_notes=midi_sliced_notes, sliced_phones=asr_sliced_phones, sil_range=sil_range, tail_sec=tail_sec)
        if not flag:
            return None, None, False

        return new_sliced_phones, new_sliced_notes, True

    # have some tail note error still
    except AssertionError as e:
        if str(e).split('|')[0] == "ExceptionB" or str(e).split('|')[0] == "ExceptionC":
            print(str(e))

        return None, None, False

    except AssertionErrorSilenceNotLegal as e:
        e.display_error()
        return None, None, False

    except ProcessNoteException as e:
        e.display_error()
        return None, None ,False 
    
    except ZeroPitchAssignException as e:
        e.display_error()
        return None, None ,False 

    except Exception as e:
        print("other unknown error")
        return None, None, False



def force_alignment_v1(asr_sliced_phones, midi_sliced_notes, vuv, f0, tail_sec):
    pitch = f0ToPitch(f0)
    sil_range = vuv_from_frame_to_second(vuv.squeeze(),phoneme_timestamp=asr_sliced_phones, thres=15)
    new_sliced_notes, new_sliced_phones, flag, = add_sil_in_notes_and_phones(f0, sliced_notes=midi_sliced_notes, sliced_phones=asr_sliced_phones, sil_range=sil_range, tail_sec=tail_sec)
    if not flag:
        return None, None, False

    legalized_sliced_phones = make_legal_phones(new_sliced_phones, refered_notes=new_sliced_notes)

    # use legal phones check the silence again 
    new_sliced_notes, flag = _recheck_sil_labels_note_with_legalphones(legalized_sliced_phones, new_sliced_notes)
    if not flag:
        return None, None, False   
    
    legalized_sliced_notes = align_note_to_words_v0(legalized_sliced_phones, pitch.squeeze(), vuv.squeeze(), new_sliced_notes)

    # did not cover big silence in phones, with multiple silence in notes
    legalized_sliced_notes = _merge_sil_labels_notes(legalized_sliced_notes, forced_legal=True)

    # for idx in range(1, len(legalized_sliced_notes)):
    #     assert legalized_sliced_notes[idx].get('start') == legalized_sliced_notes[idx - 1].get('end'), "ExceptionB|illegal notes {}".format(' '.join([str(i) for i in legalized_sliced_notes]))
    
    for idx in range(1, len(legalized_sliced_phones)):
        assert legalized_sliced_phones[idx].get('start') == legalized_sliced_phones[idx - 1].get('end'), "ExceptionC|illegal phones {}".format(' '.join([str(i) for i in legalized_sliced_phones]))
    

    return legalized_sliced_phones, legalized_sliced_notes, True
    # have some tail note error still


def force_alignment_v2(asr_sliced_phones, midi_sliced_notes, vuv, f0, tail_sec):
    pitch = f0ToPitch(f0)
    sil_range = vuv_from_frame_to_second(vuv.squeeze(),phoneme_timestamp=asr_sliced_phones, thres=15)
    new_sliced_notes, new_sliced_phones, flag, = add_sil_in_notes_and_phones(f0, sliced_notes=midi_sliced_notes, sliced_phones=asr_sliced_phones, sil_range=sil_range, tail_sec=tail_sec)
    if not flag:
        return None, None, False

    legalized_sliced_phones = make_legal_phones(new_sliced_phones, refered_notes=new_sliced_notes)

    # use legal phones check the silence again 
    new_sliced_notes, flag = _recheck_sil_labels_note_with_legalphones(legalized_sliced_phones, new_sliced_notes)
    if not flag:
        return None, None, False   
    
    legalized_sliced_notes = align_note_to_words_v0(legalized_sliced_phones, pitch.squeeze(), vuv.squeeze(), new_sliced_notes)

    # did not cover big silence in phones, with multiple silence in notes
    legalized_sliced_notes = _merge_sil_labels_notes(legalized_sliced_notes, forced_legal=True, debug_note=new_sliced_notes)
    
    legalized_syllables = notes_word_align(legalized_sliced_notes, legalized_sliced_phones)

    for idx in range(1, len(legalized_syllables)):
        assert legalized_syllables[idx].get('start') == legalized_syllables[idx - 1].get('end'), "ExceptionC|illegal phones {}".format(' '.join([str(i) for i in legalized_syllables]))
    

    return legalized_syllables, legalized_sliced_notes, True
    # have some tail note error still
   



def force_alignment_v3(asr_sliced_phones, midi_sliced_notes, vuv, f0, tail_sec):
    # eliminate some tiny sliced notes
    # 在 notes_word_align_v1 处新增一版ext修正与notes修正
    pitch = f0ToPitch(f0)
    sil_range = vuv_from_frame_to_second(vuv.squeeze(),phoneme_timestamp=asr_sliced_phones, thres=15)
    new_sliced_notes, new_sliced_phones, flag, = add_sil_in_notes_and_phones(f0, sliced_notes=midi_sliced_notes, sliced_phones=asr_sliced_phones, sil_range=sil_range, tail_sec=tail_sec)
    if not flag:
        return None, None, False

    legalized_sliced_phones = make_legal_phones(new_sliced_phones, refered_notes=new_sliced_notes)

    # use legal phones check the silence again 
    new_sliced_notes, flag = _recheck_sil_labels_note_with_legalphones(legalized_sliced_phones, new_sliced_notes)
    if not flag:
        return None, None, False   
    
    legalized_sliced_notes = align_note_to_words_v1(legalized_sliced_phones, pitch.squeeze(), vuv.squeeze(), new_sliced_notes)

    # did not cover big silence in phones, with multiple silence in notes
    legalized_sliced_notes = _merge_sil_labels_notes(legalized_sliced_notes, forced_legal=True, debug_note=new_sliced_notes)
    # legalized_sliced_notes = _finetune_sliced_notes(legalized_sliced_notes)
    
    legalized_syllables = notes_word_align_v1(legalized_sliced_notes, legalized_sliced_phones)

    for idx in range(1, len(legalized_syllables)):
        assert legalized_syllables[idx].get('start') == legalized_syllables[idx - 1].get('end'), "ExceptionC|illegal phones {}".format(' '.join([str(i) for i in legalized_syllables]))
    

    return legalized_syllables, legalized_sliced_notes, True




def phone_type_initialize(legalized_sliced_phones):
    for word in legalized_sliced_phones:
        mark_english_phone_type(word)

    return legalized_sliced_phones


def notes_word_align(legalized_sliced_notes, legalized_sliced_phones):
    # a syllable assign algorithm based on syllable rules
    # cons on the left of vowel is usually one
    # cons on the right of vowel could > 1
    # how to design ext? (actually if meet len(syll_list) < len(note), every )
    syllable_list = []
    for word in legalized_sliced_phones:
        notes, _is_sil = find_related_notes(legalized_sliced_notes, word['start'], word['end'], word['phone'])
        if _is_sil:
            syllable_list.append(word)
            continue 

        vowel_list = mark_vowel_phone_type(word.get('phone'))
        syll_list = split_word_2_syll_v0(vowel_list)

        if len(syll_list) == 1 and len(notes) == 1:
            syllable_list.append(word)
        else:
            if len(notes) > len(syll_list):
                # print("ext founded")
                pointer = 0
                num_vowel, num_ext = sum(vowel_list), len(notes) - len(syll_list)
                for idx in range(len(notes)):
                    syll_start, syll_end = notes[idx]['start'], notes[idx]['end']
                    if idx < len(syll_list) and idx == len(syll_list) - 1:
                        # if vowel_list.index(1) == len(vowel_list) - 1:
                            syllable_list.append({
                                'start': syll_start,
                                'end': syll_end,
                                'pitch': [],
                                'phone': word['phone'][pointer: pointer + syll_list[idx].index(1) + 1]
                            })
                            pointer += syll_list[idx].index(1) + 1
                            # if syll_start == syll_end:
                                # print('here')
                        # else:
                        #     syllable_list.append({
                        #         'start': syll_start,
                        #         'end': syll_end,
                        #         'pitch': [],
                        #         'phone': word['phone'][pointer: pointer + len(syll_list[idx])]
                        #     })

                    elif idx < len(syll_list):
                        syllable_list.append({
                            'start': syll_start,
                            'end': syll_end,
                            'pitch': [],
                            'phone': word['phone'][pointer: pointer + len(syll_list[idx])]
                        })
                        pointer += len(syll_list[idx])
                    else:
                        if num_ext == 1:
                            syllable_list += [{
                                'start': syll_start,
                                'end': syll_end,
                                'pitch': [],
                                'phone': ['ext'] + word['phone'][pointer:]
                            }]
                        else:
                            syllable_list += [{
                                'start': syll_start,
                                'end': syll_end,
                                'pitch': [],
                                'phone': ['ext'] 
                            }]
                            num_ext -= 1
            elif len(notes) == len(syll_list):
                pointer = 0
                for idx in range(len(notes)):
                    syll_start, syll_end = notes[idx]['start'], notes[idx]['end']
                    syllable_list.append({
                        'start': syll_start,
                        'end': syll_end,
                        'pitch': [],
                        'phone': word['phone'][pointer: pointer + len(syll_list[idx])]
                    })
                    pointer += len(syll_list[idx])
            elif len(notes) == 1:
                syllable_list.append(word)
            else:
                pointer = 0 
                for idx in range(len(notes) - 1):
                    syll_start, syll_end = notes[idx]['start'], notes[idx]['end']
                    syllable_list.append({
                        'start': syll_start,
                        'end': syll_end,
                        'pitch': [],
                        'phone': word['phone'][pointer: pointer + len(syll_list[idx])]
                    })
                    pointer += len(syll_list[idx])

                syllable_list.append({
                    'start': syll_end,
                    'end': word['end'],
                    'pitch': [], 
                    'phone': word['phone'][pointer:]
                })

    # assert len(syllable_list) == len(legalized_sliced_notes)
    if len(syllable_list) != len(legalized_sliced_notes):
        raise AlignNoteException("phone: {} note: {}".format(' '.join([str(i) for i in syllable_list]), ' '.join([str(i) for i in legalized_sliced_notes])))
    return syllable_list
 

def _merge_same_pitch_notes_in_syllable(notes, syll_list):  
    # legato test actually
    # e.g. {[1.3,1.4,67], [1.4,1.6, 67]}, {['E0ae', 'E0t']} vowel_list: [1,0], syll_list [[1,0]]

    temp = []
    for i in range(1, len(notes)):
        if notes[i]['pitch'] == notes[i-1]['pitch']:
            temp += [i - 1]
            notes[i]['start'] = notes[i-1]['start']

    return [n for idx, n in enumerate(notes) if idx not in temp] , syll_list    


def notes_word_align_v1(legalized_sliced_notes, legalized_sliced_phones):
    debug_notes = copy.deepcopy(legalized_sliced_notes)
    new_legalized_notes = []
    # a syllable assign algorithm based on syllable rules
    # cons on the left of vowel is usually one
    # cons on the right of vowel could > 1
    # how to design ext? (actually if meet len(syll_list) < len(note), every )
    syllable_list = []
    for word in legalized_sliced_phones:
        notes, _is_sil = find_related_notes(legalized_sliced_notes, word['start'], word['end'], word['phone'])
        if _is_sil:
            syllable_list.append(word)
            new_legalized_notes.extend(notes)
            continue 

        vowel_list = mark_vowel_phone_type(word.get('phone'))
        syll_list = split_word_2_syll_v1(vowel_list)

        if len(notes) > len(syll_list):
            # Remove possible Legato: refine the note list to see if it's wrong in 'ext' arrangement
            notes, syll_list = _merge_same_pitch_notes_in_syllable(notes, syll_list)

        new_legalized_notes.extend(notes)

        if len(syll_list) == 1 and len(notes) == 1:
            syllable_list.append(word)

        else:
            if len(notes) > len(syll_list):
                # print("ext founded")
                pointer = 0
                num_vowel, num_ext = sum(vowel_list), len(notes) - len(syll_list)
                for idx in range(len(notes)):
                    syll_start, syll_end = notes[idx]['start'], notes[idx]['end']
                    if idx < len(syll_list) and idx == len(syll_list) - 1:

                            syllable_list.append({
                                'start': syll_start,
                                'end': syll_end,
                                'pitch': [],
                                'phone': word['phone'][pointer: pointer + syll_list[idx].index(1) + 1]
                            })
                            pointer += syll_list[idx].index(1) + 1

                    elif idx < len(syll_list):
                        syllable_list.append({
                            'start': syll_start,
                            'end': syll_end,
                            'pitch': [],
                            'phone': word['phone'][pointer: pointer + len(syll_list[idx])]
                        })
                        pointer += len(syll_list[idx])
                    else:
                        if num_ext == 1:
                            syllable_list += [{
                                'start': syll_start,
                                'end': syll_end,
                                'pitch': [],
                                'phone': ['ext'] + word['phone'][pointer:]
                            }]
                        else:
                            syllable_list += [{
                                'start': syll_start,
                                'end': syll_end,
                                'pitch': [],
                                'phone': ['ext'] 
                            }]
                            num_ext -= 1
            elif len(notes) == len(syll_list):
                pointer = 0
                for idx in range(len(notes)):
                    syll_start, syll_end = notes[idx]['start'], notes[idx]['end']
                    syllable_list.append({
                        'start': syll_start,
                        'end': syll_end,
                        'pitch': [],
                        'phone': word['phone'][pointer: pointer + len(syll_list[idx])]
                    })
                    pointer += len(syll_list[idx])
            elif len(notes) == 1:
                syllable_list.append(word)
            else:
                pointer = 0 
                for idx in range(len(notes) - 1):
                    syll_start, syll_end = notes[idx]['start'], notes[idx]['end']
                    syllable_list.append({
                        'start': syll_start,
                        'end': syll_end,
                        'pitch': [],
                        'phone': word['phone'][pointer: pointer + len(syll_list[idx])]
                    })
                    pointer += len(syll_list[idx])

                syllable_list.append({
                    'start': syll_end,
                    'end': word['end'],
                    'pitch': [], 
                    'phone': word['phone'][pointer:]
                })

    if len(syllable_list) != len(new_legalized_notes):
        raise AlignNoteException("phone: {} note: {}".format(' '.join([str(i) for i in syllable_list]), ' '.join([str(i) for i in legalized_sliced_notes])))
    return syllable_list
 



def split_word_2_syll_v0(vowel_list):
    partitions = []
    start = 0
    end = 0


    while end < len(vowel_list):
        if vowel_list[end] == 1:
            if len(vowel_list) > end > 0 and vowel_list[end-1] == 0:
                # 当前值为1且上一个值为0，属于同一部分
                # end += 1
                while end < len(vowel_list) and vowel_list[end] == 1:
                    end += 1
                while end < len(vowel_list) and vowel_list[end] != 1:
                    end += 1
                # print(end)
                # print(partitions)
                if end != len(vowel_list):
                    # partition = vowel_list[start: end]
                # else:
                    partition = vowel_list[start:end - 1]
                    start = end - 1
                    end = end - 1
                
                    partitions.append(partition)
            else:
                # 当前值为1，属于同一部分
                end += 1
        else:
            # 当前值为0，属于同一部分
            end += 1

    if end > start:
        partition = vowel_list[start:end]
        partitions.append(partition)

    return partitions



def split_word_2_syll_v1(vowel_list):
    # 新增处理开头为1的单音节逻辑
    partitions = []
    start = 0
    end = 0

    while end < len(vowel_list):
        if vowel_list[end] == 1:
            if len(vowel_list) > end > 0 and vowel_list[end-1] == 0:
                # 当前值为1且上一个值为0，属于同一部分
                # end += 1
                while end < len(vowel_list) and vowel_list[end] == 1:
                    end += 1
                while end < len(vowel_list) and vowel_list[end] != 1:
                    end += 1

                if end != len(vowel_list):
                    
                    partition = vowel_list[start:end - 1]
                    start = end - 1
                    end = end - 1
                    partitions.append(partition)

            elif end == 0:
                while end < len(vowel_list) and vowel_list[end] == 1:
                    end += 1
                while end < len(vowel_list) and vowel_list[end] != 1:
                    end += 1
                if end != len(vowel_list):
                    partition = vowel_list[start:end - 1]
                    start = end - 1
                    end = end - 1
                    partitions.append(partition)

            else:
                # 当前值为1，属于同一部分
                end += 1
        else:
            # 当前值为0，属于同一部分
            end += 1

    if end > start:
        partition = vowel_list[start:end]
        partitions.append(partition)

    return partitions



def find_related_notes(legalized_sliced_notes, start, end, phones):
    is_sil = False
    if len(phones) == 1 and phones[0] == 'sil':
        is_sil = True

    notes = []
    note_lens = len(legalized_sliced_notes)
    i = 0
    st = None
    while i < note_lens:
        note = legalized_sliced_notes[i]
        if note.get('start') == start:
            st = i

            if note.get('end') == end:
                notes.append(note)
                break

            while i < note_lens and legalized_sliced_notes[i].get('end') != end:
                i += 1
            notes.extend(legalized_sliced_notes[st:i + 1])
            break
        
        i += 1 
    assert st is not None 
    notes = [note for note in notes if not (note['start'] == note['end'] and note['pitch'] == 'Rest')]

    return notes, is_sil




# ############### silence functions ############
def _merge_sil_labels_phones(ft_sliced_phones, forced_legal=False):
    new_sliced_phones = [] 
    idx = 0 
    while idx < len(ft_sliced_phones):
        if 'sil' not in ft_sliced_phones[idx].get('phone'):
            new_sliced_phones += [ft_sliced_phones[idx]]
            idx += 1
        else:
            start = ft_sliced_phones[idx]['start']
            while idx < len(ft_sliced_phones) and ft_sliced_phones[idx].get('phone')[0] == 'sil':
                idx += 1
            if idx >= len(ft_sliced_phones):
                end = ft_sliced_phones[idx - 1]['end']
            else:
                end = ft_sliced_phones[idx]['start']
            # make sure there is no point range silence here
            if start != end:
                new_sliced_phones.append({
                    'start':start,
                    'end': end,
                    'pitch': [],
                    'phone': ['sil']
                })

    if forced_legal:
        for idx in range(1, len(new_sliced_phones)):
            assert new_sliced_phones[idx].get('start') == new_sliced_phones[idx - 1].get('end'), (new_sliced_phones[idx], new_sliced_phones[idx - 1], new_sliced_phones)
        
    # should be double checked
    return new_sliced_phones


def _merge_sil_labels_notes(ft_sliced_notes, forced_legal=False, debug_note=None):
    new_sliced_notes = [] 
    idx = 0 
    while idx < len(ft_sliced_notes):
        if ft_sliced_notes[idx].get('pitch') != 'Rest':
            if ft_sliced_notes[idx]['start'] != ft_sliced_notes[idx]['end']:
                new_sliced_notes += [ft_sliced_notes[idx]]
            else:
                if idx < len(ft_sliced_notes) - 1 and ft_sliced_notes[idx + 1]['pitch'] != ft_sliced_notes[idx]['pitch']:
                    print(ft_sliced_notes[idx], ft_sliced_notes[idx + 1])
            idx += 1
        else:
            start = ft_sliced_notes[idx]['start']
            while idx < len(ft_sliced_notes) and ft_sliced_notes[idx].get('pitch') == 'Rest':
                idx += 1
            if idx >= len(ft_sliced_notes):
                end = ft_sliced_notes[idx - 1]['end']
            else:
                end = ft_sliced_notes[idx]['start']

            if start != end:
                new_sliced_notes.append({
                    'start':start,
                    'end': end,
                    'pitch': 'Rest',
                    'phone': []
                })

    if forced_legal:
        for idx in range(1, len(new_sliced_notes)):
            assert new_sliced_notes[idx].get('start') == new_sliced_notes[idx - 1].get('end'), "ExceptionB|illegal notes {} debug| {}".format(' '.join([str(i) for i in new_sliced_notes]), ' '.join([str(i) for i in debug_note]))
        
    # should be double checked
    return new_sliced_notes



def _recheck_sil_labels_note_v2(sliced_notes, sliced_phones=None, sil_phones_timesteps=None):
    '''
    given silence range, to return a sliced_notes list with silences, do adjust the positions related to added silence
    '''
    assert sil_phones_timesteps is not None or sliced_phones is not None
    if sil_phones_timesteps is None:
        sil_phones_timesteps = []
        new_sliced_notes = copy.deepcopy(sliced_notes)
        for v in sliced_phones:
            if 'sil' in v.get('phone') and len(v.get('phone')) == 1:
                sil_phones_timesteps.append([v.get('start'), v.get('end')])

    for j, new_itvl in enumerate(sil_phones_timesteps):
        for i, interval in enumerate(sliced_notes):
            if new_itvl[0] > interval['end']:
                continue
            elif new_itvl[1] < interval['start']:
                # handle cases like [
                # {'start': 1.74, 'end': 1.96, 'pitch': 68, 'phone': []}
                # {'start': 2.025, 'end': 2.4875, 'pitch': 'Rest', 'phone': []}
                # {'start': 2.5, 'end': 3.08, 'pitch': 73, 'phone': []}']
                if i == 0 or new_itvl[0] > sliced_notes[i-1]['end']:
                    sliced_notes.insert(i,{
                        'start': new_itvl[0],
                        'end': new_itvl[1],
                        'pitch': 'Rest',
                        'phone': [],
                    })
                    break

            elif new_itvl[1] == interval['end'] and new_itvl[0] == interval['start']:
                interval['pitch'] = 'Rest'
                break
            else:
                # handle head and tail cases
                if interval['pitch'] == 'Rest':
                    if i == (len(sliced_notes) - 1):
                        interval['start'], interval['end'] = new_itvl[0], new_itvl[1]
                        sliced_notes[i - 1]['end'] = interval['start']
                    else:
                        interval['start'] = max(new_itvl[0], interval['start'])
                        interval['end'] = min(new_itvl[1], interval['end'])
                    break
                else:
                    if interval['end'] > new_itvl[0] > interval['start']:
                        if (i < len(sliced_notes) - 1 and interval['end'] < new_itvl[1]  <= sliced_notes[i+1]['start']) or  (i == len(sliced_notes) - 1 and interval['end'] < new_itvl[1]):
                            interval['end'] =  new_itvl[0]
                            sliced_notes.insert(i + 1,{
                            'start': new_itvl[0],
                            'end': new_itvl[1],
                            'pitch': 'Rest',
                            'phone': [],
                        })
                        elif new_itvl[1] < interval['end']:
                            tmp_end = interval['end']
                            tmp_note = interval['pitch']
                            interval['end'] = new_itvl[0]
                            if tmp_end - new_itvl[1] >= 0.5:
                                sliced_notes[i+1 : i+1] = [{
                                    'start': new_itvl[0],
                                    'end': new_itvl[1],
                                    'pitch': 'Rest',
                                    'phone': [],
                                }, {
                                    'start': new_itvl[1],
                                    'end': tmp_end,
                                    'pitch': tmp_note,
                                    'phone': [],
                                }]
                            else:
                                sliced_notes[i+1 : i+1] = [{
                                    'start': new_itvl[0],
                                    'end': tmp_end,
                                    'pitch': 'Rest',
                                    'phone': [],
                                }]

                        elif new_itvl[0] < new_itvl[1] and new_itvl[1] == interval['end']:
                            interval['end'] = new_itvl[0]
                            sliced_notes.insert(i + 1,{
                            'start': new_itvl[0],
                            'end': new_itvl[1],
                            'pitch': 'Rest',
                            'phone': [],})
                        else:
                            # print("not handled when sil is so long")
                            interval['end'] = new_itvl[0] 
                            next_note = sliced_notes[i + 1]
                            if next_note['pitch'] == 'Rest':
                                next_note['start'] = new_itvl[0] 
                            else:
                                next_note['start'] = new_itvl[1]
                                sliced_notes.insert(i + 1,{
                                'start': new_itvl[0],
                                'end': new_itvl[1],
                                'pitch': 'Rest',
                                'phone': [],})

                    elif new_itvl[0] == interval['end']:
                        next_note = sliced_notes[i + 1]
                        if next_note['pitch'] == 'Rest':
                            next_note['start'] = new_itvl[0]
                        else:
                            if new_itvl[1] > next_note['start']:
                                next_note['start'] = new_itvl[1]
                            sliced_notes.insert(i+1, {
                            'start': new_itvl[0],
                            'end': new_itvl[1],
                            'pitch': 'Rest',
                            'phone': [],
                            })

                    elif new_itvl[0] <= interval['start']:
                        interval['start'] = new_itvl[1]
                        sliced_notes.insert(i, {
                            'start': new_itvl[0],
                            'end': new_itvl[1],
                            'pitch': 'Rest',
                            'phone': [],})
                    else:
                        # both {'start': 5.9375, 'end': 6.0, 'pitch': 48, 'phone': []}, {'start': 5.9375, 'end': 6.0, 'pitch': 'Rest', 'phone': []}
                        print("not implemented")
                    break

    if sliced_notes[-1]['end'] <= sliced_notes[-1]['start']:
        sliced_notes[-2]['end'] = sliced_notes[-1]['end']
        sliced_notes = sliced_notes[:-1]
        
    for idx in range(1, len(sliced_notes)):
        assert sliced_notes[idx]['start'] >= sliced_notes[idx - 1]['end'], (sliced_notes, sil_phones_timesteps, new_sliced_notes)
        # assert sliced_notes[idx]['end'] > sliced_notes[idx]['start'], ('------ wrong timeline-',sliced_notes)
        if sliced_notes[idx]['end'] < sliced_notes[idx]['start']:
            return sliced_notes, False
        
    return sliced_notes, True


def _recheck_sil_labels_phone_v2(sliced_notes, f0, sliced_phones=None, sil_phones_timesteps=None):
    assert sil_phones_timesteps is not None or sliced_phones is not None
    if sil_phones_timesteps is None:
        sil_phones_timesteps = []
        new_sliced_notes = copy.deepcopy(sliced_notes)
        for v in sliced_phones:
            if v.get('pitch') == 'Rest':
                sil_phones_timesteps.append([v.get('start'), v.get('end')])

    invalid_list = [] 
    for j, new_itvl in enumerate(sil_phones_timesteps):
        for i, interval in enumerate(sliced_notes):
            if new_itvl[0] > interval['end']:
                continue
            elif new_itvl[1] < interval['start']:
                # handle cases like [
                # {'start': 1.74, 'end': 1.96, 'pitch': 68, 'phone': []}
                # {'start': 2.025, 'end': 2.4875, 'pitch': 'Rest', 'phone': []}
                # {'start': 2.5, 'end': 3.08, 'pitch': 73, 'phone': []}']
                if i == 0 or new_itvl[0] > sliced_notes[i-1]['end']:
                    sliced_notes.insert(i,{
                        'start': new_itvl[0],
                        'end': new_itvl[1],
                        'pitch': [],
                        'phone': ['sil'],
                    })
                    break

            elif new_itvl[1] == interval['end'] and new_itvl[0] == interval['start']:
                interval['phone'] = ['sil']
                break
            else:
                # handle head and tail cases
                # if interval['pitch'] == 'Rest':
                #     if i == (len(sliced_notes) - 1):
                #         interval['start'], interval['end'] = new_itvl[0], new_itvl[1]
                #         sliced_notes[i - 1]['end'] = interval['start']
                #     else:
                #         interval['start'] = max(new_itvl[0], interval['start'])
                #         interval['end'] = min(new_itvl[1], interval['end'])
                #     break
                # else:
                if interval['end'] > new_itvl[0] > interval['start']:
                    if i < len(sliced_notes) - 1 and interval['end'] < new_itvl[1]  <= sliced_notes[i+1]['start']:
                        interval['end'] =  new_itvl[0]
                        sliced_notes.insert(i + 1,{
                        'start': new_itvl[0],
                        'end': new_itvl[1],
                        'pitch': [],
                        'phone': ['sil'],
                    })
                    elif i == len(sliced_notes) - 1 and interval['end'] < new_itvl[1]:
                        interval['end'] =  new_itvl[0]
                        sliced_notes.insert(i + 1,{
                        'start': new_itvl[0],
                        'end': new_itvl[1],
                        'pitch': [],
                        'phone': ['sil'],
                    })
                    elif new_itvl[1] < interval['end']:
                        tmp_end = interval['end']
                        tmp_phone = interval['phone']
                        interval['end'] = new_itvl[0]
                        if tmp_end - new_itvl[1] < 0.5:
                            # 要
                            # 怎么要， 分音素能分出来就直接分音素，然后加sil
                            sliced_notes[i+1 : i+1] = [{
                                'start': new_itvl[0],
                                'end': new_itvl[1],
                                'pitch': [],
                                'phone': ['sil'],
                            }, {
                                'start': new_itvl[1],
                                'end': tmp_end,
                                'pitch': [],
                                'phone': tmp_phone,
                            }]
                            if 0 in f0[int(interval['start'] * hz):int(interval['end'] * hz)] or len(f0[int(interval['start'] * hz):int(interval['end'] * hz)]) < 10:
                                interval['phone'] = ['sil']

                        else:
                            invalid_list.append(new_itvl)# 不要
                            break 
                            # sliced_notes[i+1 : i+1] = [{
                            #     'start': new_itvl[0],
                            #     'end': tmp_end,
                            #     'pitch': [],
                            #     'phone': ['sil'],
                            # }]

                    elif new_itvl[0] < new_itvl[1] and new_itvl[1] == interval['end']:
                        interval['end'] = new_itvl[0]
                        sliced_notes.insert(i + 1,{
                        'start': new_itvl[0],
                        'end': new_itvl[1],
                        'pitch': [],
                        'phone': ['sil'],})
                    else:
                        # print("not handled when sil is so long")
                        interval['end'] = new_itvl[0] 
                        next_note = sliced_notes[i + 1]
                        if next_note['phone'][0] == 'sil':
                            next_note['start'] = new_itvl[0] 
                        else:
                            if next_note['start'] < new_itvl[1]:
                                next_note['start'] = new_itvl[1]
                            sliced_notes.insert(i + 1,{
                            'start': new_itvl[0],
                            'end': new_itvl[1],
                            'pitch': [],
                            'phone': ['sil'],})

                elif new_itvl[0] == interval['end']:
                    next_note = sliced_notes[i + 1]
                    if next_note['phone'][0] == 'sil':
                        next_note['start'] = new_itvl[0]
                    else:
                        if new_itvl[1] > next_note['start']:
                            next_note['start'] = new_itvl[1]
                        sliced_notes.insert(i+1, {
                        'start': new_itvl[0],
                        'end': new_itvl[1],
                        'pitch': [],
                        'phone': ['sil'],
                        })

                elif new_itvl[0] <= interval['start']:
                    interval['start'] = new_itvl[1]
                    sliced_notes.insert(i, {
                        'start': new_itvl[0],
                        'end': new_itvl[1],
                        'pitch': [],
                        'phone': ['sil'],})
                else:
                    # both {'start': 5.9375, 'end': 6.0, 'pitch': 48, 'phone': []}, {'start': 5.9375, 'end': 6.0, 'pitch': 'Rest', 'phone': []}
                    print("not implemented")
                break

    if sliced_notes[-1]['end'] <= sliced_notes[-1]['start']:
        sliced_notes[-2]['end'] = sliced_notes[-1]['end']
        sliced_notes = sliced_notes[:-1]
        
    for idx in range(1, len(sliced_notes)):
        assert sliced_notes[idx]['start'] >= sliced_notes[idx - 1]['end'], (sliced_notes, sil_phones_timesteps, new_sliced_notes)
        # assert sliced_notes[idx]['end'] > sliced_notes[idx]['start'], ('------ wrong timeline-',sliced_notes)
        if sliced_notes[idx]['end'] < sliced_notes[idx]['start']:
            return sliced_notes, False, []
        
    return sliced_notes, True, invalid_list


def add_sil_in_notes_and_phones(f0, sliced_notes, sliced_phones, sil_range, tail_sec):
    new_sliced_notes, note_flag = _recheck_sil_labels_note_v2(sliced_notes, sil_phones_timesteps=sil_range)

    new_sliced_phones, phone_flag, invalid_list = _recheck_sil_labels_phone_v2(sliced_phones, f0.squeeze(), sliced_phones=new_sliced_notes)
    try:
        assert len(invalid_list) == 0, "ExceptionA|{}".format(' '.join(invalid_list))
    except AssertionError as e:
        if str(e).split('|')[0] == "ExceptionA":
            raise AssertionErrorSilenceNotLegal(str(e).split('|')[1])

    if not note_flag or not phone_flag: return None, None, False     

    new_sliced_notes[-1]['end'] = tail_sec
    new_sliced_phones[-1]['end'] = tail_sec
    # There's the RISK when tail note and tail ASR result are wrong, 比如asr结果或者midi结果在最后被遗漏


    return new_sliced_notes, new_sliced_phones, True


def _recheck_sil_labels_note_with_legalphones(legalized_sliced_phones, sliced_notes):
    new_sliced_notes, note_flag = _recheck_sil_labels_note_v2(sliced_notes, sliced_phones=legalized_sliced_phones)
    
    return new_sliced_notes, note_flag


def make_legal_phones(sliced_phones, refered_notes):
    # leave head in case there're some asr results miss some phones 
    # sliced_phones = _merge_sil_labels_phones(sliced_phones, forced_legal=False)
    speech_thres = 1.0
    valid_time = []
    start = sliced_phones[0].get('start')

    # detailed version ，先不用

    # for i in range(1,len(sliced_phones) - 1):
    #     if sliced_phones[i-1]['phone'][0] == 'sil' and sliced_phones[i+1]['phone'][0] == 'sil' and sliced_phones[i]['phone'][0] != 'sil' :
    #         has_speech_notes = [j for j in refered_notes if j['start'] >= sliced_phones[i-1]['end'] and j['end'] <= sliced_phones[i+1]['start']]
    #         if not has_speech_notes and sliced_phones[i + 1]['start'] - sliced_phones[i - 1]['end'] > speech_thres:
    #             valid_time.append((start, sliced_phones[i]['start']))
    #             start = sliced_phones[i].get('end')

    # if there're time ranges when midi has no value while asr has results e.g. speech
    valid_time.append((start, sliced_phones[-1].get('end')))

    result = []
    for time_range in valid_time:
        st, ed = time_range
        result.extend(legalize(sliced_phones, st, ed, 'phone'))

    return result


def legalize(leadsheets, start, end, type='phone'):
    part_leadsheets = [i for i in leadsheets if i['start'] >= start and i['end'] <= end]
    new_leadsheet = [part_leadsheets[0]] 
    special_case = ['sil']
    
    for idx in range(1, len(part_leadsheets)):
        
        if part_leadsheets[idx].get('start') >= start and part_leadsheets[idx].get('end') <= end:
            
            if part_leadsheets[idx]['phone'] == special_case and part_leadsheets[idx - 1]['phone'] == special_case:
                new_leadsheet.append(part_leadsheets[idx])
                continue
            # TBD
            # could be a risk when asr result forget some words

            if part_leadsheets[idx]['start'] != part_leadsheets[idx - 1]['end']:    
                if type == 'phone' and part_leadsheets[idx - 1]['phone'] != special_case:
                    part_leadsheets[idx - 1]['end'] = part_leadsheets[idx]['start']

                elif type == 'phone' and part_leadsheets[idx - 1]['phone'] == special_case:
                    part_leadsheets[idx]['start'] = part_leadsheets[idx - 1]['end']
                
                elif type == 'pitch' and part_leadsheets[idx - 1]['pitch'] != 'Rest':
                    part_leadsheets[idx - 1]['end'] = part_leadsheets[idx]['start']
                
                elif type == 'pitch' and part_leadsheets[idx - 1]['pitch'] == 'Rest':
                    part_leadsheets[idx]['start'] = part_leadsheets[idx - 1]['end']

            new_leadsheet.append(part_leadsheets[idx])


    new_leadsheet = _merge_sil_labels_phones(new_leadsheet, forced_legal=True)

    return new_leadsheet


def align_note_to_words_v0(legal_sliced_phones, pitch, uv, sliced_notes):
    # 是以一种向后看的思路去算的
    # word level alignment for note
    aligned_notes = []
    for word in legal_sliced_phones:
        st, ed = word['start'], word['end']
        related_notes = []
        for note in sliced_notes:
            # if note['start'] <= ed and note['start'] >= st:
            if st <= note['start'] < ed:
                overlap = max(min(note['end'], word['end']) - max(note['start'], word['start']), 0.01) / (ed - st)
                if note['pitch'] == 'Rest' and note['start'] == ed:
                    raise ProcessNoteException("phone: {} note: {}".format(' '.join([str(i) for i in legal_sliced_phones]), ' '.join([str(i) for i in sliced_notes])))
                    # continue
                related_notes += [(note, overlap)]
            elif st == note['start'] and ed == note['end'] and note['pitch']  == 'Rest':
                related_notes += [(note, 1)]
        # 有问题出在phone不是sil但是对应的时间内note内部有Rest的情况，下面的方法没有解决这种问题
        new_notes = assign_note_to_word_v0(word['phone'], related_notes, pitch, uv, st, ed, legal_sliced_phones, sliced_notes)
        if len(aligned_notes) == 0 or new_notes[0] != aligned_notes[-1]:
            aligned_notes.extend(new_notes)

    return aligned_notes


def align_note_to_words_v1(legal_sliced_phones, pitch, uv, sliced_notes):
    # 新增零碎notes 扔掉
    # word level alignment for note
    # 
    aligned_notes = []
    for word in legal_sliced_phones:
        st, ed = word['start'], word['end']
        related_notes = []
        for note in sliced_notes:
            # if note['start'] <= ed and note['start'] >= st:
            if st <= note['start'] < ed:
                overlap = max(min(note['end'], word['end']) - max(note['start'], word['start']), 0.01) / (ed - st)
                if note['pitch'] == 'Rest' and note['start'] == ed:
                    raise ProcessNoteException("phone: {} note: {}".format(' '.join([str(i) for i in legal_sliced_phones]), ' '.join([str(i) for i in sliced_notes])))
                    # continue
                related_notes += [(note, overlap)]
            elif st == note['start'] and ed == note['end'] and note['pitch']  == 'Rest':
                related_notes += [(note, 1)]
        # 有问题出在phone不是sil但是对应的时间内note内部有Rest的情况，下面的方法没有解决这种问题
        new_notes = assign_note_to_word_v0(word['phone'], related_notes, pitch, uv, st, ed, legal_sliced_phones, sliced_notes)

        if len(new_notes) > 1:
            temp = []
            for i in range(len(new_notes)):
                if new_notes[i]['end'] - new_notes[i]['start'] <= 0.04 and i != len(new_notes) - 1:
                    new_notes[i + 1]['start'] = new_notes[i]['start']
                    temp += [i]
                elif new_notes[i]['end'] - new_notes[i]['start'] <= 0.04 and i == len(new_notes) - 1:
                    new_notes[i - 1]['end'] = new_notes[i]['end']
                    temp += [i]
            new_notes = [tmp_note for idx, tmp_note in enumerate(new_notes) if idx not in temp]

        if len(aligned_notes) == 0 or new_notes[0] != aligned_notes[-1]:
            aligned_notes.extend(new_notes)

    return aligned_notes



def assign_note_to_word_v0(phones, related_notes, pitch, uv ,st, ed, sliced_phones=None, sliced_notes=None):
    # version 0 just fill in the blank part 
    st_int, ed_int = round(st * hz), round(ed * hz)
    if len(related_notes) == 0:
        if phones[0] == 'sil':
            print('here wrong silence')
            return [{
            'start': st, 
            'end':ed,
            'pitch': 'Rest',
            'phone': []
        }]
        # could be bugs for non
        if len(pitch[st_int: ed_int][uv[st_int: ed_int]]) == 0:
            if phones != ['sil']:
                value = 0

        else:
            value = int(pitch[st_int: ed_int][uv[st_int: ed_int]].mean().int())
        return [{
            'start': st, 
            'end':ed,
            'pitch': value,
            'phone': []
        }]
        
    if phones[0] == 'sil':

        # assert len(related_notes) == 1 and related_notes[0][0]['start'] == st and related_notes[0][0]['end'] == ed
        return [i[0] for i in related_notes if i[0]['pitch'] == 'Rest']
    
    elif len(related_notes) == 1 and related_notes[0][1] > 0.2:
        if related_notes[0][0].get('pitch') == 'Rest':
            if len(pitch[st_int: ed_int][uv[st_int: ed_int]]) > 0 and phones[0] != 'sil':
                value = int(pitch[st_int: ed_int][uv[st_int: ed_int]].mean().int())
                if related_notes[-1][0]['end'] == sliced_phones[-1]['end'] and related_notes[-1][0]['pitch'] == "Rest":
                    # 处理有的时候错误给结尾的asr 字，但是实际上是silence的情况
                    sliced_phones[-1]['end'] = related_notes[-1][0]['start']
                    if sliced_phones[-1] != {'start': related_notes[-1][0]['start'], 'end': related_notes[-1][0]['end'], 'phone': ['sil'], 'pitch': []}:
                        sliced_phones.append({'start': related_notes[-1][0]['start'], 'end': related_notes[-1][0]['end'], 'phone': ['sil'], 'pitch': []})
                    
                    additional_note = {'start': st, 'end': related_notes[-1][0]['start'], 'pitch': value, 'phone': []}
                    return [additional_note, related_notes[0][0]]
            else:
                raise ZeroPitchAssignException("phone: {} note: {}".format(' '.join([str(i) for i in sliced_phones]), ' '.join([str(i) for i in sliced_notes])))
        else:
            value = max(min(MAX_PITCH, related_notes[0][0].get('pitch')), 0)
        return [{
            'start': st, 
            'end':ed,
            'pitch': value,
            'phone': []
        }]
    
    elif len(related_notes) == 1 and related_notes[0][1] <= 0.2:
        if related_notes[0][0].get('pitch') == 'Rest':
            if len(pitch[st_int: ed_int][uv[st_int: ed_int]]) > 0 and phones[0] != 'sil':
                value = int(pitch[st_int: ed_int][uv[st_int: ed_int]].mean().int())
                if related_notes[-1][0]['end'] == sliced_phones[-1]['end'] and related_notes[-1][0]['pitch'] == "Rest":
                    # 处理有的时候错误给结尾的asr 字，但是实际上是silence的情况
                    sliced_phones[-1]['end'] = related_notes[-1][0]['start']
                    if sliced_phones[-1] != {'start': related_notes[-1][0]['start'], 'end': related_notes[-1][0]['end'], 'phone': ['sil'], 'pitch': []}:
                        sliced_phones.append({'start': related_notes[-1][0]['start'], 'end': related_notes[-1][0]['end'], 'phone': ['sil'], 'pitch': []})
                    
                    additional_note = {'start': st, 'end': related_notes[-1][0]['start'], 'pitch': value, 'phone': []}
                    return [additional_note, related_notes[0][0]]

            else:
                raise ZeroPitchAssignException("phone: {} note: {}".format(' '.join([str(i) for i in sliced_phones]), ' '.join([str(i) for i in sliced_notes])))
                                
        else:
            value = max(min(MAX_PITCH, related_notes[0][0].get('pitch')), 0)
        return [{
            'start': st, 
            'end':ed,
            'pitch':  value,
            'phone': []
        }]

    else:
        # notes = [] 
        related_notes[0][0]['start'] = st
        for idx in range(1, len(related_notes)):
            if related_notes[idx][0].get('start') != related_notes[idx - 1][0].get('end'):
                related_notes[idx][0]['start'] = related_notes[idx - 1][0]['end']

        related_notes[-1][0]['end'] = ed 

        if related_notes[-1][0]['end'] == sliced_phones[-1]['end'] and related_notes[-1][0]['pitch'] == "Rest":
            # 处理有的时候错误给结尾的asr 字，但是实际上是silence的情况
            sliced_phones[-1]['end'] = related_notes[-1][0]['start']
            if sliced_phones[-1] != {'start': related_notes[-1][0]['start'], 'end': related_notes[-1][0]['end'], 'phone': ['sil'], 'pitch': []}:
                sliced_phones.append({'start': related_notes[-1][0]['start'], 'end': related_notes[-1][0]['end'], 'phone': ['sil'], 'pitch': []})
            return [i[0] for i in related_notes]

        # v0 处理ext
        # incase there are some ext!
        non_rest_related_notes = [i[0] for i in related_notes if i[0]['pitch'] != 'Rest']
        for idx in range(1, len(non_rest_related_notes)):
            if non_rest_related_notes[idx].get('start') != non_rest_related_notes[idx - 1].get('end'):
                non_rest_related_notes[idx]['start'] = non_rest_related_notes[idx - 1]['end']
        
        if non_rest_related_notes[0].get('start') != st:
            # 防止开头rest
            non_rest_related_notes[0]['start'] = st

        if non_rest_related_notes[-1].get('end') != ed:
            non_rest_related_notes[-1]['end'] = ed

        assert non_rest_related_notes[0].get('start') == st and non_rest_related_notes[-1].get('end') == ed, "ExceptionD|assignNoteswrong {}".format(' '.join([str(i) for i in sliced_notes]))
        return non_rest_related_notes




def vuv_from_frame_to_second(vuv, thres=15, phoneme_timestamp=None):
    # tail_sec = audio.shape[-1] / sample_rate
    positions = get_vuv_range(vuv.float())
    unvoiced_time = [[i[0] * FRAME_PER_SECOND, i[1] * FRAME_PER_SECOND] for i in positions if (i[1] - i[0]) > thres]
    invalid = []

    if phoneme_timestamp is not None:
        for time_sec in unvoiced_time:
            start, end = time_sec
            for item in phoneme_timestamp:
                if start >= item.get('start') and end <= item.get('end') and item.get('phone')[0] != 'sil':
                    invalid.append(time_sec)
                    # 理论上应该杜绝所有在phone有真实值的时候，存在silence的情况

    return [i for i in unvoiced_time if i not in invalid]




def force_align_pretrain_note_to_sft_note_with_pitch_automidiv5(sliced_phones, sliced_notes, pitch):
    sft_sliced_notes = [] 
    for t in sliced_phones:
        value_lengths = {}
        for s in sliced_notes:
            if s['start'] <= t['end'] and s['end'] >= t['start']:
                overlap = max(min(s['end'], t['end']) - max(s['start'], t['start']), 0.01)
                if s['pitch'] in value_lengths:
                    value_lengths[s['pitch']] += overlap
                else:
                    value_lengths[s['pitch']] = overlap 
        if value_lengths:
            longest_overlap_value = max(value_lengths, key=value_lengths.get)
            if max(value_lengths.values()) < 0.2 and t['phone'] != ['sil']:
                longest_overlap_value = int(pitch[round(t['start'] * hz) : round(t['end'] * hz)].median().round().int())
                if longest_overlap_value == 0:
                    nonzero = pitch[round(t['start'] * hz) : round(t['end'] * hz)].nonzero().squeeze()
                    longest_overlap_value = int(pitch[round(t['start'] * hz) : round(t['end'] * hz)][nonzero].median().round().int())

                longest_overlap_value = max(min(MAX_PITCH, longest_overlap_value), 0)
            
            elif t['phone'] == ['sil']:
                longest_overlap_value = 'Rest'
            sft_sliced_notes.append({
                'start' : t['start'],
                'end': t['end'],
                'pitch': longest_overlap_value,
                'phone':[]
            })
        else:
            if t['phone'] != ['sil']:
                note = int(pitch[round(t['start'] * hz) : round(t['end'] * hz)].median().round().int())
                if note == 0:
                    nonzero = pitch[round(t['start'] * hz) : round(t['end'] * hz)].nonzero().squeeze()
                    note = int(pitch[round(t['start'] * hz) : round(t['end'] * hz)][nonzero].median().round().int())
                
                note = max(min(MAX_PITCH, note), 0)
            
            else:
                note = 'Rest'

            sft_sliced_notes.append({
                        'start' : t['start'],
                        'end': t['end'],
                        'pitch': note,
                        'phone':[]
                    })

    
    assert len(sliced_phones) == len(sft_sliced_notes)
    return sft_sliced_notes




class F0_Handler:
    def __init__(self, hpath, src_sample_rate, cache_dir=None) -> None:
        self.f0_model = init_rmvpe(hpath, cache_dir)
        self.source_sample_rate = src_sample_rate


    def infer_f0(self, audio):
        f0 = self.f0_model.infer_from_audio(audio.squeeze().numpy(), self.source_sample_rate, device=torch.device('cpu'), thred=0.03, use_viterbi=False)
        vuv = f0 != 0
        return f0, vuv

    def infer_pitch(self, audio):
        '''
        audio: [1, T] tensor
        '''
        f0 = self.f0_model.infer_from_audio(audio.squeeze().numpy(), self.source_sample_rate, device=torch.device('cpu'), thred=0.03, use_viterbi=False)
        pitch = f0ToPitch(f0).squeeze()
        return pitch


    def vuv_test_pretrain(self):
        
        pass




def force_align_note_phone_pipeline(asr_sliced_phones, midi_sliced_notes, vuv, f0, tail_sec, align_method=None):
    if align_method is None:
        return asr_sliced_phones, midi_sliced_notes, True
    
    try: 
        legalized_phone, legalized_notes, flag = ALIGN_MAP[align_method](asr_sliced_phones, midi_sliced_notes, vuv, f0, tail_sec)
        return legalized_phone, legalized_notes, flag
    
    except AssertionError as e:
        if str(e).split('|')[0] == "ExceptionB" or str(e).split('|')[0] == "ExceptionC" or str(e).split('|')[0] == "ExceptionD":
            print(str(e).split('|')[0])

        return None, None, False

    except AssertionErrorSilenceNotLegal as e:
        e.display_error()
        return None, None, False

    except ProcessNoteException as e:
        e.display_error()
        return None, None ,False 
    
    except ZeroPitchAssignException as e:
        e.display_error()
        return None, None ,False 

    except AlignNoteException as e:
        e.display_error()
        return None, None ,False 

    except Exception as e:
        print("other unknown error")
        return None, None, False
    



class AssertionErrorSilenceNotLegal(Exception):
    def __init__(self, message):
        self.message = message 
    
    def display_error(self):
        print("Assertion Error happened because of silence in words, ", self.message)



class ProcessNoteException(Exception):
    def __init__(self, message):
        self.message = message 
    
    def display_error(self):
        print("Note Processing Error when finding related notes using legal phone, ", self.message)


class ZeroPitchAssignException(Exception):
    def __init__(self, message):
        self.message = message 
    
    def display_error(self):
        print("Assigning notes, meet Pitch is Rest, which should be handled already, ", self.message)


class AlignNoteException(Exception):
    def __init__(self, message):
        self.message = message 
    
    def display_error(self):
        print("Aligning note, but len syllables not equals to len notes, ", self.message[:5])


    


ALIGN_MAP = {
    'V3' : force_alignment_v3,
    'V2' : force_alignment_v2,
    'V1' : force_alignment_v1,
}