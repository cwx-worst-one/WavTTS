import importlib
import os
import os.path as osp
import sys
current_dir = osp.dirname(__file__)
sys.path.append(current_dir)

def list_dir(path, extension=None):
    def remove_self_and_parent(file_dir_list):
        if '.' in file_dir_list:
            file_dir_list.remove('.')
        if '..' in file_dir_list:
            file_dir_list.remove('..')
    fn_list = os.listdir(path)
    remove_self_and_parent(fn_list)
    if extension is None:
        return fn_list
    new_fn_list = []
    if isinstance(extension, str):
        for fn in fn_list:
            if fn.endswith(extension):
                new_fn_list.append(fn)
    elif isinstance(extension, (tuple, list)):
        for fn in fn_list:
            fn_ext = osp.splitext(fn)[1]
            if fn_ext in extension:
                new_fn_list.append(fn)
    return new_fn_list

def merge_list_as_set(list1, list2):
    new_list = None
    if len(list1) == 0:
        new_list = list2
    if len(list2) == 0:
        new_list = list1
    new_list = list1
    for i2 in list2:
        if i2 not in list1:
            new_list.append(i2)
    return new_list


def init_language():
    _pad = '_'
    _eos = '§'

    punctuation = {
        'transition': set(["<unk>"]),
        'pause': set(['pau']),
        'separation': set(['sp']),
        'silence': set(['sil']),
    }

    punc_phone_mark = dict(
        pause='|',
        separation=',',
        full_stop='.',
        question_mark='?',
        exclamation_mark='!',
        silence='#',
    )

    meta_punc_2_pwpp = dict(
        transition=1,
        pause=2,
        separation=3,
        silence=4,
    )
    seperate_level = ['pause', 'separation']
    silence_level = list(meta_punc_2_pwpp.keys())

    phone_set = [_pad, _eos]
    tone_set = [_pad, _eos]

    # Word category set
    wordcateg_set = [_pad, _eos] + ['B', 'M', 'E', 'S']

    # Prosody set for phoneme and punctuation
    prosody_set = [_pad, _eos] + ['0', '1', '2', '3', '4']
    return _pad, _eos, punctuation, punc_phone_mark, seperate_level, silence_level, meta_punc_2_pwpp, \
        phone_set, tone_set, wordcateg_set, prosody_set


def get_language_name(language_path):
    language_list = []
    for lang_file in list_dir(language_path, extension='.py'):
        if not lang_file.startswith('__'):
            language = osp.splitext(lang_file)[0]
            language_list.append(language)
    lang = os.getenv("TTS_LANG")
    if lang is None:
        language_list = ['ZH', 'EN']
    else:
        language_list = lang.upper().split(',')
    return language_list


def summary_language_symbols(punctuation, tone_set):
    phone_set = []
    punc_phone = ['pause', 'separation', 'silence']
    consonant = set()
    vowel = set()
    filter_keywords_set = set()
    language_set = dict()
    language_list = get_language_name(osp.join(current_dir, 'language'))

    for language in language_list:
        module = importlib.import_module(f".{language}", ".language")
        module_var = [v for v in dir(module) if v[:2] != "__"]
        for p_k, p_v in module.punctuation.items():
            if p_k in punctuation.keys():
                punctuation[p_k] = punctuation[p_k].union(set(p_v))
            else:
                punctuation[p_k] = set(p_v)
            if 'pause_set' in module_var and p_k in module.pause_set:
                punctuation['pause'] = punctuation['pause'].union(set(p_v))
            if 'separation_set' in module_var and p_k in module.separation_set:
                punctuation['separation'] = punctuation['separation'].union(set(p_v))
            if 'silence_set' in module_var and p_k in module.silence_set:
                punctuation['silence'] = punctuation['silence'].union(set(p_v))
        if 'consonant' in module_var:
            consonant = consonant.union(module.consonant)
        if 'vowel' in module_var:
            vowel = vowel.union(module.vowel)
        language_info = dict()
        if 'phone_set' in module_var:
            if 'phone_set' in language_info:
                language_info['phone_set'] = merge_list_as_set(language_info['phone_set'], module.phone_set)
            else:
                language_info['phone_set'] = module.phone_set
            phone_set = merge_list_as_set(phone_set, module.phone_set)
        if 'tone_set' in module_var:
            if 'tone_set' in language_info:
                language_info['tone_set'] = merge_list_as_set(language_info['tone_set'], module.tone_set)
            else:
                language_info['tone_set'] = module.tone_set
            tone_set = merge_list_as_set(tone_set, module.tone_set)
        if 'filter_set' in module_var:
            filter_keywords_set = filter_keywords_set.union(module.filter_set)
            language_info['filter_set'] = module.filter_set
        if 'punc_phone' in module_var:
            if 'phone_set' in language_info:
                language_info['phone_set'] = merge_list_as_set(language_info['phone_set'], module.punc_phone)
            else:
                language_info['phone_set'] = module.punc_phone
            punc_phone = merge_list_as_set(punc_phone, module.punc_phone)
        language_set[language] = language_info
    return punc_phone, consonant, vowel, language_set, punctuation, phone_set, tone_set, filter_keywords_set


def export_language_summary():
    _pad, _eos, punctuation, punc_phone_mark, seperate_level, silence_level, meta_punc_2_pwpp, \
        init_phone_set, tone_set, wordcateg_set, prosody_set = init_language()
    punc_phone, consonant, vowel, language_set, punctuation, core_phone_set, tone_set, filter_keywords_set = \
        summary_language_symbols(punctuation, tone_set)
    phone_set = init_phone_set + punc_phone + core_phone_set

    filters_set = []
    for key in filter_keywords_set:
        filters_set += list(punctuation[key])

    seperate_set = []
    for key in seperate_level:
        seperate_set += punctuation[key]
    if 'EN' in language_set:
        EN_tones_2_CMU = dict((tone, str(int(tone) - 10))
                            for tone in language_set['EN']['tone_set'])
    else:
        EN_tones_2_CMU = None

    punc_2_pwpp = dict((sil_symbols, meta_punc_2_pwpp[sil_level])
                       for sil_level in silence_level for sil_symbols in punctuation[sil_level])

    phone_to_int = {}
    int_to_phone = {}
    for idx, item in enumerate(phone_set):
        if item in punc_phone:
            for item_mark in punctuation[item]:
                phone_to_int[item_mark] = idx
        else:
            phone_to_int[item] = idx
        if item in punc_phone_mark:
            int_to_phone[idx] = punc_phone_mark[item]
        else:
            int_to_phone[idx] = item

    tone_to_int = {}
    int_to_tone = {}
    for idx, item in enumerate(tone_set):
        tone_to_int[item] = idx
        int_to_tone[idx] = item

    wordcateg_to_int = {}
    int_to_wordcateg = {}
    for idx, item in enumerate(wordcateg_set):
        wordcateg_to_int[item] = idx
        int_to_wordcateg[idx] = item

    prosody_to_int = {}
    prosodicword_to_int = {}
    int_to_prosody = {}
    for idx, item in enumerate(prosody_set):
        prosody_to_int[item] = idx
        if (item == '2' or item == '3' or item == '4'):
            prosodicword_to_int[item] = prosodicword_to_int['1']
        else:
            prosodicword_to_int[item] = idx
        int_to_prosody[idx] = item
    return _pad, _eos, punctuation, filters_set, seperate_set, phone_set, \
        tone_set, wordcateg_set, prosody_set, EN_tones_2_CMU, punc_2_pwpp, \
        phone_to_int, tone_to_int, wordcateg_to_int, prosody_to_int, \
        prosodicword_to_int, int_to_phone, int_to_tone, int_to_wordcateg, \
        int_to_prosody, consonant, vowel


_pad, _eos, punctuation, filters_set, seperate_set, phone_set, tone_set, \
    wordcateg_set, prosody_set, EN_tones_2_CMU, punc_2_pwpp, phone_to_int, \
    tone_to_int, wordcateg_to_int, prosody_to_int, prosodicword_to_int, int_to_phone, \
    int_to_tone, int_to_wordcateg, int_to_prosody, consonant, vowel = export_language_summary()
