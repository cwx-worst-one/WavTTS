# coding: utf-8
import numpy as np

from .labels import sanity_check_label
from .symbols import (phone_to_int, tone_to_int, wordcateg_to_int, prosody_to_int, prosodicword_to_int)

import os

def enc_durations_label(boundaries_path, meta, cfg):
    if not os.path.exists(boundaries_path):
        print(f"Bounaries label No such file or directory: '{boundaries_path}'")
        return None
    else:
        print(f"-------->{boundaries_path}")

    with open(boundaries_path, 'r') as f:
        lines = f.readlines()
        SCALE = 1e-7
        last_time = 0.
        num_symbols = len(lines)
        durations = []
        for i, line in enumerate(lines):
            metas = line.strip().split(' ')
            end_time = metas[1] * SCALE
            duration = end_time - last_time
            if i == num_symbols - 1:
                assert metas[-1] == 'sp'
                continue
            durations.append(duration)
        duration_length = len(durations)
        durations = np.asarray(durations, np.int32)
        durations = durations.astype(np.float32)
    to_bytes = getattr(cfg, 'to_bytes', True)
    if to_bytes:
        durations = durations.tobytes()
    return (durations, duration_length)


def enc_taco_label(lab_path, meta, cfg):
    """ Encode the label to indices

    Args:
        lab_path (str): label file path

    Returns:
        tuple: (phones_enc, tones_enc, word_categs_enc, prosodys_enc, labs_len)
               (np.int32, np.int32, np.int32, np.int32, int)
    """
    if not os.path.exists(lab_path):
        print(f"Encode_label No such file or directory: '{lab_path}'")
        return None
    else:
        pass
    use_prsdword = cfg.get('use_prsdword', False)
    forced_refix = cfg.get('forced_refix', False)
    with open(lab_path, 'r') as f:
        lines = f.readlines()
        pre_metas = None
        phones_enc = list()
        tones_enc = list()
        word_categs_enc = list()
        prosodys_enc = list()

        for i, line in enumerate(lines):
            metas = line.strip().split('\t')
            metas, should_fix_previous = sanity_check_label(metas, pre_metas,
                                                            lab_path, i, len(lines), num_meta=5, allow_fix=forced_refix)
            if forced_refix and should_fix_previous:
                phone, tone, wordpost, wordcateg, prosody = ",", "0", "0.0 0.0 0.0 0.0", "S", "3"
                phones_enc[-1] = phone_to_int[phone]
                tones_enc[-1] = tone_to_int[tone]
                word_categs_enc[-1] = wordcateg_to_int[wordcateg]
                if use_prsdword:
                    prosodys_enc[-1] = prosodicword_to_int[prosody]
                else:
                    prosodys_enc[-1] = prosody_to_int[prosody]
                pre_metas = metas
                continue

            if metas is None:
                return None
            else:
                phone, tone, wordpost, wordcateg, prosody = metas
            pre_metas = metas
            phones_enc.append(phone_to_int[phone])
            tones_enc.append(tone_to_int[tone])
            word_categs_enc.append(wordcateg_to_int[wordcateg])

            use_prsdword = cfg.get('use_prsdword', False)
            if use_prsdword:
                prosodys_enc.append(prosodicword_to_int[prosody])
            else:
                prosodys_enc.append(prosody_to_int[prosody])
        # makding data, extra processing is forbidden
        # phones_enc.append(phone_to_int[_eos])
        # tones_enc.append(tone_to_int[_eos])
        # word_categs_enc.append(wordcateg_to_int[_eos])
        # prosodys_enc.append(prosody_to_int[_eos])
        label_length = len(lines)

        phones = np.asarray(phones_enc, np.int32)
        tones = np.asarray(tones_enc, np.int32)
        word_categs = np.asarray(word_categs_enc, np.int32)
        prosodies = np.asarray(prosodys_enc, np.int32)
        to_bytes = cfg.get('to_bytes', True)
        if to_bytes:
            phones = phones.tobytes()
            tones = tones.tobytes()
            word_categs = word_categs.tobytes()
            prosodies = prosodies.tobytes()

    return (phones, tones, word_categs, prosodies, label_length)


def enc_taco_label_no_bytes(lab_path, meta, cfg):
    """ Encode the label to indices

    Args:
        lab_path (str): label file path

    Returns:
        tuple: (phones_enc, tones_enc, word_categs_enc, prosodys_enc, labs_len)
               (np.int32, np.int32, np.int32, np.int32, int)
    """
    if meta is not None:
        lines = meta
    else:
        if not os.path.exists(lab_path):
            print(f"Encode_label No such file or directory: '{lab_path}'")
            return None
        else:
            with open(lab_path, 'r') as f:
                lines = f.readlines()
    
    use_prsdword = cfg.get('use_prsdword', False)
    forced_refix = cfg.get('forced_refix', False)
        
    pre_metas = None
    phones_enc = list()
    tones_enc = list()
    word_categs_enc = list()
    prosodys_enc = list()

    for i, line in enumerate(lines):
        metas = line.strip().split('\t')
        metas, should_fix_previous = sanity_check_label(metas, pre_metas,
                                                        lab_path, i, len(lines), num_meta=5, allow_fix=forced_refix)
        if forced_refix and should_fix_previous:
            phone, tone, wordpost, wordcateg, prosody = ",", "0", "0.0 0.0 0.0 0.0", "S", "3"
            phones_enc[-1] = phone_to_int[phone]
            tones_enc[-1] = tone_to_int[tone]
            word_categs_enc[-1] = wordcateg_to_int[wordcateg]
            if use_prsdword:
                prosodys_enc[-1] = prosodicword_to_int[prosody]
            else:
                prosodys_enc[-1] = prosody_to_int[prosody]
            pre_metas = metas
            continue

        if metas is None:
            return None
        else:
            phone, tone, wordpost, wordcateg, prosody = metas
        pre_metas = metas
        phones_enc.append(phone_to_int[phone])
        tones_enc.append(tone_to_int[tone])
        word_categs_enc.append(wordcateg_to_int[wordcateg])

        use_prsdword = cfg.get('use_prsdword', False)
        if use_prsdword:
            prosodys_enc.append(prosodicword_to_int[prosody])
        else:
            prosodys_enc.append(prosody_to_int[prosody])

    label_length = len(lines)

    phones = np.asarray(phones_enc, np.int32)
    tones = np.asarray(tones_enc, np.int32)
    word_categs = np.asarray(word_categs_enc, np.int32)
    prosodies = np.asarray(prosodys_enc, np.int32)
    to_bytes = cfg.get('to_bytes', True)

    return (phones, tones, word_categs, prosodies, label_length)


def enc_fs_label(lab_path, meta, cfg):
    """ Encode the label to indices

    Args:
        lab_path (str): label file path

    Returns:
        tuple: (phones_enc, tones_enc, word_categs_enc, prosodys_enc, labs_len)
               (np.int32, np.int32, np.int32, np.int32, int)
    """
    if not os.path.exists(lab_path):
        print(f"Encode_label No such file or directory: '{lab_path}'")
        return None
    else:
        pass
    use_prsdword = cfg.get('use_prsdword', False)
    with open(lab_path, 'r') as f:
        lines = f.readlines()
        pre_metas = None
        phones_enc = list()
        tones_enc = list()
        word_categs_enc = list()
        prosodys_enc = list()
        durations = list()

        for i, line in enumerate(lines):
            metas = line.strip().split('\t')
            metas, should_fix_previous = sanity_check_label(metas, pre_metas,
                                                            lab_path, i, len(lines), num_meta=6, allow_fix=True)
            if metas is None:
                return None
            else:
                phone, tone, wordpost, wordcateg, prosody, duration = metas
            if should_fix_previous:
                if use_prsdword:
                    prosodys_enc[-1] = prosodicword_to_int[prosody]
                else:
                    prosodys_enc[-1] = prosody_to_int[prosody]
            pre_metas = metas
            phones_enc.append(phone_to_int[phone])
            tones_enc.append(tone_to_int[tone])
            word_categs_enc.append(wordcateg_to_int[wordcateg])

            if use_prsdword:
                prosodys_enc.append(prosodicword_to_int[prosody])
            else:
                prosodys_enc.append(prosody_to_int[prosody])

            durations.append(int(duration))
        # makding data, extra processing is forbidden
        # phones_enc.append(phone_to_int[_eos])
        # tones_enc.append(tone_to_int[_eos])
        # word_categs_enc.append(wordcateg_to_int[_eos])
        # prosodys_enc.append(prosody_to_int[_eos])
        # durations.append(0)
        # label_length = len(lines) + 1
        label_length = len(lines)

        phones = np.asarray(phones_enc, np.int32)
        tones = np.asarray(tones_enc, np.int32)
        word_categs = np.asarray(word_categs_enc, np.int32)
        prosodies = np.asarray(prosodys_enc, np.int32)
        durations = np.asarray(durations, np.int32)

        to_bytes = cfg.get('to_bytes', True)
        if to_bytes:
            phones = phones.tobytes()
            tones = tones.tobytes()
            word_categs = word_categs.tobytes()
            prosodies = prosodies.tobytes()
            durations = durations.tobytes()

    return (phones, tones, word_categs, prosodies, durations, label_length)
