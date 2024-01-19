import os

from samantha.utils.sami_tacolabel.symbols import (
    EN_tones_2_CMU,
    consonant,
    phone_to_int,
    prosody_set,
    punctuation,
    seperate_set,
    tone_to_int,
    wordcateg_set,
)

silence_pause = punctuation["silence"]

all_pause = list(silence_pause) + seperate_set


def sanity_check_label(
    metas, pre_metas, lab_path, line_idx, num_line, num_meta=5, allow_fix=False
):
    should_fix_previous = False
    assert (
        len(metas) == num_meta
    ), f"{num_meta} elemenets are expected in each line but {lab_path} has {len(metas)} {metas}"
    code_phone, tone, wordpost, wordcateg, prosody = metas[:5]
    if line_idx == 0:  # Check head lines, assure head code_phone to be silence
        if not (
            code_phone in silence_pause
            and (
                metas[1:5] == ["0", "0.0 0.0 0.0 1.0", "S", "0"]
                or metas[1:] == ["0", "0.0 0.0 0.0 0.0", "S", "0"]
            )
        ):
            print(f"The head meta of {lab_path} is invalid and will be discarded")
            print(f"detail info {code_phone} {tone} {wordpost} {wordcateg} {prosody}")
            metas = None
    elif (
        line_idx == num_line - 1
    ):  # Check tail lines, assure tail code_phone in long_pauseset
        if not (
            code_phone in silence_pause
            and (
                metas[1:5] == ["0", "0.0 0.0 0.0 1.0", "S", "4"]
                or metas[1:] == ["0", "0.0 0.0 0.0 0.0", "S", "4"]
            )
        ):
            print(f"The tail meta of {lab_path} is invalid and will be discarded")
            print(f"detail info {code_phone} {tone} {wordpost} {wordcateg} {prosody}")
            metas = None
    else:
        # Check every line, each meta belongs to correponding set
        if not (
            (code_phone in phone_to_int)
            and (tone in tone_to_int)
            and (wordcateg in wordcateg_set)
            and (prosody in prosody_set)
        ):
            import pdb

            pdb.set_trace()
            print(
                f"The meta at line {line_idx} of {lab_path} is invalid and will be discarded"
            )
            print(f"detail info {code_phone} {tone} {wordpost} {wordcateg} {prosody}")
            metas = None
        # Check consecutive pause not exists, pause's meta[-1] are same to previous
        if code_phone in all_pause:
            if not (
                (pre_metas[0] not in all_pause)
                and ((metas[4] == pre_metas[4]) or (pre_metas[0][:2] == "E0"))
            ):
                if allow_fix:
                    print(
                        f"The pause-meta at line {line_idx} of {lab_path} is invalid and will be fixed"
                    )
                    should_fix_previous = True
                else:
                    assert pre_metas[0] in silence_pause, (
                        "the previous symbols should be one of long pauset. \n"
                        + f"but previous symbols: {pre_metas[0]} vs "
                        + f"long pauset: {silence_pause} at line {line_idx} of {lab_path}"
                    )
                    print(
                        f"The pause-meta at line {line_idx} of {lab_path} is invalid and will be discarded"
                    )
                    metas = None

    return metas, should_fix_previous


def load_tacolabel_to_kaldi(lab_path):
    """Encode the label to indices

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
    output_lines = []
    with open(lab_path, "r") as f:
        lines = f.readlines()
        pre_metas = None
        phonemes = []
        for i, line in enumerate(lines):
            metas = line.strip().split("\t")
            metas, should_fix_previous = sanity_check_label(
                metas, pre_metas, lab_path, i, len(lines), num_meta=5, allow_fix=True
            )
            if metas is None:
                return None
            elif (i == 0) or (i == len(lines) - 1):
                pass
            else:
                code_phone, tone, _, _, _ = metas
                if should_fix_previous:
                    if phonemes[-1] == "sp" and code_phone == "sil":
                        output_lines[-1] = (
                            "\t".join([",", "0", "0.0 0.0 0.0 0.0", "S", "3"]) + "\n"
                        )
                        continue
                pre_metas = metas
                if code_phone in all_pause:
                    phonemes.append("sp")
                else:
                    if code_phone.startswith("C") or code_phone.startswith("E"):
                        phone = code_phone[0] + code_phone[2:]
                    elif code_phone.startswith("PB"):
                        phone = code_phone[:2] + code_phone[3:]
                    else:
                        # print(f"unrecognized language {code_phone} at {lab_path}")
                        phone = code_phone
                    if code_phone.startswith("E") and tone in EN_tones_2_CMU:
                        tone = EN_tones_2_CMU[tone]
                    if code_phone in consonant:
                        phonemes.append(phone)
                    else:
                        phonemes.append(phone + tone)
            output_lines.append(line)
    phonemes_str = " ".join(phonemes)
    return phonemes_str, output_lines
