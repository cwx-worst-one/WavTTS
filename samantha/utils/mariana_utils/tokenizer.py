def is_cjk(character):
    """
    This checks for CJK character.

        >>> CJKChars().ranges
        [(4352, 4607), (11904, 42191), (43072, 43135), (44032, 55215),
        (63744, 64255), (65072, 65103), (65381, 65500), (131072, 196607)]
        >>> is_cjk(u'\u33fe')
        True
        >>> is_cjk(u'\uFE5F')
        False
    """
    if any(
        [
            start <= ord(character) <= end
            for start, end in [
                (4352, 4607),
                (11904, 42191),
                (43072, 43135),
                (44032, 55215),
                (63744, 64255),
                (65072, 65103),
                (65381, 65500),
                (131072, 196607),
            ]
        ]
    ):
        return True

    if character in "，。？,；;！!（）()":
        return True
    return False


def recover_text(origin_text):
    if origin_text[:2] == "##":
        origin_text = origin_text[2:]
    text = origin_text.replace(" ##", "").replace("[UNK]", "").replace("[EOS]", "")
    text = text.replace("●●●", "●●")
    text = text.replace("●", "\n")
    char_list = list(text)
    idx = 1
    final_text = ""
    if len(char_list) > 0:
        final_text = char_list[0]
    while idx < len(char_list) - 1:
        if char_list[idx] != " ":
            final_text += char_list[idx]
        elif char_list[idx - 1] == char_list[idx + 1] and char_list[idx + 1] in "-*~`":
            # 前后一样
            pass
        elif is_cjk(char_list[idx - 1]) and is_cjk(char_list[idx + 1]):
            # 前后中文
            pass
        elif char_list[idx - 1] in "123456789" and char_list[idx + 1] == ",":
            pass
        elif char_list[idx - 1] in "123456789" and char_list[idx + 1] == "]":
            pass
        elif char_list[idx + 1] in "123456789" and char_list[idx - 1] == "[":
            pass
        elif char_list[idx + 1] in "123456789" and char_list[idx - 1] == ",":
            pass
        elif char_list[idx + 1] in ".：":
            # 后是.
            pass
        else:
            final_text += char_list[idx]
        idx += 1
    if len(char_list) > 1:
        final_text += char_list[-1]

    return final_text.strip()
