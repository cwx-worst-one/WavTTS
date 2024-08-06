from enum import Enum

from samantha.dataio.lite.utils.punctuation import punctuation_all


class LangID(Enum):
    EN = "en"
    ZH = "zh"
    ZH_EN = "zh_en"


def is_chinese_char(char):
    return "\u4e00" <= char <= "\u9fff"


def is_english_char(char):
    return ("\u0041" <= char <= "\u005a") or ("\u0061" <= char <= "\u007a")


def is_english_spanish_char(char):
    special_Spanish_chars_list = [
        "á",
        "é",
        "í",
        "ó",
        "ú",
        "Á",
        "É",
        "Í",
        "Ó",
        "Ú",
        "ñ",
        "Ñ",
        "¡",
        "¿",
        "ü",
        "Ü",
    ]
    return (
        ("\u0041" <= char <= "\u005a")
        or ("\u0061" <= char <= "\u007a")
        or char in special_Spanish_chars_list
    )


# This implementation is not work for everywhere,
# feel free to update it with backward compatibility.
def get_lang_by_text(text, detail=False):

    text = text.replace("'", "")
    # en, zh
    en_word_cnt = 0
    zh_char_cnt = 0
    i = 0
    while i < len(text):
        x = text[i]
        if x in punctuation_all:  # punc
            i += 1
            continue
        elif is_chinese_char(x):  # zh

            zh_char_cnt += 1
            i += 1
        elif is_english_spanish_char(x):  # en with little spanish
            i += 1
            if i >= len(text):
                en_word_cnt += 1
                break
            while is_english_spanish_char(text[i]):
                i += 1
                if i >= len(text):
                    break
            en_word_cnt += 1
            continue
        else:
            i += 1

    # TODO: japan
    if zh_char_cnt > 0:
        if en_word_cnt > 0:
            lang = LangID.ZH_EN.value
        else:
            lang = LangID.ZH.value
    else:
        if en_word_cnt > 0:
            lang = LangID.EN.value
        else:
            raise NotImplementedError
    if detail:
        lang = (lang, zh_char_cnt, en_word_cnt)
    return lang
