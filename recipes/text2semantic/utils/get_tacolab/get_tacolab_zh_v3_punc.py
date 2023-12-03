import sys, os
sys.path.append('/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_ref_enc/')
from recipes.text2semantic.datasets.sami_tacolabel import generate_tacolabels_from_textstr_punc
from zhon.hanzi import punctuation
import string
punctuation_all = punctuation + string.punctuation

def is_english_char(char):
    if (u'\u0041'<= char <= u'\u005a') or (u'\u0061'<= char <= u'\u007a'):
        return True
    else:
        return False

def is_english_spanish_char(char):
    special_Spanish_chars_list = ['á', 'é', 'í', 'ó', 'ú', 'Á', 'É', 'Í', 'Ó', 'Ú', 'ñ', 'Ñ', '¡', '¿', 'ü', 'Ü']
    if (u'\u0041'<= char <= u'\u005a') or (u'\u0061'<= char <= u'\u007a') or char in special_Spanish_chars_list:
        return True
    else:
        return False

def get_lang_by_text(text):
    text = text.replace('\'', '')
    # en, zh
    len_en_word = 0
    len_zh_char = 0
    i = 0
    while i < len(text):
        x = text[i]
        if x in punctuation_all: # punc
            i += 1
            continue
        elif u'\u4e00' <= x <= u'\u9fff': # zh
            len_zh_char += 1
            i += 1
        elif is_english_spanish_char(x): # en with little spanish
            i += 1
            if i >= len(text):
                len_en_word += 1
                break
            while is_english_spanish_char(text[i]):
                i += 1
                if i >= len(text):
                    break
            len_en_word += 1
            continue
        else:
            i += 1

    lang = 'en'
    if len_zh_char > len_en_word:
        lang = 'zh'

    return lang

def generate_tacolabels_engine(text_str):
    lang_key = get_lang_by_text(text_str)
    if lang_key == None:
        print(f"{text_str}: Wrong lang_key")
        return None
    tacolab = None
    if lang_key in ['zh', 'zh_en']:
        tacolab = generate_tacolabels_from_textstr_punc(text_str, "Chinese_v3_punc")
    elif lang_key in ['en']:
        tacolab = generate_tacolabels_from_textstr_punc(text_str, "English_v3_punc")

    return tacolab


# f = open(in_text_path)
# lines = f.readlines()
# f.close()

# for line in lines:
# utt, text = line.strip().split('\t')
text = "你的感受是完全可以理解的。如果你确信自己没有做错，那么坚守自己的立场是很重要的。不过，这里有一些建议你可以考虑： 1. 分辨行为和感受：即使你不认为自己做错了，你仍然可以为在公开场合发生冲突这种行为表示遗憾，而不是为你的观点道歉。 2. 积极沟通：试图理解对方为什么会有这样的反应，以及他们的观点是什么。通过沟通，你们或许可以找到一个共同的解决方案。 3. 避免公开场合的冲突：尽管你可能觉得自己是对的，但在公开场合争吵可能会损害双方的形象。未来最好避免在众人面前发生争执。 最终，每个人都有自己的底线和立场，啊重要的是找到一个平衡，啊既维护自己的观点，又能保持和同事之间的和谐关系。"
infer_tacolab = generate_tacolabels_engine(text)
print("infer_tacolab: ", infer_tacolab.decode())
