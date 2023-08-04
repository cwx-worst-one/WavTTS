from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
import json
import string


phoneme_dir = Path('/mnt/bn/audio-diffusion/ashaw/tokenizers/phoneme')
phoneme_vocab_size = 80 # 69 phonemes + x special tokens
phoneme_padding_value = 70
phoneme_unk_value = 71
phoneme_newline_value = 72

@lru_cache(maxsize=1)
def init_dict():
    diction_fp = phoneme_dir/'diction_word_phoneme.json'
    if (diction_fp).exists():
        with open(diction_fp, 'r') as f:
            return json.load(f)
    else:
        with open(phoneme_dir/'own_dict.txt', 'r') as f:
            file_dict = f.read()
        file_dict = file_dict.split('\n')

        diction_word_phoneme = {}
        diction_phoneme_id = {}
        idx = 0
        for temp in tqdm(file_dict):
            arr = temp.split(' ')
            word = arr[0].lower()
            diction_word_phoneme[word] = []
            for x in arr[1:]:
                x = x.lower()
                if x=='' or x==' ':
                    continue
                if x not in diction_phoneme_id:
                    diction_phoneme_id[x] = idx
                    idx+=1
                phoneme_id = diction_phoneme_id[x]
                diction_word_phoneme[word].append(phoneme_id)
            

        print('len diction:',      len(diction_phoneme_id))
        print('len word phoneme:', len(diction_word_phoneme))
        with open(diction_fp, 'w') as f:
            json.dump(diction_word_phoneme, f)
        return diction_word_phoneme

diction_word_phoneme = init_dict()

_punc_translator_space = str.maketrans(string.punctuation, ' '*len(string.punctuation))
_punc_translator = str.maketrans('', '', string.punctuation)
def remove_punctuation(text, replace_with_space=True):
    if replace_with_space:
        return text.translate(_punc_translator_space)
    else:
        return text.translate(_punc_translator)

def text_to_words(text):
    words = text.lower().split(' ')
    return [w.strip() for w in words if w.strip()]

def word_to_phonemes(word, allow_unknown=True):
    if word == '<n>':
        return [phoneme_newline_value]
    if word in diction_word_phoneme:
        return diction_word_phoneme[word]
    norm_word = remove_punctuation(word, replace_with_space=False)
    if norm_word == word:
        if allow_unknown: return [phoneme_unk_value]
        raise Exception(f'Could not find word {word}. Must allow_unknown or add word to dictionary')
    if norm_word in diction_word_phoneme:
        return diction_word_phoneme[norm_word]
    norm_word_space = remove_punctuation(word, replace_with_space=True).split(' ')
    phonemes = [word_to_phonemes(n) for n in norm_word_space]
    return sum(phonemes, [])

def convert_text(text, allow_unknown=True):
    words = text_to_words(text)
    phonemes = [word_to_phonemes(w, allow_unknown) for w in words]
    return sum(phonemes, [])

class PhonemeTokenizer(object):
    def __init__(self, allow_unknown=False):
        self.allow_unknown = allow_unknown
        self.pad_id = phoneme_padding_value
        self.unk_id = phoneme_unk_value
        self.vocab_size = phoneme_vocab_size

    def __call__(self, text):
        return convert_text(text, self.allow_unknown)
