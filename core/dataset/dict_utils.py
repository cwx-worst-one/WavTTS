"""
dict utils.
"""

import re

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    '''tokenize line.'''
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def safe_readline(f):
    '''safe readline.'''
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def process_bpe_symbol(sentence: str, bpe_symbol: str):
    '''process bpe symbol.'''
    if bpe_symbol == 'sentencepiece':
        sentence = sentence.replace(' ', '').replace('\u2581', ' ').strip()
    elif bpe_symbol is not None:
        sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip()
    return sentence
