import os, json
from itertools import groupby
from transformers import PreTrainedTokenizerFast


class CharacterTokenizerWithAudioTokens():
    def __init__(self, audio_tokens_num=100, audio_token_dedup=True):
        self.audio_tokens_num = audio_tokens_num
        self.en_chars = {char: idx+audio_tokens_num+1 for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz ',.?!")}
        self.vocab_size = len(self.en_chars) + audio_tokens_num + 3 + 50 # <sep> <s> </s> <pad> and 50 placeholders
        self.bos = self.vocab_size - 1
        self.eos = self.vocab_size - 1
        self.sep = self.vocab_size - 2
        self.pad = 0
        self.audio_token_dedup = audio_token_dedup

    def tokenize(self, sentence, input_key):
        if input_key == 'inputs':
            if self.audio_token_dedup:
                tokens = [int(item.replace('<audio',''))+1 for item in sentence.split('>')[:-1]]
                return [key for key, _group in groupby(tokens)]
            else:
                return [int(item.replace('<audio',''))+1 for item in sentence.split('>')[:-1]]
        elif input_key == 'targets':
            return [self.en_chars[char] for char in sentence]
        else:
            return None

    def token2text(self, tokens):
        inv_map = {v: k for k, v in self.en_chars.items()}
        return ''.join([inv_map[idx] for idx in tokens if idx in inv_map])
    
    def token2audio(self, tokens):
        return ' '.join([str(idx-1) for idx in tokens])



class BPETokenizerWithAudioTokens():
    def __init__(self, bpe_tokenizer_file, audio_tokens_num=100, audio_token_dedup=True):
        self.audio_tokens_num = audio_tokens_num
        self.bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_file=bpe_tokenizer_file)
        self.vocab_size = self.bpe_tokenizer.vocab_size + audio_tokens_num + 3 + 50 # <sep> <s> </s> <pad> and 50 placeholders
        self.bos = self.vocab_size - 1
        self.eos = self.vocab_size - 1
        self.sep = self.vocab_size - 2
        self.pad = 0
        self.audio_token_dedup = audio_token_dedup

    def tokenize(self, sentence, input_key):
        if input_key == 'inputs':
            if self.audio_token_dedup:
                tokens = [int(item.replace('<audio',''))+1 for item in sentence.split('>')[:-1]]
                return [key for key, _group in groupby(tokens)]
            else:
                return [int(item.replace('<audio',''))+1 for item in sentence.split('>')[:-1]]
        elif input_key == 'targets':
            return [item + self.audio_tokens_num + 1 for item in self.bpe_tokenizer.encode(sentence)] # pad and audio tokens
        else:
            return None

    def token2text(self, tokens):
        return self.bpe_tokenizer.decode([x - self.audio_tokens_num - 1 for x in tokens]).replace(' '+self.bpe_tokenizer._tokenizer.model.continuing_subword_prefix, '')
    
    def token2audio(self, tokens):
        return ' '.join([str(idx-1) for idx in tokens])
