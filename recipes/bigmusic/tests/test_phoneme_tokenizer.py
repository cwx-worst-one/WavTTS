import torch
from recipes.bigmusic.datasets.tokenizers.phoneme import MAX_PHONE_LEN, Wav2VecPhonemeTokenizer


def test_wav2vec_phoneme_tokenizer():
    tokenizer = Wav2VecPhonemeTokenizer()
    
    tokens = tokenizer("Hi there")
    assert tokens.shape == (1, tokenizer.max_phone_len)
    assert tokens[0, -1] == tokenizer.pad_token_id


    tokens_2 = tokenizer("hi there")
    torch.testing.assert_close(tokens, tokens_2)
