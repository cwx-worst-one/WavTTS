import os

import torch

from recipes.bigmusic.datasets.tokenizers.phoneme import MAX_PHONE_LEN, Wav2VecPhonemeTokenizer


def test_wav2vec_phoneme_tokenizer():
    tokenizer = Wav2VecPhonemeTokenizer()
    
    tokens = tokenizer("Hi there")
    assert tokens.shape == (1, tokenizer.max_phone_len)
    assert tokens[0, -1] == tokenizer.pad_token_id


    tokens_2 = tokenizer("hi there")
    torch.testing.assert_close(tokens, tokens_2)


def test_sami_tokenizer():
    from recipes.datasets.mcc.sami_tokenizer import SamiTokenizer
    from sami_tts_api.sail import download_model
    fe_version="42.0"
    fe_task="tts_chinese_frontend_model"
    # fe_task="tts_chinese_frontend_model"
    model_dir = os.path.join(
        os.environ['AI_MUSIC_DIR'],
        '../../dump/ai_music/20240223.svs.dump'
    )
    fe = download_model(fe_task, model_dir, fe_version)

    tokenizer = SamiTokenizer(fe=fe)
    result = tokenizer("We love you, oh, Lord. Lord, we love you. Hmm? Lord. We love you, Lord.")
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
