import torch
import numpy as np
import random
from functools import lru_cache
from scipy.io import wavfile

from recipes.text2semantic.lit_modules.lit_infer_lang_spk import BigTTSWVAEInferLangSpk

import io
import logging
from bytedance import easycycle
import uuid

device = "cuda:0"

tos_url_expires = 60 * 60 * 24 * 1000
tos_bucket = "bigspeech-platform"
input_sample_rate = 24000


meta_lst = "/mnt/bn/jdy-lq-3/AudioGPT/repo/bigtts_testset/inner_testset_zh/meta.lst.prompt1_rand200"
ckpt_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/huangzhiying.92/exp/samantha_bigtts_wvae_sft_tobe_merge/text2semantic/sft_WFVAE_v2_labv3_punc_data_id307_bt14000_16A100_accu5_byteT5_scr0.0_freezeTrue_langFalse/checkpoints/epoch=00-step=98500-kl_loss=0.12.ckpt"
wvae_decoder_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_decoder_%d.pt"
wvae_encoder_path = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/wangxin.colin/ckpts/wvae_1.0/wavevae_encoder_%d.pt"
output_dir = "taozi_outputs"
#bpe_dir = "hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/panjunjie.jeff/resource/models/byte-T5-base"
bpe_dir = "resource/models/byte-T5-base"
spk2id = "recipes/text2semantic/datasets/dict/spk2id.json"
spk_name = "duibiao/sinong_conversation"


def preload_models():
    return load_cached_models()


@lru_cache(maxsize=2)
def load_cached_models():
    logging.info("========== start loading model ==========")
    pl_module = BigTTSWVAEInferLangSpk(
        ar_model_name="VAET2SLangSpkModule",
        ckpt_path=ckpt_path,
        wvae_encoder=wvae_encoder_path,
        wvae_decoder=wvae_decoder_path,
        output_dir=None,
        tokenizer_type='byte-T5-base',
        use_bpe=True,
        bpe_dir=bpe_dir,
        use_spk_id=True,
        spk2id=spk2id,
        infer_spk_name=spk_name,
        tag_id=3,
        infer_mode='online'
    )

    pl_module.to(device).eval()
    logging.info("========== load model success ==========")
    return pl_module

@torch.no_grad()
def api_main(
    prompt_wav: bytes,
    prompt_type: str,
    timbre_name: str,
    text: str,
):
    logging.info(f"receive request, prompt_type:{prompt_type}, timbre_name:{timbre_name}, text:{text}")
    if len(text) == 0:
        return gen_error_response("Input text can not be empty.")

    setup_seed(1024)

    # load models
    pl_module = preload_models()

    # predict
    pl_module.infer_spk_name = timbre_name

    uttid = uuid.uuid4().hex
    gen_wav = pl_module.predict_step((uttid, text), batch_idx=1)

    # numpy to bytes
    output_wav = convert_to_bytes_wav(gen_wav, input_sample_rate)

    prompt_wav_url = None
    generated_wav_url = upload_wav_to_tos(uttid, output_wav)
    return gen_response(output_wav, prompt_wav_url, text, generated_wav_url)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def gen_response(result, prompt_wav_url, prompt_text, generated_wav_url):
    return [result, prompt_wav_url, prompt_text, generated_wav_url, '']


def gen_error_response(err_msg):
    return [b'', '', '', '', err_msg]


def upload_wav_to_tos(uttid, output_wav):
    generated_wav_name = f"{uttid}-generated.wav"
    generated_wav_url = easycycle.upload_data_and_get_public_url(easycycle.Host.CN, 'wangtuo.todd', output_wav,
                                                                 tos_bucket,
                                                                 generated_wav_name, tos_url_expires)
    logging.info(f"upload generated wav to tos, key:{generated_wav_name}, url:{generated_wav_url}")
    return generated_wav_url


def convert_to_bytes_wav(audio, sr=24000):
    audio = audio * 32768.0
    audio = audio.astype("int16")
    bytes_wav = bytes()
    bytes_io = io.BytesIO(bytes_wav)
    wavfile.write(bytes_io, sr, audio)
    return bytes_io.read()


if __name__ == "__main__":
    setup_seed(1024)

    # timbre_name = "taozi_multi_style_wvae"
    timbre_name = "maomao_conversation_wvae"
    # text = "明月几时有，把酒问青天，不知天上宫阙，今夕是何年"
    text = "以下是一些中国美食的推荐：1. 麻婆豆腐：四川菜的代表之一，以豆腐和辣椒豆瓣酱为主要原料，口感麻辣，非常有特色。2. 红烧肉：传统的中国菜肴，将猪肉先炖煮再烧制，入口酥软，肉质鲜嫩，汁浓味美。3. 北京烤鸭：北京的特色菜肴，以选用优质的鸭子制作而成，皮薄肉嫩，口感酥脆，搭配葱、酱等配料食用更佳。"
    res = api_main(b'', "prefab", timbre_name, text)
