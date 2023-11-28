import torch
import numpy as np
import random
from functools import lru_cache
from scipy.io import wavfile

import os
import io
import sys
import logging
from bytedance import easycycle
import uuid
from recipes.serving.utils.setup import get_model_configs

device = "cuda:0"
tos_url_expires = 60 * 60 * 24 * 1000
tos_bucket = "bigspeech-platform"
input_sample_rate = 24000


def preload_models():
    load_cached_models()


@lru_cache(maxsize=2)
def load_cached_models():
    logging.info("***** start loading model *****")
    try:
        app = os.getenv("SERVER_APP", "BigTTS")
        configs = get_model_configs(app)
    except Exception as err:
        logging.error(err)
        sys.exit(0)

    pl_module = configs.pl_module

    pl_module.to(device).eval()
    logging.info("***** load model success *****")
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
    pl_module = load_cached_models()

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

    text = "明月几时有，把酒问青天，不知天上宫阙，今夕是何年"
    res = api_main(b'', "prefab", 'duibiao/sinong_conversation', text)
