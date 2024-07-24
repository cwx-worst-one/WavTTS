import logging
import os
import sys
import time
import uuid
from functools import lru_cache

from bytedance import easycycle

from recipes.research.diff.diff_instrumental import DiffInstrumental
from recipes.research.diff.prod import sample
from recipes.serving.utils.setup import get_model_configs
from samantha.data.av_audio import audio_write

device = "cuda:0"

tos_url_expires = 60 * 60 * 24 * 1000
tos_bucket = "bigspeech-platform"


def preload_models() -> DiffInstrumental:
    return load_cached_models()


@lru_cache(maxsize=2)
def load_cached_models() -> DiffInstrumental:
    logging.info("***** start loading model *****")
    try:
        app = os.getenv("SERVER_APP", "Research")
        configs = get_model_configs(app)
    except Exception as err:
        logging.error(err)
        sys.exit(0)
    logging.info(f"model config: {configs}")

    pl_module = configs.pl_module

    pl_module.to(device).eval()

    logging.info("***** load model success *****")
    return pl_module


def api_main(prompt: str):

    logging.info(
        f"instrumental request, prompt:{prompt}")

    start = time.time()
    pl_module = preload_models()
    logging.info(f"load model cost {time.time() - start} s")

    urls = []

    pred_audio, sample_rate = sample(
        pl_module,
        text_prompt=prompt,
        seconds_start=0,
        seconds_total=60,
        t=50,
        cfg_weight=2.5,
        schedule_tau=1.0,
        batch_size=1,
        solver="dpmpp-3m-sde",
    )
    pred_audio = pred_audio.cpu()
    
    for a in pred_audio:
        bytes_mp3 = audio_write(a, sample_rate, format="mp3", mp3_rate=320, normalize=True)
        fileid = uuid.uuid4().hex
        audio_name = f"{fileid}.mp3"
        audio_url = easycycle.upload_data_and_get_public_url(
            easycycle.Host.US, 'lixingxing.cs',
            bytes_mp3,
            tos_bucket,
            audio_name,
            tos_url_expires
        )
        urls.append(audio_url)
    return gen_response(b'', sample_rate, ','.join(urls))


def gen_response(audio, sr, urls):
    return [audio, sr, urls, '']


def gen_error_response(err_msg):
    return [b'', 0, '', err_msg]


if __name__ == "__main__":
    prompt = "EDM Disco"
    audio, sr, urls, err_msg = api_main(prompt)
    logging.info(f"inference result: {urls}")
