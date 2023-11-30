from functools import lru_cache
import json
import torch
import logging
import time
import sys
import os
import io
import torchaudio
from scipy.io import wavfile
from recipes.bigmusic.datasets.inference import inference_dataset_from_prompt
from recipes.bigmusic.datasets.lyrics import LyricsDataModule
from recipes.serving.utils.setup import get_model_configs

from bytedance import easycycle
import uuid

device = "cuda:0"
default_batch_size = 1

tos_url_expires = 60 * 60 * 24 * 1000
tos_bucket = "bigspeech-platform"


def torch_save_wav_to_binary(audio, sr=24000):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    buffer_ = io.BytesIO()
    torchaudio.save(buffer_, audio, sr, format="wav")

    buffer_.seek(0)
    return buffer_.read()


def torch_fp32_to_numpy_int16(audio: torch.Tensor):
    return (audio * 32767).to(torch.int16).T.cpu().numpy()


def preload_models():
    return load_cached_models()


@lru_cache(maxsize=2)
def load_cached_models():
    logging.info("***** start loading model *****")
    try:
        app = os.getenv("SERVER_APP", "Instrumental")
        configs = get_model_configs(app)
    except Exception as err:
        logging.error(err)
        sys.exit(0)
    logging.info(f"model config: {configs}")

    pl_module = configs.pl_module
    extra_params = configs.extra_params
    trainer = configs.trainer

    pl_module.to(device).eval()

    logging.info("***** load model success *****")
    return pl_module, extra_params, trainer


def api_main(prompt, duration, chorus):
    duration = int(duration)
    assert (
        duration in {30, 45, 60, 75, 90, 105, 120}
    ), f"Invalid duration: {duration}"

    if chorus == "random":
        chorus = None
    else:
        chorus = json.loads(chorus)
        assert isinstance(chorus, list)
        chorus = [
            ["chorus", float(x[0]), float(x[1])] for x in chorus
        ]

    logging.info(f"instrumental request, prompt:{prompt} duration:{duration} chorus:{chorus}")

    start = time.time()
    pl_module, extra_params, trainer = preload_models()
    logging.info(f"load model cost {time.time() - start} s")
    # Override extra_params.duration, used by dataloader
    extra_params.duration = duration

    prompts = {
        'style_text': [prompt],
        'structure': [json.dumps(chorus)],
    }
    inference_dataset = inference_dataset_from_prompt(
        prompts,
        conditions="style_text,duration,structure",
        batch_size=1,
        extra_params=extra_params,
    )
    
    pl_datamodule = LyricsDataModule(predict_dataset=inference_dataset, num_workers=0)
    predictions = trainer.predict(pl_module, pl_datamodule)
    output_wavs = predictions[0]['generated_audio']

    sr = extra_params.sample_rate
    urls = []
    for output_wav in output_wavs:
        bytes_wav = torch_save_wav_to_binary(output_wav.cpu().float(), sr)

        fileid = uuid.uuid4().hex
        audio_name = f"{fileid}.wav"
        audio_url = easycycle.upload_data_and_get_public_url(easycycle.Host.US, 'lixingxing.cs', bytes_wav,
                                                             tos_bucket, audio_name, tos_url_expires)
        urls.append(audio_url)

    return gen_response(b'', sr, ','.join(urls))


def gen_response(audio, sr, urls):
    return [audio, sr, urls, '']


def gen_error_response(err_msg):
    return [b'', 0, '', err_msg]


def convert_wav_to_bytes(audio, sr=24000):
    bytes_wav = bytes()
    bytes_io = io.BytesIO(bytes_wav)
    wavfile.write(bytes_io, sr, audio)
    return bytes_io.read()


if __name__ == "__main__":
    prompt = "EDM Disco"
    duration = "30"
    chorus = "[[0, 10], [20, 30]]"
    #chorus = "random"
    #chorus = "[]"
    audio, sr, urls, err_msg = api_main(prompt, duration, chorus)
    logging.info(f"inference result: {urls}")
