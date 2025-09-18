from functools import lru_cache
import torch
import logging
import time
import sys
import os
import io
import torchaudio
from scipy.io import wavfile
from samantha.dataio.bigmusic.music_batch_transform import FullTokenCollate
from hyperpyyaml import load_hyperpyyaml

from bytedance import easycycle
import uuid
import pandas as pd
import json


device = "cuda:0"

tos_url_expires = 60 * 60 * 24 * 1000
tos_bucket = "bigspeech-platform"

class DotDict(dict):
    """Dictionary that supports dot notation.

    Arguments
    --------
    py_dict: dict, {}
        A python dict object, will be recursively converted to DotDict.

    Example
    --------
    >>> d = {"key1": "val1", "key2": {"key3": "val3"}}
    >>> dot_d = DotDict(d)
    >>> dot_d.key1
    'val1'
    >>> dot_d["key1"]
    'val1'
    >>> dot_d.key2.key3
    'val3'
    >>> dot_d["key2"]["key3"]
    'val3'
    >>> dot_d.key2.key3 = "new_val"
    >>> dot_d.key2.key3
    'new_val'
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, py_dict: dict = {}):
        for key, value in py_dict.items():
            if isinstance(value, list):
                for i in range(len(value)):
                    if isinstance(value[i], dict):
                        value[i] = DotDict(value[i])
            if isinstance(value, dict):
                value = DotDict(value)
            self[key] = value


def torch_save_wav_to_binary(audio, sr=24000):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    buffer_ = io.BytesIO()
    torchaudio.save(buffer_, audio, sr, format="wav")

    buffer_.seek(0)
    return buffer_.read()


@lru_cache(maxsize=2)
def load_cached_models(config, ckpt=None):
    logging.info("***** start loading model *****")
    if ckpt is None:
        over_rides = {}
    else:
        over_rides = {"extra_params": {"partial_pretrain": ckpt}}
    try:
        with open(config, "r", encoding="utf-8") as f:
            configs = DotDict(load_hyperpyyaml(f, overrides=over_rides))
    except Exception as err:
        logging.error(err)
        sys.exit(0)
    logging.info(f"model config: {configs}")

    pl_module = configs.pl_module
    extra_params = configs.extra_params
    trainer = configs.trainer
    batch_transform = configs.data.predict_batch_transform

    pl_module.to(device).eval()

    pl_module.trainer = trainer
    pl_module.setup('predict')
    logging.info("***** setup completed *****")
    
    logging.info("***** load model success *****")
    return pl_module, extra_params, trainer, batch_transform


class M8inferenceInterface:
    def __init__(self, ckpt=None, config='hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/ziqian/configs/m8/infer_s002_genreinstructure_freeform_callback.yaml'):
        init_start = time.time()
        # 若为hdfs config，先下载至本地
        if config.startswith('hdfs://'):
            os.system(f"hdfs dfs -get {config} ./")
            config = config.split('/')[-1]
        self.pl_module, self.extra_params, self.trainer, self.batch_transform = load_cached_models(config, ckpt)
        if hasattr(self.pl_module, 'semantic_module'):
            self.pl_module.semantic_module.__class__.__del__ = lambda self: None
        logging.info(f"init model cost {time.time() - init_start} s")

    def process(self, 
                request
        ):
        output_dir = os.path.join('./results', request['index'])
        
        self.pl_module.predict_step_seed = request['seed']
        self.pl_module.extra_params.controller_cfg_gamma = request['cfg']

        prompt = request['prompt']
        lyrics = request['lyrics']
        if '~' in prompt:
            prompt = ''
        if '~' in lyrics:
            lyrics = ''
        positive_prompt = prompt.split('|')[0]
        # get the second one if len > 1 else ""
        negative_prompt = prompt.split('|')[1] if len(prompt.split('|')) > 1 else ""
        positive_lyrics = lyrics.split('|')[0]
        negative_lyrics = lyrics.split('|')[1] if len(lyrics.split('|')) > 1 else ""

        batch_in = [{
            'input_strings_pl': [request['prompt'], request['lyrics']],
            'input_strings_p': [request['prompt']],
            'input_strings_l': [request['lyrics']],
            'input_strings_n': [],
            'input_strings': [positive_prompt, positive_lyrics],
            'input_strings_uncond': [negative_prompt, negative_lyrics]
        }]
        batch_out = {}

        for transform in self.batch_transform:
            if transform['type'] == 'FullTokenCollate':
                config_without_type = {k: v for k, v in transform.items() if k != 'type'}
                collator = FullTokenCollate(**config_without_type)
            collator(batch_in, batch_out)

        for key, value in batch_out.items():
            if isinstance(value, torch.Tensor):
                batch_out[key] = value.to(device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                batch_out[key] = [v.to(device) for v in value]

        precision_plugin = self.trainer.precision_plugin

        for callback in self.trainer.callbacks:
            if hasattr(callback, 'on_predict_start'):
                try:
                    callback.on_predict_start(self.trainer, self.pl_module)
                except Exception as e:
                    logging.warning(f"Callback {type(callback).__name__} on_predict_start failed: {e}")

        # 手动执行predict_step

        with torch.no_grad():
            for callback in self.trainer.callbacks:
                if hasattr(callback, 'on_predict_batch_start'):
                    try:
                        callback.on_predict_batch_start(self.trainer, self.pl_module, batch_out, 0)
                    except Exception as e:
                        logging.warning(f"Callback {type(callback).__name__} on_predict_batch_start failed: {e}")
            
            with precision_plugin.forward_context():
                pred = self.pl_module.predict_step(batch_out, 0)

            output_wav = pred['generated_audio'][0]
            sr = self.extra_params.sample_rate
            bytes_wav = torch_save_wav_to_binary(output_wav.cpu().float(), sr)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'output.wav'), 'wb') as f:
                f.write(bytes_wav)
        return None

    def __call__(
            self,
            request: dict
    ):
        return self.process(request)


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Debug server listening on all interfaces, port 5678")
    # debugpy.wait_for_client()  
    prompt = 'An aggressive trap track built on a looping, melancholic melodic sample and hard-hitting 808s, featuring a dynamic arrangement that shifts between intense, fast-paced rap verses and a more atmospheric, sung outro with layered vocals. hip hop, trap rap, gritty texture, intense workout session, unapologetic, rain-slicked city streets at night, confrontational, energetic, dynamic, sharp high-end, consistent dynamics, monotonic, speech-like, f:min, vocal-forward mix, compressed dynamics, syncopated kick, trap beat, slow tempo, 4/4 time signature, duration duration 245, deep, nightclub, non-vocal|vocal'
    lyrics = '[intro]\n[verse]\n[inst]\n[verse]\n[chorus]\n[verse]\n[chorus]\n[outro]'
    input = {
        'prompt': prompt,
        'lyrics': lyrics,
        'index': 'test',
        'seed': 1234,
        'cfg': 3.0
    }
    
    start_time = time.time()
    
    client = M8inferenceInterface(config="sc002_vocoderv2_posneg_webdemo.yaml")
    
    start_time = time.time()
    client(input)
    print(f"inference cost {time.time() - start_time} s")
    breakpoint()

    start_time = time.time()
    input['index'] = 'test2'
    input['seed'] = 1235
    input['cfg'] = 3.0
    client(input)
    print(f"inference cost {time.time() - start_time} s")
