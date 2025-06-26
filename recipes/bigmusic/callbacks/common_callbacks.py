import pytorch_lightning as pl
from tqdm import tqdm
import logging
import requests
from typing import Any, Union
from pathlib import Path
import json
import uuid
from recipes.bigmusic.utils.format_utils import update_json, load_json_locked
from recipes.bigmusic.scripts.tos import upload_to_easycycle
from recipes.bigmusic.utils.upload_to_easycycle import Host, upload_data_and_get_public_url
import re
import random
import time
import datetime
import os
from functools import wraps
import warnings
import inspect
from multiprocessing.pool import ThreadPool
import numpy as np
from bytedance import easycycle

def metadata_check_decorator(func):
    """装饰器：检查 metadata.json 是否包含指定字段，兼容没有 output_dir 参数的情况"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        print("Checking dependency...")

        # 获取方法的参数签名
        func_params = inspect.signature(func).parameters

        # 判断 output_dir 是否在参数列表中
        if 'output_dir' in func_params:
            output_dir = kwargs.get('output_dir', None)  # 优先从关键字参数中获取
            if output_dir is None:
                try:
                    output_dir = args[list(func_params).index('output_dir')]  # 从位置参数获取
                except IndexError:
                    output_dir = None
        else:
            output_dir = None  # 这个方法本来就没有 output_dir 这个参数

        # 如果 output_dir 仍然为空，尝试从 pl_module 获取
        if output_dir is None and 'pl_module' in func_params:
            try:
                pl_module = kwargs.get('pl_module', args[list(func_params).index('pl_module')])
                output_dir = getattr(pl_module, 'extra_params', {}).get('output_dir', None)
            except (IndexError, AttributeError):
                output_dir = None

        # 如果 output_dir 仍然是 None，则不执行检查，直接调用原方法
        if output_dir is None:
            return func(self, *args, **kwargs)

        # 进行 metadata.json 文件检查
        generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
        
        for generated_output_fp in generated_output_fps:
            metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
            
            try:
                with open(metadata_fp, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"⚠️ [WARNING] Metadata file {metadata_fp} not found or invalid JSON. Ensure it exists before proceeding.")
                raise FileNotFoundError
            
            # 需要检查的字段
            required_fields = ['easycycle_url', 'force_align', 'asr_lyrics']
            missing_fields = [field for field in required_fields if field not in metadata]

            if missing_fields:
                print(
                    f"⚠️ [WARNING] Metadata file {metadata_fp} is missing fields: {missing_fields}. "
                    "Ensure you have run ForceAlignCallback, UploadToEasyCycleCallback, and ASRCallback."
                )
                raise FileNotFoundError
        
        return func(self, *args, **kwargs)  # 继续执行原方法
    
    return wrapper

class ForceAlignCallback(pl.Callback):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def run_force_align_online(url, gt_lyrics, verbose=True):
        base_url = 'http://speech.byted.org/api/v1/vc'
        appid = 'ks9x6edfo8x1h0ts'
        token = 'lQpuXhcIeG_khGJRPdKK-5XpsR7Z6pDm'
        caption_type = "singing"
        max_retry_num = 5
        try_num = 0
        base_delay = 2
        if verbose:
            print(url)
        gt_lyrics = re.sub(r'\[.+?\]\n', '', gt_lyrics)
        gt_lyrics = gt_lyrics.replace('\n', ',')
        if verbose:
            print(gt_lyrics)
        while try_num < max_retry_num:
            try:
                print(f"Try num: {try_num}")
                data = {
                    # "url": "https://tosv.byted.org/obj/tostest/mingfei_zh_music_sample_30.wav",
                    # "audio_text": "曾梦想仗剑走天涯,看一看世界的繁华,年少的心总有些轻狂,如今你四海为家"
                    "url": url,
                    "audio_text": gt_lyrics
                }
                response = requests.post(
                                '{base_url}/ata/submit'.format(base_url=base_url),
                                params=dict(
                                    appid=appid,
                                    token=token,
                                    caption_type=caption_type,                                 
                                    with_confidence='True',
                                    verbose='True'
                                ),
                                json=data,
                                headers={
                                'content-type': 'application/json',
                                "Authorization": 'Bearer; {access_token}'.format(access_token=token)
                                }
                            )
                job_id = response.json()['id']
                print(response)
                time.sleep(3)
                response = requests.get(
                        '{base_url}/query'.format(base_url=base_url),
                        params=dict(
                            appid=appid,
                            token=token,
                            id=job_id,
                        ),
                        headers={
                            "Authorization": 'Bearer; {access_token}'.format(access_token=token)
                        }
                )
                print(response)
                aligned_utts = response.json()['utterances'][0]
                try_num = max_retry_num + 1      
            except Exception as e:
                try_num += 1
                if try_num < max_retry_num:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** (try_num - 1)) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    print(f"Request failed after {max_retry_num} retries.")
                    aligned_utts = []
        return aligned_utts 
        
    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None):
        
        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        if (trainer is None             # for offline mode
            or trainer.is_global_zero): # for multi-GPU mode

            print("Begin to call force align model...")

            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()  

            for i, generated_output_fp in enumerate(tqdm(generated_output_fps)):

                # get lyrics
                metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                metadata = json.load(open(metadata_fp, 'r', encoding='utf-8'))
                lyrics = metadata['lyrics']

                # check if exists
                if 'force_align' in metadata and isinstance(metadata['force_align'], dict):
                    print(f"Force align already exists for {generated_output_fp}")
                    continue

                with open(generated_output_fp, "rb") as f:
                    wav = f.read()

                # NOTE: This is a hack to suppress the info level logging in the easycycle package.
                # If we don't do this, the entire byte string will be printed in the log.
                logger = logging.getLogger('bytedance.easycycle.bigspeech')
                logger.setLevel(logging.CRITICAL)
                url = metadata.get("easycycle_url", None)
                logger.setLevel(logging.INFO)
                if not lyrics.endswith('\n'):
                    lyrics += '\n'
                asr_timestamp = self.run_force_align_online(url, lyrics, verbose=False)

                if len(str(asr_timestamp)) < 10:
                    print(f"Force align failed for {generated_output_fp}")

                update_json(metadata_fp, { 'force_align': asr_timestamp })

    def run_parallel(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None, parallel=10):
        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir
        if (trainer is None or trainer.is_global_zero):
            print("Begin to call force align model...")

            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()

            def process_single_file(generated_output_fp):
                metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')
                metadata = load_json_locked(metadata_fp)
                lyrics = metadata['lyrics']

                # check if exists
                if 'force_align' in metadata and isinstance(metadata['force_align'], dict):
                    print(f"Force align already exists for {generated_output_fp}")
                    return

                logger = logging.getLogger('bytedance.easycycle.bigspeech')
                logger.setLevel(logging.CRITICAL)
                url = metadata.get("easycycle_url", None)
                logger.setLevel(logging.INFO)

                if not lyrics.endswith('\n'):
                    lyrics += '\n'
                asr_timestamp = self.run_force_align_online(url, lyrics, verbose=False)

                if len(str(asr_timestamp)) < 10:
                    print(f"Force align failed for {generated_output_fp}")

                update_json(metadata_fp, { 'force_align': asr_timestamp })

            def process_group(file_list):
                for fp in file_list:
                    process_single_file(fp)

            pool = ThreadPool(parallel)
            file_groups = np.array_split(generated_output_fps, parallel)
            rets = []
            for group in file_groups:
                ret = pool.apply_async(process_group, args=(group,))
                rets.append(ret)
            pool.close()
            for ret in rets:
                ret.get()
            pool.join()
            print("Force align done.")

# if __name__ == "__main__":
#     import argparse
#     callback = ForceAlignCallback()
#     callback.on_predict_end(None, None, "/mnt/bn/music-llm-nas-lq/zh/output/v5_sft_freeform/rl_b_1k1/zh_200_n20_seed1000_v2/minp/t0.9_p0.1_group_cfg/minp_temp0.8_pbase0.075_20250326-1015544576/")

class UploadToEasyCycleCallback(pl.Callback):
    def __init__(self, space_name=None, expires=None) -> None:
        super().__init__()
        self.space_name = space_name
        self.expires = expires

    @staticmethod
    def upload_file(filepath, space_name=None, expires=None):
        with open(filepath, "rb") as f:
            wav = f.read()
        
        _, format = os.path.splitext(os.path.basename(filepath))
        file_name = "my_file_" + uuid.uuid4().hex + format

        if expires is None:
            expires = 3600 * 24 * 1000

        logger = logging.getLogger('bytedance.easycycle.bigspeech')
        logger.setLevel(logging.CRITICAL)
        url = easycycle.upload_data_and_get_public_url_v2(wav, file_name, expires)
        logger.setLevel(logging.INFO)
        return url

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None):

        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir
        if (trainer is None             # for offline mode
            or trainer.is_global_zero): # for multi-GPU mode
            print("Begin to upload to easycycle...")
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()
            for i, generated_output_fp in enumerate(tqdm(generated_output_fps)):
                # get lyrics
                metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')

                metadata = json.load(open(metadata_fp, 'r', encoding='utf-8'))

                if 'easycycle_url' in metadata:
                    print(f"Easycycle url already exists for {generated_output_fp}")
                    continue

                url = self.upload_file(generated_output_fp, self.space_name, self.expires)

                update_json(metadata_fp, { 'easycycle_url': url })

    def run_parallel(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None, parallel=10):
        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        if (trainer is None or trainer.is_global_zero):
            print("Begin to upload to EasyCycle...")
            start = time.time()

            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()

            def process_single_file(generated_output_fp):
                metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')
                metadata = load_json_locked(metadata_fp)
                if 'easycycle_url' in metadata:
                    print(f"Easycycle url already exists for {generated_output_fp}")
                    return
                url = self.upload_file(generated_output_fp, self.space_name, self.expires)
                update_json(metadata_fp, {'easycycle_url': url})

            def process_group(file_list):
                for fp in file_list:
                    process_single_file(fp)

            logging.basicConfig(level=logging.WARNING)

            pool = ThreadPool(parallel)
            file_groups = np.array_split(generated_output_fps, parallel)
            rets = []
            for group in file_groups:
                ret = pool.apply_async(process_group, args=(group,))
                rets.append(ret)
            pool.close()
            for ret in rets:
                ret.get()
            pool.join()

            logging.basicConfig(level=logging.INFO)
            print(f"All audios uploaded to EasyCycle, cost {time.time() - start} seconds.")

# if __name__ == "__main__":
#     import argparse
#     callback = UploadToEasyCycleCallback()
#     callback.on_predict_end(None, None, "/mnt/bn/music-llm-nas-lq/yixiao/cn_lyrics2song_High-quality")

class ASRCallback(pl.Callback):
    def __init__(self, asr_model_path="zh-CN") -> None:
        super().__init__()
        self.asr_model_path = asr_model_path 

    @staticmethod
    def run_asr_lyrics_sa_online(filepath, language='zh-CN'):
        #base_url = 'http://speech-test.byted.org/api/v1/vc'
        #appid = "api_dev"
        #token = "lv_token"
        base_url = 'http://speech.byted.org/api/v1/vc'
        appid = 'bkseig7309i0'
        token = 'lv_token'
        access_token = 'YLp0eHdvLZH_IvgNSEHQrHsBeTdliqe0'
        #caption_type = "singing"
        caption_type = "auto"
        max_retry_num = 5
        try_num = 0
        base_delay = 2

        text = ''
        while try_num < max_retry_num:
            try:
                with open(filepath, 'rb') as fp:
                    data = fp.read()
                    response = requests.post(
                                '{base_url}/submit'.format(base_url=base_url),
                                params=dict(
                                    appid=appid,
                                    token=token,
                                    language=language,
                                    caption_type=caption_type,
                                    use_itn='False',
                                    dirt_filter='False',
                                    use_capitalize='False',
                                    use_spell_correct='False',
                                    with_gender_info='False',
                                    with_speaker_info='False',
                                    max_lines=1,
                                    words_per_line=15,
                                    with_confidence='True',
                                    verbose='True'
                                ),
                                data=data,
                                headers={
                                    'content-type': 'audio/m4a',
                                    "Authorization": 'Bearer; {access_token}'.format(access_token=access_token)
                                }
                            )
                    job_id = response.json()['id']
                    response = requests.get(
                            '{base_url}/query'.format(base_url=base_url),
                            params=dict(
                                appid=appid,
                                token=token,
                                id=job_id,
                            ),
                            headers={
                                "Authorization": 'Bearer; {access_token}'.format(access_token=access_token)
                            }
                    )
                    for item in response.json()['utterances']:
                        text += item['text'] + ' '
                    try_num = max_retry_num + 1
            except Exception as e:
                try_num += 1
                if try_num < max_retry_num:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** (try_num - 1)) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    print(f"Request failed after {max_retry_num} retries.")
                    aligned_utts = []
        return text.strip(), None

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None):

        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        if (trainer is None             # for offline mode
            or trainer.is_global_zero): # for multi-GPU mode

            print("Begin to call asr model...")

            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()  

            for i, generated_output_fp in enumerate(tqdm(generated_output_fps)):

                # get lyrics
                metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
                metadata = json.load(open(metadata_fp, 'r', encoding='utf-8'))

                if 'asr_lyrics' in metadata:
                    print(f"ASR already exists for {generated_output_fp}")
                    continue

                asr_lyrics, timestamps = self.run_asr_lyrics_sa_online(generated_output_fp, self.asr_model_path)

                update_json(metadata_fp, {'asr_lyrics': asr_lyrics})

                if timestamps is not None:
                    update_json(metadata_fp, {'asr_timestamps': timestamps})

    def run_parallel(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", output_dir=None, parallel=10):
        
        if output_dir is None:
            output_dir = pl_module.extra_params.output_dir

        if (trainer is None or trainer.is_global_zero):

            print("Begin to call asr model...")

            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
            generated_output_fps.sort()

            def process_single_file(generated_output_fp):
                metadata_fp = str(generated_output_fp).replace('generated.wav','metadata.json')

                metadata = load_json_locked(metadata_fp)

                if 'asr_lyrics' in metadata:
                    print(f"ASR already exists for {generated_output_fp}")
                    return

                asr_lyrics, timestamps = self.run_asr_lyrics_sa_online(generated_output_fp, self.asr_model_path)
                
                update_json(metadata_fp, {'asr_lyrics': asr_lyrics})

                if timestamps is not None:
                    update_json(metadata_fp, {'asr_timestamps': timestamps})

            def process_group(file_list):
                for fp in file_list:
                    process_single_file(fp)

            pool = ThreadPool(parallel)
            file_groups = np.array_split(generated_output_fps, parallel)
            rets = []
            for group in file_groups:
                ret = pool.apply_async(process_group, args=(group,))
                rets.append(ret)
            pool.close()
            for ret in rets:
                ret.get()
            pool.join()
            



# if __name__ == "__main__":
#     import argparse
#     callback = UploadToEasyCycleCallback()
#     callback.run_parallel(None, None, "/mnt/bn/music-llm-nas-lq/zoupei/infer_results/250429_v5_rltice/rl_pitch_700")
