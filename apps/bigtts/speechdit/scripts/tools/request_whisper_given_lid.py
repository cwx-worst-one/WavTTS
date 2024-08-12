# !/usr/bin/python
# encoding=utf-8

"""
This module provide configure file management service in i18n environment.
Authors: denglelai
Date:  2024/6/27 18:34

Describe: request_whisper_given_lid.py
    * 
"""
# ASR接入文档 https://bytedance.feishu.cn/wiki/wikcnLkC8HFMSlLGF3HOTPb3OIh
import base64
import functools
import hmac
import sys
import os
import json
import logging
import multiprocessing
import subprocess
import time
import uuid
import wave
from enum import Enum
from hashlib import sha256
from io import BytesIO
from typing import List
from urllib.parse import urlparse

import requests


class AudioType(Enum):
    LOCAL = 1  # 使用本地音频文件
    URL = 2  # 使用音频url


class TransferMode(Enum):
    URL = 1  # 请求参数audio里面使用url
    DATA = 2  # 请求参数audio里面是base64编码的音频数据
    STREAM = 3  # 请求参数audio里面是base64编码的音频数据, 并且流式发包


def read_wav_info(data: bytes = None) -> (int, int, int, int, bytes):
    with BytesIO(data) as _f:
        wave_fp = wave.open(_f, 'rb')
        nchannels, sampwidth, framerate, nframes = wave_fp.getparams()[:4]
        wave_bytes = wave_fp.readframes(nframes)
    return nchannels, sampwidth, framerate, nframes, wave_bytes


def judge_wav(ori_date):
    if len(ori_date) < 44:
        return False
    if ori_date[0:4] == b"RIFF" and ori_date[8:12] == b"WAVE":
        return True
    return False


def convert_wav_with_path(audio_path, sample_rate):
    try:
        cmd = ['ffmpeg', '-v', 'quiet', '-y', '-i', audio_path, '-acodec',
               'pcm_s16le', '-ac', '1', '-ar', str(sample_rate), '-f', 'wav', '-']
        process = subprocess.run(cmd, stdout=subprocess.PIPE, timeout=60)
        if process.returncode != 0:
            return []
        return process.stdout
    except Exception as e:
        logging.warn(e)
        return []


def convert_wav_with_url(url, sample_rate):
    if str(url).startswith('https'):
        url = url.replace('https', 'http')
    cmd = ['ffmpeg', '-v', 'quiet', '-y', '-i', url, '-acodec',
           'pcm_s16le', '-ac', '1', '-ar', str(sample_rate), '-f', 'wav', '-']
    try:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, timeout=60)
        if process.returncode != 0:
            return []
        return process.stdout
    except Exception as e:
        logging.warn(e)
        return []


class AsrHttpClient:
    def __init__(self, audio_path, cluster, **kwargs):
        """
        :param config: config
        """
        # 音频路径, 本地或者音频链接
        self.audio_path = audio_path
        self.cluster = cluster
        self.success_code = 1000  # success code, default is 1000
        self.transfer_mode = kwargs.get("transfer_mode", TransferMode.DATA)
        self.audio_type = kwargs.get("audio_type", AudioType.LOCAL)
        # 如果格式是mp3、ogg, 则seg_duration是音频发送的字节数量
        self.seg_duration = int(kwargs.get("seg_duration", 15000))
        self.nbest = int(kwargs.get("nbest", 1))
        self.appid = kwargs.get("appid", "ailab_test")
        self.token = kwargs.get("token", "access_token")
        # 如果psm在lab.speech.asr下面，则base_url是https://speech-test.byted.org/api/v1/asr
        # 如果psm在lab.speech.asr_test下面，则base_url是https://speech-test.byted.org/api/v1/asr_test
        self.base_url = kwargs.get(
            "base_url", "https://speech-test.byted.org/api/v1/asr")
        self.uid = kwargs.get("uid", "ailab")
        self.workflow = kwargs.get(
            "workflow", "audio_in,resample,partition,vad,fe,decode")
        # 如果为True,则asr_service不打印日志,默认False
        self.skip_logging = kwargs.get("skip_logging", False)
        self.vad_signal = kwargs.get("vad_signal", True)
        self.show_language = kwargs.get("show_language", False)
        self.show_utterances = kwargs.get("show_utterances", False)
        self.show_word_additions = kwargs.get("show_word_additions", False)
        self.result_type = kwargs.get("result_type", "full")
        self.format = kwargs.get("format", "wav")
        self.rate = kwargs.get("sample_rate", 16000)
        self.language = kwargs.get("language", "en-US")
        self.language_tag = kwargs.get("language_tag", "<en>")
        self.bits = kwargs.get("bits", 16)
        self.channel = kwargs.get("channel", 1)
        self.codec = kwargs.get("codec", "raw")
        self.secret = kwargs.get("secret", "access_secret")
        self.auth_method = kwargs.get("auth_method", "none")
        # 热词, str格式
        self.hot_words = kwargs.get("hot_words", None)
        self.use_google = kwargs.get("use_google", False)

    @staticmethod
    def slice_data(data: bytes, chunk_size: int) -> (list, bool):
        """
        slice data
        :param data: wav data
        :param chunk_size: the segment size in one request
        :return: segment data, last flag
        """
        data_len = len(data)
        offset = 0
        while offset + chunk_size < data_len:
            yield data[offset: offset + chunk_size], False
            offset += chunk_size
        else:
            yield data[offset: data_len], True

    def construct_request(self, seq, reqid, data=None):
        req = {
            'app': {
                'appid': self.appid,
                'cluster': self.cluster,
                'token': self.token
            },
            'user': {
                'uid': self.uid
            },
            'request': {
                'reqid': reqid,
                'sequence': seq,
                'nbest': self.nbest,
                'workflow': self.workflow,
                "vad_signal": self.vad_signal,
                'skip_logging': self.skip_logging,
                'show_language': self.show_language,
                'show_utterances': self.show_utterances,
                'show_word_additions': self.show_word_additions,
                'result_type': self.result_type,
                'language_tag': self.language_tag,
            },
            'audio': {
                'format': self.format,
                'rate': self.rate,
                'language': self.language,
                'bits': self.bits,
                'channel': self.channel,
                'codec': self.codec
            }
        }
        if self.transfer_mode == TransferMode.URL:
            req['audio']['url'] = self.audio_path
        else:
            req['audio']['data'] = base64.b64encode(data).decode("utf-8")
        if self.hot_words and abs(seq) == 1:
            req["request"]['hot_words'] = self.hot_words
        if self.use_google:
            # google api 文档: https://cloud.google.com/speech-to-text/docs/basics
            req['request']["key"] = "AIzaSyAZdFKs_JgnpEpM1Lql2BmKfB3SNVa-3hY"
            req['request']['model'] = "default"  # 目前有default, command_and_search, video 三种模式, 具体解释可以找google语音api
            req['request']['workflow'] = "audio_in,resample,partition,google_asr"
        return req

    def token_auth(self):
        return {'Authorization': 'Bearer; {}'.format(self.token)}

    def signature_auth(self, data):
        header_dicts = {
            'Custom': 'auth_custom',
        }

        url_parse = urlparse(self.base_url)
        input_str = 'POST {} HTTP/1.1\n'.format(url_parse.path)
        auth_headers = 'Custom'
        for header in auth_headers.split(','):
            input_str += '{}\n'.format(header_dicts[header])
        input_str += json.dumps(data)
        mac = base64.urlsafe_b64encode(
            hmac.new(self.secret.encode('utf-8'), input_str.encode('utf-8'), digestmod=sha256).digest())
        header_dicts['Authorization'] = 'HMAC256; access_token="{}"; mac="{}"; h="custom"'.format(
            self.token, str(mac, 'utf-8'))
        return header_dicts

    def real_processor(self, request_params):
        header = None
        if self.auth_method == "token":
            header = self.token_auth()
        elif self.auth_method == "signature":
            header = self.signature_auth(request_params)
        result = requests.post(
            self.base_url, headers=header, json=request_params).json()
        return result

    def url_processor(self, reqid) -> dict:
        """
        use url mode to send request
        :return: real_processor return
        """
        request_params = self.construct_request(-1, reqid)
        result = self.real_processor(request_params)
        return result

    def segment_data_processor(self, wav_data: bytes, segment_size: int):
        reqid = str(uuid.uuid4())
        result = {}
        for seq, (chunk, last) in enumerate(AsrHttpClient.slice_data(wav_data, segment_size), 1):
            start = time.time()
            sequence = -seq if last else seq
            request_params = self.construct_request(sequence, reqid, chunk)
            result = self.real_processor(request_params)
            if result["code"] != self.success_code:
                return result
            if sequence < 0:
                break
            if self.transfer_mode == TransferMode.STREAM:
                sleep_time = max(
                    0, (self.seg_duration / 1000 - (time.time() - start)))
                time.sleep(sleep_time)
        return result

    def execute(self) -> dict:
        """
        :return: request result
        """
        reqid = str(uuid.uuid4())
        if self.transfer_mode == TransferMode.URL:
            return self.url_processor(reqid)
        if self.audio_type == AudioType.LOCAL:
            with open(self.audio_path, 'rb') as _f:
                data = _f.read()
        else:
            data = requests.get(self.audio_path).content
        audio_data = bytes(data)
        if self.format == "mp3" or self.format == "ogg":
            # mp3和ogg 按seg_duration大小的字节数量发送数据
            segment_size = self.seg_duration
            return self.segment_data_processor(audio_data, segment_size)
        if self.format == "any":
            # any格式按全量发送
            segment_size = len(data)
            return self.segment_data_processor(audio_data, segment_size)
        if self.format != "wav" and self.format != "pcm":
            raise Exception("format should in wav, pcm or mp3")
        if not judge_wav(audio_data):
            if self.audio_type == AudioType.LOCAL:
                audio_data = convert_wav_with_path(self.audio_path, self.rate)
            else:
                audio_data = convert_wav_with_url(self.audio_path, self.rate)
        nchannels, sampwidth, framerate, nframes, wave_bytes = read_wav_info(
            audio_data)
        size_per_sec = nchannels * sampwidth * framerate
        segment_size = int(size_per_sec * self.seg_duration / 1000)
        if self.format == "pcm":
            audio_data = wave_bytes
        return self.segment_data_processor(audio_data, segment_size)


def execute_one(audio_item, cluster, **kwargs):
    """

    :param audio_item: {"id": xxx, "path": "xxx"}
    :param cluster: 集群名称
    :param kwargs {  # 可选参数
        "transfer_mode": TransferMode.DATA(默认值) | TransferMode.STREAM | TransferMode.URL  # 使用音频数据传输 ｜ 流式传输 ｜ 直接使用音频URL
        "base_url" : "https://speech-test.byted.org/api/v1/asr" (默认) | "https://speech-test.byted.org/api/v1/asr_test",
        "audio_type": AudioType.LOCAL | AudioType.URL  # 根据 是否以http开头判断是否是URL类型音频
        "seg_duration": 15000 (ms) # 音频分包时长
        "workflow": "audio_in,resample,partition,vad,fe,decode"  # 默认
        "sample_rate"： 16000(默认) ｜ 8000
        "hot_words": ""  # 默认空，热词文本
    }
    :return:
    """
    assert 'id' in audio_item
    assert 'path' in audio_item
    logging.info("execute audio id: {} start".format(audio_item['id']))
    audio_id = audio_item['id']
    audio_path = audio_item['path']
    if str(audio_path).startswith("http"):
        audio_type = AudioType.URL
    else:
        audio_type = AudioType.LOCAL
    asr_http_client = AsrHttpClient(
        audio_path=audio_path,
        cluster=cluster,
        audio_type=audio_type,
        **kwargs
    )
    result = asr_http_client.execute()
    logging.info("execute audio id: {} end".format(audio_item['id']))
    return {"id": audio_id, "path": audio_path, "result": result}


def execute_multi(audio_list: List[dict], cluster, parallel: int, **kwargs):
    """
    :param audio_list: [{"id": xxx, "path": "xxx"}]
    :param cluster: 集群
    :param parallel: 10
    :param kwargs: :param kwargs {  # 可选参数
        "transfer_mode": TransferMode.DATA(默认值) | TransferMode.STREAM | TransferMode.URL  # 使用音频数据传输 ｜ 流式传输 ｜ 直接使用音频URL
        "base_url" : "https://speech-test.byted.org/api/v1/asr" (默认) | "https://speech-test.byted.org/api/v1/asr_test",
        "audio_type": AudioType.LOCAL | AudioType.URL  # 根据 是否以http开头判断是否是URL类型音频
        "seg_duration": 15000 (ms) # 音频分包时长
        "workflow": "audio_in,resample,partition,vad,fe,decode"  # 默认
        "sample_rate"： 16000(默认) ｜ 8000
        "hot_words": ""  # 默认空，热词文本
    }
    :return:
    """
    task = functools.partial(execute_one, cluster=cluster, **kwargs)
    with multiprocessing.Pool(parallel) as _pool:
        results = _pool.map(task, audio_list)
    return results


def main():
    input_dir = sys.argv[1]
    language_tag = sys.argv[2]

    audio_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            uttid = os.path.splitext(filename)[0]
            audio_path = os.path.join(input_dir, filename)
            audio_list.append({
                'id': uttid,
                'path': audio_path
            })
    print(f'Total {len(audio_list)} wav files found.')

    results = execute_multi(
        audio_list,
        cluster='vc_whisper_lid_test',
        parallel=10,
        base_url='https://speech-test.byted.org/api/v1/asr_test',
        workflow='audio_in,resample,partition,fe,decode',
        language_tag=language_tag,
    )

    output_file = os.path.join(input_dir, 'whisper_hyp.txt')
    with open(output_file, 'w') as f:
        cnt = 0
        for result in results:
            if cnt == 0:
                cnt = 1
            else:
                f.write('\n')
            uttid = result['id']
            try:
                seg_hyp_list = [res_seg['text'] for res_seg in result['result']['result']]
                hyp = ' '.join(seg_hyp_list)
            except Exception as e:
                logging.warning(f'Error while processing {result}, set hyp to be empty string.', e)
                hyp = ''
            f.write(f'{uttid} {hyp}')
            cnt += 1


logging.basicConfig(level=logging.INFO)
if __name__ == '__main__':
    main()
