"""
ASR-related services.

Function recipes.bigmusic.utils.asr.run_asr_lyrics_sa_online is also a wrapper for the ASR service,
but some of the parameters are hard-coded.
"""

import io
import logging
import requests
import time
import uuid
from typing import Optional, Union

import torch

from recipes.bigmusic.utils.audio_utils import audio_array_to_bytes
from recipes.bigmusic.scripts.tos import upload_to_easycycle

logger = logging.getLogger(__file__)


class ASRTransformError(Exception):
    pass


class ASROnlineConfig:
    BASE_URL = "http://speech.byted.org/api/v1/vc"
    APP_ID = "bkseig7309i0"
    TOKEN = "lv_token"
    ACCESS_TOKEN = "YLp0eHdvLZH_IvgNSEHQrHsBeTdliqe0"
    CAPTION_TYPE = "singing"


class ForceAlignConfig:
    BASE_URL = "http://speech.byted.org/api/v1/vc"
    APP_ID = "ks9x6edfo8x1h0ts"
    TOKEN = "lQpuXhcIeG_khGJRPdKK-5XpsR7Z6pDm"
    CAPTION_TYPE = "singing"


class ASRTransform:
    def __init__(self, language: str = "zh-CN", sample_rate: Optional[int] = None):
        self.language = language
        self.sample_rate = sample_rate

    def __call__(
        self, wav: Union[torch.Tensor, bytes], sample_rate: Optional[int] = None
    ) -> dict:
        if isinstance(wav, torch.Tensor):
            sample_rate = sample_rate or self.sample_rate
            if sample_rate is None:
                raise ASRTransformError(
                    "sample_rate must be provided when wav is a tensor"
                )
            wav = _tensor_to_bytes(wav, sample_rate)
        return _run_asr_lyrics_sa_online(wav, language=self.language)


class ForceAlignTransform:
    def __init__(self, sample_rate: Optional[int] = None):
        self.sample_rate = sample_rate

    def __call__(
        self, wav: Union[torch.Tensor, bytes], text: str, sample_rate: Optional[int] = None
    ) -> dict:
        if isinstance(wav, torch.Tensor):
            sample_rate = sample_rate or self.sample_rate
            if sample_rate is None:
                raise ASRTransformError(
                    "sample_rate must be provided when wav is a tensor"
                )
            wav = _tensor_to_bytes(wav, sample_rate)
        # NOTE: This is a hack to suppress the info level logging in the easycycle package.
        # If we don't do this, the entire byte string will be printed in the log.
        logger = logging.getLogger('bytedance.easycycle.bigspeech')
        logger.setLevel(logging.CRITICAL)
        url = upload_to_easycycle(wav, f"{uuid.uuid4()}", "zh_vocal_offline", "wav")
        logger.setLevel(logging.INFO)
        return _run_force_align_online(url, text)


def _run_asr_lyrics_sa_online(
    data, max_n_retries: int = 5, delay_in_sec: int = 10, language="zh-CN"
) -> dict:
    response = None
    for try_num in range(max_n_retries):
        try:
            response = requests.post(
                "{base_url}/submit".format(base_url=ASROnlineConfig.BASE_URL),
                params=dict(
                    appid=ASROnlineConfig.APP_ID,
                    token=ASROnlineConfig.TOKEN,
                    language=language,
                    caption_type=ASROnlineConfig.CAPTION_TYPE,
                    use_itn="False",
                    dirt_filter="False",
                    use_capitalize="False",
                    use_spell_correct="False",
                    with_gender_info="False",
                    with_speaker_info="False",
                    max_lines=1,
                    words_per_line=15,
                    with_confidence="True",
                    verbose="True",
                ),
                data=data,
                headers={
                    "content-type": "audio/m4a",
                    "Authorization": "Bearer; {access_token}".format(
                        access_token=ASROnlineConfig.ACCESS_TOKEN
                    ),
                },
            )
            job_id = response.json()["id"]
            response = requests.get(
                "{base_url}/query".format(base_url=ASROnlineConfig.BASE_URL),
                params=dict(
                    appid=ASROnlineConfig.APP_ID, token=ASROnlineConfig.TOKEN, id=job_id
                ),
                headers={
                    "Authorization": "Bearer; {access_token}".format(
                        access_token=ASROnlineConfig.ACCESS_TOKEN
                    )
                },
            )
            break
        except Exception as e:
            logger.info(f"ASR failed, try_num={try_num}, error={e}")
            time.sleep(delay_in_sec)
    if response is None:
        raise ASRTransformError(f"ASR failed after {max_n_retries} retries")
    return response.json()


def _run_force_align_online(
    url, gt_lyrics, max_n_retries: int = 5, delay_in_sec: int = 10
):
    for try_num in range(max_n_retries):
        try:
            response = requests.post(
                "{base_url}/ata/submit".format(base_url=ForceAlignConfig.BASE_URL),
                params=dict(
                    appid=ForceAlignConfig.APP_ID,
                    token=ForceAlignConfig.TOKEN,
                    caption_type=ForceAlignConfig.CAPTION_TYPE,
                    with_confidence="True",
                    verbose="True",
                ),
                json={
                    # "url": "https://tosv.byted.org/obj/tostest/mingfei_zh_music_sample_30.wav",
                    # "audio_text": "曾梦想仗剑走天涯,看一看世界的繁华,年少的心总有些轻狂,如今你四海为家"
                    "url": url,
                    "audio_text": gt_lyrics,
                },
                headers={
                    "content-type": "application/json",
                    "Authorization": "Bearer; {access_token}".format(
                        access_token=ForceAlignConfig.TOKEN
                    ),
                },
            )
            job_id = response.json()["id"]
            time.sleep(3)
            response = requests.get(
                "{base_url}/query".format(base_url=ForceAlignConfig.BASE_URL),
                params=dict(
                    appid=ForceAlignConfig.APP_ID,
                    token=ForceAlignConfig.TOKEN,
                    id=job_id,
                ),
                headers={
                    "Authorization": "Bearer; {access_token}".format(
                        access_token=ForceAlignConfig.TOKEN
                    )
                },
            )
            break
        except Exception as e:
            logger.info(f"Force alignment failed, try_num={try_num}, error={e}")
            time.sleep(delay_in_sec)
    if response is None:
        raise ASRTransformError(f"Force alignment failed after {max_n_retries} retries")
    return response.json()


def _tensor_to_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> io.BytesIO:
    """
    Args:
        audio_tensor: A tensor with shape of (num_channels, num_samples)
        sample_rate: The audio sample rate
    """
    audio_tensor = audio_tensor.T
    audio_numpy = audio_tensor.numpy()
    return audio_array_to_bytes(audio_numpy, sample_rate)
