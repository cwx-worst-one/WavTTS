import os
from bytedance import easycycle
import uuid
import datetime
import time
import torchaudio
import io
import logging


try:
    import bytedtos
    TOS_INSTALL = True
except:
    TOS_INSTALL = False

from bytedance.easycycle import Region, get_current_region


TOS_INFO = {
    Region.CN: {
        "bucket": "sa-music-model-zoo",
        "ak": "9NC3OBANMDH4TTPTO52E",
        "endpoint": "https://tosv.byted.org"
    },
    Region.I18n:{
        "bucket": "bigspeech-data-us",
        "ak": "XZ9QU5ORVWK",
        "endpoint": "https://tos-us.byted.org"
    }
}


def retry(times, delay=3):
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    if attempt > 0:
                        time.sleep(attempt * delay)
                    return func(*args, **kwargs)
                except Exception as ex:
                    print(f"Exception {ex} thrown when attempting to run {func}, attempt {attempt} of {times}")
                    attempt += 1
            return func(*args, **kwargs)
        return newfn
    return decorator


def audio_tensor_to_bytes(audio, sr, format="wav"):
    
    if audio.ndim == 1:
        audio = audio.unsqueeze(0).repeat((2,1))

    handle = io.BytesIO()
    torchaudio.save(handle, audio, sr, format=format)
    handle.seek(0)
    return handle.read()


@retry(times=3)
def upload_to_easycycle(data, fname, space_name=None, format="wav"):
    if space_name is None:
        space_name = datetime.datetime.now().date().strftime("%Y-%m-%d")
    
    file_name = fname + "." + uuid.uuid4().hex + "." + format
    expires = 60 * 60 * 24 * 365 * 10   # 10 years
    host = easycycle.Host.CN if os.getenv("ARNOLD_REGION", "US") == "CN" else easycycle.Host.US
    url = easycycle.upload_data_and_get_public_url(
        host, 'wangtuo.todd', data, space_name, file_name, expires
    )
    return url

@retry(times=3)
def upload_to_easycycle_v2(data, fname, space_name=None, format="wav"):
    
    file_name = fname + "." + uuid.uuid4().hex + "." + format
    expires = 60 * 60 * 24 * 180 # half year
    logger = logging.getLogger('bytedance.easycycle.bigspeech')
    logger.setLevel(logging.CRITICAL)
    url = easycycle.upload_data_and_get_public_url_v2(data, file_name, expires)
    logger.setLevel(logging.INFO)
    return url


def try_get_toscli():
    if os.path.exists("/mnt/bn/bigmusic-lf/user/zh/scripts/1.0.0.20/toscli"):
        tos_cli = "/mnt/bn/bigmusic-lf/user/zh/scripts/1.0.0.20/toscli"
    elif os.path.exists("/opt/tiger/1.0.0.20/toscli"):
        tos_cli = "/opt/tiger/1.0.0.20/toscli"
    else:
        os.system("hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/zhaohang.ai/scripts/1.0.0.20 /opt/tiger/")
        os.system("chmod +x /opt/tiger/1.0.0.20/toscli")
        tos_cli = "/opt/tiger/1.0.0.20/toscli"
    return tos_cli


def upload_to_tos(audio_fp, prefix, bucket="sa-music-model-zoo", ak="9NC3OBANMDH4TTPTO52E"):
    tos_info = TOS_INFO.get(get_current_region())
    if tos_info is None:
        raise ValueError(f"Invalid region {get_current_region()}")
    endpoint = tos_info["endpoint"]
    bucket = tos_info["bucket"]
    ak = tos_info["ak"]
    tos_cli = try_get_toscli()
    os.system(f'{tos_cli} -bucket {bucket} -accessKey {ak} put -prefix {prefix} {audio_fp}')
    file_name = os.path.basename(audio_fp)
    url = f'{endpoint}/obj/{bucket}/{prefix}/{file_name}'
    return url


@retry(times=3)
def upload_to_tos_v2(audio_fp, prefix, bucket="sa-music-model-zoo", ak="9NC3OBANMDH4TTPTO52E"):
    tos_info = TOS_INFO.get(get_current_region())
    if tos_info is None:
        raise ValueError(f"Invalid region {get_current_region()}")
    endpoint = tos_info["endpoint"]
    bucket = tos_info["bucket"]
    ak = tos_info["ak"]
    if TOS_INSTALL:
        file_name = os.path.basename(audio_fp)
        client = bytedtos.Client(bucket, ak, timeout=300, connect_timeout=300)
        url = f'{endpoint}/obj/{bucket}/{prefix}/{file_name}'
        resp = client.put_object(f"{prefix}/{file_name}", open(audio_fp, 'rb').read())
        print(f"upload {audio_fp} to {url} successful")
        return url
    else:
        return upload_to_tos(audio_fp, prefix, bucket, ak)


@retry(times=3)
def upload_obj_to_tos(object, path, verbose=False):
    tos_info = TOS_INFO.get(get_current_region())
    if tos_info is None:
        raise ValueError(f"Invalid region {get_current_region()}")
    endpoint = tos_info["endpoint"]
    bucket = tos_info["bucket"]
    ak = tos_info["ak"]
    url = f'{endpoint}/obj/{bucket}/{path}'
    client = bytedtos.Client(
        bucket, 
        ak, 
        timeout=int(os.getenv("SAMANTHA_TOS_TIMEOUT", 300)), 
        connect_timeout=int(os.getenv("SAMANTHA_TOS_CONNECT_TIMEOUT", 300)),
    )
    resp = client.put_object(path, object)
    if resp.status_code == 200:
        if verbose:
            print(f'[TOS UPLOAD] {url} successful')
        return url
    else:
        raise Exception(f'[TOS UPLOAD] {url} failed, status code: {resp.status_code}')