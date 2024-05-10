import os
from bytedance import easycycle
import uuid
import datetime
import time
import torchaudio
import io


def retry(times, delay=1):
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
    tos_cli = try_get_toscli()
    os.system(f'{tos_cli} -bucket {bucket} -accessKey {ak} put -prefix {prefix} {audio_fp}')
    file_name = os.path.basename(audio_fp)
    url = f'https://tosv.byted.org/obj/{bucket}/{prefix}/{file_name}'
    return url
