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
    url = easycycle.upload_data_and_get_public_url(
        easycycle.Host.US, 'wangtuo.todd', data, space_name, file_name, expires
    )
    return url
