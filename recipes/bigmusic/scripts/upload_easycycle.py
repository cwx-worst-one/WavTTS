from bytedance import easycycle
import uuid
import datetime
import time
import io
import pandas as pd
from pathlib import Path
import os
import glob


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


def load_file_to_bytes(filepath):
    with open(filepath, "rb") as f:
        b = f.read()
    return b


@retry(times=3)
def upload_to_easycycle(data, fname, space_name=None, format="wav"):
    if space_name is None:
        space_name = datetime.datetime.now().date().strftime("%Y-%m-%d")

    file_name = fname + "." + uuid.uuid4().hex + "." + format
    expires = 60 * 60 * 24 * 365 * 10   # 10 years
    url = easycycle.upload_data_and_get_public_url(
        easycycle.Host.CN, 'wangtuo.todd', data, space_name, file_name, expires
    )
    return url



if __name__ == "__main__":
    # The dir and prefix string are customizable    
    # audio_fps = Path("assets/cnVocalSongTest_SunoV3/set1").glob("*.mp3")
    audio_fps = Path("generated/eval0418_120_1min_cfg3").glob("*/*.wav")
    prefix = "product_v2_engine"

    mapping = {
        "file_name": [],
        "url": [],
    }

    for audio_fp in audio_fps:
        # command = "ffmpeg-normalize '%s' -t -16 --keep-loudness-range-target -c:a libmp3lame -o '%s' -f" % (audio_fp, audio_fp)
        # command = "ffmpeg-normalize '%s' -t -16 --keep-loudness-range-target -o '%s' -f" % (audio_fp, audio_fp)
        # os.system(command)
        audio_data = load_file_to_bytes(audio_fp)        
        filename = str(Path(audio_fp).stem)
        url = upload_to_easycycle(audio_data, f"{prefix}_{filename}", "wav")
        mapping["file_name"].append(filename)
        mapping["url"].append(url)
        print(f"Uploaded: {filename} to {url}")

    # Save the results to CSV
    df = pd.DataFrame.from_dict(mapping)
    df.to_csv(f"{prefix}.csv")