import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pandas as pd
import soundfile as sf
import time

REGION = "CN"
if REGION == "US":
    # NOTE US TOS CONFIG
    TOS_PATH = "  http://tosv.byted.org/obj/cb-bucket-us/"
    TOS_bucket = "cb-bucket-us"
    accessKey = "RJ40ZMJQBZ27UYZX48R1"
else:
    # NOTE CN TOS CONFIG
    TOS_PATH = "  http://tosv.byted.org/obj/cb-bucket/"
    TOS_bucket = "cb-bucket"
    accessKey = "XN651TDBI79VVQR4JON3"
# Init TOS
import bytedtos

tos = bytedtos.Client(TOS_bucket, accessKey, timeout=99999999999)

def upload_one_file_to_tos(key: str, data: str):
    """
    key: the expected tos url, e.g. the TOS url will be
        f"  http://tosv.byted.org/obj/cb-bucket-us/{key}"
    data: the file path of data to be uploaded
    """
    try:
        tos.head_object(key)
        print(f"File {key} already exist")
    except:
        # if we don't have this file on the cloud, upload it
        try:
            tos.put_object(key, open(data, "rb"))
            time.sleep(1)
        except:
            print(f"Bad {data}")
import os
upload_audio_files = os.listdir("audio_res_1214")
for audio_file in tqdm(upload_audio_files):
    audio_path = f"audio_res_1214/{audio_file}"
    audio_key = audio_file
    upload_one_file_to_tos(audio_key, audio_path)
