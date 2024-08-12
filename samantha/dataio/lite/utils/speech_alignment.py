import json
import os
import re
import time
import wave

import bytedance.context
import euler
import logid
import soundfile as sf
import thriftpy2
from scipy.signal import resample

from recipes.voicebox.datasets.utils.frontend import phone_to_int

base_thrift = thriftpy2.load(
    os.path.join(os.path.dirname(__file__), "./server/base.thrift")
)
sami_thrift = thriftpy2.load(
    os.path.join(os.path.dirname(__file__), "./server/sami.thrift")
)

euler.install_thrift_import_hook()


def get_euler_client(timeout=180):
    client = euler.Client(
        sami_thrift.SamiService,
        "sd://lab.sami.gateway?cluster=release_thrift",
        transport="ttheader",
        timeout=timeout,
    )
    bytedance.context.set("logid", logid.generate_v2())
    return client


ppe_env = ""

"""
    LanguagePTBR Language = "pt-BR"
    LanguageJP   Language = "jp"
    LanguageID   Language = "id"
    LanguageES   Language = "es"
"""


#
# !! 具体调用参数见 https://bytedance.larkoffice.com/wiki/wikcnfSENNBUR41CJhYuQ7fZ5Ce
def SpeechAlignment(audio_data, input_json):
    req = sami_thrift.InvokeRequest(
        Base=base_thrift.Base(),
        access_key="SKrwIGqYYx",
        method="SpeechAlignment",
        payload=json.dumps(input_json),
    )

    if ppe_env != "":
        req.Base.Extra = {"env": ppe_env}

    req.data = audio_data

    cnt = 0
    while cnt < 50:
        try:
            resp = get_euler_client().Invoke(req)
        except Exception as e:
            print(e)
            return None
        if resp.BaseResp.StatusMessage == "OK":
            return resp
        cnt += 1
        time.sleep(2)
    print("Run out of cnt")
    return None


def remove_last_digits(input_str):
    # 使用正则表达式匹配最后一个数字
    pattern = r"\d+$"
    match = re.search(pattern, input_str)
    if match:
        # 去掉最后一个数字
        output_str = input_str[: match.start()]
    else:
        output_str = input_str
    return output_str


def GetAlignment(wav_path, text):
    with open(wav_path, mode="rb") as file:  # b is important -> binary
        audio_data = file.read()

    input_json = {
        "model": "alignment_tts",
        "extra": {"text": text, "enable_phone": True},
    }
    result = SpeechAlignment(audio_data, input_json)
    result = json.loads(result.payload)

    words = result["results"][0]["alternatives"][0]["words"]
    phones = result["results"][0]["alternatives"][0]["phones"]
    words = [
        (x["word"], x["start_time"], x["end_time"], x["confidence"]) for x in words
    ]
    phones = [
        (phone_to_int[remove_last_digits(x["phone"])], x["start_time"], x["end_time"])
        for x in phones
    ]
    return words, phones


if __name__ == "__main__":
    # 16k wav!!!
    filePath = "HE-dongbei_gran.wav"
    tmp_filePath = "tmp/HE-dongbei_gran.wav"
    tmp_filePath = "tmp.wav"

    filePath = "/mnt/bn/jdy-lq-2/caorong/data/dit/overdub/dit_long_video_test_50/split_wav/Dit_07_en_05_0000.wav"

    wav, sr = sf.read(filePath)
    if sr != 16000:
        wav = resample(wav, int(len(wav) * 16000 / sr))
        # wav = (wav * 32767).astype("int16")
    sf.write(tmp_filePath, wav, 16000)
    # audio_data = wav.tobytes()
    # import pdb; pdb.set_trace()

    with open(filePath, mode="rb") as file:  # b is important -> binary
        audio_data = file.read()

    input_json = {
        # "lang": "enzh",
        "model": "alignment_tts",
        "extra": {
            "text": (
                "Hey, guys. Today i'm sharing three incredible ways to clean with pine salt. "
                "If you've heard of all of these, i will be shocked. Yes, i pour this directly "
                "into my washing machine. Stick around. You don't want to miss it. "
            ),
            "enable_phone": "true",
        },
    }
    result = SpeechAlignment(audio_data, input_json)
    print(result.BaseResp.StatusCode)
    print(json.loads(result.payload))
