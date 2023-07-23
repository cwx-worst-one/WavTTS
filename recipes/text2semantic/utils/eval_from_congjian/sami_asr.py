import euler
import json
euler.install_thrift_import_hook()
from server.sami_thrift import SAMI, InvokeRequest
from server.base_thrift import Base
import sys

client = euler.Client(SAMI, 'sd://lab.sami.gateway?cluster=release_thrift', timeout=1200)
    
b = Base()

def sami_asr(wav_path):
    req = InvokeRequest(
        Base=b,
        access_key="xEdeQmXyuj",
        method="ASR",
        data=open(wav_path, "rb").read(),
        payload='{"lang": "en"}'
    )
    result = client.Invoke(req)
    data = json.loads(result.payload)
    text = [x["text"] for x in data['results']]
    text = "".join(text)
    return text

if __name__ == '__main__':
    wav_path = sys.argv[1]
    text = sami_asr(wav_path)
#    text = sami_asr("/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/testset/youtube/prompt3.wav")
    print(text)
