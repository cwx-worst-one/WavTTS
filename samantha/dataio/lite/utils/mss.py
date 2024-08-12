import os
from time import time

import euler
import thriftpy2

euler.install_thrift_import_hook()


base_thrift = thriftpy2.load(
    os.path.join(os.path.dirname(__file__), "./server/base.thrift")
)
sami_thrift = thriftpy2.load(
    os.path.join(os.path.dirname(__file__), "./server/sami.thrift")
)

client = euler.Client(
    sami_thrift.SamiService,
    "sd://lab.sami.gateway?cluster=release_thrift",
    timeout=1200,
)


def mss(input_wav, output_wav):
    b = base_thrift.Base()

    input_data = open(input_wav, "rb").read()

    # data by url
    req = sami_thrift.InvokeRequest(
        Base=b,
        data=input_data,
        access_key="FeFJSBZgnq",
        method="MusicSourceSeparate",  # MSS
        payload="""{
            "model": "mel_4track_vocal",
            "extra":{"output_sample_rate":44100,"output_channel":1,"output_format":"wav"}
        }""",
    )
    invoke_res = client.Invoke(req)

    print(invoke_res.BaseResp.StatusCode)
    print(len(invoke_res.data))
    with open(output_wav, "wb") as f:
        f.write(invoke_res.data)


if __name__ == "__main__":
    input_dir = "/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/edit_test/EN_44k"
    output_vocal_dir = (
        "/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/edit_test/EN_vocal_44k"
    )
    output_acc_dir = (
        "/mnt/bn/jdy-lq-2/bigtts-nar/repo/bigtts_testset/edit_test/EN_acc_44k"
    )
    os.makedirs(output_vocal_dir, exist_ok=True)
    os.makedirs(output_acc_dir, exist_ok=True)

    wav_names = os.listdir(input_dir)
    for wav_name in wav_names:
        if not wav_name.endswith(".wav"):
            continue
        mss(os.path.join(input_dir, wav_name), os.path.join(output_vocal_dir, wav_name))
