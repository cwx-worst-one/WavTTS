# -*- coding:utf-8 -*-


import euler

euler.install_thrift_import_hook()
import json
from base_thrift import *
from sami_new_app_thrift import *
import argparse

client = euler.Client(SamiService, 'sd://lab.sami.gateway?cluster=release_thrift', timeout=30000)
# client = euler.Client(SamiService, 'tcp://10.174.241.189:9224', timeout=3000000) # debug
# ppe_env = "ppe_video_transcribe"
# ppe_env = "boe_samipyl"
ppe_env = ""

def SmartPunct(json_input):
    req = InvokeRequest(
        Base=Base(),
        access_key="LfpkmuCMVP",
        method="SmartPunct",
        payload=json.dumps(json_input)
    )

    if ppe_env != "":
        req.Base.Extra = {
            "env": ppe_env
        }

    result = client.Invoke(req)
    #print(result)
    #print(result.BaseResp.StatusCode)
    #print(result.payload)
    return result, result.BaseResp.StatusCode, result.payload


if __name__ == '__main__':
    #SmartPunct(None)
    #'''
    import time
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_text', type=str, required='True')
    parser.add_argument('--out_text', type=str, required='True')
    args = parser.parse_args()

    with open(args.in_text) as f:
        lines = [l.strip() for l in f]
    names = [l.split('\t')[0] for l in lines]
    texts = [l.split('\t')[1] for l in lines]

    new_text = []
    f = open(args.out_text, 'w')
    for i in tqdm(range(len(texts))):
        name = names[i]
        text = texts[i].replace(', ', ' ')
        retry = 0
        while retry < 10:
            try:
                json_input = {
                    "language": "en",
                     "text": text,
                    "extra": {
                        "enable_true_case": False,
                        "maximum_seq_len": 100
                    }
                }
                res, code, payload= SmartPunct(json_input)
                x = eval(payload)
                assert code == 0
                r = x['results'][0]['text']
                new_text.append([name, r])
                f.write("{}\t{}\n".format(name, r))
                break
            except:
                print("Retry {}...".format(name))
                retry += 1
                time.sleep(0.1)
        time.sleep(0.05)
    #'''