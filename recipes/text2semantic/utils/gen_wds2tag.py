import sys, os
import json

url_lst_wds_path = sys.argv[1]
wds2tag_path = sys.argv[2]

dataname2tag = {
    "librilight": 2,
    "rp27_99": 2,
    "11labs": 3,
    "BigSpeech_EN-TTS": 1,
    "bc2013": 2,
    "duibiao/Angel_conversation": 1,
    "duibiao/Corey": 1,
    "duibiao/DaceyNew0717": 1,
    "duibiao/Dacey_conversation": 1,
    "duibiao/SherrieNew0725": 1,
    "libritts_clean_460": 2,
    "BigSpeech_ZH-TTS": 1,
    "duibiao/DaceyGPT": 1,
    "duibiao/DaceyNew0801": 1,
    "duibiao/Dacey_emotion": 3,
    "duibiao/Dina": 1,
    "duibiao/MimicJason": 3,
    "duibiao/MimicTim": 3,
    "duibiao/MimicTim_short": 1,
    "duibiao/Sherrie": 1,
    "duibiao/Tim_emotion": 3,
    "duibiao/Tim_normal": 1,
    "duibiao/lyf_0810": 1,
    "duibiao/taozi_1700": 1,
    "duibiao/taozi_chatgpt": 1,
    "duibiao/taozi_conversation": 1,
    "duibiao/taozi_emotion": 3,
    "fanqie_20230725": 2,
    }

wds2tag = dict()
urls = [x.strip() for x in open(url_lst_wds_path).readlines()]
for url in urls:
    tag = None
    for dataname in dataname2tag.keys():
        if dataname in url:
            tag = dataname2tag[dataname]
    
    if tag:
        wds2tag[url] = tag
    else:
        print(url, "not found tag, exit.")
        exit()

with open(wds2tag_path, "w") as f_w:
    json.dump(wds2tag, f_w, ensure_ascii=False, indent=2)