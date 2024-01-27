# coding=utf-8
# flake8: noqa
import json
import os.path as osp
from collections import OrderedDict

import euler
from tqdm import tqdm

euler.install_thrift_import_hook()
import os

from samantha.dataio.lite.utils.punctuation import punctuation_all
from samantha.utils.sami_tacolabel.tacofrontend.server.base_thrift import Base
from samantha.utils.sami_tacolabel.tacofrontend.server.sami_thrift import (
    SAMI,
    InvokeRequest,
)

_client = None
_base = None

GATEWAYS = ["sd://lab.sami.gateway", "sd://lab.sami.gateway.service.hl"]


def InvokeServer(file_id, text, speaker):
    payload_obj = {
        "audio_info": {
            "format": "wav",
            "sample_rate": 24000,
            "pitch_rate": 0,
            "speech_rate": 0,
            "speaker": speaker,
            "need_alignment": True,
            "silence_duration": 0,
        },
        # 'text': '??',
        # 'text': '��������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������á�?'
        # 'text': ''''1. You're listening to Faith Radio Online-Simply to Relax, I'm Faith. When you're faced with so many negative and draining situations, realize how minuscule problems will seem when you view your life as a whole--and remember the positive things.''',
        # 'text': '''Although you'll need a warm coat weather this time of year hardly ever dips below freezing For warmer weather without throngs of tourists and the sweltering humidity come in May or September High average temperatures flit between the mid-70s and the lower 80s''',
        "text": text,
    }
    payload_str = json.dumps(payload_obj)

    global _base
    if _base is None:
        _base = Base()
    req = InvokeRequest(
        Base=_base, access_key="flKJmCtkYc", method="TTS", payload=payload_str
    )

    global _client
    if _client is None:
        for gateway in GATEWAYS:
            _client = euler.Client(
                SAMI, gateway + "?cluster=release_thrift", timeout=1200
            )
            result = _client.Invoke(req)
            if result.BaseResp.StatusMessage == "ServerFailedInvoke":
                continue
            else:
                break
    result = _client.Invoke(req)
    return result.data, file_id, result.BaseResp.StatusMessage


def InvokeServerPunc(file_id, text, speaker):
    payload_obj = {
        "audio_info": {
            "format": "wav",
            "sample_rate": 24000,
            "pitch_rate": 0,
            "speech_rate": 0,
            "speaker": speaker,
            "need_alignment": True,
            "silence_duration": 0,
        },
        "internal": {"lab_version": "V3", "enable_recover_puncts": True},
        # 'text': '??',
        # 'text': '��������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������á�?'
        # 'text': ''''1. You're listening to Faith Radio Online-Simply to Relax, I'm Faith. When you're faced with so many negative and draining situations, realize how minuscule problems will seem when you view your life as a whole--and remember the positive things.''',
        # 'text': '''Although you'll need a warm coat weather this time of year hardly ever dips below freezing For warmer weather without throngs of tourists and the sweltering humidity come in May or September High average temperatures flit between the mid-70s and the lower 80s''',
        "text": text,
    }
    payload_str = json.dumps(payload_obj)

    global _base
    if _base is None:
        _base = Base()
    req = InvokeRequest(
        Base=_base, access_key="flKJmCtkYc", method="TTS", payload=payload_str
    )

    global _client
    if _client is None:
        for gateway in GATEWAYS:
            _client = euler.Client(
                SAMI, gateway + "?cluster=release_thrift", timeout=1200
            )
            result = _client.Invoke(req)
            if result.BaseResp.StatusMessage == "ServerFailedInvoke":
                continue
            else:
                break
    result = _client.Invoke(req)
    return result.data, file_id, result.BaseResp.StatusMessage


def InvokeServerSplitText(file_id, text, speaker, max_paragraph_phoneme_size=240):
    payload_obj = {
        "speaker": speaker,
        # "internal": {"max_paragraph_phoneme_size": max_paragraph_phoneme_size},
        # 'text': '??',
        # 'text': '��������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������á�?'
        # 'text': ''''1. You're listening to Faith Radio Online-Simply to Relax, I'm Faith. When you're faced with so many negative and draining situations, realize how minuscule problems will seem when you view your life as a whole--and remember the positive things.''',
        # 'text': '''Although you'll need a warm coat weather this time of year hardly ever dips below freezing For warmer weather without throngs of tourists and the sweltering humidity come in May or September High average temperatures flit between the mid-70s and the lower 80s''',
        "text": text,
        "enable_text_seg": True,
    }
    payload_str = json.dumps(payload_obj)

    global _base
    if _base is None:
        _base = Base()
    req = InvokeRequest(
        Base=_base, access_key="oTVoyjjBAU", method="TTS", payload=payload_str
    )

    global _client
    if _client is None:
        for gateway in GATEWAYS:
            _client = euler.Client(
                SAMI, gateway + "?cluster=release_thrift", timeout=1200
            )
            result = _client.Invoke(req)
            if result.BaseResp.StatusMessage == "ServerFailedInvoke":
                continue
            else:
                break
    result = _client.Invoke(req)
    return (
        json.loads(result.payload)["text_segmentation"],
        file_id,
        result.BaseResp.StatusMessage,
    )


def parse_raw_text(text_filepath):
    text_dict = OrderedDict()
    f = open(text_filepath)
    lines = f.readlines()
    print("lines: ", len(lines))
    for index, line in enumerate(lines):
        metas = line.strip().split("\t")
        if len(metas) == 2:
            text_dict[metas[0]] = metas[1]
        else:
            text_dict[f"{index:08}"] = metas[0]
    return text_dict


def generate_tacolabels_from_textstr(text: str, language="Chinese"):
    if language == "Chinese" or language == "English":
        speaker = "front_end"
    elif language == "Chinese_new":
        speaker = "front_end_zh"
    elif language == "English_new":
        speaker = "front_end_en"
    elif language == "Japanese":
        speaker = "front_end_jp"
    elif language == "BrazilPortuguese":
        speaker = "front_end_bp"
    elif language == "SouthKorean":
        speaker = "front_end_kr"
    else:
        raise ValueError("language error : {}".format(language))

    lab_data, file_id, invoke_response = InvokeServer(None, text, speaker)
    if lab_data is None:
        print(
            f"file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)"
        )
    return lab_data


def generate_tacolabels_from_text(text_filepath, lab_output_dir, language="Chinese"):
    os.makedirs(lab_output_dir, exist_ok=True)
    text_dict = parse_raw_text(text_filepath)
    if language == "Chinese" or language == "English":
        speaker = "front_end"
    elif language == "Japanese":
        speaker = "front_end_jp"
    elif language == "BrazilPortuguese":
        speaker = "front_end_bp"
    elif language == "SouthKorean":
        speaker = "front_end_kr"
    else:
        raise ValueError("language error : {}".format(language))
    sucess_labs = []
    for file_id, text in tqdm(text_dict.items()):
        output_path = osp.join(lab_output_dir, f"{file_id}.lab")
        if os.path.exists(output_path):
            sucess_labs.append(osp.abspath(output_path))
            continue
        lab_data, file_id, invoke_response = InvokeServer(file_id, text, speaker)
        if lab_data is None:
            print(
                f"file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)"
            )
            continue
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(lab_data)
        sucess_labs.append(osp.abspath(output_path))
    return sucess_labs


def generate_tacolabels_from_text_by_split(
    text_filepath, utt2split, lab_output_dir_prefix, language="Chinese"
):
    # mkdir_or_exist(lab_output_dir)
    text_dict = parse_raw_text(text_filepath)
    if language == "Chinese" or language == "English":
        speaker = "front_end"
    elif language == "Japanese":
        speaker = "front_end_jp"
    elif language == "BrazilPortuguese":
        speaker = "front_end_bp"
    elif language == "SouthKorean":
        speaker = "front_end_kr"
    else:
        raise ValueError("language error : {}".format(language))
    sucess_labs = []
    print("text_dict.items(): ", len(text_dict.items()))
    for file_id, text in tqdm(text_dict.items()):
        if file_id not in utt2split.keys():
            print(f"file_id {file_id} not in utt2split, skip.")
            continue
        split = utt2split[file_id]
        lab_output_dir = osp.join(lab_output_dir_prefix, split)
        os.makedirs(lab_output_dir, exist_ok=True)
        output_path = osp.join(lab_output_dir, f"{file_id}.lab")
        if os.path.exists(output_path):
            continue
        lab_data, file_id, invoke_response = InvokeServer(file_id, text, speaker)
        if lab_data is None:
            print(
                f"file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)"
            )
            continue
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(lab_data)
        sucess_labs.append(osp.abspath(output_path))
    return sucess_labs


def generate_tacolabels_from_textstr_punc(text: str, language="Chinese_v3_punc"):
    if language == "Chinese_v3_punc":
        speaker = "front_end_zh"
    elif language == "English_v3_punc":
        speaker = "front_end_en"
    elif language == "Japanese_v3_punc":
        speaker = "front_end_jp"
    else:
        raise ValueError("language error : {}".format(language))

    lab_data = None
    while lab_data is None:
        lab_data, file_id, invoke_response = InvokeServerPunc(
            None, text, speaker
        )  # TODO: add japan
        if lab_data is None:
            print(
                f"file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)"
            )
            print(f"file_id {file_id} retry...")
    return lab_data


def text_all_punc(text):
    for x in text:
        if x not in punctuation_all:
            return False
    return True


def split_text_engine(
    text: str, language="Chinese_v3_punc", max_paragraph_phoneme_size=240
):
    if language == "Chinese_v3_punc":
        speaker = "front_end_zh"
    elif language == "English_v3_punc":
        speaker = "front_end_en"
    else:
        raise ValueError("language error : {}".format(language))

    texts, file_id, invoke_response = InvokeServerSplitText(
        None, text, speaker, max_paragraph_phoneme_size=max_paragraph_phoneme_size
    )
    if texts is None:
        print(
            f"file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)"
        )
    texts = [t for t in texts if len(t) > 0]
    # punc
    new_texts = []
    for text in texts:
        if not text_all_punc(text):
            new_texts.append(text)
    texts = new_texts
    return texts


if __name__ == "__main__":
    text = "好啊，我觉得《追捕野蛮人》就不错，适合长时间工作之后放松身心。它讲的是一个为了不被抓进儿童收容所的“熊孩子”，和一个为了不被抓进监狱的怪叔叔的森林逃亡之旅。里面有顺手拈来的戏谑、回味无穷的冷幽默，比如，瑞奇被寄养在贝拉家的时候，他趁天黑逃跑，但因为不熟悉地形，折腾了一晚上，发现只跑出去不到200米，醒来后贝拉就坐在他旁边盯着他，让人忍俊不禁，另外，在这部影片中你还可以欣赏新西兰壮丽的风景，把你工作中的疲惫一扫光。"
    # text = "当回答\"What are your weaknesses?\"时，可以说： \"My weakness is that I tend to be overly critical of my own work, but I've been working on improving my self-confidence and trusting my abilities more.\" 这里的关键词是\"weaknesses\"（弱点）、\"overly\"（过于）、\"critical\"（批评）、\"self-confidence\"（自信）、\"trusting\"（信任）。"
    # text = "Hey there! Long time no see! How've you been? I heard you landed that awesome job at the tech startup. That's killer! So, spill the beans, what's it like working there? Um Do you get to wear one of those cool hoodies every day, or what? I'm still stuck in the corporate grind, you know how it is. Anyway, Um I wanted to ask if you're up for a camping trip this summer. Remember the good ol' days when we used to go camping every summer? Let's relive those memories, man! Ah We can roast marshmallows, tell ghost stories, and just chill by the campfire. What do you say? Yep, it's been too long! Hmmm, those campfire stories were epic, LOL! And oops, I almost forgot to mention, I heard you became a master at grilling, so we'll definitely need your skills for some tasty BBQ. Yay, this trip is going to be amazing!"
    texts = split_text_engine(
        text, language="Chinese_v3_punc", max_paragraph_phoneme_size=240
    )
    # texts = split_text_engine(text, language='Chinese_v3_punc', max_paragraph_phoneme_size=20)
    # texts = split_text_engine(text, language='English_v3_punc', max_paragraph_phoneme_size=240)
    print(f"{len(texts)}, {texts}")
