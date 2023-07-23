# coding=utf-8
import os.path as osp
import json
from collections import OrderedDict
from tqdm import tqdm
import euler
euler.install_thrift_import_hook()
from .server.sami_thrift import SAMI, InvokeRequest
from .server.base_thrift import Base
import os

_client = None
_base = None

GATEWAYS = [
    'sd://lab.sami.gateway',
    'sd://lab.sami.gateway.service.hl'
]



def InvokeServer(file_id, text, speaker):
    payload_obj = {
        'audio_info': {'format': 'wav', 'sample_rate': 24000, 'pitch_rate': 0, 'speech_rate': 0, 'speaker': speaker,
                       'need_alignment': True, "silence_duration": 0},
        # 'text': '??',
        # 'text': '��������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������á�?'
        # 'text': ''''1. You're listening to Faith Radio Online-Simply to Relax, I'm Faith. When you're faced with so many negative and draining situations, realize how minuscule problems will seem when you view your life as a whole--and remember the positive things.''',
        # 'text': '''Although you'll need a warm coat weather this time of year hardly ever dips below freezing For warmer weather without throngs of tourists and the sweltering humidity come in May or September High average temperatures flit between the mid-70s and the lower 80s''',
        'text': text,
    }
    payload_str = json.dumps(payload_obj)

    global _base
    if _base is None:
        _base = Base()
    req = InvokeRequest(
        Base=_base,
        access_key="flKJmCtkYc",
        method="TTS",
        payload=payload_str,
    )

    global _client
    if _client is None:
        for gateway in GATEWAYS:
            _client = euler.Client(SAMI, gateway + '?cluster=release_thrift', timeout=1200)
            result = _client.Invoke(req)
            if result.BaseResp.StatusMessage == 'ServerFailedInvoke':
                continue
            else:
                break
    result = _client.Invoke(req)
    return result.data, file_id, result.BaseResp.StatusMessage


def parse_raw_text(text_filepath):
    text_dict = OrderedDict()
    f = open(text_filepath)
    lines = f.readlines()
    print("lines: ", len(lines))
    for index, line in enumerate(lines):
        metas = line.strip().split('\t')
        if len(metas) == 2:
            text_dict[metas[0]] = metas[1]
        else:
            text_dict[f'{index:08}'] = metas[0]
    return text_dict

def generate_tacolabels_from_textstr(text:str, language='Chinese'):
    if language == 'Chinese' or language == 'English':
        speaker = 'front_end'
    elif language == 'Chinese_new' or language == 'English_new':
        speaker = 'front_end_en'
    elif language == 'Japanese':
        speaker = 'front_end_jp'
    elif language == 'BrazilPortuguese':
        speaker = 'front_end_bp'
    elif language == 'SouthKorean':
        speaker = 'front_end_kr'
    else:
        raise ValueError('language error : {}'.format(language))
    
    lab_data, file_id, invoke_response = InvokeServer(None, text, speaker)
    if lab_data is None:
        print(f'file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)')
    return lab_data

def generate_tacolabels_from_text(text_filepath, lab_output_dir, language='Chinese'):
    os.makedirs(lab_output_dir, exist_ok=True)
    text_dict = parse_raw_text(text_filepath)
    if language == 'Chinese' or language == 'English':
        speaker = 'front_end'
    elif language == 'Japanese':
        speaker = 'front_end_jp'
    elif language == 'BrazilPortuguese':
        speaker = 'front_end_bp'
    elif language == 'SouthKorean':
        speaker = 'front_end_kr'
    else:
        raise ValueError('language error : {}'.format(language))
    sucess_labs = []
    for file_id, text in tqdm(text_dict.items()):
        output_path = osp.join(lab_output_dir, f'{file_id}.lab')
        if os.path.exists(output_path):
            sucess_labs.append(osp.abspath(output_path))
            continue
        lab_data, file_id, invoke_response = InvokeServer(file_id, text, speaker)
        if lab_data is None:
            print(f'file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)')
            continue
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(lab_data)
        sucess_labs.append(osp.abspath(output_path))
    return sucess_labs


def generate_tacolabels_from_text_by_split(text_filepath, utt2split, lab_output_dir_prefix, language='Chinese'):
    # mkdir_or_exist(lab_output_dir)
    text_dict = parse_raw_text(text_filepath)
    if language == 'Chinese' or language == 'English':
        speaker = 'front_end'
    elif language == 'Japanese':
        speaker = 'front_end_jp'
    elif language == 'BrazilPortuguese':
        speaker = 'front_end_bp'
    elif language == 'SouthKorean':
        speaker = 'front_end_kr'
    else:
        raise ValueError('language error : {}'.format(language))
    sucess_labs = []
    print("text_dict.items(): ", len(text_dict.items()))
    for file_id, text in tqdm(text_dict.items()):
        if file_id not in utt2split.keys():
            print(f'file_id {file_id} not in utt2split, skip.')
            continue
        split = utt2split[file_id]
        lab_output_dir = osp.join(lab_output_dir_prefix, split)
        os.makedirs(lab_output_dir, exist_ok=True)
        output_path = osp.join(lab_output_dir, f'{file_id}.lab')
        if os.path.exists(output_path):
            continue
        lab_data, file_id, invoke_response = InvokeServer(file_id, text, speaker)
        if lab_data is None:
            print(f'file_id {file_id} failed to get results. Status: {invoke_response}(`speaker` represents language)')
            continue
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(lab_data)
        sucess_labs.append(osp.abspath(output_path))
    return sucess_labs